import copy
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub


class QuantizedModelWrapper(nn.Module):
    def __init__(self, model):
        super(QuantizedModelWrapper, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        return self.dequant(x)


# Method that performs static quantization on a model
def static_quantization(model, dataset, backend="qnnpack", fuse=False):
    # Decouple the quantized model from the original model
    model = copy.deepcopy(model)

    # Wrap the model in a QuantizedModelWrapper
    quantized_model = QuantizedModelWrapper(model)

    # Fuse layers if specified
    if fuse:
        modules_to_fuse = get_modules_to_fuse(model)
        torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)

    # Set the backend to use for quantization
    # 'fbgemm' for server (x86), 'qnnpack' for mobile (ARM)
    # TODO: Create option to set backend in front-end, e.g. "what device will the model be deployed on?"
    model.qconfig = torch.quantization.get_default_qconfig(backend)

    # Prepare the model for quantization
    torch.quantization.prepare(quantized_model, inplace=True)

    # Calibrate with the training set
    print("Calibrating...")
    calibrate(quantized_model, dataset.train_loader)
    print("Calibration complete.")

    # Convert the model to a quantized model
    torch.quantization.convert(quantized_model, inplace=True)

    return quantized_model


# Method that calibrates a model for quantization
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            model(data)


# Method to find modules to fuse
# Uses a depth-first search approach to traverse the model's hierarchy and identify layers to fuse
def get_modules_to_fuse(model, modules_to_fuse=None, prefix=""):
    # Initialize layers_to_fuse as an empty list if not provided
    if modules_to_fuse is None:
        modules_to_fuse = []

    # Iterate over the named children (layers) of the current model/module
    for name, layer in model.named_children():
        # Construct the full layer name by appending the current name to the prefix
        layer_name = f"{prefix}{name}"

        # Check if the current layer is a Conv2d or Linear layer
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            # If it's the first layer or not a direct successor of the previous layer,
            # create a new list to store the fusion candidate layers
            if (len(modules_to_fuse) == 0) or (modules_to_fuse[-1][-1] != layer_name):
                modules_to_fuse.append([layer])

        # Check if the current layer is a BatchNorm2d or ReLU layer
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.ReLU):
            # If there are layers in layers_to_fuse, append the current layer to the last group
            if modules_to_fuse:
                modules_to_fuse[-1].append(layer)

        # Recursively call find_layers_to_fuse for child layers, updating the prefix
        get_modules_to_fuse(layer, modules_to_fuse, prefix=f"{layer_name}.")

    # Filter out lists with less than 2 layers (not a valid fusion candidate)
    modules_to_fuse = [group for group in modules_to_fuse if len(group) >= 2]

    return modules_to_fuse


# Method that performs dynamic quantization on a model
def dynamic_quantization(model, backend="qnnpack", layers_to_quantize={torch.nn.Linear}, dtype=torch.qint8):
    # Decouple the quantized model from the original model
    model = copy.deepcopy(model)

    # Set the model to evaluation mode
    model.eval()

    # Set the backend to use for quantization
    torch.backends.quantized.engine = backend

    # Quantize the model using dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,                 # Model to be quantized
        layers_to_quantize,    # Set of layers to quantize
        dtype=dtype            # Data type for quantized weights
    )

    return quantized_model
