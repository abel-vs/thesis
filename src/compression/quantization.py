import copy
from enum import Enum
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, FakeQuantize
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import QConfigMapping
from tqdm import tqdm
import general
from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig, QConfigMapping


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




def static_quantization(model, dataset, backend="x86", device=None, calibration_cap=20):
    # Decouple the quantized model from the original model
    quantized_model = copy.deepcopy(model)

    example_inputs = general.get_example_inputs(dataset.train_loader, device=device)

    # Set the backend to use for quantization
    # 'fbgemm' for server (x86), 'qnnpack' for mobile (ARM)
    # TODO: Create option to set backend in front-end, e.g. "what device will the model be deployed on?"

    # Set qconfig for the entire model to use per_tensor_affine
    qconfig = get_default_qconfig(backend)
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    
    # Prepare the model for quantization
    quantized_model = prepare_fx(quantized_model, qconfig_mapping, example_inputs)

    # Calibrate with the training set
    calibrate(quantized_model, dataset.train_loader, cap=calibration_cap, device=device)

    # Set the model to cpu
    quantized_model.cpu()

    # Convert the model to a quantized model
    quantized_model = convert_fx(quantized_model)
    

    return quantized_model

# Method that calibrates a model for quantization
def calibrate(model, data_loader, cap=None, device=None):
    if device is None:
        device = general.get_device()
    model.to(device)
    model.eval()
    with torch.no_grad():
        i = 0
        for data, _ in tqdm(data_loader, desc="Calibration", position=0, leave=True, dynamic_ncols=True):
            data = data.to(device)
            model(data)
            if cap is not None:
                i += 1
                if i >= cap:
                    break
    print("Calibration complete.")


def fuse_modules(model):
    modules_to_fuse = get_modules_to_fuse(model)
    torch.quantization.fuse_modules(model, modules_to_fuse, inplace=True)


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
                modules_to_fuse.append([layer_name])

        # Check if the current layer is a BatchNorm2d or ReLU layer
        if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.ReLU):
            # If there are layers in layers_to_fuse, append the current layer to the last group
            if modules_to_fuse:
                modules_to_fuse[-1].append(layer_name)

        # Recursively call find_layers_to_fuse for child layers, updating the prefix
        get_modules_to_fuse(layer, modules_to_fuse, prefix=f"{layer_name}.")

    # Filter out lists with less than 2 layers (not a valid fusion candidate)
    modules_to_fuse = [group for group in modules_to_fuse if len(group) >= 2]

    return modules_to_fuse


def is_quantized(model):
    if isinstance(model, QuantizedModelWrapper):
        return True
    for module in model.modules():
        if isinstance(module, QuantStub) or isinstance(module, DeQuantStub) or isinstance(module, FakeQuantize):
            return True
    return False

# Method that performs dynamic quantization on a model
def dynamic_quantization(model, dataset, backend="x86",  dtype=torch.qint8):
    # Decouple the quantized model from the original model
    model = copy.deepcopy(model)

    example_inputs = general.get_example_inputs(dataset.train_loader)

    # qconfig_mapping = (QConfigMapping()
    #     .set_object_type(nn.Embedding, float_qparams_weight_only_qconfig)
    #     .set_object_type(nn.LSTM, default_dynamic_qconfig)
    #     .set_object_type(nn.Linear, default_dynamic_qconfig)
    #     .set_object_type(nn.Conv2d, default_dynamic_qconfig)
    # )
    qconfig_mapping = {"": torch.quantization.default_dynamic_qconfig}  # An empty key denotes the default applied to all modules


    # Set the model to evaluation mode
    model.eval()


    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
    
    # Set the model to cpu
    prepared_model.cpu()

    quantized_model = convert_fx(prepared_model)

    # # Quantize the model using dynamic quantization
    # quantized_model = torch.quantization.quantize_dynamic(
    #     model,                 # Model to be quantized
    #     layers_to_quantize,    # Set of layers to quantize
    #     dtype=dtype            # Data type for quantized weights
    # )

    return quantized_model


def qat_prepare(model):
    # Specify the quantization configuration
    qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    # Prepare the model for QAT
    model.qconfig = qconfig
    torch.quantization.prepare_qat(model, inplace=True)

    return model


def qat_convert(model):
    model = torch.quantization.convert(model, inplace=True)
    return model