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
def static_quantization(model, dataset, backend="qnnpack"):
    # Decouple the quantized model from the original model
    model = copy.deepcopy(model)

    # Wrap the model in a QuantizedModelWrapper
    quantized_model = QuantizedModelWrapper(model)

    # TODO: Figure out how to fuse modules
    # modules_to_fuse = find_modules_to_fuse(model)
    # if not len(modules_to_fuse) == 0:
    #     model = torch.quantization.fuse_modules(model, modules_to_fuse)

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


def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            model(data)


# Method that performs dynamic quantization on a model
def dynamic_quantization(model, backend="qnnpack"):
    torch.backends.quantized.engine = backend
    quantized_model = torch.quantization.quantize_dynamic(
        model, dtype=torch.qint8)
    return quantized_model


# TODO: Intelligently return a list of modules to fuse
# Model Fusion combines multiple sequential modules (eg: [Conv2d, BatchNorm, ReLU]) into one.
def find_modules_to_fuse(model):
    modules_to_fuse = []
    for name, module in model.named_modules():
        pass
    return modules_to_fuse


# # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
# # This operation does not change the numerics
# def fuse_model(self):
#     for m in self.modules():
#         if type(m) == ConvBNReLU:
#             torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
#         if type(m) == InvertedResidual:
#             for idx in range(len(m.conv)):
#                 if type(m.conv[idx]) == nn.Conv2d:
#                     torch.ao.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)],inplace=True)
