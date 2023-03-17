import torch
import general

# Method that performs static quantization on a model
def static_quantization(model, dataset, backend="qnnpack"):
    model.eval()

    modules_to_fuse = find_modules_to_fuse(model)
    if not len(modules_to_fuse) == 0:
        model = torch.quantization.fuse_modules(model, modules_to_fuse)

    model.qconfig = torch.ao.quantization.default_qconfig
    print(model.qconfig)
    # model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    quantized_model = torch.quantization.prepare(model, inplace=False)

    # Calibrate with the training set
    general.test(model, dataset)
    print("Static Quantization: Calibration done")

    quantized_model = torch.quantization.convert(quantized_model, inplace=False)

    return quantized_model


# Method that performs dynamic quantization on a model
def dynamic_quantization(model, backend="qnnpack"):
    torch.backends.quantized.engine = backend
    quantized_model = torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
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
