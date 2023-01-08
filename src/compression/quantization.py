import torch

# Method that performs static quantization on a model
def static_quantization(model, train_loader, fbgemm=True):
    modules_to_fuse = find_modules_to_fuse(model)
    model = torch.quantization.fuse_modules(model, modules_to_fuse)
    if fbgemm:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        model.qconfig = torch.quantization.default_qconfig
    torch.quantization.prepare(model, inplace=True)
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            model(data)
    torch.quantization.convert(model, inplace=True)


# TODO: Intelligently return a list of modules to fuse
def find_modules_to_fuse(model):
    modules_to_fuse = []
    for name, module in model.named_modules():
        pass
    return modules_to_fuse