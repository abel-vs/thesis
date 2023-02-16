import torch

from compression_models import CompressionAction, CompressionType

PRUNE_PERCENTAGE = 0.5


def get_modules(model):
    modules = []
    for name, module in model.named_modules():
        if name != "":
            modules.append(module)
    return modules


def get_module_type(module):
    return type(module).__name__


def get_model_types(model):
    modules = get_modules(model)
    types = []
    for module in modules:
        types.append(get_module_type(module))
    return types


def get_conv_layers(model):
    modules = get_modules(model)
    conv_layers = []
    for module in modules:
        if get_module_type(module) == "Conv2d":
            conv_layers.append(module)
    return conv_layers


# Method that ranks the filters of convolutional layers according to their L1-norm
def rank_filters(model, data_loader, device):
    # Get the convolutional layers of the model
    conv_layers = get_conv_layers(model)

    # For each convolutional layer
    for layer in conv_layers:
        # Get the layer's weight
        weight = layer.weight.data

        # Get the L1-norm of the layer's weight
        l1_norm = torch.sum(weight.abs(), dim=(1, 2, 3))

        # Get the number of filters
        num_filters = weight.shape[0]

        # Get the number of filters to prune
        num_filters_to_prune = int(num_filters * PRUNE_PERCENTAGE)

        # Get the indices of the filters to prune
        _, filter_indices_to_prune = torch.topk(l1_norm, num_filters_to_prune)

        # Set the filters to prune to zero
        weight[filter_indices_to_prune, :, :, :] = 0

    return model


def analyze(
    model_state,
    model_architecture,
    compression_goal,
    compression_target,
    performance_metric,
    perfomance_target,
):
    """Method that analyzes the given model and returns suggested compression actions."""
    compression_actions = []
    # TODO: Implement the analyze method
    compression_actions.append(
        CompressionAction(type=CompressionType.distillation, name="Logits Distillation")
    )
    compression_actions.append(
        CompressionAction(type=CompressionType.pruning, name="Magnitude Pruning")
    )
    compression_actions.append(
        CompressionAction(
            type=CompressionType.quantization, name="INT-8 Dynamic Quantization"
        )
    )
    return compression_actions
