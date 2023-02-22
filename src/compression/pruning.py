import torch
import torch.nn.utils.prune as prune

import torch_pruning as tp

# Method that returns the parameters to prune
# TODO: This should be done intelligently, returning the parameters that are the least important
# For now, we just return the parameters that are the smallest
def get_parameters_to_prune(model, pruning_rate):
    # Get the parameters to prune
    parameters_to_prune = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameters_to_prune.append((name, parameter))
    # Sort the parameters by their absolute value
    parameters_to_prune.sort(key=lambda x: torch.sum(torch.abs(x[1])))
    for x in parameters_to_prune:
        print(x[0], torch.sum(torch.abs(x[1])))
    # Return the parameters to prune
    return parameters_to_prune[: int(len(parameters_to_prune) * pruning_rate)]


# Method that randomly prunes a module by a given rate from 0% to 100%
def random_pruning(module, rate):
    prune.random_unstructured(module, name="weight", amount=rate)


# Method that randomly prunes a module by a given rate from 0% to 100%
def pruning_global(model, rate):
    for module in model.children():
        print("module", module)
        prune.l1_unstructured(module, name="weight", amount=rate)
        prune.remove(module, "weight")

    return model


# Method that prunes the lowest magnitude weights of a module
def magnitude_pruning(module, rate):
    prune.l1_unstructured(module, name="weight", amount=rate)


# Method that prunes the lowest magnitude weights globally
def magnitude_pruning_global(model, pruning_rate):
    parameters_to_prune = get_parameters_to_prune(model, pruning_rate)
    print("parameters_to_prune", parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )


from torchpruner.attributions import (
    RandomAttributionMetric,
)  # or any of the methods above


def get_attribution_scores(model, data_loader, criterion, device):
    attr = RandomAttributionMetric(model, data_loader, criterion, device)
    for module in model.children():
        if len(list(module.children())) == 0:  # leaf module
            scores = attr.run(module)
            print(scores)


def structured_pruning(model, example_inputs):
    # 1. build dependency graph for resnet18
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=example_inputs)

    # 2. Specify the to-be-pruned channels. Here we prune those channels indexed by [2, 6, 9].
    pruning_idxs = [2, 6, 9]
    pruning_group = DG.get_pruning_group(
        model.conv1, tp.prune_conv_out_channels, idxs=pruning_idxs
    )

    # 3. prune all grouped layer that is coupled with model.conv1 (included).
    if DG.check_pruning_group(pruning_group):  # avoid full pruning, i.e., channels=0.
        pruning_group.exec()

    return model


# TODO: Define a custom pruning method that
# Below is an example of a custom pruning method
class CustomMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor"""

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask


import mnist

# Main method
def main():
    model = mnist.MnistModel()
    model.eval()

    model = structured_pruning(model, torch.randn(1, 1, 28, 28))

    print(model)


if __name__ == "__main__":
    main()
