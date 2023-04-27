import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp
import general
from dataset_models import DataSet

""" UNSTRUCTURED PRUNING """

# Method that randomly prunes a module by a given rate from 0% to 100%


def random_pruning(model, rate):
    for module in model.children():
        prune.random_unstructured(module, name="weight", amount=rate)
        prune.remove(module, "weight")

    return model

# Method that randomly prunes a module by a given rate from 0% to 100%


def magnitude_pruning_global_unstructured(model, rate):
    for module in model.children():
        prune.l1_unstructured(module, name="weight", amount=rate)
        prune.remove(module, "weight")

    return model


""" STRUCTURED PRUNING """


# Method to get the layers that should be ignored.
# TODO: this method should find the final layer in a general way, we can't assume the final layer is the last module, since it depends on the forward function.
def get_layers_not_to_prune(model):
    layers_not_to_prune = []
    previous_module = None

    for module in model.children():
        # Skip input and output layers
        if isinstance(module, nn.Conv2d) and (previous_module is None or isinstance(previous_module, nn.Linear)):
            layers_not_to_prune.append(module)
            continue

        # Skip batch normalization layers
        if isinstance(module, nn.BatchNorm2d):
            layers_not_to_prune.append(module)
            continue

        # Skip shortcut connections and first and last layers in each residual block
        if isinstance(module, nn.Conv2d):
            if isinstance(previous_module, nn.ReLU) or isinstance(previous_module, nn.Conv2d):
                layers_not_to_prune.append(module)
                continue

        previous_module = module

    return layers_not_to_prune


def magnitude_pruning_structured(model, dataset: DataSet, sparsity: float, fineTune=False, iterative_steps=3):
    example_inputs = general.get_example_input(dataset.train_loader)

    # 0. importance criterion for parameter selections
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = get_layers_not_to_prune(model)

    # 2. Pruner initialization
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        # If False, a uniform sparsity will be assigned to different layers.
        global_pruning=False,
        importance=imp,  # importance criterion for parameter selection
        # the number of iterations to achieve target sparsity
        iterative_steps=iterative_steps,
        ch_sparsity=sparsity,
        ignored_layers=ignored_layers,
    )

    for i in range(iterative_steps):
        # pruner.step will remove some channels from the model with least importance
        pruner.step()
        if fineTune:
            general.train(model, dataset)

    return model
