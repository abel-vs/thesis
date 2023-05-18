import copy
from enum import Enum
import logging
from typing import List
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp
import general
from src import plot
import src.analysis as analysis
from src.interfaces.dataset_models import DataSet
from src.interfaces.techniques import PruningTechnique


""" UNSTRUCTURED PRUNING """

# Method that randomly prunes a module by a given rate from 0% to 100%


def unstructured_random_pruning(model, rate):
    for module in model.children():
        prune.random_unstructured(module, name="weight", amount=rate)
        prune.remove(module, "weight")

    return model

# Method that randomly prunes a module by a given rate from 0% to 100%


def unstructured_magnitude_pruning(model, rate):
    for module in model.children():
        prune.l1_unstructured(module, name="weight", amount=rate)
        prune.remove(module, "weight")

    return model


""" STRUCTURED PRUNING """

# Method that gets first and last layer
# TODO: this method should find the final layer in a general way, we can't assume the final layer is the last module, since it depends on the forward function.


def get_first_last_layers(model):
    if isinstance(model, list):
        layers = model
    elif isinstance(model, torch.nn.Module):
        layers = analysis.flatten_layers(model)
    else:
        raise Exception("Model must be a list or torch.nn.Module")
    return [layers[0], layers[-1]]


# Method to get the layers that should be ignored.
def get_layers_not_to_prune(model):
    layers_not_to_prune = []
    previous_module = None

    first_last_layers = get_first_last_layers(model)
    layers_not_to_prune.extend(first_last_layers)

    # Flatten if not yet flattened
    if not isinstance(model, list):
        model = analysis.flatten_layers(model)

    for module in model:
        # # Skip input and output layers
        # if isinstance(module, nn.Conv2d) and (previous_module is None or isinstance(previous_module, nn.Linear)):
        #     layers_not_to_prune.append(module)
        #     continue

        # # Skip batch normalization layers
        # if isinstance(module, nn.BatchNorm2d):
        #     layers_not_to_prune.append(module)
        #     continue

        # # Skip shortcut connections and first and last layers in each residual block
        # if isinstance(module, nn.Conv2d):
        #     if isinstance(previous_module, nn.ReLU) or isinstance(previous_module, nn.Conv2d):
        #         layers_not_to_prune.append(module)
        #         continue

        previous_module = module

    return layers_not_to_prune


# Method that estimates the required channel sparsity to reach a target global sparsity
def calculate_channel_sparsity(model: nn.Module, target_global_sparsity: float) -> List[float]:
    total_parameters = 0
    total_conv_parameters = 0

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer_parameters = torch.prod(
                torch.tensor(layer.weight.size())).item()
            total_parameters += layer_parameters

            if isinstance(layer, nn.Conv2d):
                total_conv_parameters += layer_parameters

    target_conv_sparsity = (1 - (1 - target_global_sparsity)
                            * total_parameters / total_conv_parameters)

    channel_sparsity = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            num_channels = layer.out_channels
            num_channels_to_prune = int(num_channels * target_conv_sparsity)
            layer_channel_sparsity = num_channels_to_prune / num_channels
            channel_sparsity.append(layer_channel_sparsity)

    return channel_sparsity


# Method that constructs a pruner object
def get_pruner(model, example_inputs, type, ignored_layers, settings):
    if type == "random":
        imp = tp.importance.RandomImportance()
        pruner = tp.pruner.MagnitudePruner
    elif type == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner = tp.pruner.MagnitudePruner
    elif type == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner = tp.pruner.MagnitudePruner
    elif type == "slim":
        imp = tp.importance.BNScaleImportance()
        pruner = tp.pruner.MagnitudePruner
    elif type == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner = tp.pruner.GroupNormPruner
    else:
        raise NotImplementedError

    return pruner(model,
                  example_inputs,
                  importance=imp,
                  iterative_steps=settings["iterative_steps"],
                  ch_sparsity=settings["sparsity"],
                  ignored_layers=ignored_layers)


# Method that applies channel pruning using a given technique
def structure_pruning(
        model,
        dataset: DataSet,
        technique: PruningTechnique,
        sparsity: float,
        fineTune=False,
        iterative_steps=3,
        prunable_layers=None,
        optimizer=None,
        inPlace=True,
        writer=None,
        save_path=None,
        device=None,
        ignored_layers=[],
        **kwargs):
    
    if not inPlace:
        model = copy.deepcopy(model)
    model.to(device)
    example_inputs = general.get_example_inputs(dataset.train_loader, device=device)

    # TODO: optimizer should be able to be created in a smart way
    if optimizer is None:
        learning_rate = 0.1 * sparsity/iterative_steps
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    if prunable_layers is None:
        ignored_layers.extend(get_layers_not_to_prune(model))
    else:
        ignored_layers.extend(get_layers_not_to_prune(prunable_layers))

    print("Ignored layers: ", list(type(x).__name__ for x in ignored_layers))
    logging.info(
        f"Ignored layers: {list(type(x).__name__ for x in ignored_layers)}")

    # 2. Pruner initialization
    pruner = get_pruner(model, example_inputs, technique, ignored_layers, {
                        "iterative_steps": iterative_steps, "sparsity": sparsity})

    # 3. Pruning
    for i in range(iterative_steps):
        pruner.step()  # Removes the least important channels from the model
        if fineTune:
            plot.print_header(f"Pruning step {i+1}/{iterative_steps}")
            general.finetune(model, dataset, target=99, patience=3, optimizer=optimizer,
                             writer=writer, writer_tag=f"pruning_{(i+1)/iterative_steps*sparsity}", **kwargs)
            if save_path is not None:
                torch.save(
                    model, f"{save_path}/pruned_{(i+1)/iterative_steps*sparsity}.pt")

    return model
