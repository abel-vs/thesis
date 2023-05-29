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
from src.interfaces.objectives import CompressionObjective
from src.interfaces.strategies import PruningStrategy
from src.interfaces.techniques import PruningTechnique
import src.evaluation as eval

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
        imp = tp.importance.MagnitudeImportance(p=2)
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
    elif type == "group_lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner = tp.pruner.GroupNormPruner
    else:
        raise NotImplementedError

    return pruner(model,
                  example_inputs,
                  importance=imp,
                  iterative_steps=settings["iterative_steps"],
                  ch_sparsity=settings["sparsity"],
                  ignored_layers=ignored_layers,
                  global_pruning=True
                  )


# Method that applies channel pruning using a given technique
def structure_pruning(
        model,
        dataset: DataSet,
        sparsity: float,
        technique: PruningTechnique,
        strategy: PruningStrategy,
        objective: CompressionObjective,
        finetune=False,
        finetune_epochs=1,
        iterative_steps=100,
        optimizer=None,
        inplace=True,
        writer=None,
        save_path=None,
        device=None,
        ignored_layers=None,
        **kwargs):

    if not inplace:
        model = copy.deepcopy(model)
    model.to(device)
    example_inputs = general.get_example_inputs(
        dataset.train_loader, device=device)
    
    if objective == "flops":
        iterative_steps = 20

    # TODO: optimizer should be able to be created in a smart way
    if optimizer is None:
        learning_rate = 0.1 * sparsity/iterative_steps
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = set() if ignored_layers is None else set(ignored_layers.copy())
    ignored_layers.update(analysis.get_layers_to_ignore(model, strategy))

    first_layer, last_layer = analysis.get_first_last_layers(model)
    ignored_layers.update([first_layer, last_layer])

    print(f"Ignored layers: {list(type(x).__name__ for x in ignored_layers)}")

    logging.info(
        f"Ignored layers: {list(type(x).__name__ for x in ignored_layers)}")

    # 2. Pruner initialization
    pruner = get_pruner(model, example_inputs, technique, ignored_layers, {
                        "iterative_steps": iterative_steps, "sparsity": 0.9})

    if objective == "size":
        original_params = eval.get_params(model)
        target = int(original_params * (1 - sparsity))
    elif objective == "flops":
        original_flops = eval.get_flops(model, example_inputs)
        target = int(original_flops * (1 - sparsity))
    elif objective == "time":
        original_time = eval.get_inference_time(model, dataset)
        target = original_time * (1 - sparsity)
    else:
        raise NotImplementedError

    # 3. Pruning
    for i in range(iterative_steps):
        pruner.step()  # Removes the least important channels from the model

        if writer is not None:
            score = general.validate(model, dataset, device=device)[1]
            writer.add_scalars(f"score/pruning", {"Score": score}, i+1)

        if finetune:
            plot.print_header(f"Pruning step {i+1}/{iterative_steps}")
            general.finetune(model, dataset, target=99, patience=1, optimizer=optimizer,
                             writer=writer, writer_tag=f"pruning_{(i+1)/iterative_steps*sparsity}", max_it=finetune_epochs, ** kwargs)
            if save_path is not None:
                torch.save(
                    model, f"{save_path}/pruned_{(i+1)/iterative_steps*sparsity}.pt")


        if objective == "size":
            params = eval.get_params(model)
            if params < target:
                break
        elif objective == "flops":
            flops = eval.get_flops(model, example_inputs)
            if flops < target:
                break
        elif objective == "time":
            time = eval.get_inference_time(model, dataset)
            if time < target:
                break

    # 4. Remove the pruner from the model
    del pruner

    return model
