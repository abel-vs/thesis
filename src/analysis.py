from enum import Enum
import torch
import torch.nn as nn
from src import metrics

from src.interfaces.compression_actions import CompressionAction, CompressionCategory, DistillationAction, PruningAction, QuantizationAction
from src.interfaces.objectives import CompressionObjective
from src.interfaces.strategies import PruningStrategy, strategy_names
from src.interfaces.techniques import DistillationTechnique, PruningTechnique, QuantizationTechnique, technique_names
import src.general as general
import src.evaluation as eval

class ModelCategory(str, Enum):
    cnn = "cnn"
    transformer = "transformer"


""" General """

# Method that returns list of layers of a model
def flatten_layers(model):
    flat_layers = []
    for layer in model.children():
        children = list(layer.children())
        if len(children) > 0:
            flat_layers.extend(flatten_layers(layer))
        else:
            flat_layers.append(layer)
    return flat_layers

def get_first_last_layers(model):
    if isinstance(model, list):
        layers = model
    elif isinstance(model, torch.nn.Module):
        layers = flatten_layers(model)
    else:
        raise Exception("Model must be a list or torch.nn.Module")
    return [layers[0], layers[-1]]


""" Analyzing Layers """

# Method that returns list of types of layers of a model
def get_layer_types_with_counts(model):
    modules = flatten_layers(model)
    types = {}
    for module in modules:
        module_type = type(module)
        if module_type in types:
            types[module_type] += 1
        else:
            types[module_type] = 1
    return types

# Method that returns list of types of layers of a model with their parameter counts
def get_layer_types_with_parameter_counts(model, ignored_layers=[]):
    modules = flatten_layers(model)
    types_with_params = {}
    
    for module in modules:
        if module in ignored_layers:
            continue

        module_type = type(module)
        num_parameters = eval.get_params(module)
        
        if module_type in types_with_params:
            types_with_params[module_type] += num_parameters
        else:
            types_with_params[module_type] = num_parameters

    types_with_params = sorted(types_with_params.items(), key=lambda x: x[1], reverse=True)
            
    return types_with_params


# Method that returns a dict with the types of layers of a model and their percentage of the total parameter count
def get_layer_types_with_parameter_percentage(model, ignored_layers=[]):
    types_with_params = get_layer_types_with_parameter_counts(model, ignored_layers=ignored_layers)
    total_params = sum([param_count for layer_type, param_count in types_with_params])
    types_with_percentage = []

    for layer_type, param_count in types_with_params:
        percentage = (param_count / total_params) * 100
        types_with_percentage.append((layer_type, percentage))

    types_with_percentage.sort(key=lambda x: x[1], reverse=True)

    return types_with_percentage


# Method that returns a list of layers to prune to reach a certain compression goal
def get_layers_to_prune(model, compression_goal, ignored_layers=[]):
    sorted_layers = get_layer_types_with_parameter_percentage(model, ignored_layers=ignored_layers)
    accumulated_percentage = 0
    layers_to_prune = []

    for layer in sorted_layers:
        # allow 5 percent at minimum to remain
        prunable_percentage = layer[1]*0.95
        accumulated_percentage += prunable_percentage
        layers_to_prune.append(layer[0])

        if accumulated_percentage >= compression_goal:
            break

    return layers_to_prune

# Method that selects a strategy based on the types of layers in a model
def get_pruning_strategy(prunable_layers, objective: CompressionObjective, compression_target: int, performance_target: int):
    has_conv = any(isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for layer in prunable_layers)
    has_linear = any(isinstance(layer, nn.Linear) for layer in prunable_layers)

    if has_conv and has_linear:
        return PruningStrategy.Global
    elif has_conv:
        return PruningStrategy.OnlyConv
    elif has_linear:
        return PruningStrategy.OnlyLinear
    else:
        return PruningStrategy.Global


def get_pruning_technique(model, dataset, objective: CompressionObjective, strategy: PruningStrategy, compression_target: int, performance_target: int, ):
    architecture = detect_model_category(model)

    if architecture == ModelCategory.cnn:
        if objective == CompressionObjective.Size:
            if strategy == PruningStrategy.OnlyLinear:
                return PruningTechnique.LAMP
            elif strategy == PruningStrategy.Global:
                return PruningTechnique.L1
            elif strategy == PruningStrategy.OnlyConv:
                return PruningTechnique.GroupNorm
            else:
                raise Exception("Unknown strategy")
        elif objective == CompressionObjective.Time:
            if strategy == PruningStrategy.OnlyLinear:
                return PruningTechnique.LAMP
            elif strategy == PruningStrategy.Global:
                return PruningTechnique.GroupNorm
            elif strategy == PruningStrategy.OnlyConv:
                return PruningTechnique.GroupNorm
            else:
                raise Exception("Unknown strategy")
        elif objective == CompressionObjective.Computations:
            if strategy == PruningStrategy.OnlyLinear:
                return PruningTechnique.LAMP
            elif strategy == PruningStrategy.Global:
                return PruningTechnique.GroupNorm
            elif strategy == PruningStrategy.OnlyConv:
                return PruningTechnique.GroupNorm
            else:
                raise Exception("Unknown strategy")
        else:
            raise Exception("Unknown objective")
    elif architecture == ModelCategory.transformer:
        if objective == CompressionObjective.Size:
            return PruningTechnique.GroupNorm
        elif objective == CompressionObjective.Computations:
            return PruningTechnique.GroupNorm
        elif objective == CompressionObjective.Time:
            return PruningTechnique.GroupNorm
        else: 
            raise Exception("Unknown objective")
    else:  # shallow or unique architecture
        return PruningTechnique.Random


def get_layers_to_ignore(model, strategy: PruningStrategy):
    if strategy == PruningStrategy.OnlyLinear:
        return get_conv_layers(model)
    elif strategy == PruningStrategy.OnlyConv:
        return get_linear_layers(model)
    else:
        return []



""" Getting Specific Layers """

def get_conv_layers(model):
    modules = flatten_layers(model)
    conv_layers = [m for m in modules if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d))]
    return conv_layers


def get_linear_layers(model):
    modules = flatten_layers(model)
    linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
    return linear_layers


def detect_model_category(model):
    is_transformer = any(isinstance(layer, (nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder)) for layer in model.modules())
    is_cnn = any(isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for layer in model.modules())

    if is_transformer:
        return ModelCategory.transformer
    elif is_cnn:
        return ModelCategory.cnn
    else:
        return None
    
def get_distillation_technique(compute_available, metric):
    print(metric)
    if not compute_available:
        return DistillationTechnique.HardTarget
    else:
        if metric in [metrics.accuracy]:
            return DistillationTechnique.HardTarget
        elif metric in ["f1", "perplexity", ""]: 
            return DistillationTechnique.SoftTarget
        else:
            return DistillationTechnique.CombinedLoss

def get_quantization_technique(backend, compute_available, performance_drop):
    if backend == "CPU":
        if compute_available: 
            return QuantizationTechnique.QAT
        else:
            if performance_drop>2:
                return QuantizationTechnique.Static
            else: 
                return QuantizationTechnique.Dynamic
        
    else:
        return None

def analyze(
    model, 
    dataset, 
    objective: CompressionObjective,
    compression_target,
    performance_target,
    settings,
    compute_available = True,
    backend = "CUDA",
    device = None,
):

    compression_actions = []

    # Calculate heuristics
    current_score = general.test(model, dataset, device=device)[1]
    performance_drop = performance_target - current_score

    first_last_layer = get_first_last_layers(model)

    layers_to_prune = get_layers_to_prune(model, compression_target, ignored_layers=first_last_layer)

    pruning_strategy = get_pruning_strategy(layers_to_prune, objective, compression_target, performance_target)

    pruning_technique = get_pruning_technique(model, dataset, objective, pruning_strategy, compression_target, performance_target)

    distillation_technique = get_distillation_technique(compute_available, dataset.metric)

    quantization_technique = get_quantization_technique(backend, compute_available, performance_drop)


    compression_actions.append(
    PruningAction( name=(strategy_names[pruning_strategy] + " " + technique_names[pruning_technique]), 
                    technique=pruning_technique, sparsity=compression_target, 
                    strategy=pruning_strategy, objective=objective, settings={
                            "performance_target": performance_target,
                            "compression_target": compression_target, 
                            })
    )

    compression_actions.append(
        DistillationAction(name=technique_names[distillation_technique], technique=distillation_technique,  settings={
                        "performance_target": performance_target,
                        "compression_target": compression_target, 
                        "patience": 1})
    )

    if quantization_technique is not None:
        compression_actions.append(
            QuantizationAction(name=technique_names[quantization_technique], technique=quantization_technique)
        )

    print(compression_actions)

    return compression_actions
2