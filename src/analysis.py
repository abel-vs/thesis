from enum import Enum
import torch
import torch.nn as nn

from src.interfaces.compression_actions import CompressionAction, CompressionType, DistillationAction, PruningAction, QuantizationAction
from src.interfaces.strategies import PruningStrategy
from src.interfaces.techniques import DistillationTechnique, PruningTechnique
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
def get_layer_types_with_parameter_counts(model):
    modules = flatten_layers(model)
    types_with_params = {}
    
    for module in modules:
        module_type = type(module)
        num_parameters = eval.get_params(module)
        
        if module_type in types_with_params:
            types_with_params[module_type] += num_parameters
        else:
            types_with_params[module_type] = num_parameters
            
    return types_with_params


# Method that returns a dict with the types of layers of a model and their percentage of the total parameter count
def get_layer_types_with_parameter_percentage(model):
    types_with_params = get_layer_types_with_parameter_counts(model)
    total_params = sum(types_with_params.values())
    types_with_percentage = {}

    for layer_type, param_count in types_with_params.items():
        percentage = (param_count / total_params) * 100
        types_with_percentage[layer_type] = percentage

    return types_with_percentage




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


def analyze(
    model, 
    dataset, 
    type,
    compression_target,
    settings
):

    compression_actions = []

    resources_available = False

    if type == "performance":
        # Focus on actions that minimally affect performance
        compression_actions.append(
        PruningAction( name="Magnitude Pruning", technique=PruningTechnique.L1, sparsity=0.1, strategy=PruningStrategy.OnlyLinear, settings={
                              "performance_target": compression_target,
                              "compression_target": compression_target})
        )
        compression_actions.append(
            DistillationAction(name="Logits Distillation", technique=DistillationTechnique.HardTarget,  settings={
                            "performance_target": compression_target,
                            "compression_target": compression_target})
        )

    if type == "size":
        # Focus of fully connected layers, as they contain most of the parameters
        compression_actions.append(
        PruningAction( name="Magnitude Pruning", technique=PruningTechnique.L1, sparsity=0.5, strategy=PruningStrategy.OnlyLinear,settings={
                              "performance_target": compression_target,
                              "compression_target": compression_target, 
                              })
        )

        compression_actions.append(
            DistillationAction(name="Combined Distillation", technique=DistillationTechnique.CombinedLoss,  settings={
                            "performance_target": compression_target,
                            "compression_target": compression_target, 
                            "patience": 5})
        )

    elif type == "time":
        # Pruning of filters
        compression_actions.append(
        PruningAction( name="Magnitude Pruning", technique=PruningTechnique.L1, sparsity=0.1, strategy=PruningStrategy.OnlyConv, settings={
                              "performance_target": compression_target,
                              "compression_target": compression_target, 
                              }))


    elif type == "computations":
        # Focus on making it fit on a CPU
        compression_actions.append(
            QuantizationAction(name="INT-8 Dynamic Quantization")
        )
        compression_actions.append(
        PruningAction( name="Magnitude Pruning", technique=PruningTechnique.L1, sparsity=0.1,  strategy=PruningStrategy.OnlyConv, settings={
                              "performance_target": compression_target,
                              "compression_target": compression_target}))
    else:
        raise Exception("Unknown compression type")

    
    
    return compression_actions
