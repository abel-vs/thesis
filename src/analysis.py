from enum import Enum
import torch
import torch.nn as nn

from src.interfaces.compression_actions import CompressionAction, CompressionType, DistillationAction, PruningAction, QuantizationAction
from src.interfaces.strategies import PruningStrategy
from src.interfaces.techniques import DistillationTechnique, PruningTechnique

class ModelCategory(str, Enum):
    cnn = "cnn"
    transformer = "transformer"


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
        pass
    if type == "size":
        # Focus of fully connected layers, as they contain most of the parameters
        compression_actions.append(
        PruningAction( name="Magnitude Pruning", technique=PruningTechnique.L1, sparsity=0.1, settings={
                              "performance_target": compression_target,
                              "compression_target": compression_target, "strategy": PruningStrategy.OnlyLinear})
    ),

        pass
    elif type == "time":
        # Pruning of filters
        pass
    elif type == "computations":
        # Focus on making it fit on a CPU
        compression_actions.append(
            QuantizationAction(name="INT-8 Dynamic Quantization")
        )
    else:
        raise Exception("Unknown compression type")

    
    compression_actions.append(
        PruningAction( name="Magnitude Pruning", technique=PruningTechnique.L1, sparsity=0.1, settings={
                              "performance_target": compression_target,
                              "compression_target": compression_target})
    ),

    compression_actions.append(
        DistillationAction(name="Logits Distillation", technique=DistillationTechnique.HardTarget,  settings={
                          "performance_target": compression_target,
                          "compression_target": compression_target})
    )

    return compression_actions
