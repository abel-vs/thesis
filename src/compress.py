
import copy
import logging
import torch.nn as nn
import src.compression.quantization as quant
import src.compression.distillation as distil
import src.compression.pruning as prune
from src.interfaces.strategies import PruningStrategy
import src.plot as plot
import src.evaluation as eval

from typing import List
from src.interfaces.compression_actions import CompressionAction, DistillationAction, PruningAction, QuantizationAction


def compress_model(model, dataset, compression_actions: List[CompressionAction]):
    """Main method for compressing a model via API"""
    
    compressed_model = copy.deepcopy(model)

    for action in compression_actions:
        if type(action) == PruningAction:
            plot.print_header("PRUNING STARTED")

            old_params = eval.get_params(compressed_model)
            old_size = eval.get_size(compressed_model)

            if action.strategy == PruningStrategy.OnlyLinear:
                layers =  prune.flatten_layers(model)
                prunable_layers = [module for module in layers if isinstance(module, nn.Linear)]
            elif action.strategy == PruningStrategy.OnlyConv:
                layers =  prune.flatten_layers(model)
                prunable_layers = [module for module in layers if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))]
            else:
                prunable_layers = None
                
            compressed_model = prune.channel_pruning(compressed_model, dataset, action.technique, action.sparsity, prunable_layers=prunable_layers, **action.settings)

            new_params = eval.get_params(compressed_model)
            new_size = eval.get_size(compressed_model)
            
            logging.info("Pruning Results:")
            logging.info("Old Params: %d, New Params: %d, Reduction: %.2f", old_params, new_params, 1 - (new_params / old_params))
            logging.info("Old Size: %.2f, New Size: %.2f, Reduction: %.2f", old_size, new_size, 1 - (new_size / old_size))
        
        if type(action) == DistillationAction:
            plot.print_header("DISTILLATION STARTED")
            compressed_model = distil.perform_distillation(model, dataset, compressed_model,  action.settings)

            
        if type(action) ==  QuantizationAction:
            plot.print_header("QUANTIZATION STARTED")
            compressed_model = quant.dynamic_quantization(compressed_model)

    return compressed_model
