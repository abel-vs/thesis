
import copy
import logging
import torch.nn as nn
import src.compression.quantization as quant
import src.compression.distillation as distil
import src.compression.pruning as prune
from src.interfaces.strategies import PruningStrategy
import src.plot as plot
import src.evaluation as eval
import src.analysis as analysis

from typing import List
from src.interfaces.compression_actions import CompressionAction, DistillationAction, PruningAction, QuantizationAction, order_compression_actions
from torch.utils.tensorboard import SummaryWriter


def compress_model(model, dataset, compression_actions: List[CompressionAction], writer: SummaryWriter = None, device=None, save_path=None):
    """Main method for compressing a model via API"""
    
    compressed_model = copy.deepcopy(model)
    model.eval() # Original model shouldn't be changed

    print("Compression Actions:", compression_actions)

    compression_actions = order_compression_actions(compression_actions)

    for action in compression_actions:
        if type(action) == PruningAction:
            plot.print_header("PRUNING STARTED")

            old_params = eval.get_params(compressed_model)
            old_size = eval.get_size(compressed_model)

            if action.sparsity > 1:
                sparsity = action.sparsity / 100
            else:
                sparsity = action.sparsity 
                
            compressed_model = prune.structure_pruning(compressed_model, dataset, sparsity, action.technique,  action.strategy, action.objective, writer=writer, device=device, **action.settings)

            new_params = eval.get_params(compressed_model)
            new_size = eval.get_size(compressed_model)
            
            print("Pruning Results:")
            print("Old Params: %d, New Params: %d, Reduction: %.2f", old_params, new_params, 1 - (new_params / old_params))
            print("Old Size: %.2f, New Size: %.2f, Reduction: %.2f", old_size, new_size, 1 - (new_size / old_size))


            logging.info("Pruning Results:")
            logging.info("Old Params: %d, New Params: %d, Reduction: %.2f", old_params, new_params, 1 - (new_params / old_params))
            logging.info("Old Size: %.2f, New Size: %.2f, Reduction: %.2f", old_size, new_size, 1 - (new_size / old_size))
        
        if type(action) == DistillationAction:
            plot.print_header("DISTILLATION STARTED")
            compressed_model = distil.perform_distillation(model, dataset, technique=action.technique, student_model=compressed_model,  settings = action.settings, save_path=save_path, writer=writer, device=device)

            
        if type(action) ==  QuantizationAction:
            plot.print_header("QUANTIZATION STARTED")
            compressed_model = quant.dynamic_quantization(compressed_model)

    if writer is not None:
        writer.close()

    return compressed_model
