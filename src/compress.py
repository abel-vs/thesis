
import copy
import src.compression.quantization as quant
import src.compression.distillation as distil
import src.compression.pruning as prune
import src.plot as plot



from typing import List

from src.models.compression_actions import CompressionAction, DistillationAction, PruningAction, QuantizationAction


def compress_model(model, dataset, compression_actions: List[CompressionAction]):
    """Main method for compressing a model via API"""
    
    compressed_model = copy.deepcopy(model)
    print("Compression Actions:", list(c.name for c in compression_actions))
    for action in compression_actions:
        if type(action) == PruningAction:
            plot.print_header("PRUNING STARTED")
            compressed_model = prune.channel_pruning(compressed_model, dataset, action.technique, action.sparsity, **action.settings)
            # action_settings = action["settings"]
            # compressed_model = prune.channel_pruning(compressed_model, dataset, sparsity=action_settings.get(
            #     "sparsity"), fineTune=action_settings.get("fineTune", False), strategy="NO_CONV")
        
        if type(action) == DistillationAction:
            plot.print_header("DISTILLATION STARTED")
            compressed_model = distil.perform_distillation(model, dataset, compressed_model,  action.settings)
            
        if type(action) ==  QuantizationAction:
            plot.print_header("QUANTIZATION STARTED")
            compressed_model = quant.dynamic_quantization(compressed_model)

    return compressed_model
