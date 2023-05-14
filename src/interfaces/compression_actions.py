
import torch.nn.functional as F
from enum import Enum
from src.interfaces.strategies import PruningStrategy
from src.interfaces.techniques import DistillationTechnique, PruningTechnique, QuantizationTechnique


class CompressionType(str, Enum):
    pruning = "pruning"
    quantization = "quantization"
    distillation = "distillation"

class CompressionAction:
    """ Class that represents a compression action."""
    def __init__(self, type: CompressionType, name: str):
        self.type = type
        self.name = name


class PruningAction(CompressionAction):
    """ Class that represents a pruning action. """
    def __init__(self, name: str, technique: PruningTechnique,  sparsity: float, strategy: PruningStrategy, settings: dict = {}):
        super().__init__(CompressionType.pruning, name)

        self.technique = technique 
        self.strategy = strategy
        self.sparsity = sparsity
        self.settings = settings  # Use settings to set all the necessary kwargs

    def __str__(self):
        return f"PruningAction(name={self.name}, technique={self.technique}, sparsity={self.sparsity}, strategy={self.strategy})"
    
    def __repr__(self):
        return str(self)
    


class QuantizationAction(CompressionAction):
    """ Class that represents a quantization action. """
    def __init__(self, name: str, technique: QuantizationTechnique, settings: dict = {}):
        super().__init__(CompressionType.quantization, name)

        self.technique = technique 
        self.quantization_bits = settings.get("quantization_bits", 8)
        self.settings = settings  # Use settings to set all the necessary kwargs
    
    def __str__(self):
        return f"QuantizationAction(name={self.name}, technique={self.technique}, quantization_bits={self.quantization_bits})"
    
    def __repr__(self):
        return str(self)


class DistillationAction(CompressionAction):
    """ Class that represents a distillation action. """
    def __init__(self, name: str, technique: DistillationTechnique, settings: dict = {}):
        super().__init__(CompressionType.distillation, name)

        self.technique = technique
        self.distillation_loss = settings.get("distillation_loss", F.mse_loss)
        self.settings = settings  # Use settings to set all the necessary kwargs

    def __str__(self):
        return f"DistillationAction(name={self.name}, technique={self.technique}, distillation_loss={self.distillation_loss})"

    def __repr__(self):
        return str(self)