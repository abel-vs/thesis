
from enum import Enum


class CompressionType(str, Enum):
    pruning = "pruning"
    quantization = "quantization"
    distillation = "distillation"

class CompressionAction:
    """ Class that represents a compression action."""
    def __init__(self, type: CompressionType, name: str):
        self.type = type
        self.name = name