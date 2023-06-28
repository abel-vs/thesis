from enum import Enum

class PruningStrategy(str, Enum):
    Global = "global"
    OnlyLinear = "only_linear"
    OnlyConv = "only_conv"