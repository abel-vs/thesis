from enum import Enum

class PruningStrategy(str, Enum):
    Global = "global"
    OnlyLinear = "only_linear"
    OnlyConv = "only_conv"
    OnlyAttention = "only_attention"

strategy_names = {
    PruningStrategy.Global: "Global",
    PruningStrategy.OnlyLinear: "Linear",
    PruningStrategy.OnlyConv: "Convolutional",
    PruningStrategy.OnlyAttention: "Attention"
}

