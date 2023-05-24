from enum import Enum

class CompressionObjective(str, Enum):
    Size = "size"
    Time = "time"
    Computations = "flops"