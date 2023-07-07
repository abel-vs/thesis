from enum import Enum


class PruningTechnique(str, Enum):
    Random = "random"
    L1 = "l1"
    LAMP = "lamp"
    SLIM = "slim"
    GroupNorm = "group_norm"

class DistillationTechnique(str, Enum):
    SoftTarget = "soft_target"
    HardTarget = "hard_target"
    CombinedLoss = "combined_loss"

class QuantizationTechnique(str, Enum):
    Dynamic = "dynamic"
    Static = "static"
    QAT = "qat"
