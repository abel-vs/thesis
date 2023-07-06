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


technique_names = {
    PruningTechnique.Random: "Random Pruning",
    PruningTechnique.LAMP: "LAMP Pruning",
    PruningTechnique.L1: "Magnitude Pruning",
    PruningTechnique.SLIM: "Batch Norm Pruning",
    PruningTechnique.GroupNorm: "Group Norm Pruning",
    DistillationTechnique.SoftTarget: "Soft Target Distillation",
    DistillationTechnique.HardTarget: "Hard Target Distillation",
    DistillationTechnique.CombinedLoss: "Combined Loss Distillation",
    QuantizationTechnique.Dynamic: "Dynamic Training",
    QuantizationTechnique.Static: "Static Training",
    QuantizationTechnique.QAT: "Quantization Aware Training"
}