
import torch.nn.functional as F
from enum import Enum
from src.interfaces.objectives import CompressionObjective
from src.interfaces.strategies import PruningStrategy
from src.interfaces.techniques import DistillationTechnique, PruningTechnique, QuantizationTechnique

class CompressionCategory(str, Enum):
    pruning = "pruning"
    quantization = "quantization"
    distillation = "distillation"

class CompressionAction:
    """ Class that represents a compression action."""
    def __init__(self, type: CompressionCategory, name: str):
        self.type = type
        self.name = name


class PruningAction(CompressionAction):
    """ Class that represents a pruning action. """
    def __init__(self, name: str, technique: PruningTechnique,  sparsity: float, strategy: PruningStrategy, objective: CompressionObjective, settings: dict = {}):
        super().__init__(CompressionCategory.pruning, name)

        self.objective = objective 
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
        super().__init__(CompressionCategory.quantization, name)

        self.technique = technique 
        self.quantization_bits = settings.get("quantization_bits", 8)
        self.settings = settings  # Use settings to set all the necessary kwargs
    
    def __str__(self):
        return f"QuantizationAction(name={self.name}, technique={self.technique}, quantization_bits={self.quantization_bits})"
    
    def __repr__(self):
        return str(self)


class DistillationAction(CompressionAction):
    """ Class that represents a distillation action. """
    def __init__(self, name: str, technique: DistillationTechnique, performance_target: float, settings: dict = {}):
        super().__init__(CompressionCategory.distillation, name)

        self.technique = technique
        self.performance_target = settings.get("performance_target", 100)
        if technique == "soft_target":
            self.distillation_loss = settings.get("distillation_loss", F.kl_div)
        elif technique == "hard_target":
            self.distillation_loss = settings.get("distillation_loss", F.cross_entropy)  
        elif technique == "combined_loss":
            self.distillation_loss = settings.get("distillation_loss", F.kl_div)
        else:
            self.distillation_loss = settings.get("distillation_loss", F.mse_loss)  
        self.settings = settings  # Use settings to set all the necessary kwargs

    def __str__(self):
        return f"DistillationAction(name={self.name}, technique={self.technique}, distillation_loss={self.distillation_loss})"

    def __repr__(self):
        return str(self)
    
def order_compression_actions(compression_actions):
    # Define the order of the compression categories
    category_order = {
        CompressionCategory.pruning: 0,
        CompressionCategory.distillation: 1,
        CompressionCategory.quantization: 2
    }

    # Sort the actions based on the category order
    sorted_actions = sorted(compression_actions, key=lambda action: category_order[action.type])

    return sorted_actions
    


# Method to create compression actions from config file
def create_compression_action(action_dict):
    action_type = action_dict["type"]
    if action_type == CompressionCategory.pruning:
        return PruningAction(
            name=action_dict["name"],
            technique=action_dict["technique"],
            sparsity=action_dict["sparsity"],
            strategy=action_dict["strategy"],
            objective=action_dict["objective"],
            settings=action_dict.get("settings", {}),
        )
    elif action_type == CompressionCategory.quantization:
        return QuantizationAction(
            name=action_dict["name"],
            technique=action_dict["technique"],
            settings=action_dict.get("settings", {}),
        )
    elif action_type == CompressionCategory.distillation:
        return DistillationAction(
            name=action_dict["name"],
            technique=action_dict["technique"],
            performance_target = action_dict["performance_target"],
            settings=action_dict.get("settings", {}),
        )
    else:
        raise ValueError(f"Unknown compression action type: {action_type}")