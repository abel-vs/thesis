import torch
import torch.nn.utils.prune as prune

# Method that returns the parameters to prune
# TODO: This should be done intelligently, returning the parameters that are the least important
# For now, we just return the parameters that are the smallest
def get_parameters_to_prune(model, pruning_rate):
    # Get the parameters to prune
    parameters_to_prune = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameters_to_prune.append((name, parameter))
    # Sort the parameters by their absolute value
    parameters_to_prune.sort(key=lambda x: torch.sum(torch.abs(x[1])))
    for x in parameters_to_prune:
        print(x[0], torch.sum(torch.abs(x[1])))
    # Return the parameters to prune
    return parameters_to_prune[:int(len(parameters_to_prune) * pruning_rate)]


# Method that randomly prunes a module by a given rate from 0% to 100%
def random_pruning(module, rate):
    prune.random_unstructured(module, name="weight", amount=rate)

# Method that prunes the lowest magnitude weights of a module 
def magnitude_pruning(module, rate):
    prune.l1_unstructured(module, name="weight", amount=rate)

# Method that prunes the lowest magnitude weights globally
def magnitude_pruning_global(model, pruning_rate):
    parameters_to_prune = get_parameters_to_prune(model, pruning_rate)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )

# TODO: Define a custom pruning method that 
# Below is an example of a custom pruning method
class CustomMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor"""
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0 
        return mask