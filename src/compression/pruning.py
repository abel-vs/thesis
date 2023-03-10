import torch
import torch.nn.utils.prune as prune

import torch_pruning as tp


""" UNSTRUCTURED PRUNING """

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
    return parameters_to_prune[: int(len(parameters_to_prune) * pruning_rate)]


# Method that randomly prunes a module by a given rate from 0% to 100%
def random_pruning(module, rate):
    prune.random_unstructured(module, name="weight", amount=rate)


# Method that randomly prunes a module by a given rate from 0% to 100%
def pruning_global(model, rate):
    for module in model.children():
        print("module", module)
        prune.l1_unstructured(module, name="weight", amount=rate)
        prune.remove(module, "weight")

    return model


# Method that prunes the lowest magnitude weights of a module
def magnitude_pruning(module, rate):
    prune.l1_unstructured(module, name="weight", amount=rate)


# Method that prunes the lowest magnitude weights globally
def magnitude_pruning_global(model, pruning_rate):
    parameters_to_prune = get_parameters_to_prune(model, pruning_rate)
    print("parameters_to_prune", parameters_to_prune)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=pruning_rate,
    )


""" STRUCTURED PRUNING """


# Method to get the layers that should be ignored.
# For example the final classifying layer
# TODO: this method should find the final layer in a general way, we can't assume the final layer is the last module, since it depends on the forward function.
def get_ignored_layers(model):
    ignored_layers = []
    for i, m in enumerate(model.modules()):
        if i == len(list(model.modules())) -1:
            ignored_layers.append(m) 


def magnitude_pruning(model, example_inputs):
    # 0. importance criterion for parameter selections
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')

    # 1. ignore some layers that should not be pruned, e.g., the final classifier layer.
    ignored_layers = get_ignored_layers(model)

            
    # 2. Pruner initialization
    iterative_steps = 5 # You can prune your model to the target sparsity iteratively.
    pruner = tp.pruner.MagnitudePruner(
        model, 
        example_inputs, 
        global_pruning=False, # If False, a uniform sparsity will be assigned to different layers.
        importance=imp, # importance criterion for parameter selection
        iterative_steps=iterative_steps, # the number of iterations to achieve target sparsity
        ch_sparsity=0.5, # remove 50% channels
        ignored_layers=ignored_layers,
    )

    for i in range(iterative_steps):
        # pruner.step will remove some channels from the model with least importance
        pruner.step()

    return model

# TODO: Define a custom pruning method that
# Below is an example of a custom pruning method
class CustomMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor"""

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask
