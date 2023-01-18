import os
import time
import torch
from flopth import flopth

# Method that returns the model size in MB


def get_model_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    model_size = os.path.getsize("tmp.pt") / (1024*1024)
    os.remove("tmp.pt")
    return round(model_size, 2)


# Method that returns the number of parameters in the model
def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Method that returns the model siz in RAM in MB
def get_model_ram(model):
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024*1024)


# Method that returns the number of FLOPS the model executes
def get_model_flops(model, input_size):
    return flopth(model, input_size)


# Method that returns the number of pruned parameters in the model
def get_model_pruned_parameters(model):
    # For each parameter check whether its value is zero
    return sum(p.numel() for p in model.parameters() if p.requires_grad and p.eq(0).all())


# Method that returns the sparsity of a module
def get_module_sparsity(module):
    100. * float(torch.sum(module.weight == 0)) / \
        float(module.weight.nelement())


# Method that
def time_model_evaluation(model, configs, tokenizer):
    eval_start_time = time.time()
    # evaluate(configs, model, tokenizer, prefix="")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))
    return eval_duration_time
