import copy
import os
import time
import torch
from flopth import flopth
import general
import metrics
import torch.nn.functional as F
from op_counter import count_ops_and_params

# Method that returns the model size in MB


def get_model_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    model_size = os.path.getsize("tmp.pt") / (1024 * 1024)
    os.remove("tmp.pt")
    return round(model_size, 2)


# Method that returns the number of parameters in the model
def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Method that returns the number of FLOPS the model executes
def get_model_flops(model, input_size):
    model_copy = copy.deepcopy(model)

    return flops


def get_MACS(model, example_input):
    macs, params = count_ops_and_params(model, example_input)
    return macs


# Method that returns the number of pruned parameters in the model
def get_model_pruned_parameters(model):
    # For each parameter check whether its value is zero
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad and p.eq(0).all()
    )


# Method that returns the sparsity of a module
def get_module_sparsity(module):
    100.0 * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())


# Method that
def time_model_evaluation(model, configs, tokenizer):
    eval_start_time = time.time()
    # evaluate(configs, model, tokenizer, prefix="")
    eval_end_time = time.time()
    eval_duration_time = eval_end_time - eval_start_time
    print("Evaluate total time (seconds): {0:.1f}".format(eval_duration_time))
    return eval_duration_time


def get_metrics(model, test_loader):

    device = general.get_device()
    input_batch = next(iter(test_loader))
    example_input = input_batch[0][0]
    batch_size = input_batch[0].shape[0]

    loss, score, duration, batch_duration, data_duration = general.test(
        model, device, test_loader, criterion=F.nll_loss, metric=metrics.accuracy
    )

    evaluation_metrics = {
        "model": model,
        "loss": loss,
        "score": score,
        "duration": duration,
        "batch_duration": batch_duration,
        "data_duration": data_duration,
        "batch_size": batch_size,
        "example_input": example_input,  # Currently error with flops calculation
    }

    return evaluation_metrics


def get_results(model, test_loader):
    metrics = get_metrics(model, test_loader)

    results = {
        "loss": metrics["loss"],
        "score": metrics["score"],
        "batch_duration": metrics["batch_duration"],
        "data_duration": metrics["data_duration"],
        "model_size": get_model_size(model),
        "params": get_model_parameters(model),
        # "flops": get_model_flops(model, metrics["example_input"].size()),
        "macs": get_MACS(model, metrics["example_input"]),
    }

    return results
