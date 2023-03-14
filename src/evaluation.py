import copy
import os
import time
import torch
from flopth import flopth
import general
import metrics
import torch.nn.functional as F
from op_counter import count_ops_and_params
from dataset_models import DataSet

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

    flops, params = flopth(model_copy, input_size, bare_number=True, show_detail=False)

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


def test_and_get_metrics(model, dataset: DataSet):

    device = general.get_device()
    example_input = general.get_example_input(dataset.test_loader)
    batch_size = example_input.shape[0]

    loss, score, duration, batch_duration, data_duration = general.test(model, device, dataset)

    evaluation_metrics = {
        "model": model,
        "loss": loss,
        "score": score,
        "duration": duration,
        "batch_duration": batch_duration,
        "data_duration": data_duration,
        "batch_size": batch_size,
        "example_input": example_input,  # Currently error with flops calculation
        "input_size": example_input[0].size(),
    }

    return evaluation_metrics


def get_results(model, dataset: DataSet):
    metrics = test_and_get_metrics(model, dataset)

    flops = -1
    try:
        flops = get_model_flops(model, metrics["input_size"])
    except:
        print("Could not calculate FLOPS")

    macs, params = count_ops_and_params(model, metrics["example_input"])

    results = {
        "loss": metrics["loss"],
        "score": metrics["score"],
        "batch_duration": metrics["batch_duration"],
        "data_duration": metrics["data_duration"],
        "model_size": get_model_size(model),
        "params": round(params),
        "flops": round(flops),
        "macs": round(macs),
    }

    return results
