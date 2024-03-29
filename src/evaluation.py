import copy
import os
import torch
import general
import torch.nn.functional as F
from op_counter import count_ops_and_params
from src.interfaces.dataset_models import DataSet
import time

# Method that returns the model size in MB
def get_size(model):
    torch.save(model.state_dict(), "tmp.pt")
    model_size = os.path.getsize("tmp.pt") / (1024 * 1024)
    os.remove("tmp.pt")
    return round(model_size, 2)


# Method that returns the number of parameters in the model
def get_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Method that returns the number of FLOPS the model executes
def get_flops(model, example_input):
    model = copy.deepcopy(model).cpu()
    example_input = example_input.cpu()
    flops, _ = count_ops_and_params(model, example_input)
    return flops

# Method that returns both the number of FLOPS and parameters
def get_flops_and_params(model, example_input):
    flops, params = count_ops_and_params(model, example_input)
    return flops, params


# Method that returns the number of pruned parameters in the model
def get_model_pruned_parameters(model):
    # For each parameter check whether its value is zero
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad and p.eq(0).all()
    )


# Method that returns the sparsity of a module
def get_module_sparsity(module):
    100.0 * float(torch.sum(module.weight == 0)) / \
        float(module.weight.nelement())


# Method that returns the batch and data duration of a model
def get_inference_time(model, dataset: DataSet, device=None, quantized=False, cap=3):
    if device is None:
        device = general.get_device()
    model.to(device)
    model.eval()
    start_time = time.time()
    i = 0
    with torch.no_grad():
        for data, target in dataset.test_loader:
            if quantized:
                data = data.to(torch.uint8)
            data, target = data.to(device), target.to(device)
            model(data)
            i += 1
            if i >= cap:
                break
        end_time = time.time()
    batch_duration = ((end_time - start_time) / cap ) * 1000
    data_duration = batch_duration / data.shape[0]
    return batch_duration, data_duration

    

# Method that tests the model and returns the metrics
def get_metrics(model, dataset: DataSet, device=None):
    example_input = general.get_example_inputs(dataset.test_loader, device=device)
    batch_size = example_input.shape[0]

    loss, score, duration, batch_duration, data_duration = general.test(
        model, dataset, device=device)

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


# Method that tests the model and returns the results
def get_results(model, dataset: DataSet, device=None):
    metrics = get_metrics(model, dataset, device=device)

    macs, params = count_ops_and_params(model, metrics["example_input"])
    results = {
        "loss": metrics["loss"],
        "score": metrics["score"],
        "batch_duration": metrics["batch_duration"],
        "data_duration": metrics["data_duration"],
        "model_size": get_size(model),
        "params": round(params),
        "macs": round(macs),
    }

    return results
