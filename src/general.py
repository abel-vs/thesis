import itertools
from typing import List
import torch
import inspect
from tqdm import tqdm
import compression.distillation as distil
import compression.quantization as quant
import compression.pruning as prune
import evaluation as eval
import copy
import torch.optim as optim
import plot
import time
from src.interfaces.dataset_models import DataSet


# General train function
def train(model, dataset: DataSet, optimizer=None, device=None):
    if device is None:
        device = get_device()
    model.to(device)
    train_loader = dataset.train_loader
    criterion = dataset.criterion
    metric = dataset.metric

    if optimizer == None:
        # TODO: Build automated optimizer constructor
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

    model.train()
    st = time.time()
    train_loss = 0
    train_score = 0

    for (data, target) in tqdm(train_loader, desc="Train"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if metric is not None:
            score = metric(output, target)
            train_score += score

    et = time.time()
    duration = (et - st) * 1000
    batch_duration = duration / len(train_loader)
    data_duration = batch_duration / data.shape[0]
    train_loss /= len(train_loader)
    train_score /= len(train_loader)

    print("Train loss: {:.4f}".format(train_loss))
    print("Train score: {:.4f}".format(train_score))

    return train_loss, train_score, duration, batch_duration, data_duration

# General test function
def test(model, dataset , validate=False, device=None):
    if device is None:
        device = get_device()
    model.to(device)

    if validate:
        test_loader = dataset.val_loader
    else:
        test_loader = dataset.test_loader

    if dataset.cap:
        test_loader = itertools.islice(test_loader, dataset.cap)

    criterion = dataset.criterion
    metric = dataset.metric

    model.eval()
    test_loss = 0
    test_score = 0
    st = time.time()

    with torch.no_grad():
        for (data, target) in tqdm(test_loader, desc=("Test" if not validate else "Validate")):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            if metric is not None:
                score = metric(output, target)
                test_score += score

    et = time.time()
    duration = (et - st) * 1000
    no_batches = dataset.cap if dataset.cap is not None else len(test_loader)
    batch_duration = duration / no_batches
    data_duration = batch_duration / data.shape[0]
    test_loss /= no_batches
    test_score /= no_batches

    print("Test loss: {:.4f}".format(test_loss))
    print("Test score: {:.4f}".format(test_score))

    return test_loss, test_score, duration, batch_duration, data_duration


# General validation method
def validate(model, dataset, device=None):
    return test(model, dataset, validate=True, device=device)

# General finetune method
def finetune(model, dataset, target, epochs=None, patience=3, save_path=None, device=None, optimizer=None):
    epochs_without_improvement = 0
    iterations = 0

    start_metrics = validate(model, dataset, device=device)
    score = start_metrics[1]
    best_score = score
    best_model = copy.deepcopy(model)

    while score < target and epochs_without_improvement < patience:
        train(model, dataset, optimizer=optimizer, device=device)
        metrics = validate(model, dataset, device=device)
        score = metrics[1]

        if score > best_score:
            best_model = copy.deepcopy(model)
            if save_path is not None:
                torch.save(best_model, save_path)
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        iterations += 1

        if epochs is not None and iterations >= epochs:
            print("Maximum number of iterations reached")
            break


    if epochs_without_improvement >= patience:
        print("Finetuning stopped due to early stopping with patience = {}".format(patience))
    else:
        print("Finetuning stopped due to reaching the target score")

    print("Finetuning finished after {} iterations".format(iterations))
    print("Best score: {:.4f}".format(best_score))
    return best_model

# Method that imports the classes from a module to the globals dictionary of a process
def import_module_classes(module, globals):
    # Get all classes in the module
    classes = [obj[1] for obj in inspect.getmembers(module, inspect.isclass)]

    # Import the classes that are Modules
    for cls in classes:
        if issubclass(cls, torch.nn.Module):
            # Add the class to this package's variables
            globals()[cls.__name__] = cls


def get_module_classes(module):
    # Get all classes in the module
    classes = [obj[1] for obj in inspect.getmembers(module, inspect.isclass)]

    # Only retain classes that are Modules
    for cls in classes:
        if not issubclass(cls, torch.nn.Module):
            classes.remove(cls)

    return classes


def get_device(no_cuda=False):
    use_cuda = not no_cuda and torch.cuda.is_available()

    return torch.device("cuda" if use_cuda else "cpu")


def get_example_input_batch(data_loader):
    return next(iter(data_loader))


def get_example_inputs(data_loader, device=None):
    if device is None:
        device = get_device()
    input_batch = get_example_input_batch(data_loader)
    return input_batch[0].to(device)