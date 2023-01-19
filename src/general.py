import time
import torch
import importlib
import inspect
from tqdm import tqdm
import src.metrics as metrics
import src.plot as plot

LOGGING_STEPS = 1000

# General train function


def train(model, device, train_loader, criterion, optimizer, metric=None):
    model.train()
    st = time.time()
    train_loss = 0
    train_score = 0

    for batch_id, (data, target) in enumerate(tqdm(train_loader, desc='Train')):
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

        if batch_id % LOGGING_STEPS == 0 and batch_id > 0:
            plot.print_progress("Train", batch_id, data, train_loader, loss.item(),
                                score if metric is not None else None)

    et = time.time()
    duration = (et - st) * 1000
    batch_duration = duration/len(train_loader)
    data_duration = batch_duration/data.shape[0]
    train_loss /= len(train_loader)
    train_score /= len(train_loader)

    plot.print_performance("Train", train_loss, duration,
                           batch_duration, data_duration,  metric, train_score)


# General test function
def test(model, device, test_loader, criterion, metric=None):
    model.eval()
    test_loss = 0
    test_score = 0
    st = time.time()

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(tqdm(test_loader, desc='Test')):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            if metric is not None:
                score = metric(output, target)
                test_score += score

            if batch_id % LOGGING_STEPS == 0 and batch_id > 0:
                plot.print_progress("Test", 1, batch_id, data, test_loader, loss,
                                    score if metric is not None else None)

    et = time.time()
    duration = (et - st) * 1000
    batch_duration = duration/len(test_loader)
    data_duration = batch_duration/data.shape[0]
    test_loss /= len(test_loader)
    test_score /= len(test_loader)

    plot.print_performance("Test", test_loss, duration,
                           batch_duration, data_duration, metric, test_score)

    return test_loss, test_score, duration, batch_duration, data_duration


# Method that imports the classes from a module to the globals dictionary of a process
def import_module_classes(module, globals):
    # Get all classes in the module
    classes = [
        obj[1] for obj in inspect.getmembers(module, inspect.isclass)
    ]

    # Import the classes that are Modules
    for cls in classes:
        if issubclass(cls, torch.nn.Module):
            # Add the class to this package's variables
            globals()[cls.__name__] = cls


def get_module_classes(module):
    # Get all classes in the module
    classes = [
        obj[1] for obj in inspect.getmembers(module, inspect.isclass)
    ]

    # Only retain classes that are Modules
    for cls in classes:
        if not issubclass(cls, torch.nn.Module):
            classes.remove(cls)

    return classes


def get_device(no_cuda=False):
    use_cuda = not no_cuda and torch.cuda.is_available()
    print(f"Using cuda: {use_cuda}")

    return torch.device("cuda" if use_cuda else "cpu")


def save_model(model, path):
    torch.save(model.state_dict(), path)
