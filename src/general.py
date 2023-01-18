import time
import torch
import importlib
import inspect

LOGGING_STEPS = 100

# General train function
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    st = time.time()

    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_id % LOGGING_STEPS == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Distillation Loss: {:.4f}'.format(
                epoch, batch_id * len(data), len(train_loader.dataset),
                100. * batch_id / len(train_loader), loss.item()))
                
    et = time.time()
    duration = (et - st) * 1000
    batch_duration = duration/len(train_loader)
    
    print_performance("Train Set", loss, duration, batch_duration)


# General test function
def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    st = time.time()

    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            if batch_id % LOGGING_STEPS == 0 or (batch_id == len(test_loader) - 1):
                print_progress("Test", epoch, batch_id, data, test_loader, loss)

    et = time.time()
    duration = (et - st) * 1000
    batch_duration = duration/len(test_loader)
    test_loss /= len(test_loader)

    print_performance("Test Set", test_loss, duration, batch_duration)


# Method that prints the progress of the training/testing
def print_progress(title, epoch, batch_id, data, data_loader, loss):
    print('{} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    title, epoch, batch_id * len(data), len(data_loader.dataset),
                    100. * batch_id / len(data_loader), loss.item()))


# Method that prints the average loss and the elapsed time
def print_performance(title, loss, duration, batch_duration):
    print("========================================= PERFORMANCE =============================================")
    print('{}: Average loss: {:.4f}'.format(title, loss))
    print('Elapsed time = {:.2f} milliseconds ({:.2f} per batch)'.format(duration, batch_duration))
    print("====================================================================================================")


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


def get_device(no_cuda = False):
    use_cuda = not no_cuda and torch.cuda.is_available()
    print(f"Using cuda: {use_cuda}")

    return torch.device("cuda" if use_cuda else "cpu")

def save_model(model, path):
    torch.save(model.state_dict(), path)

