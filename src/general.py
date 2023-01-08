import torch
import importlib
import inspect

#Define the training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#Define the test function
def test(model, device, test_loader, criterion):
    # Set model to evaluation mode
    model.eval()

    # Initialize test loss and accuracy
    test_loss = 0.0
    test_acc = 0.0

    # Disable gradients (to save memory)
    with torch.no_grad():
        # Loop over batches of test data
        for inputs, labels in iter(test_loader):
            # Print number of inputs
            print("Inputs: " + str(len(inputs)))
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update test loss and accuracy
            test_loss += loss.item()
            test_acc += (outputs.argmax(dim=1) == labels).float().mean()

    
    return test_loss, test_acc




def load_model(model_state, model_class):
    # Import the module class
    module = importlib.import_module(model_class)

    # Get all classes in the module
    classes = [
        obj[1] for obj in inspect.getmembers(module, inspect.isclass)
    ]

    # Import the classes that are Modules
    for cls in classes:
        if issubclass(cls, torch.nn.Module):
            # Add the class to this package's variables
            globals()[cls.__name__] = cls

    model = torch.load(model_state, map_location=torch.device('cpu'))

    return model