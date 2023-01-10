import time
import torch
import importlib
import inspect

#Define the training function
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
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
        for batch_idx, (data, target) in enumerate(test_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Update test loss and accuracy
            test_loss += loss.item()
            test_acc += (outputs.argmax(dim=1) == target).float().mean()

    
    return test_loss, test_acc



def test(model, device, test_loader, criterion, quantize=False, fbgemm=False):
    model.to(device)
    model.eval()
    
    print(model)
    test_loss = 0
    correct = 0

    st = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Update test loss
            test_loss += loss.item()

            # Count 
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    et = time.time()

    
    test_loss /= len(test_loader.dataset)
    
    print("========================================= PERFORMANCE =============================================")
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Elapsed time = {:0.4f} milliseconds'.format((et - st) * 1000))
    print("====================================================================================================")