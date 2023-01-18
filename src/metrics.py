import torch


# Currently just for classification like MNIST
def accuracy(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / len(output)


def mean_absolute_error(output, target):
    with torch.no_grad():
        return torch.mean(torch.abs(output - target))


def mean_squared_error(output, target):
    with torch.no_grad():
        return torch.mean((output - target) ** 2)


NAMES = {
    accuracy: 'Accuracy',
    mean_absolute_error: 'Mean Absolute Error',
    mean_squared_error: 'Mean Squared Error'
}
