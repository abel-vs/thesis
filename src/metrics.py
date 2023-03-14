import torch

def mean_absolute_error(output, target):
    with torch.no_grad():
        return torch.mean(torch.abs(output - target))


def mean_squared_error(output, target):
    with torch.no_grad():
        return torch.mean((output - target) ** 2)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0].item()


NAMES = {
    accuracy: 'Accuracy',
    mean_absolute_error: 'Mean Absolute Error',
    mean_squared_error: 'Mean Squared Error'
}
