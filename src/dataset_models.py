import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import metrics
import torchvision as tv


class DataSet:
    """ Class that represents a compression action."""

    def __init__(self, name: str, criterion, metric, train_loader, test_loader, cap=None):
        self.name = name
        self.criterion = criterion
        self.metric = metric
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cap = cap


""" Supported Datasets """

use_cuda = False
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}



train_cifar_transform = tv.transforms.Compose([
    tv.transforms.RandomCrop(
        32, padding=4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2023, 0.1994, 0.2010])
])

test_cifar_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                    std=[0.2023, 0.1994, 0.2010])
        ])

supported_datasets = {
    "MNIST": DataSet(
        name="MNIST",
        criterion=F.nll_loss,
        metric=metrics.accuracy,
        train_loader=DataLoader(tv.datasets.MNIST('../data', train=True, download=True, transform=tv.transforms.ToTensor(),),
                                batch_size=64, shuffle=True, **kwargs),
        test_loader=DataLoader(tv.datasets.MNIST('../data', train=False, download=True, transform=tv.transforms.ToTensor(),),
                               batch_size=1000, shuffle=True, **kwargs),
    ),
    "CIFAR-10": DataSet(
        name="CIFAR-10",
        criterion=F.nll_loss,
        metric=metrics.accuracy,
        train_loader=DataLoader(tv.datasets.CIFAR10('../data', train=True, download=True, transform=train_cifar_transform),
                                batch_size=64, shuffle=True, **kwargs),
        test_loader=DataLoader(tv.datasets.CIFAR10('../data', train=False, download=True, transform=test_cifar_transform),
                               batch_size=64, shuffle=True, **kwargs),
    ),
}


