import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import metrics
import torchvision as tv


class DataSet:
    """ Class that represents a compression action."""

    def __init__(self, name: str, criterion, metric, train_loader, test_loader):
        self.name = name
        self.criterion = criterion
        self.metric = metric
        self.train_loader = train_loader
        self.test_loader = test_loader


""" Supported Datasets """

use_cuda = False
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

supported_datasets = {
    "MNIST": DataSet(
        name="MNIST", 
        criterion=F.nll_loss,
        metric=metrics.accuracy,
        train_loader=DataLoader(tv.datasets.MNIST('../data', train=True, download=True, transform=tv.transforms.ToTensor(),),
                                batch_size=64, shuffle=True, **kwargs),
        test_loader=DataLoader(tv.datasets.MNIST('../data', train=False, download=True, transform=tv.transforms.ToTensor(),),
                               batch_size=64, shuffle=True, **kwargs),
        ),
    "CIFAR-10": DataSet(
        name="CIFAR-10",
        criterion=F.nll_loss,
        metric=metrics.accuracy,
        train_loader=DataLoader(tv.datasets.CIFAR10('../data', train=True, download=True, transform=tv.transforms.ToTensor(),),
                                batch_size=64, shuffle=True, **kwargs),
        test_loader=DataLoader(tv.datasets.CIFAR10('../data', train=False, download=True, transform=tv.transforms.ToTensor(),),
                               batch_size=64, shuffle=True, **kwargs),
    ),
    "IMAGENET": DataSet(
        name="CIFAR-10",
        criterion=F.cross_entropy,
        metric=metrics.accuracy,
        train_loader=DataLoader(tv.datasets.CIFAR10('../data', train=True, download=True, transform=tv.transforms.ToTensor(),),
                                batch_size=64, shuffle=True, **kwargs),
        test_loader=DataLoader(tv.datasets.CIFAR10('../data', train=False, download=True, transform=tv.transforms.ToTensor(),),
                               batch_size=64, shuffle=True, **kwargs),
    )

}
