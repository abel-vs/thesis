import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import metrics
import torchvision as tv
from torchvision import transforms
from transformers import glue_convert_examples_to_features, glue_output_modes, glue_processors, AutoTokenizer
from torchvision.datasets import CocoDetection, ImageNet, CIFAR10, MNIST
from transformers import glue_tasks_num_labels, GlueDataset, SquadDataset, SquadDataTrainingArguments
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from datasets import load_dataset


""" General Variables """

use_cuda = True
kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

# DATA_DIR = "../../volume/data"
# CACHE_DIR = "../../volume/cache"
DATA_DIR = "/workspace/volume/data"
CACHE_DIR = "/workspace/volume/cache"

""" General Dataset Class """


class DataSet:
    def __init__(self, name: str, criterion, metric, train_batch_size, test_batch_size, data_loaders, transforms, cap=None):
        self.name = name
        self.criterion = criterion
        self.metric = metric
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.data_loaders = data_loaders
        self.transforms = transforms
        self.train_loader, self.val_loader, self.test_loader = data_loaders(
            train_batch_size, test_batch_size, transforms)
        self.cap = cap

    def set_batch_sizes(self, train_batch_size, test_batch_size):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_loader, self.val_loader, self.test_loader = self.data_loaders(
            train_batch_size, test_batch_size, self.transforms)

    def set_transforms(self, transforms):
        self.transforms = transforms
        self.train_loader.dataset.transform = transforms
        self.test_loader.dataset.transform = transforms


class HuggingFaceDataset(DataSet):
    def __init__(self, name: str, criterion, metric, train_batch_size, test_batch_size, transforms, dataset_name, tokenizer_name, cap=None):
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.dataset = None
        self.tokenizer = None
        self.init_dataset_and_tokenizer()

        super().__init__(name, criterion, metric, train_batch_size, test_batch_size, self.hf_data_loaders, transforms, cap)

    def init_dataset_and_tokenizer(self):
        self.dataset = load_dataset(self.dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

    def hf_data_loaders(self, train_batch_size, test_batch_size, transforms):
        train_loader = self.get_data_loader('train', train_batch_size, transforms)
        val_loader = self.get_data_loader('validation', test_batch_size, transforms)
        test_loader = self.get_data_loader('test', test_batch_size, transforms)
        return train_loader, val_loader, test_loader

    def get_data_loader(self, split_name, batch_size, transforms):
        def encode(examples):
            return self.tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

        dataset = self.dataset[split_name]
        dataset = dataset.map(encode, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader



""" Supported Datasets Classes """


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]['image']

        # Convert grayscale image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.ds[idx]['label'], dtype=torch.long)
        return image, label


""" Helper Methods """


def get_train_val_sampler(dataset, shuffle=False):
    valid_size = 0.1
    random_seed = 42

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # Shuffling is done by the sampler, don't do it twice in Dataloader
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, val_sampler


""" Transforms """

imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cifar10_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

""" Supported Dataloaders """


def get_cifar_data_loaders(train_batch_size, test_batch_size, transforms):
    train_dataset = CIFAR10(DATA_DIR, train=True,
                            download=True, transform=transforms)
    test_dataset = CIFAR10(DATA_DIR, train=False,
                           download=True, transform=transforms)
    train_sampler, val_sampler = get_train_val_sampler(
        train_dataset, shuffle=False)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=train_sampler, **kwargs)
    val_loader = DataLoader(
        train_dataset, batch_size=test_batch_size, sampler=val_sampler, **kwargs)
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader, test_loader


def get_mnist_data_loaders(train_batch_size, test_batch_size, transforms):
    train_dataset = MNIST(DATA_DIR, train=True,
                          download=True, transform=transforms)
    test_dataset = MNIST(DATA_DIR, train=False,
                         download=True, transform=transforms)
    train_sampler, val_sampler = get_train_val_sampler(
        train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, sampler=train_sampler, **kwargs)
    val_loader = DataLoader(
        train_dataset, batch_size=test_batch_size, sampler=val_sampler, **kwargs)
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader, test_loader


def get_imagenet_loaders(train_batch_size, test_batch_size, transforms):
    dataset = load_dataset(
        'imagenet-1k', data_dir=DATA_DIR, cache_dir=CACHE_DIR)
    train_dataset = ImageNetDataset(dataset['train'], transform=transforms)
    val_dataset = ImageNetDataset(dataset['validation'], transform=transforms)
    # The test dataset is not available for ImageNet, all labels are -1, so using validation
    test_dataset = ImageNetDataset(
        dataset['validation'], transform=imagenet_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True, **kwargs)
    return train_loader, val_loader, test_loader


# # Add this function for GLUE dataset processin
# def get_glue_data_loader(task, split, tokenizer, batch_size=64, shuffle=True, **kwargs):
#     processor = glue_processors[task]()
#     output_mode = glue_output_modes[task]
#     label_list = processor.get_labels()
#     examples = processor.get_examples(split)
#     features = glue_convert_examples_to_features(examples, tokenizer, max_length=128, label_list=label_list, output_mode=output_mode)
#     dataset = GlueDataset(features)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

# # Add this function for SQUAD dataset processing
# def get_squad_data_loader(data_args, tokenizer, split, batch_size=64, shuffle=True, **kwargs):
#     dataset = SquadDataset(data_args, tokenizer=tokenizer, cache_dir=None, mode=split)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

""" Suppored Datasets """


def get_supported_dataset(name):
    if name == "MNIST":
        return DataSet(
            name="MNIST",
            criterion=F.cross_entropy,
            metric=metrics.accuracy,
            train_batch_size=128,
            test_batch_size=256,
            data_loaders=get_mnist_data_loaders,
            transforms=mnist_transform,
        )
    elif name == "CIFAR-10":
        return DataSet(
            name="CIFAR-10",
            criterion=F.cross_entropy,
            metric=metrics.accuracy,
            train_batch_size=64,
            test_batch_size=64,
            data_loaders=get_cifar_data_loaders,
            transforms=cifar10_transform,
        )
    elif name == "ImageNet":
        return DataSet(
            name="ImageNet",
            criterion=F.cross_entropy,
            metric=metrics.accuracy,
            train_batch_size=32,
            test_batch_size=128,
            data_loaders=get_imagenet_loaders,
            transforms=imagenet_transform,
        )
    # elif name == 'SQUAD':
    #     return DataSet(
    #         name="SQUAD",
    #         criterion=F.cross_entropy,  # or whatever loss function is appropriate for your task
    #         metric=,  # or whatever metric is appropriate for your task
    #         train_batch_size=16,
    #         test_batch_size=64,
    #         data_loaders=get_squad_data_loaders,
    #         transforms=None,  # Not used for text data
    #     )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

