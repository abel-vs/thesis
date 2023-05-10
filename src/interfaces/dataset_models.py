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

DATA_DIR = "/workspace/volume/data"
CACHE_DIR = "/workspace/volume/cache"

""" General Dataset Class """

class DataSet:
    def __init__(self, name: str, criterion, metric, train_loader, val_loader, test_loader, cap=None):
        self.name = name
        self.criterion = criterion
        self.metric = metric
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cap = cap

    def set_transforms(self, transforms):
        self.train_loader.dataset.transform = transforms
        self.test_loader.dataset.transform = transforms


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


""" General Variables """

use_cuda = True
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mnist_transform = mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

""" Supported Dataloaders """

def get_cifar_data_loaders():
    train_dataset = CIFAR10(DATA_DIR, train=True, download=True, transform=cifar10_transform)
    test_dataset = CIFAR10(DATA_DIR, train=False, download=True, transform=cifar10_transform)
    train_sampler, val_sampler = get_train_val_sampler(train_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, **kwargs)
    val_loader = DataLoader(train_dataset, batch_size=64, sampler=val_sampler, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, **kwargs)
    return train_loader, val_loader, test_loader

def get_mnist_data_loaders():
    train_dataset = MNIST(DATA_DIR, train=True, download=True, transform=mnist_transform)
    test_dataset = MNIST(DATA_DIR, train=False, download=True, transform=mnist_transform)
    train_sampler, val_sampler = get_train_val_sampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler, **kwargs)
    val_loader = DataLoader(train_dataset, batch_size=64, sampler=val_sampler, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, **kwargs)
    return train_loader, val_loader, test_loader

def get_imagenet_loaders():
    dataset = load_dataset('imagenet-1k', data_dir=DATA_DIR, cache_dir=CACHE_DIR)
    train_dataset = ImageNetDataset(dataset['train'], transform=imagenet_transform)
    val_dataset = ImageNetDataset(dataset['validation'], transform=imagenet_transform)
    # The test dataset is not available for ImageNet, all labels are -1, so using validation
    test_dataset = ImageNetDataset(dataset['validation'], transform=imagenet_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, **kwargs)
    return train_loader, val_loader, test_loader


# Add this function for COCO dataset processing
def get_coco_data_loader(root, annFile, transform, batch_size=64, shuffle=True, **kwargs):
    coco = CocoDetection(root, annFile, transform=transform)
    return DataLoader(coco, batch_size=batch_size, shuffle=shuffle, **kwargs)

# Add this function for GLUE dataset processing
def get_glue_data_loader(task, split, tokenizer, batch_size=64, shuffle=True, **kwargs):
    processor = glue_processors[task]()
    output_mode = glue_output_modes[task]
    label_list = processor.get_labels()
    examples = processor.get_examples(split)
    features = glue_convert_examples_to_features(examples, tokenizer, max_length=128, label_list=label_list, output_mode=output_mode)
    dataset = GlueDataset(features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

# Add this function for SQUAD dataset processing
def get_squad_data_loader(data_args, tokenizer, split, batch_size=64, shuffle=True, **kwargs):
    dataset = SquadDataset(data_args, tokenizer=tokenizer, cache_dir=None, mode=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

# supported_datasets = {
    # "ImageNet": DataSet(
    #     name="ImageNet",
    #     criterion=F.cross_entropy,
    #     metric=metrics.accuracy,
    #     train_loader=DataLoader(ImageNet('../data/ImageNet', split='train'), batch_size=64, shuffle=True, **kwargs),
    #     test_loader=DataLoader(ImageNet('../data/ImageNet', split='val'), batch_size=64, shuffle=True, **kwargs),
    # ),
    # "COCO": DataSet(
    #     name="COCO",
    #     criterion=None,  # Define appropriate criterion for COCO
    #     metric=None,  # Define appropriate metric for COCO
    #     train_loader=get_coco_data_loader(root="../data/coco/train2017", annFile="../data/coco/annotations/instances_train2017.json"),
    #     test_loader=get_coco_data_loader(root="../data/coco/val2017", annFile="../data/coco/annotations/instances_val2017.json"),
    # ),
    # "GLUE": {
    #     task: DataSet(
    #         name=f"GLUE-{task}",
    #         criterion=None,  # Define appropriate criterion for GLUE
    #         metric=None,  # Define appropriate metric for GLUE
    #         train_loader=get_glue_data_loader(task=task, split="train", tokenizer=AutoTokenizer.from_pretrained("bertbase-uncased"), batch_size=64, shuffle=True, **kwargs),
    #         test_loader=get_glue_data_loader(task=task, split="validation", tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"), batch_size=64, shuffle=True, **kwargs),
    #     )
    #     for task in glue_tasks_num_labels.keys()
    # },
    # "SQUAD": DataSet(
    #         name="SQUAD",
    #         criterion=None,  # Define appropriate criterion for SQUAD
    #         metric=None,  # Define appropriate metric for SQUAD
    #         train_loader=get_squad_data_loader(data_args=SquadDataTrainingArguments(data_dir="../data/squad"), tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"), split=split, 
    #             batch_size=64, shuffle=True, **kwargs),
    #         test_loader=None,
    #     )     
# }
    



""" Older Supported Datasets """

cifar_train_loader, cifar_val_loader, cifar_test_loader = get_cifar_data_loaders()
mnist_train_loader, mnist_val_loader, mnist_test_loader = get_mnist_data_loaders()
imagenet_train_loader, imagenet_val_loader, imagenet_test_loader = get_imagenet_loaders()

supported_datasets = {
    "MNIST": DataSet(
        name="MNIST",
        criterion=F.cross_entropy,
        metric=metrics.accuracy,
        train_loader=mnist_train_loader,
        val_loader=mnist_val_loader,
        test_loader=mnist_test_loader,
    ),
    "CIFAR-10": DataSet(
        name="CIFAR-10",
        criterion=F.cross_entropy,
        metric=metrics.accuracy,
        train_loader=cifar_train_loader,
        val_loader=cifar_val_loader,
        test_loader=cifar_test_loader
    ),
    "ImageNet": DataSet(
        name="ImageNet",
        criterion=F.cross_entropy,
        metric=metrics.accuracy,
        train_loader=imagenet_train_loader,
        val_loader=imagenet_val_loader,
        test_loader=imagenet_test_loader
    ),
}


