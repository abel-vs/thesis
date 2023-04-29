import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import metrics
import torchvision as tv
from transformers import glue_convert_examples_to_features, glue_output_modes, glue_processors, AutoTokenizer
from torchvision.datasets import CocoDetection, ImageNet, CIFAR10
from transformers import glue_tasks_num_labels, GlueDataset, SquadDataset, SquadDataTrainingArguments


""" General Dataset Class """

class DataSet:
    def __init__(self, name: str, criterion, metric, train_loader, test_loader, cap=None):
        self.name = name
        self.criterion = criterion
        self.metric = metric
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cap = cap

    def set_transforms(self, transforms):
        self.train_loader.dataset.transform = transforms
        self.test_loader.dataset.transform = transforms


""" General Variables """

use_cuda = False
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}



""" Supported Datasets """


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

supported_datasets = {
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
        
}


""" Older Supported Datasets """

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

supported_datasets.update({
    "MNIST": DataSet(
        name="MNIST",
        criterion=F.nll_loss,
        metric=metrics.accuracy,
        train_loader=DataLoader(tv.datasets.MNIST('/workspace/volume/data', train=True, download=True, transform=tv.transforms.ToTensor(),),
                                batch_size=64, shuffle=True, **kwargs),
        test_loader=DataLoader(tv.datasets.MNIST('/workspace/volume/data', train=False, download=True, transform=tv.transforms.ToTensor(),),
                               batch_size=1000, shuffle=True, **kwargs),
    ),
    "CIFAR-10": DataSet(
        name="CIFAR-10",
        criterion=F.cross_entropy,
        metric=metrics.accuracy,
        train_loader=DataLoader(CIFAR10('/workspace/volume/data', train=True, download=True, transform=train_cifar_transform),
                                batch_size=16, shuffle=True, **kwargs),
        test_loader=DataLoader(CIFAR10('/workspace/volume/data', train=False, download=True, transform=test_cifar_transform),
                               batch_size=64, shuffle=True, **kwargs),
    ),
})


