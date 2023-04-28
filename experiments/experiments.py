import torch
import torchvision
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class ExperimentManager:
    """
    Class for easily executing experiments
    
    """
    def __init__(self, experiment_config):
        self.config = experiment_config

    def load_model(self):
        model_type = self.config['model_type']
        architecture = self.config['architecture']
        task = self.config['task']

        if model_type == 'CNN':
            model = CNNModel(architecture, task)
        elif model_type == 'Transformer':
            model = TransformerModel(architecture, task)

        return model

    def load_data(self):
        # Add your data loading logic here, e.g. DataLoader in PyTorch
        pass

    def train(self, model, data):
        # Add your training logic here
        pass

    def compress(self, model):
        # Add your compression logic here
        pass

    def evaluate(self, model, data):
        # Add your evaluation logic here
        pass

    def save_results(self, results):
        # Add your result saving logic here, e.g. using pandas to save as CSV
        pass

    def run_experiment(self):
        model = self.load_model()
        data = self.load_data()
        self.train(model, data)
        self.compress(model)
        results = self.evaluate(model, data)
        self.save_results(results)

class CNNModel:
    def __init__(self, architecture):
        self.model = torchvision.models.__dict__[architecture](pretrained=True)

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
)

class TransformerModel:
    def __init__(self, architecture, task):
        self.tokenizer = AutoTokenizer.from_pretrained(architecture)
        if task == "sequence_classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(architecture)
        elif task == "token_classification":
            self.model = AutoModelForTokenClassification.from_pretrained(architecture)
        elif task == "question_answering":
            self.model = AutoModelForQuestionAnswering.from_pretrained(architecture)
        else:
            raise ValueError(f"Unknown task type: {task}")