import torch
import yaml
import logging
import os
import sys
import json

import src.evaluation as eval
import src.analysis as analysis
import src.plot as plot
import src.general as general
import src.compression.quantization as quant
import src.interfaces.dataset_models as data

from src.interfaces.compression_actions import CompressionType, DistillationAction, PruningAction, QuantizationAction
from src.compress import compress_model
from torch.utils.tensorboard import SummaryWriter


LOG_DIR = "/workspace/volume/experiments/"

class ExperimentManager:
    """
    Class for easily executing experiments
    
    """
    def __init__(self, experiment_config, device=None):
        self.config = experiment_config
        self.foldername = self.config["name"].lower()
        self.device = device if device is not None else general.get_device()

    def setup_logging(self):
        os.makedirs(os.path.join(LOG_DIR, self.foldername), exist_ok=True)

        log_filename = f"{self.get_experiment_name()}.log"
        log_filepath = os.path.join(LOG_DIR, self.foldername, log_filename)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%H:%M:%S',
                            handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()])

        self.writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, self.foldername, "tensorboard"))
        

    def load_model(self):
        model = torch.load(self.config['model_path'])
        return model

    def load_data(self):
        dataset = data.supported_datasets[self.config['dataset']]

        return dataset
    def compress(self, model, dataset):
        # Check if config has compression_actions key
        if "compression_actions" in self.config:
            compression_actions = self.config["compression_actions"]
        else:
            compression_actions = analysis.analyze(model, dataset, type="size", compression_target=self.config["target"], settings={})
        logging.info("Compression Actions:")
        for action in compression_actions:
            logging.info(action)
        compressed_model = compress_model(model, dataset, compression_actions, writer=self.writer, device=self.device)
        return compressed_model

    def evaluate(self, model, data):
        if quant.is_quantized(model):
            device = torch.device("cpu")
        else:
            device = self.device
        results = eval.get_results(model, data, device=device)
        return results

    def get_experiment_name(self):
        name = self.config["name"]
        target = self.config["target"]
        return f"{name}_{target:.0f}".lower()

    def save_model(self, model):
        filename = self.get_experiment_name() +".pt"
        torch.save(model, os.path.join(LOG_DIR, self.foldername, filename))

    def save_results(self, results, tag):
        filename = self.get_experiment_name() + tag +".json"
        with open(os.path.join(LOG_DIR, self.foldername, filename), 'w') as f:
            json.dump(results, f)

    def run_experiment(self):
        self.setup_logging()
        
        logging.info('Loading model...')
        model = self.load_model()
        
        logging.info('Loading dataset...')
        dataset = self.load_data()

        logging.info('Evaluating original model...')
        original_results = self.evaluate(model, dataset)
        logging.info("Original Results:" + str(original_results))
        plot.print_results(**original_results)

        logging.info('Saving original results...')
        self.save_results(original_results, "_original_results")
        
        logging.info('Compressing model...')
        compressed_model = self.compress(model, dataset)
        
        logging.info('Saving compressed model...')
        self.save_model(compressed_model)
        
        logging.info('Evaluating compressed model...')
        compressed_results = self.evaluate(compressed_model, dataset)
        logging.info("Compressed Results:" + str(compressed_results))
        plot.print_before_after_results(original_results, compressed_results)
        
        logging.info('Saving evaluation results...')
        self.save_results(compressed_results, "_compressed_results")
        
        logging.info('Experiment completed successfully')

        self.writer.close()


# Method to load config files
def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as error:
            print(f"Error parsing YAML file: {error}")
            return None

# Method to create compression actions from config file      
def create_compression_action(action_dict):
    action_type = action_dict["type"]
    if action_type == CompressionType.pruning:
        return PruningAction(
            name=action_dict["name"],
            technique=action_dict["technique"],
            sparsity=action_dict["sparsity"],
            strategy=action_dict["strategy"],
            settings=action_dict.get("settings", {}),
        )
    elif action_type == CompressionType.quantization:
        return QuantizationAction(
            name=action_dict["name"],
            technique=action_dict["technique"],
            settings=action_dict.get("settings", {}),
        )
    elif action_type == CompressionType.distillation:
        return DistillationAction(
            name=action_dict["name"],
            technique=action_dict["technique"],
            settings=action_dict.get("settings", {}),
        )
    else:
        raise ValueError(f"Unknown compression action type: {action_type}")


def process_configs(experiment_config, compression_actions_config):
    compression_sets = {}
    for action_set in compression_actions_config["compression_sets"]:
        compression_sets[action_set["name"]] = [
            create_compression_action(action_dict)
            for action_dict in action_set["actions"]
        ]

    for experiment in experiment_config["experiments"]:
        if experiment["type"] in compression_sets:
            experiment["compression_actions"] = compression_sets[experiment["type"]]


    return experiment_config


def main():
    experiment_config = load_yaml_file("config.yml")
    compression_actions_config = load_yaml_file("compression_actions.yml")
    processed_config = process_configs(experiment_config, compression_actions_config)

    experiments = processed_config["experiments"]

    for experiment_config in experiments:
        print("Running experiment: ", experiment_config["name"])
        experiment = ExperimentManager(experiment_config)
        experiment.run_experiment()



if __name__ == "__main__":
    main()
