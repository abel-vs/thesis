import sys

sys.path.append('../')
sys.path.append('../src/')


from torch.utils.tensorboard import SummaryWriter
from src.compress import compress_model
from src.interfaces.compression_actions import CompressionCategory, DistillationAction, PruningAction, QuantizationAction, create_compression_action
import src.interfaces.dataset_models as data
import src.compression.quantization as quant
import src.general as general
import src.plot as plot
import src.analysis as analysis
import src.evaluation as eval
import json
import os
import logging
import yaml
import torch
import argparse


LOG_DIR = "/workspace/volume/experiments/"


class ExperimentManager:
    """
    Class for easily executing experiments

    """

    def __init__(self, experiment_config, device=None, transforms=None):
        self.config = experiment_config
        if self.config["foldername"] is not None:
            self.foldername = self.config["foldername"] + "/" + self.config["name"] 
        else:
            self.foldername = self.config["dataset"] + "/" + self.config["model"].lower() + "/" + self.config["objective"] + "/" + str(self.config["compression_target"])
        self.device = device if device is not None else general.get_device()
        self.transforms = transforms

    # Method to set up logging and tensorboard
    def setup_logging(self):
        os.makedirs(os.path.join(LOG_DIR, self.foldername), exist_ok=True)

        log_filename = f"{self.get_experiment_name()}.log"
        log_filepath = os.path.join(LOG_DIR, self.foldername, log_filename)

        # Create a new logger instance
        self.logger = logging.getLogger(self.get_experiment_name())
        self.logger.setLevel(logging.INFO)

        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create handlers and formatter
        file_handler = logging.FileHandler(log_filepath)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        # Set the logger for the global logging module
        logging.root = self.logger

        logging.info("EXPERIMENT: " + self.get_experiment_name())

        self.writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, self.foldername, "tensorboard"))

    # Method to load model
    def load_model(self):
        model = torch.load(self.config['model_path'], map_location=self.device)
        return model

    # Method to load dataset
    def load_data(self):
        dataset = data.get_supported_dataset(self.config['dataset'])

        if "settings" in self.config:
            settings = self.config["settings"]
            train_batch_size = settings["train_batch_size"]
            test_batch_size = settings["test_batch_size"]
            dataset.set_batch_sizes(train_batch_size, test_batch_size)

        return dataset

    # Method to compress model
    def compress(self, model, dataset):
        # Check if config has compression_actions key
        if "compression_actions" in self.config:
            compression_actions = self.config["compression_actions"]
        else:
            compression_actions = analysis.analyze(
                model, dataset, type="size", compression_target=self.config["target"], settings={})
        logging.info("Compression Actions:")
        for action in compression_actions:
            logging.info(action)

        save_path = os.path.join(LOG_DIR, self.foldername, self.get_experiment_name() + ".pt")
        compressed_model = compress_model(
            model, dataset, compression_actions, writer=self.writer, device=self.device, save_path=save_path)
        return compressed_model

    # Method to evaluate model
    def evaluate(self, model, data):
        if quant.is_quantized(model):
            device = torch.device("cpu")
        else:
            device = self.device
        results = eval.get_results(model, data, device=device)
        return results

    # Method to create experiment name based on config file
    def get_experiment_name(self):
        if self.config["name"] is None:
            model = self.config["model"]
            target = self.config["compression_target"]
            return f"{model}_{target:.0f}".lower()
        else:
            return self.config["name"].lower()

    # Method to save model
    def save_model(self, model):
        filename = self.get_experiment_name() + ".pt"
        torch.save(model, os.path.join(LOG_DIR, self.foldername, filename))

    # Method to save results
    def save_results(self, results, tag):
        filename = self.get_experiment_name() + tag + ".json"
        with open(os.path.join(LOG_DIR, self.foldername, filename), 'w') as f:
            json.dump(results, f, indent=4)

    # Method to save experiment settings
    def save_settings(self):
        filename = self.get_experiment_name() + "_settings.json"
        serialized_config = serialize_config(self.config)
        with open(os.path.join(LOG_DIR, self.foldername, filename), 'w') as f:
            json.dump(serialized_config, f, indent=4)

    # Method to run experiment, this is the main method
    def run_experiment(self):
        self.setup_logging()

        logging.info('Loading model...')
        model = self.load_model()

        logging.info('Loading dataset...')
        dataset = self.load_data()

        if self.transforms is not None:
            dataset.set_transforms(self.transforms)

        logging.info('Saving experiment settings...')
        self.save_settings()

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
        # Reset global logger
        logging.getLogger().handlers = []

        


# Method to load config files
def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as error:
            print(f"Error parsing YAML file: {error}")
            return None


#  Method to process config files
def process_configs(experiment_config, compression_actions_config):
    action_sets_dict = {action_set["name"]: action_set["actions"] for action_set in compression_actions_config["compression_sets"]}

    for experiment in experiment_config["experiments"]:
        if "action_set" in experiment:
            actions = action_sets_dict.get(experiment["action_set"])
            if actions:
                compression_actions = [
                    create_compression_action({
                        **action_dict, 
                        "objective": experiment.get("objective"), 
                        "performance_target": experiment.get("performance_target"), 
                        "compression_target": experiment.get("compression_target")
                    })
                    for action_dict in actions
                ]
                experiment["compression_actions"] = compression_actions
            else:
                raise ValueError(f"Action set {experiment['action_set']} not found in compression actions config")
        else:
            raise ValueError(f"Experiment doesn't contain action_set field")

    return experiment_config


# Method that prepares the config file for saving
def serialize_config(config):
    config = config.copy()
    actions = config.pop("compression_actions", None)
    if actions is not None:
        config["compression_actions"] = [str(action) for action in actions]
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True,
                        help='Specify the device to use (e.g., "cuda:0")')
    parser.add_argument('--config', required=True,
                        help='Path to the configuration file')
    parser.add_argument('--actions', required=False,
                        help='Path to the actions configuration file')
    parser.add_argument('--experiment', required=False,
                        help='Index of the experiment to run')
                        

    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_yaml_file(args.config)

    if args.actions is not None:
        compression_actions_config = load_yaml_file(args.actions)
    else:
        compression_actions_config = load_yaml_file("experiments/compression_actions_config.yml")
    
    processed_config = process_configs(config, compression_actions_config)
    experiments = processed_config["experiments"]


    if args.experiment is None:
        for experiment in experiments:
            experiment = ExperimentManager(
                experiment, device=device)
            experiment.run_experiment()
    else:
        experiment = ExperimentManager(
            experiments[int(args.experiment)], device=device)
        experiment.run_experiment()


if __name__ == "__main__":
    main()
