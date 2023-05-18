import sys
sys.path.append('../')
sys.path.append('../src/')


from torch.utils.tensorboard import SummaryWriter
from src.compress import compress_model
from src.interfaces.compression_actions import CompressionType, DistillationAction, PruningAction, QuantizationAction
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
        self.foldername = self.config["dataset"] + "/" + self.config["name"].lower() + "/" + self.config["type"] + "/" + str(self.config["target"])
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

        # Create handlers and formatter
        file_handler = logging.FileHandler(log_filepath)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

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
        name = self.config["name"]
        target = self.config["target"]
        return f"{name}_{target:.0f}".lower()

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


#  Method to process config files
def process_configs(experiment_config, compression_actions_config):
    compression_sets = {}
    for action_set in compression_actions_config["compression_sets"]:
        compression_sets[action_set["name"]] = [
            create_compression_action(action_dict)
            for action_dict in action_set["actions"]
        ]

    for experiment in experiment_config["experiments"]:
        if experiment["action_set"] in compression_sets:
            experiment["compression_actions"] = compression_sets[experiment["action_set"]]

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
    parser.add_argument('--experiment', required=True,
                        help='Index of the experiment to run')

    args = parser.parse_args()

    device = torch.device(args.device)
    config = load_yaml_file(args.config)

    compression_actions_config = load_yaml_file(
        "experiments/compression_actions_config.yml")
    processed_config = process_configs(config, compression_actions_config)

    experiments = processed_config["experiments"]

    experiment = ExperimentManager(
        experiments[int(args.experiment)], device=device)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
