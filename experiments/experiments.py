import torch
import yaml
import logging
import os
import sys
import json
sys.path.append('..')
sys.path.append('../src')

import src.interfaces.dataset_models as data
import src.evaluation as eval
import src.analysis as analysis
import src.plot as plot

from src.compress import compress_model



SAVE_PATH = 'results/'

class ExperimentManager:
    """
    Class for easily executing experiments
    
    """
    def __init__(self, experiment_config):
        self.config = experiment_config

    def setup_logging(self):
        log_folder = os.path.join(SAVE_PATH, self.config["name"])
        os.makedirs(log_folder, exist_ok=True)

        log_filename = f"{self.get_experiment_name()}.log"
        log_filepath = os.path.join(log_folder, log_filename)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(message)s',
                            datefmt='%H:%M:%S',
                            handlers=[logging.FileHandler(log_filepath), logging.StreamHandler()])
        
        # #  Redirect print statements to the logger
        # logger = logging.getLogger(__name__)
        # sys.stdout = LoggerWriter(logger, logging.INFO)

    def load_model(self):
        model = torch.load(self.config['model_path'])
        return model

    def load_data(self):
        return data.supported_datasets[self.config['dataset']]

    def compress(self, model, dataset):
        compression_actions = analysis.analyze(model, dataset, type="size", compression_target=self.config["target"], settings={})
        logging.info("Compression Actions:" + str(list(c.name for c in compression_actions)))
        compressed_model = compress_model(model, dataset, compression_actions)
        return compressed_model

    def evaluate(self, model, data):
        results = eval.get_results(model, data)
        return results

    def get_experiment_name(self):
        name = self.config["name"]
        target = self.config["target"]
        return f"{name}_{target:.0f}"

    def save_model(self, model):
        foldername = self.config["name"]
        filename = self.get_experiment_name() +".pt"
        torch.save(model, os.path.join(SAVE_PATH, foldername, filename))

    def save_results(self, results):
        foldername = self.config["name"]
        filename = self.get_experiment_name() +".json"
        with open(os.path.join(SAVE_PATH, foldername, filename), 'w') as f:
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
        
        logging.info('Compressing model...')
        compressed_model = self.compress(model, dataset)
        
        logging.info('Saving compressed model...')
        self.save_model(compressed_model)
        
        logging.info('Evaluating compressed model...')
        compressed_results = self.evaluate(compressed_model, dataset)
        logging.info("Compressed Results:" + str(compressed_results))
        
        logging.info('Saving evaluation results...')
        self.save_results(compressed_results)
        
        logging.info('Experiment completed successfully')


def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as error:
            print(f"Error parsing YAML file: {error}")
            return None
        

def main():
    config = load_yaml_file("config.yml")
    experiments = config["experiments"]

    for experiment_config in experiments:
        print("Running experiment: ", experiment_config["name"])
        experiment = ExperimentManager(experiment_config)
        experiment.run_experiment()


if __name__ == "__main__":
    main()
