# Bring in lightweight dependencies
import sys
sys.path.append('/home/abel/Development/thesis')
import json
from tempfile import NamedTemporaryFile
from typing import List

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import analysis
import evaluation as eval
import plot
import utils
from src.compress import compress_model
from src.interfaces.compression_actions import create_compression_action
from src.interfaces.dataset_models import get_supported_dataset

HOST = "0.0.0.0"
PORT = 8000

app = FastAPI()

origins = ["http://" + HOST + ":" +
           str(PORT), "*", "http://localhost:3000", "http://localhost:3001", "http://localhost:6969"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class APIModel(BaseModel):
    """Base model for API endpoints"""

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class AnalysisSettings(APIModel):
    # Dataset used for the analysis.
    dataset: str
    # Goal of the compression (model_size, inference_time, energy_usage).
    compression_goal: str
    # Target value that the model should achieve after compression, as percentage of the original value.
    compression_target: float
    # Metric used to measure the performance of the model.
    performance_metric: str
    # Target value that the model should achieve after compression.
    performance_target: float


class CompressionAction(APIModel):
    type: str          # Type of compression
    name: str          # TName of specific technique
    settings: dict     # Extra settings dependent on the compression action


class CompressionSettings(APIModel):
    actions: List[CompressionAction]
    dataset: str
    performance_target: float
    compression_type: str
    compression_target: float


class ModelDefinition(APIModel):
    name: str
    type: str


@app.get("/")
async def home():
    return {"message": "Tool X API is running!"}


# Analze the given model and return suggested compression actions
@app.post("/analyze")
def analyze(
    settings: AnalysisSettings = Form(...),
    model_definition: ModelDefinition = Form(...),
    model_state: UploadFile = File(...),
    model_architecture: UploadFile = File(...),
):
    print("Settings:", settings)

    # Load the model
    model_files = utils.prepare_model_params(model_state, model_architecture)
    model = utils.import_model(*model_files, model_definition)
    # Load the dataset
    dataset = get_supported_dataset(settings.dataset)

    compression_actions = analysis.analyze(
        model,
        dataset,
        settings.compression_goal,
        settings.compression_target,
        settings.performance_target,
        settings,
    )

    return {"compression_actions": compression_actions, "settings": settings}


# Analze the given model and return suggested compression actions
@app.post("/compress")
def compress(
    settings: CompressionSettings = Form(...),
    model_definition: ModelDefinition = Form(...),
    model_state: UploadFile = File(...),
    model_architecture: UploadFile = File(...),
):

    # Load the model
    model_files = utils.prepare_model_params(model_state, model_architecture)
    model = utils.import_model(*model_files, model_definition)

    dataset = get_supported_dataset(settings.dataset)


    compression_actions = list(map(lambda action: create_compression_action(action, settings.compression_type), settings.actions))


    plot.print_header("Compression Started")

    # Compress the model
    compressed_model = compress_model(model, dataset, compression_actions)

    plot.print_header("Compression Complete")

    # Evaluate the compressed model
    compressed_results = eval.get_results(compressed_model, dataset)

    # Save the compressed model into a temporary file
    compressed_model_file = NamedTemporaryFile(suffix=".pth", delete=False)
    # compressed_model_file.name = "compressed_model.pth"
    torch.save(model, compressed_model_file.name)

    return {
        "compressed_results": compressed_results,
        "compressed_architecture": str(compressed_model),
        "compressed_model": compressed_model_file,
    }


# Evaluate the performance of the model and return the results
@app.post("/evaluate")
def evaluate(
    dataset: str = Form(...),
    model_definition: ModelDefinition = Form(...),
    model_state: UploadFile = File(...),
    model_architecture: UploadFile = File(...),
):
    
    print("Evaluating model...")

    # Load the model
    model_files = utils.prepare_model_params(model_state, model_architecture)
    print("Check")
    model = utils.import_model(*model_files, model_definition)

    dataset = get_supported_dataset(dataset)

    print("Model:", model)
    print("Dataset:", dataset)

    # Evaluate the model
    results = eval.get_results(model, dataset)

    print("Results:", results)

    return results


# Analze the given model and return suggested compression actions
@app.post("/get-modules-methods")
async def get_modules(
    file: UploadFile = File(...),
):

    modules = utils.extract_modules(file)
    methods = utils.extract_methods(file)

    return {"modules": modules, "methods": methods}


def main(host, port):
    uvicorn.run("api:app", host=host, port=port, reload=True)


# Run the file to start the api server
if __name__ == "__main__":
    main(HOST, PORT)
