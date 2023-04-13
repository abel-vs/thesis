# Bring in lightweight dependencies
from enum import Enum
import os
from typing import List, Union
from fastapi import Depends, FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import analysis
from pydantic import BaseModel
import json
import general
import plot
import utils
import importlib
import torch
from tempfile import SpooledTemporaryFile, NamedTemporaryFile
import mnist
import torchvision
import torch.nn.functional as F
import metrics
import evaluation as eval
from dataset_models import supported_datasets

HOST = "127.0.0.1"
PORT = 8000

app = FastAPI()

origins = ["http://" + HOST + ":" +
           str(PORT), "http://localhost:3000", "http://localhost:3001", "http://localhost:3002"]

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

    compression_actions = analysis.analyze(
        model_state,
        model_architecture,
        settings.compression_goal,
        settings.compression_target,
        settings.performance_metric,
        settings.performance_target,
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

    dataset = supported_datasets[settings.dataset]

    # Compress the model
    compressed_model = general.compress_model(
        model, dataset, settings.actions, settings)

    plot.print_header("Compression Complete")

    # Evaluate the compressed model
    original_results = eval.get_results(model, dataset)
    compressed_results = eval.get_results(compressed_model, dataset)

    # Save the compressed model into a temporary file
    compressed_model_file = NamedTemporaryFile(suffix=".pth", delete=False)
    # compressed_model_file.name = "compressed_model.pth"
    torch.save(model, compressed_model_file.name)

    return {
        "original_results": original_results,
        "compressed_results": compressed_results,
        "compressed_model": compressed_model_file,
    }


# Evaluate the performance of the model and return the results
@app.post("/evaluate")
def compress(
    dataset: str = Form(...),
    model_definition: ModelDefinition = Form(...),
    model_state: UploadFile = File(...),
    model_architecture: UploadFile = File(...),
):

    # Load the model
    model_files = utils.prepare_model_params(model_state, model_architecture)
    model = utils.import_model(*model_files, model_definition)

    dataset = supported_datasets[dataset]

    print("Model:", model)
    print("Dataset:", dataset)

    # Evaluate the compressed model
    results = eval.get_results(model, dataset)

    print("Results:", results)

    return results


# Analze the given model and return suggested compression actions
@app.post("/get-modules-methods")
async def analyze(
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
