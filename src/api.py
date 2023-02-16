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
import torch.nn.functional as F
import metrics
import evaluation as eval

HOST = "127.0.0.1"
PORT = 8000

app = FastAPI()

origins = ["http://" + HOST + ":" + str(PORT), "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisModel(BaseModel):
    compression_goal: str  # Goal of the compression (model_size, inference_time, energy_usage).
    compression_target: float  # Target value that the model should achieve after compression, as percentage of the original value.
    performance_metric: str  # Metric used to measure the performance of the model.
    performance_target: float  # Target value that the model should achieve after compression.

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            print("String:", value)
            return cls(**json.loads(value))
        else:
            print("Object:", value.keys())
            return value


class CompressionModel(BaseModel):
    type: str  # Type of compression
    name: str  # Target value that the model should achieve after compression, as percentage of the original value.
    # target: float = 69         # Target value that the model should achieve after compression.

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class CompressionList(BaseModel):
    """Passing list of compression actions directly to API is not possible, therefore this model"""

    actions: List[CompressionModel]

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.get("/")
async def home():
    return {"message": "Wow it works!"}


# Analze the given model and return suggested compression actions
@app.post("/analyze")
def analyze(
    settings: AnalysisModel = Form(...),
    model_state: UploadFile = File(...),
    model_architecture: UploadFile = File(...),
):
    print("Settings", settings)
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
    compression_actions: CompressionList = Form(...),
    model_state: UploadFile = File(...),
    model_architecture: UploadFile = File(...),
):

    model_state_file: NamedTemporaryFile = utils.spooled_to_named(
        model_state.file, suffix=".pth"
    )
    model_architecture_file: NamedTemporaryFile = utils.spooled_to_named(
        model_architecture.file, suffix=".py"
    )

    model = torch.load(model_state_file.name)

    # Compress the model
    compressed_model = general.compress_model(model, compression_actions.actions)

    # Evaluate the compressed model
    original_results = eval.get_results(model, mnist.test_loader)
    compressed_results = eval.get_results(compressed_model, mnist.test_loader)

    # Save the compressed model into a temporary file
    compressed_model_file = NamedTemporaryFile(suffix=".pth", delete=False)
    # compressed_model_file.name = "compressed_model.pth"
    torch.save(model, compressed_model_file.name)

    return {
        "original_results": original_results,
        "compressed_results": compressed_results,
        "compressed_model": compressed_model_file,
    }


def main(host, port):
    uvicorn.run("api:app", host=host, port=port, reload=True)


# Run the file to start the api server
if __name__ == "__main__":
    main(HOST, PORT)
