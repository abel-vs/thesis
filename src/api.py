# Import additional required modules
import redis
import uuid
import pickle
import sys
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
from src.compression.quantization import is_quantized
from src.interfaces.compression_actions import create_compression_action
from src.interfaces.dataset_models import get_supported_dataset
from utils import import_model

HOST = "localhost"
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
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class AnalysisSettings(APIModel):
    dataset: str
    compression_goal: str
    compression_target: float
    performance_metric: str
    performance_target: float


class CompressionAction(APIModel):
    type: str
    name: str
    settings: dict


class CompressionSettings(APIModel):
    actions: List[CompressionAction]
    dataset: str
    performance_target: float
    compression_type: str
    compression_target: float


class ModelDefinition(APIModel):
    name: str
    type: str

r = redis.Redis(host='localhost', port=6379, db=0)


@app.get("/")
async def home():
    return {"message": "EasyCompress API is running!"}


@app.post("/register")
async def register_model_files(
    model_state: UploadFile = File(...),
    model_architecture: UploadFile = File(...),
    model_definition: ModelDefinition = Form(...)
):
    # Read files content
    state_contents = await model_state.read()
    arch_contents = await model_architecture.read()
    model_definition_dict = model_definition.dict()

    # Generate a unique id for these files
    model_id = uuid.uuid4().hex

    # Store files and definition in a dictionary and pickle it
    model_data = {
        "state": state_contents,
        "architecture": arch_contents,
        "definition": model_definition_dict,
    }
    pickled_model_data = pickle.dumps(model_data)

    # Cache the pickled data in Redis
    r.set(f"model:{model_id}", pickled_model_data)
    r.expire(f"model:{model_id}", 3600)  # 3600 seconds = 1 hour

    # Return the id so the client can use it in future requests
    return {"model_id": model_id}


def load_model_by_id(model_id: str):
    # Retrieve the pickled data from Redis
    pickled_model_data = r.get(f"model:{model_id}")
    
    if pickled_model_data is None:
        raise ValueError("Model not found")

    # Unpickle the data
    model_data = pickle.loads(pickled_model_data)

    # Extract each piece of data
    state_contents = model_data["state"]
    arch_contents = model_data["architecture"]
    model_definition = ModelDefinition(**model_data["definition"])

    # Prepare the model parameters
    model_files = utils.prepare_model_params(state_contents, arch_contents)
    # Import the model
    model = import_model(*model_files, model_definition)

    return model


@app.post("/analyze")
async def analyze_model(
    model_id: str = Form(...),  
    settings: AnalysisSettings = Form(...),
):
    model = load_model_by_id(model_id)

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


@app.post("/compress")
async def compress(
        model_id: str = Form(...),
    settings: CompressionSettings = Form(...),
):
    model = load_model_by_id(model_id)

    dataset = get_supported_dataset(settings.dataset)


    compression_actions = list(map(lambda action: create_compression_action(action, settings.compression_type), settings.actions))


    plot.print_header("Compression Started")

    # Compress the model
    compressed_model = compress_model(model, dataset, compression_actions)

    plot.print_header("Compression Complete")


    # for name, module in compressed_model.named_modules():
    #     print(f"Name: {name}, Class: {type(module).__module__}")

    print(compressed_model)

    print("Is Quantized", is_quantized(compressed_model))
    if is_quantized(compressed_model):
        device = "cpu"
    else: 
        device = "cuda"

    # Evaluate the compressed model
    compressed_results = eval.get_results(compressed_model, dataset, device=device)

    # Save the compressed model into a temporary file
    compressed_model_file = NamedTemporaryFile(suffix=".pth", delete=False)
    # compressed_model_file.name = "compressed_model.pth"
    # torch.save(model.state_dict, compressed_model_file.name)

    return {
        "compressed_results": compressed_results,
        "compressed_architecture": str(compressed_model),
        "compressed_model": compressed_model_file,
    }

@app.post("/evaluate")
def evaluate(
    dataset: str = Form(...),
    model_id: str = Form(...),
):

    model = load_model_by_id(model_id)
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



if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
