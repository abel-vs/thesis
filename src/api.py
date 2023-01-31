# Bring in lightweight dependencies
from enum import Enum
from typing import List, Union
from fastapi import Depends, FastAPI, File, UploadFile
import uvicorn
import analysis
from pydantic import BaseModel

app = FastAPI()


class CompressionMethod(str, Enum):
    pruning = "pruning"
    quantization = "quantization"
    distillation = "distillation"

class AnalysisModel(BaseModel):
    compression_goal: str = "model_size"    # Goal of the compression (model_size, inference_time, energy_usage).
    compression_target: float = 0.5         # Target value that the model should achieve after compression, as percentage of the original value.
    performance_metric: str = "accuracy"    # Metric used to measure the performance of the model.
    performance_target: float = 0.9         # Target value that the model should achieve after compression.

class Compression(BaseModel):
    type: str = "pruning"                   # Type of compression
    name: float = 0.5                       # Target value that the model should achieve after compression, as percentage of the original value.
    performance_metric: str = "accuracy"    # Metric used to measure the performance of the model.
    performance_target: float = 0.9         # Target value that the model should achieve after compression.

# GET = Read
# POST = Create
# PUT = Update
# DELETE = Delete

@app.get("/")
async def home():
    return {"message": "Hello"}

# Analze the given model and return suggested compression actions
@app.post("/analyze")
def analyze(settings: AnalysisModel = Depends(), files: List[UploadFile] = File(...)):
    model_state = files[0].file
    model_architecture = files[1].file

    # compression_actions = []
    compression_actions = analysis.analyze(
        model_state, 
        model_architecture, 
        settings.compression_goal,
        settings.compression_target, 
        settings.performance_metric, 
        settings.performance_target) 

    return {"suggested_compression_actions": compression_actions}

# Run the file to start the api server
if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)