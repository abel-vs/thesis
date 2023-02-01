# Bring in lightweight dependencies
from enum import Enum
from typing import List, Union
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import analysis
from pydantic import BaseModel


HOST = "127.0.0.1"
PORT = 8000

app = FastAPI()

origins = [
    "http://" + HOST + ":" + str(PORT),
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisModel(BaseModel):
    compression_goal: str   # Goal of the compression (model_size, inference_time, energy_usage).
    compression_target: float       # Target value that the model should achieve after compression, as percentage of the original value.
    performance_metric: str    # Metric used to measure the performance of the model.
    performance_target: float       # Target value that the model should achieve after compression.

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
    return {"message": "Wow it works!"}


# Analze the given model and return suggested compression actions
@app.post("/analyze")
def analyze(settings: AnalysisModel = Depends(), model_state: UploadFile = File(...), model_architecture: UploadFile = File(...)):
    compression_actions = analysis.analyze(
        model_state, 
        model_architecture, 
        settings.compression_goal,
        settings.compression_target, 
        settings.performance_metric, 
        settings.performance_target) 

    return {"compression_actions": compression_actions, "settings": settings}


def main(host, port):
    uvicorn.run("api:app", host=host, port=port, reload=True)

# Run the file to start the api server
if __name__ == "__main__":
    main(HOST, PORT)