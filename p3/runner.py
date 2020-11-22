import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd

from kedro.context import load_context
from kedro.extras.datasets.api import APIDataSet
from src.bike.pipelines.predictapi_pipeline import pipeline
import json


app = FastAPI()
@app.get("/{model}/predictor")
async def predictor(model):
    if model == 'rf_model':
        context = load_context("")
        output = context.run(pipeline_name='predict_api')

    return output



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=6789)