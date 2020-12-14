import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd

from kedro.context import load_context
from kedro.extras.datasets.api import APIDataSet
from src.bike.pipelines.predictapi_pipeline import pipeline
import json






app = FastAPI()
@app.post("/{model}/predictor")
async def predictor(model: str = 'rf_model'):
    """
        API для предикта по модели

        outputs:
            row_index: int
            data: string
            predict: float

    """
    if model == "rf_model":
        context = load_context("")
        output = context.run(pipeline_name='predict_api')

    return output



if __name__ == "__main__":
    #predictor('rf_model')
    uvicorn.run(app, host="0.0.0.0", port=6789)