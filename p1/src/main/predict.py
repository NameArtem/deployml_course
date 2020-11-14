import pandas as pd
import logging
import joblib
from config.config import *


def make_prediction(input_data):

    # загрузка модели
    model = joblib.load(filename=f"{DATA_PATH}{MODEL_NAME}")
    
    results = model.predict(input_data)

    return results

   
if __name__ == '__main__':
    pass

