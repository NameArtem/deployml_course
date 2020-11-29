import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def get_api_date(data):
    return data.headers['date'], \
           data.json()['index'], \
           pd.DataFrame.from_dict(data.json()['data'], orient='index').T


def make_prediction(test_x,
                    model):
    # импорты в функции

    predict = model.predict(test_x)

    return predict


def serve_result(predict_date, row_index, predict):
    """
    answer = {"predict_date": _90.headers['date'],
              "row_index": _90.json()['index'],
              "predict": float(rf_model.predict(df))
              }
    """

    return predict_date, row_index, float(predict) if len(predict) == 1 else list(predict)


