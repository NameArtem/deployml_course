import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics


def make_prediction(test_x,
                    model):
    # импорты в функции

    predict = model.predict(test_x)

    return predict



# score
def rmse_cv(test_x, test_y, model):
    # Определяем функцию для проверки качества: RMSE
    rmse = np.sqrt(-cross_val_score(model,
                                    test_x, test_y.values.ravel(),
                                    scoring='neg_mean_squared_error',
                                    cv=3))

    return rmse.mean()