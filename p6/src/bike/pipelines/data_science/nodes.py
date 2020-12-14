import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


def run_training(train_x,
                 train_y,
                 model_params):

    # обучаем модель и сохраняем её
    rf = RandomForestRegressor(**model_params)
    rf_model = rf.fit(train_x, train_y.values.ravel())

    return rf_model


def run_training_wo_params(train_x,
                           train_y):
    # обучаем модель и сохраняем её
    rf = RandomForestRegressor()
    rf_model = rf.fit(train_x, train_y.values.ravel())

    return rf_model