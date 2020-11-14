import numpy as np
import pandas as pd

import joblib

from config.config import *
from sklearn.model_selection import train_test_split
from src.main.preprocessors import *


def run_training(X_train, y_train):
    # обучаем модель и сохраняем её

    rf = RFModel(n_estimators = N_ESTIMATORS,
                 max_depth = MAX_DEPTH,
                 min_samples_leaf = MIN_SAMPLES_LEAF,
                 criterion = CRITERION,
                 random_state = SEED)

    model = rf.fit(X_train, y_train)


    joblib.dump(model, f"{DATA_PATH}{MODEL_NAME}")


if __name__ == '__main__':
    pass
