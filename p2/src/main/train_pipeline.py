import numpy as np
import pandas as pd

import joblib

from sklearn.model_selection import train_test_split


def run_training(X_train, y_train, DATA_PATH = None):
    from src.main.preprocessors import RFModel
    from config.config import N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF, CRITERION, SEED, MODEL_NAME
    if DATA_PATH == None:
        from config.config import DATA_PATH

    # обучаем модель и сохраняем её

    rf = RFModel(n_estimators = N_ESTIMATORS,
                 max_depth = MAX_DEPTH,
                 min_samples_leaf = MIN_SAMPLES_LEAF,
                 criterion = CRITERION,
                 random_state = SEED)

    model = rf.fit(X_train, y_train)

    joblib.dump(model, f"{DATA_PATH}{MODEL_NAME}")

    return model

if __name__ == '__main__':
    import sys
    import os

    PACKAGE_PARENT = '../..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    CURROOT = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

    from config.config import *
    from __init__ import __version__

    data = pd.read_csv(r"{}/data{}".format(CURROOT,
                                           CLEAN_DATA)).sample(10)


    X_train, X_test, y_train, y_test = train_test_split(
                                                        data[MODEL_FEATURES],
                                                        data[TARGET_NAME],
                                                        test_size=0.3,
                                                        random_state=SEED)

    run_training(X_train, y_train, DATA_PATH="{}/data".format(CURROOT))

    X_train.to_csv("{}/data/split_data/xtrain{}.csv".format(CURROOT,__version__), index=False, header=True)
    X_test.to_csv("{}/data/split_data/xtest{}.csv".format(CURROOT,__version__), index=False, header=True)
    y_train.to_csv("{}/data/split_data/ytrain{}.csv".format(CURROOT,__version__), index=False, header=True)
    y_test.to_csv("{}/data/split_data/ytest{}.csv".format(CURROOT,__version__), index=False, header=True)

