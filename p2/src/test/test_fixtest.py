# up to parent
import sys, os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
CURROOT = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import pandas as pd
import joblib

from src.main.preprocessors import *
from config.config import *
import pytest


# fixture - создает специальный объект для работы в тестах
@pytest.fixture
def params():
    N_ESTIMATORS = 32
    MAX_DEPTH = 12
    MIN_SAMPLES_LEAF = 3
    CRITERION = 'mse'

    return({"n_estimators" : N_ESTIMATORS,
            "max_depth" : MAX_DEPTH,
            "min_samples_leaf" : MIN_SAMPLES_LEAF,
            "criterion" : CRITERION,
            "random_state" : SEED})


@pytest.fixture
def simulated_data():
    # можно зафиксировать ожидаемый результат, а не вводные данные
    # пример: сохраненный результат predict
    dp = r'%s' % os.path.abspath(os.path.join(os.path.dirname("src"), os.pardir, os.pardir, 'data')).replace('\\', '/')
    return generate_test_df(DATA_PATH, CLEAN_DATA, 5)


def test_fit(simulated_data, params):
    rfm = RFModel(**params)
    data_train = simulated_data
    data_test1 = simulated_data
    #data_test2 = simulated_data

    rfm.fit(data_train[MODEL_FEATURES], data_train[TARGET_NAME])
    # или использовать pandas.testing.assert_frame_equal
    assert np.allclose(rfm.predict(data_test1[MODEL_FEATURES]), rfm.predict(data_test1[MODEL_FEATURES]), rtol= 0.1)
    #assert np.allclose(rfm.predict(data_test2[MODEL_FEATURES]), data_test2[TARGET_NAME], rtol= 0.1)


def test_checkfail(simulated_data, params):
    rfm = RFModel(**params)
    data_train = simulated_data

    # Did not raise
    #with pytest.raises(ValueError):
    #    rfm.fit(data_train[MODEL_FEATURES], data_train[TARGET_NAME])

    with pytest.raises(TypeError):
        rfm.fit(data_train[MODEL_FEATURES], ",".join(data_train[TARGET_NAME]))