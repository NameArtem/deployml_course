# up to parent
import sys
sys.path.append("../..")

import numpy as np
import pandas as pd
import joblib

from src.main.preprocessors import *
from config.config import *
import pytest

# передача параметров: имя - список вариантов
# можно зафиксировать ожидаемые результаты
# пример: средние резульаты по предикту по группам (разделенные по индекс или параметрам)
@pytest.mark.parametrize("n_estimators, max_depth, min_samples_leaf",
                         [(N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_LEAF),
                          (10, 10, 0.1),
                          (100, 2, 0)])

def test_parameteric(n_estimators, max_depth, min_samples_leaf):
    dp = r'%s' % os.path.abspath(os.path.join(os.path.dirname("src"), os.pardir, os.pardir, 'data')).replace('\\', '/')
    data_train = generate_test_df(DATA_PATH, CLEAN_DATA, 5)

    rfm = RFModel(n_estimators = n_estimators,
                 max_depth = max_depth,
                 min_samples_leaf = min_samples_leaf)

    if n_estimators == 32:
        rfm.fit(data_train[MODEL_FEATURES], data_train[TARGET_NAME])
        assert np.allclose(rfm.predict(data_train[MODEL_FEATURES]), rfm.predict(data_train[MODEL_FEATURES]), rtol=0.1)
    else:
        with pytest.raises(TypeError):
            rfm.fit(data_train[MODEL_FEATURES], ",".join(data_train[TARGET_NAME]))