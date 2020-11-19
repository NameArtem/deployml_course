# up to parent
import sys, os
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
CURROOT = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import pandas as pd
import joblib

from src.main.preprocessors import RFModel
from config.config import *
from sklearn.model_selection import train_test_split

def test_fittest():
    dp = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "src" ), os.pardir, os.pardir, 'data')).replace('\\','/')
    data = pd.read_csv("{}{}".format(DATA_PATH, CLEAN_DATA)).sample(10)

    X_train, X_test, y_train, y_test = train_test_split(
                                            data[MODEL_FEATURES],
                                            data[TARGET_NAME],
                                            test_size=0.3,
                                            random_state=SEED)


    rfm = joblib.load(filename='{}/model/rf_model'.format(DATA_PATH))

    results = rfm.predict(X_test)

    # https://numpy.org/doc/stable/reference/generated/numpy.allclose.html

    # варианты сравнения:
    # - среднее, мин-макс отклонение, стандартное отклонение
    # - разница с каждым значением , не боль чем n% от значения
    assert np.allclose(results, results)