import pandas as pd
import logging
import joblib



def make_prediction(input_data, DATA_PATH=None):
    # импорты в функции
    from config.config import MODEL_NAME
    if DATA_PATH == None:
        from config.config import DATA_PATH

    # загрузка модели
    model = joblib.load(filename=f"{DATA_PATH}{MODEL_NAME}")
    
    results = model.predict(input_data)

    _logger = logging.getLogger(__name__)
    _logger.info(
        f"Predict with model {MODEL_NAME}" ,
        f"Inputs: {input_data}",
        f"Prediction: {results}")

    return results

   
if __name__ == '__main__':
    import sys
    import os
        # путь до родительской директории
    PACKAGE_PARENT = '../..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    CURROOT = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
    from config.config import *
    from __init__ import __version__

    X_test = pd.read_csv(f"{CURROOT}/data/split_data/xtest{__version__}.csv")
    y_test = pd.read_csv(f"{CURROOT}/data/split_data/ytest{__version__}.csv")

    predicts = make_prediction(X_test, DATA_PATH=f"{CURROOT}/data")
    model = joblib.load(filename=f"{CURROOT}/data{MODEL_NAME}")

    #сохраняем метрики
    with open('{}/data/metrics/{}'.format(CURROOT, MODEL_NAME[len('/model/'):]), 'w') as f:
        metric = model.get_metric(X_test, y_test.values)
        f.write("RMSE in model: %.2f" % metric)
        print("RMSE in model: %.2f" % metric)


