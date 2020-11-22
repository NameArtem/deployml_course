from config.config import *
from src.main.quality import *
from src.main.simple_etl import *
from src.main.preprocessors import *
from src.main.train_pipeline import *
from src.main.predict import *
import logging
import pandas as pd
import numpy as np
import yaml




if __name__ == "__main__":
    import time
    import datetime
    from __init__ import __version__

    # логгирование
    _logger = logging.getLogger(__name__)
    _logger.disabled = False

    # сомтрим на существование файла с параметрами
    try:
        params = yaml.safe_load(open('modelparam.yaml'))
        _logger.info(
            f"File 'modelparam.yaml' was found",
        )
    except FileNotFoundError:
        _logger.warning(
            f"File 'modelparam.yaml' was not found | Create new",
            )

        params = {'model_results':{'metric': {'rmse': None}}}

    dif_val = 7
    try:
        # смотрим последнюю дату изменения файла
        creation_date = os.path.getatime(f"{DATA_PATH}{CLEAN_DATA}")
        now = datetime.datetime.now().timestamp()
        days = (now - creation_date) / 60 / 60 / 24
        # если больше 7 дней, то пересоздаем
        if days >= dif_val:
            _logger.info(
                f"'{CLEAN_DATA}' is old | Create new",
            )
            # сохранить дата сет
            etl_as_is(DATA_FILE_STATION,
                      DATA_FILE_WEATHER,
                      STATION_NES_NAMES,
                      ZIP_CODE,
                      DATA_FILE_TRIP,
                      TRIP_NES_NAMES,
                      MAP_ID, MIND, MAXD, FREQ,
                      FEATURE_NAME,
                      TARGET_NAME,
                      DATA_PATH, SEED)
    except FileNotFoundError:
        _logger.warning(
            f"File '{CLEAN_DATA}' was not found | Create new",
        )
        etl_as_is(DATA_FILE_STATION,
                  DATA_FILE_WEATHER,
                  STATION_NES_NAMES,
                  ZIP_CODE,
                  DATA_FILE_TRIP,
                  TRIP_NES_NAMES,
                  MAP_ID, MIND, MAXD, FREQ,
                  FEATURE_NAME,
                  TARGET_NAME,
                  DATA_PATH, SEED)

    # учим модель
    data = pd.read_csv(f"{DATA_PATH}{CLEAN_DATA}")
    X_train, X_test, y_train, y_test = train_test_split(
                                            data[MODEL_FEATURES],
                                            data[TARGET_NAME],
                                            test_size=0.3,
                                            random_state=SEED)
    run_training(X_train, y_train)

    # делаем предикт
    predicts = make_prediction(X_test)
    model = joblib.load(filename=f"{DATA_PATH}{MODEL_NAME}")
    print("RMSE in model: %.2f" % model.get_metric(X_test, y_test))
    pred = round(model.get_metric(X_test, y_test), 3)
    params['model_results']['metric']['rmse'] = round(model.get_metric(X_test, y_test), 3)
    params['model_results']['best_params'] = model.get_params()

    # сохраняем метрики
    with open('modelparam.yaml', 'w') as f:
        yaml.dump(params, f)

    _logger.warning(
        f"""Predict with model {MODEL_NAME[len('/model/'):]}\nPrediction metric : {pred}"""
    )

    # print("RMSE in model: %.2f"  % rf.get_metric(X_test, y_test))
    #
    # # Root Mean Squared Error
    # print("RMSE fact: %.2f"  % math.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))
    #
    # metric = Metrics(model, X_test, y_test)
    # print("RMSE in Metrics: %.2f"  % metric.rmse_cv())

import random