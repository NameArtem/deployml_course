from src.main.simple_etl import *
from src.main.train_pipeline import *
from src.main.predict import *

import logging

import pandas as pd
import numpy as np




if __name__ == "__main__":
    import datetime

    dif_val = 7
    try:
        # смотрим последнюю дату изменения файла
        creation_date = os.path.getatime(f"{DATA_PATH}{CLEAN_DATA}")
        now = datetime.datetime.now().timestamp()
        days = (now - creation_date) / 60 / 60 / 24

        # если больше 7 дней, то пересоздаем
        if days >= dif_val:
            # в etl / сохранить дата сет
            etl_as_is(DATA_FILE_STATION,
                      DATA_FILE_WEATHER,
                      STATION_NES_NAMES,
                      ZIP_CODE,
                      DATA_FILE_TRIP,
                      TRIP_NES_NAMES,
                      MAP_ID, MIND, MAXD, FREQ,
                      FEATURE_NAME,
                      TARGET_NAME,
                      DATA_PATH, CLEAN_DATA, SEED)

    except FileNotFoundError:
        # если файл не найдет, то создаем его
        etl_as_is(DATA_FILE_STATION,
                  DATA_FILE_WEATHER,
                  STATION_NES_NAMES,
                  ZIP_CODE,
                  DATA_FILE_TRIP,
                  TRIP_NES_NAMES,
                  MAP_ID, MIND, MAXD, FREQ,
                  FEATURE_NAME,
                  TARGET_NAME,
                  DATA_PATH, CLEAN_DATA, SEED)

    # учим модель
    data = pd.read_csv(f"{DATA_PATH}{CLEAN_DATA}")

    X_train, X_test, y_train, y_test = train_test_split(
                                            data[MODEL_FEATURES],
                                            data[TARGET_NAME],
                                            test_size=0.3,
                                            random_state=SEED)
    run_training(X_train, y_train)

    # предикт
    predicts = make_prediction(X_test)
    model = joblib.load(filename=f"{DATA_PATH}{MODEL_NAME}")
    print("RMSE in model: %.2f" % model.get_metric(X_test, y_test))
