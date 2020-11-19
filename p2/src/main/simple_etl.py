import logging
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

_logger = logging.getLogger(__name__)

def etl_as_is(DATA_FILE_STATION,
              DATA_FILE_WEATHER,
              STATION_NES_NAMES,
              ZIP_CODE,
              DATA_FILE_TRIP,
              TRIP_NES_NAMES,
              MAP_ID, MIND, MAXD, FREQ,
              FEATURE_NAME,
              TARGET_NAME,
              ROOT, CLEAN_DATA, SEED):

    from config.logging_conf import get_console_header
    from src.main.quality import DataChecker
    from src.main.preprocessors import FeatureGen, Pipeline
    from config.config import CLEAN_DATA

    df_checker = DataChecker()

    # данные о станциях
    station_data = pd.read_csv(filepath_or_buffer=DATA_FILE_STATION,
                               sep=',',
                               header=0,
                               names=STATION_NES_NAMES)

    station_data['Zip'] = station_data['City'].map(ZIP_CODE)
    station_data.drop(['Lat', 'Long', 'City'], inplace=True, axis=1)
    station_data.drop_duplicates(inplace=True)

    # проверка пустот (можно оставить для фиксации статусов)
    # print('check if any data is missing:', df_checker.checkNull(station_data))

    trip_data = pd.read_csv(filepath_or_buffer=DATA_FILE_TRIP,
                            sep=',',
                            infer_datetime_format=True,
                            dayfirst=True,
                            parse_dates=['Start_Date', 'End_Date']  #
                            )[TRIP_NES_NAMES]

    trip_data['Start_Station'].replace(to_replace=MAP_ID, inplace=True)
    trip_data['End_Station'].replace(to_replace=MAP_ID, inplace=True)

    weather_data = pd.read_csv(filepath_or_buffer=DATA_FILE_WEATHER,
                               sep=',',
                               infer_datetime_format=True,
                               dayfirst=True,
                               parse_dates=['Date'])  #

    # preprocessing
    # генерация новых параметров
    fg = FeatureGen(station_data, 'Station', MIND, MAXD, FREQ)

    # NaN => 'Clear'
    weather_data['Events'] = fg.fill_one(weather_data['Events'], "Clear")
    weather_data = fg.fill_mean_by_col(weather_data, "Date", interpolate=True, method='time')
    weather_data.set_index(['Date', 'Zip'], inplace=True)

    # init pipeline
    pp = Pipeline(trip_data,
                  [("Start_Station", "Start_Date"), ("End_Station", "End_Date")],
                  FREQ,
                  fg.idx,
                  FEATURE_NAME,
                  TARGET_NAME)

    pp.transform(station_data, weather_data)
    full_df = pp.fit(3, SEED)

    full_df.to_csv(f"{ROOT}{CLEAN_DATA}", header=True, index=False)


if __name__ == '__main__':
    import argparse
    import sys
    import os
    import time, os, yaml
    import datetime

    #для чтения агрументов, если надо перезагрузить данные
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=bool)
    args = parser.parse_args()

    PACKAGE_PARENT = '../..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    CURROOT = os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

    from src.main.quality import *
    from config.config import *
    from src.main.preprocessors import *

    # установка на кол-во дней
    dif_val = 1

    try:
        creation_date = os.path.getatime(f"{CURROOT}{CLEAN_DATA}")
        now = datetime.datetime.now().timestamp()
        days = (now - creation_date) / 60 / 60 / 24

        if days >= dif_val:
            # сохранить дата сет
            etl_as_is(f"{CURROOT}/{DATA_FILE_STATION}",
                      f"{CURROOT}/{DATA_FILE_WEATHER}",
                      STATION_NES_NAMES,
                      ZIP_CODE,
                      f"{CURROOT}/{DATA_FILE_TRIP}",
                      TRIP_NES_NAMES,
                      MAP_ID, MIND, MAXD, FREQ,
                      FEATURE_NAME,
                      TARGET_NAME,
                      f"{CURROOT}/data", CLEAN_DATA, SEED)
        if args.r == True:
            etl_as_is(f"{CURROOT}/{DATA_FILE_STATION}",
                      f"{CURROOT}/{DATA_FILE_WEATHER}",
                      STATION_NES_NAMES,
                      ZIP_CODE,
                      f"{CURROOT}/{DATA_FILE_TRIP}",
                      TRIP_NES_NAMES,
                      MAP_ID, MIND, MAXD, FREQ,
                      FEATURE_NAME,
                      TARGET_NAME,
                      f"{CURROOT}/data", CLEAN_DATA, SEED)
        else:
            print("Data still actual =) ")
    except FileNotFoundError:
        etl_as_is(f"{CURROOT}/{DATA_FILE_STATION}",
                  f"{CURROOT}/{DATA_FILE_WEATHER}",
                  STATION_NES_NAMES,
                  ZIP_CODE,
                  f"{CURROOT}/{DATA_FILE_TRIP}",
                  TRIP_NES_NAMES,
                  MAP_ID, MIND, MAXD, FREQ,
                  FEATURE_NAME,
                  TARGET_NAME,
                  f"{CURROOT}/data", CLEAN_DATA, SEED)

