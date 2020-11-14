from src.main.quality import *
from src.main.preprocessors import *

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np



def etl_as_is(DATA_FILE_STATION,
              DATA_FILE_WEATHER,
              STATION_NES_NAMES,
              ZIP_CODE,
              DATA_FILE_TRIP,
              TRIP_NES_NAMES,
              MAP_ID, MIND, MAXD, FREQ,
              FEATURE_NAME,
              TARGET_NAME,
              ROOT,
              CLEAN_DATA, SEED):

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


if __name__ == "__main__":
    pass