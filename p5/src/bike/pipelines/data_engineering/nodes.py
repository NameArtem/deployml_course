from typing import Any, Dict

import math
from functools import partial
import pandas as pd

from sklearn.model_selection import train_test_split

#step 1
def preparator(station_data,
               trip_data,
               weather_data,
               STATION_NES_NAMES,
               ZIP_CODE,
               TRIP_NES_NAMES,
               MAP_ID):
    # данные о станциях
    station_data.columns = ['Station_id', 'Station', 'Lat', 'Long', 'Dock_Count', 'City', 'Inst_date']
    station_data['Zip'] = station_data['City'].map(ZIP_CODE)
    station_data.drop(['Lat', 'Long', 'City'], inplace=True, axis=1)
    station_data.drop_duplicates(inplace=True)

    trip_data = trip_data[['Trip_ID', 'Start_Date', 'Start_Station', 'End_Date', 'End_Station', 'Subscriber_Type']]
    trip_data['Start_Station'].replace(to_replace=MAP_ID, inplace=True)
    trip_data['End_Station'].replace(to_replace=MAP_ID, inplace=True)


    return (station_data, trip_data, weather_data)


# step 2
def index_creator(df, mind, maxd, freq):
    by = 'Station'
    newCol = 'Datetime'
    return pd.MultiIndex \
             .from_product([df[by].sort_values().unique(),
                            pd.Index(pd.date_range(mind,
                                                   maxd,
                                                   freq=freq))],
                            names=[by, newCol])

# step 3 (weather_data)
def fill_one(series, value):
        #заполнение пропусков в колонке значением
        return series.fillna(value)

def fill_mean_by_col(df, col, interpolate=True, method='linear'):
    # расчет среднего
    # заполнение им пропусков в колонке
    mean_val = df.set_index([col]) \
                          .sort_index() \
                          .groupby(level=[col]).mean()

    df = df.set_index([col]) \
                     .sort_index() \
                     .fillna(value=mean_val, axis=0)
    if interpolate:
        df = df.sort_index() \
               .interpolate(method=method) \
               .reset_index()

    return df


def cleanning(df):
    df['Events'] = fill_one(df['Events'], "Clear")
    df = fill_mean_by_col(df, "Date", interpolate=True, method='time')
    df.set_index(['Date', 'Zip'], inplace=True)
    return df


# step 4 (as pipeline was)
def resample(df, idx, resamplinglist, freq):
    resample_df = pd.DataFrame()

    for col, by in resamplinglist:
        if resample_df.empty:
            resample_df = df.groupby([col]) \
                .resample(freq, on=by) \
                .size().to_frame().rename(columns={0: 'num_trips'}) \
                .reindex(idx, fill_value=0)
        else:
            tmp = df.groupby([col]) \
                .resample(freq, on=by) \
                .size().to_frame().rename(columns={0: 'num_trips'}) \
                .reindex(idx, fill_value=0)
            resample_df = tmp.merge(resample_df,
                                    left_index=True, right_index=True,
                                    suffixes=('_end', '_start'))
    return resample_df

# step 5
def transform(resample_df, station_data, weather_data, trip_data, idx):
    df_agg = resample_df
    # извлекаем дни
    df_agg['Date'] = getDate(df_agg, idx, 'd')
    # извлекаем часы
    df_agg['Hour'] = getDate(df_agg, idx, 'h')
    df_agg.eval('net_rate=num_trips_end-num_trips_start', inplace=True)
    df_agg.drop(['num_trips_end', 'num_trips_start'], inplace=True, axis=1)

    # соединяем данные
    # данные по станциям
    df_agg = df_agg.join(station_data.set_index('Station'))
    # погода
    df_agg.reset_index(inplace=True)
    df_agg.set_index(['Date', 'Zip'], inplace=True)

    df_agg = df_agg.join(weather_data) \
                    .reset_index() \
                    .set_index(['Station', 'Datetime'])

    # определяем тип покупателя (!)
    #df_agg = get_dummies(df_agg, 'Subscriber_Type', dropOrigin=True)
    # Определяем время дня
    df_agg['Is_night'] = df_agg['Hour'].apply(lambda h: 1 if h < 5 or h > 20 else 0)
    # Определяем день недели
    df_agg['Day_of_week'] = df_agg['Date'].apply(lambda dt: dt.weekday())
    # Рабочий или выходной
    df_agg['Is_weekday'] = df_agg['Day_of_week'].apply(lambda s: 0 if s in [5, 6] else 1)

    # Сизоны: зима (0), осень (1), лето (2), осень (3)
    df_agg['Season'] = df_agg['Date'].apply(lambda dt: (dt.month % 12 + 3) // 3 - 1)

    df = get_dummies(trip_data, 'Subscriber_Type', dropOrigin=True)
    end = get_stop(df, idx, 'H', 'End_Station', 'End_Date', drop=True, dropId="Trip_ID")
    start = get_stop(df, idx, 'H', 'Start_Station', 'Start_Date', drop=True, dropId="Trip_ID")

    # общий набор
    df_agg = df_agg.join(end).join(start, lsuffix='_end', rsuffix='_start')
    #self.df_agg = self.df_agg.sample(frac=0.1)

    # Net rate за преведующий час
    df_agg['net_rate_previous_hour'] = df_agg.groupby(['Station', 'Date'])['net_rate'] \
                                                .shift(1).fillna(0)
    df_agg = cyclic_feature(df_agg)

    # OHE Events
    df_agg = get_dummies(df_agg,'Events')

    # OHE Station IDs
    df_agg.reset_index(inplace=True)
    df_agg = get_dummies(df_agg, 'Station', prefix='Station')
    df_agg = get_dummies(df_agg, 'Zip', prefix='Zip')

    df_agg.drop(['Hour', 'Day_of_week', 'Season', 'Datetime', 'Date'],
                       inplace=True, axis=1)

    return df_agg


def split(df_agg, random_state):
    FEATURE_NAME = ['Hour_cosine', 'Hour_sine', 'Day_of_week_cosine', 'Day_of_week_sine', 'Is_weekday',
                    'Is_night', 'Season_cosine', 'Season_sine', 'net_rate_previous_hour', 'Dock_Count',
                    'Fog', 'Fog-Rain', 'Clear', 'Rain', 'Rain-Thunderstorm', 'CloudCover', 'PrecipitationIn',
                    'WindDirDegrees', 'Max Dew PointF', 'Max Gust SpeedMPH', 'Max Humidity', 'Max Sea Level PressureIn',
                    'Max TemperatureF', 'Max VisibilityMiles', 'Max Wind SpeedMPH', 'Mean Humidity',
                    'Mean Sea Level PressureIn', 'Mean TemperatureF', 'Mean VisibilityMiles', 'Mean Wind SpeedMPH',
                    'MeanDew PointF', 'Min DewpointF', 'Min Humidity', 'Min Sea Level PressureIn', 'Min TemperatureF',
                    'Min VisibilityMiles', 'Zip_94041', 'Zip_94063', 'Zip_94107', 'Zip_94301', 'Zip_95113']

    TARGET_NAME = 'net_rate'

    X = df_agg[FEATURE_NAME]
    y = df_agg[TARGET_NAME]

    # Разделение на train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_state)

    return (X_train,
            y_train,
            X_test,
            y_test
            )






def get_stop(df, idx, freq, stop_col, by, drop = True, dropId = "Trip_ID"):

    tmp = df.groupby([stop_col]) \
            .resample(freq, on=by) \
            .sum() \
            .reindex(idx, fill_value=0)

    if drop == True:
        tmp.drop([dropId,], axis=1, inplace = True)
    return tmp


def getDate(df_agg, idx, tp = 'd'):
    if tp == 'd':
        return pd.Series(df_agg.index.get_level_values('Datetime').date) \
                 .apply(pd.Timestamp).to_frame().set_index(idx)
    if tp == 'h':
        return df_agg.index.get_level_values('Datetime').hour.values


def f_sine(t, T):
   return math.sin(2 * math.pi * t / T)

def f_cosine(t, T):
   return math.cos(2 * math.pi * t / T)

def cyclic_feature(df_agg):
    # Час в циклическую фичу
    partial_sine = partial(f_sine, T=23)
    partial_cosine = partial(f_cosine, T=23)
    df_agg['Hour_sine'] = df_agg['Hour'].apply(partial_sine)
    df_agg['Hour_cosine'] = df_agg['Hour'].apply(partial_cosine)
    # День недели в циклическую фичу
    partial_sine = partial(f_sine, T=6)
    partial_cosine = partial(f_cosine, T=6)
    df_agg['Day_of_week_sine'] = df_agg['Day_of_week'].apply(partial_sine)
    df_agg['Day_of_week_cosine'] = df_agg['Day_of_week'].apply(partial_cosine)
    # Сизон в циклическую фичу
    partial_sine = partial(f_sine, T=3)
    partial_cosine = partial(f_cosine, T=3)
    df_agg['Season_sine'] = df_agg['Season'].apply(partial_sine)
    df_agg['Season_cosine'] = df_agg['Season'].apply(partial_cosine)
    return df_agg


def get_dummies(df, col, prefix=None, dropOrigin = True):
    if prefix != None:
        df = df.join(pd.get_dummies(df[col], prefix = prefix))
    else:
        df = df.join(pd.get_dummies(df[col]))

    if dropOrigin:
        df.drop([col], inplace=True, axis=1)

    return df



####################################################################################################
####################################################################################################
####################################################################################################





