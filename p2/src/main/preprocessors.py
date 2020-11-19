import math
import pickle

from functools import partial

import numpy as np
import pandas as pd

from collections.abc import Iterable
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection._search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from .quality import *


# получаем имя естиматора
def extract_estimator_name(estimator):

    # проверка естиматор == объект pipeline
    if isinstance(estimator, Pipeline):
        data_type = type(estimator._final_estimator)

    # проверка естиматор == grid search объект
    elif isinstance(estimator, GridSearchCV):
        # проверка естиматор == объект pipeline
        if isinstance(estimator.best_estimator_, Pipeline):
            data_type = type(estimator.best_estimator_._final_estimator)
        else:
            data_type = type(estimator.best_estimator_)

    # если это не pipeline и не grid search объект
    else:
        data_type = type(estimator)
    name = ''.join(filter(str.isalnum, str(data_type).split('.')[-1]))

    return name

# генератор сэмплов для теста
def generate_test_df(DATA_PATH, CLEAN_DATA, sample_prc=5):
    return pd.read_csv("{}{}".format(DATA_PATH, CLEAN_DATA)).sample(10)

# функция удаления старых файлов
def remove_old(*, keep_files):
    """
        Реазиловать по необходимости
    """
    pass



#########################################################
# Classes
#########################################################
class Estimators(GridSearchCV):
    """
    Класс вложенной кросс-валидации на основе k-fold

    Parameters
    ----------
    estimator : array, shape = [n_samples]
                true class, integers in [0, n_classes - 1)
    X : array,   shape = [n_samples, n_classes]
    y : array,   shape = [n_samples, n_classes]
    outer_cv :   shape = [n_samples, n_classes]
    inner_cv :   shape = [n_samples, n_classes]
    param_grid : shape = [n_samples, n_classes]
    scoring :    shape = [n_samples, n_classes]
    n_jobs : int, default 1
    debug : boolean, default Fasle

    Returns
    -------
    grid : GridSearchCV
    """

    def __init__(self, X, y,
                 estimator, param_grid,
                 inner_cv, outer_cv,
                 scoring, n_jobs):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.name = extract_estimator_name(self.estimator)
        self.param_grid = param_grid,
        self.inner_cv = inner_cv, #cv
        self.outer_cv = outer_cv,
        self.scoring = scoring,
        self.n_jobs = n_jobs
        self.debug = False


    def fit(self, X, y):
        # для sklearn pipeline
        return self



    def grid_search(self):
        # Кросс-валидаия

        # резуьтаты общего выполнения
        outer_scores = []

        # разделение трейн-теста
        for k, (training_samples, test_samples) in enumerate(self.outer_cv.split(self.X, self.y)):

            # x
            if isinstance(self.X, pd.DataFrame):
                x_train = self.X.iloc[training_samples]
                x_test = self.X.iloc[test_samples]
            else:
                x_train = self.X[training_samples]
                x_test = self.X[test_samples]

            # y
            if isinstance(self.y, pd.Series):
                y_train = self.y.iloc[training_samples]
                y_test = self.y.iloc[test_samples]
            else:
                y_train = self.y[training_samples]
                y_test = self.y[test_samples]

            # Строим классификатор на лучших параметрах
            print('fold-', k + 1, 'model fitting...')

            self.fit(x_train, y_train)

            # получаем параметры лучшей модели
            if self.debug:
                print('\n\t', self.best_estimator_.get_params()[self.name])

            # оценка качества
            score = self.score(x_test, y_test)

            outer_scores.append(abs(score))
            print('\n\tModel validation score:', abs(score), '\n')

        print('\nHyper-paramters of best model:\n\n',
              self.best_estimator_.get_params()[self.name])

        print('\nFinal model evaluation (mean cross-val scores):',
              np.array(outer_scores).mean())

        return self




class FeatureGen:
    """
        Класс с набором функций по генерации фичей
        для данной модели (для регрессионых моделей)

        Parameters
        ----------
        df : pd.DataFrame
        by :  column name
        mind : дата начала периода
        maxd : дата конца периода
        freq : частота агрегации

        Returns
        -------
        idx : index columns (pd.MultiIndex)

        """

    def __init__(self, df, by,  mind, maxd, freq):
        self.newCol = 'Datetime'
        self.idx = self.index_creator(df, by,  mind, maxd, freq)

    def fill_one(self, series, value):
        #заполнение пропусков в колонке значением
        return series.fillna(value)

    def fill_mean_by_col(self, df, col, interpolate=True, method='linear'):
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

    def index_creator(self, df, by,  mind, maxd, freq):
        # Делаем таблицу с мультИндексами по ID|trip date
        # https://stackoverflow.com/questions/28132668/
        # how-can-i-keep-this-pandas-column-type-as-datetime-when-using-multiindex

        return pd.MultiIndex \
                 .from_product([df[by].sort_values().unique(),
                                pd.Index(pd.date_range(mind,
                                                       maxd,
                                                       freq=freq))],
                                names=[by, self.newCol])


    def getDate(self, df, tp = 'd'):
        if tp == 'd':
            return pd.Series(df.index.get_level_values(self.newCol).date) \
                     .apply(pd.Timestamp).to_frame().set_index(self.idx)
        if tp == 'h':
            return df.index.get_level_values(self.newCol).hour.values

    def get_dummies(self, df, col, dropOrigin = True):
        # OHE по колонкам
        df = df.join(pd.get_dummies(df[col]))
        if dropOrigin:
            df.drop([col], inplace=True, axis=1)

        return df



class Pipeline(KFold):
    """
            Класс реализующий pipeline
            для генрации данных на основе KFolde

            Parameters
            ----------
            df : pd.DataFrame
            resampling :
            freq : частота агрегата
            idx : индексы  pd.MultiIndex
            feature_names : имена столбцов
            target: целевая переменная в столбце (иначе создаем

            Returns
            -------
            pd.DataFrame

            """

    def __init__(self, df, resampling, freq, idx, feature_names, target):
        self.df = df
        self.df_agg = None
        self.freq = freq
        self.newCol = 'Datetime'
        self.feature_names = feature_names
        self.target = target

        # KFold param
        self.n_splits = 1
        self.random_state = None

        try:
            # если объект None или не мультииндекс (для теста)
            if idx != None:
                self.resampleobject(idx, resampling)
        except ValueError:
            # если обхект мультииндекс
            self.resampleobject(idx, resampling)


    def resampleobject(self, idx, resampling):
        self.idx = idx

        if isinstance(resampling, Iterable):
            self.resamplinglist = resampling
        else:
            self.resamplinglist = [resampling, ]

    def resample(self):
        resample_df = pd.DataFrame()

        for col, by in self.resamplinglist:
            if resample_df.empty:
                resample_df = self.df.groupby([col]) \
                                    .resample(self.freq, on=by) \
                                    .size().to_frame().rename(columns={0: 'num_trips'}) \
                                    .reindex(self.idx, fill_value=0)
            else:
                tmp = self.df.groupby([col]) \
                                    .resample(self.freq, on=by) \
                                    .size().to_frame().rename(columns={0: 'num_trips'}) \
                                    .reindex(self.idx, fill_value=0)
                resample_df = tmp.merge(resample_df,
                                          left_index=True, right_index=True,
                                          suffixes=('_end', '_start'))
        return resample_df

    def get_stop(self, stop_col, by, drop = True, dropId = "Trip_ID"):
        tmp = self.df.groupby([stop_col]) \
                .resample(self.freq, on=by) \
                .sum() \
                .reindex(self.idx, fill_value=0)

        if drop == True:
            tmp.drop([dropId,], axis=1, inplace = True)

        return tmp

    def getDate(self, tp = 'd'):
        if tp == 'd':
            return pd.Series(self.df_agg.index.get_level_values(self.newCol).date) \
                     .apply(pd.Timestamp).to_frame().set_index(self.idx)
        if tp == 'h':
            return self.df_agg.index.get_level_values(self.newCol).hour.values


    def f_sine(self, t, T):
       return math.sin(2 * math.pi * t / T)

    def f_cosine(self, t, T):
       return math.cos(2 * math.pi * t / T)

    def cyclic_feature(self):
        # Час в циклическую фичу
        partial_sine = partial(self.f_sine, T=23)
        partial_cosine = partial(self.f_cosine, T=23)

        self.df_agg['Hour_sine'] = self.df_agg['Hour'].apply(partial_sine)
        self.df_agg['Hour_cosine'] = self.df_agg['Hour'].apply(partial_cosine)

        # День недели в циклическую фичу
        partial_sine = partial(self.f_sine, T=6)
        partial_cosine = partial(self.f_cosine, T=6)

        self.df_agg['Day_of_week_sine'] = self.df_agg['Day_of_week'].apply(partial_sine)
        self.df_agg['Day_of_week_cosine'] = self.df_agg['Day_of_week'].apply(partial_cosine)

        # Сизон в циклическую фичу
        partial_sine = partial(self.f_sine, T=3)
        partial_cosine = partial(self.f_cosine, T=3)

        self.df_agg['Season_sine'] = self.df_agg['Season'].apply(partial_sine)
        self.df_agg['Season_cosine'] = self.df_agg['Season'].apply(partial_cosine)

        return self


    def get_dummies(self, df, col, prefix=None, dropOrigin = True):

        if prefix != None:
            df = df.join(pd.get_dummies(df[col], prefix = prefix))
        else:
            df = df.join(pd.get_dummies(df[col]))


        if dropOrigin:
            df.drop([col], inplace=True, axis=1)

        return df


    def transform(self, station_data, weather_data):
        self.df_agg = self.resample()
        # извлекаем дни
        self.df_agg['Date'] = self.getDate('d')
        # извлекаем часы
        self.df_agg['Hour'] = self.getDate('h')
        self.df_agg.eval('net_rate=num_trips_end-num_trips_start', inplace=True)
        self.df_agg.drop(['num_trips_end', 'num_trips_start'], inplace=True, axis=1)

        # соединяем данные
        # данные по станциям
        self.df_agg = self.df_agg.join(station_data.set_index('Station'))
        # погода
        self.df_agg.reset_index(inplace=True)
        self.df_agg.set_index(['Date', 'Zip'], inplace=True)

        self.df_agg = self.df_agg.join(weather_data) \
                        .reset_index() \
                        .set_index(['Station', 'Datetime'])

        # определяем тип покупателя
        self.df = self.get_dummies(self.df, 'Subscriber_Type', dropOrigin=True)
        # Определяем время дня
        self.df_agg['Is_night'] = self.df_agg['Hour'].apply(lambda h: 1 if h < 5 or h > 20 else 0)
        # Определяем день недели
        self.df_agg['Day_of_week'] = self.df_agg['Date'].apply(lambda dt: dt.weekday())
        # Рабочий или выходной
        self.df_agg['Is_weekday'] = self.df_agg['Day_of_week'].apply(lambda s: 0 if s in [5, 6] else 1)

        # Сизоны: зима (0), осень (1), лето (2), осень (3)
        self.df_agg['Season'] = self.df_agg['Date'].apply(lambda dt: (dt.month % 12 + 3) // 3 - 1)

        end = self.get_stop('End_Station', 'End_Date', drop=True, dropId="Trip_ID")
        start = self.get_stop('Start_Station', 'Start_Date', drop=True, dropId="Trip_ID")

        # общий набор
        self.df_agg = self.df_agg.join(end).join(start, lsuffix='_end', rsuffix='_start')
        #self.df_agg = self.df_agg.sample(frac=0.1)

        # Net rate за преведующий час
        self.df_agg['net_rate_previous_hour'] = self.df_agg.groupby(['Station', 'Date'])['net_rate'] \
                                                    .shift(1).fillna(0)
        self.cyclic_feature()

        # OHE Events
        self.df_agg = self.get_dummies(self.df_agg,'Events')

        # OHE Station IDs
        self.df_agg.reset_index(inplace=True)
        self.df_agg = self.get_dummies(self.df_agg, 'Station', prefix='Station')
        self.df_agg = self.get_dummies(self.df_agg, 'Zip', prefix='Zip')

        self.df_agg.drop(['Hour', 'Day_of_week', 'Season', 'Datetime', 'Date'],
                           inplace=True, axis=1)

        assert self.df_agg.isnull().any().sum() / len(self.df_agg.columns) == 0, 'DataFrame includes NULL'
        assert self.df_agg.isnull().any(axis=1).sum() / len(self.df_agg) == 0,  'DataFrame includes NULL'

        return self


    def fit(self, n_splits, random_state):
        self.n_splits = n_splits
        self.random_state = random_state

        # Stratified K-Fold CV
        outer_kfold_cv = KFold(n_splits=self.n_splits, random_state=self.random_state)
        inner_kfold_cv = KFold(n_splits=self.n_splits - 1, random_state=self.random_state)

        X = self.df_agg[self.feature_names]
        y = self.df_agg[self.target]

        # Разделение на train / test
        #return train_test_split(X, y, test_size=0.30, random_state=self.random_state)
        # return X, y
        return self.df_agg





class RFModel(RandomForestRegressor):
    """
        Класс на основе модели
        RandomForestRegressor
    """
    def __init__(self, n_estimators=100, *,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 ccp_alpha=0.0,
                 max_samples=None):
        super().__init__(n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
            ccp_alpha=ccp_alpha)

    def get_metric(self, x_test, y_test):
        return math.sqrt(np.mean((self.predict(x_test) - y_test) ** 2))


    def save_model(self, model, file_name):
        pickle.dump(model, open(file_name, 'wb'))
        return self

    def load_model(self, file_name):
        return pickle.load(open(file_name, 'rb'))





