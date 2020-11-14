# https://scikit-learn.org/stable/modules/classes.html

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import metrics


class Metrics:
    """
        Класс-набор метрики для оценки качества моделей

    Parameters
    model = sc.Model
    X = pd.DataFrame
    y = np.array | list | pd.Series
    scorer = scorer params

    Output:
    value of metrics (float)
    """

    def __init__(self, model,
                       X, y,
                       scorer='neg_mean_squared_error'):
        self.model = model
        self.X = X
        self.y = y
        self.scorer = scorer


    def rmse_cv(self):
        # Определяем функцию для проверки качества: RMSE
        rmse = np.sqrt(-cross_val_score(self.model,
                                        self.X, self.y,
                                        scoring=self.scorer,
                                        cv=3))

        return rmse.mean()



class DataChecker:
    """
    Класс проверки данных
    Содержит функции для DataQuality
    """
    def __init__(self):
        pass

    def checkNull(self, df):
        return df.isnull().values.any()