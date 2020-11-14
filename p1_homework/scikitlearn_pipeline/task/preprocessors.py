import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Шаблон для обработки пропусков
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass


    def fit(self, X, y=None):
        #
        pass


    def transform(self, X):
        #
        X = X.copy()
        pass


#  шаблон для обработки категориальных данных
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass

    def fit(self, X, y=None):
        #
        pass

    def transform(self, X):
        X = X.copy()
        pass


# шаблон для обработки числовых значений
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass

    def fit(self, X, y=None):
        #

        pass

    def transform(self, X):

        X = X.copy()
        pass


# шаблон для извлечения данных
class ExtractSome(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass

    def fit(self, X, y=None):
        #
        pass

    def transform(self, X):
        X = X.copy()
        pass


# шаблон для обработки категорий (набор из OHE и т.п. методов)
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        pass

    def fit(self, X, y=None):

        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        
        return self

    def transform(self, X):

        X = X.copy()


        return X
