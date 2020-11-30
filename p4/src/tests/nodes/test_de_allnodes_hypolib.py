import numpy as np
import pandas as pd
import pytest
import datetime

from sklearn.model_selection import train_test_split

from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames, column

from bike.pipelines.data_engineering import nodes


#@given()  - объект передачи
#@settings() - кол-во экземпляров объекта given
# """
#     Если объект собственный - т.е. класс
#     То можно создать объект самому
#
#     class Book(object):
#         def __init__(self, title, author, price):
#             self.title = title
#             self.author = author
#             if not price >= 0.00:
#                 price = 0.00
#             self.price = Decimal(price)
#
#         def __unicode__(self):
#             return u"%s by %s ($%s)" % (self.title, self.author, self.price)
#
#         def __repr__(self):
#             return self.__unicode__().encode("utf-8")
#
#     books = st.builds(
#                         Book,
#                         title=st.text(),
#                         author=st.text(),
#                         price=st.decimals(
#                                         allow_nan=False,
#                                         allow_infinity=False,
#                                         places=2,
#                                         min_value=0.00,
#                                         max_value=1000000.00,
#                                         ),
#                                     )
#     @given(books)
#     def ...
# """

# Pandas примеры
# Тест на создание мультииндекса
@given(
    data_frames(
        [
            column('Station', dtype=str,
                   elements=st.text()),
            column('DateOfPeriod',
                   elements=st.datetimes(min_value=datetime.datetime.strptime('2019-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'),
                                         max_value=datetime.datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'))),
        ]
    ),
    st.datetimes(min_value=datetime.datetime.strptime('2019-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'),
                 max_value=datetime.datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')),
    st.datetimes(min_value=datetime.datetime.strptime('2019-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'),
                 max_value=datetime.datetime.strptime('2021-01-01 00:00:00', '%Y-%m-%d %H:%M:%S'))
)
def test_index_creator(df, mn, mx):
    assert nodes.index_creator(df, mn, mx, 'D') is not None
    assert nodes.index_creator(df, mn, mx, 'D').__class__ == pd.core.indexes.multi.MultiIndex
    assert isinstance(nodes.index_creator(df, mn, mx, 'D'), pd.MultiIndex) == True
    with pytest.raises(AssertionError):
        pd.testing.assert_index_equal(nodes.index_creator(df, mn, mx, 'D'),
                                      pd.core.indexes.multi.MultiIndex(levels=[['', '0'], []],
                                                                       names=['Station', 'Datetime'],
                                                                       codes=[[], []]))


# split test
def split(df_agg, random_state):
    X = df_agg[[i for i in df_agg.columns if i != 'target']]
    y = df_agg['target']

    # Разделение на train / test
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.30,
                                                        random_state=random_state)

    return (X_train,
            y_train,
            X_test,
            y_test)


@given(
    data_frames(
        [
            column('col1', dtype=float,
                   elements=st.floats(allow_nan=True,
                                      allow_infinity=False)),
            column('col2', dtype=float,
                   elements=st.floats(allow_nan=True,
                                      allow_infinity=False)),
            column('target', dtype=int,
                   elements=st.integers(min_value=-100, max_value=100)),
        ]
    ),
    st.integers(min_value=-100, max_value=100)
)
def test_split(df_agg, random_state):
    try:
        assert len(split(df_agg, random_state)) == 4
        assert isinstance(split(df_agg, random_state)[1], pd.Series)
        assert isinstance(split(df_agg, random_state)[3], pd.Series)

        assert (len(split(df_agg, random_state)[0]) + len(split(df_agg, random_state)[1])) == len(df_agg)

        # если разбивка будет собственная, то использовать такую проверку
        #pd.testing.assert_frame_equal(expected_train_X, train_X)
        #pd.testing.assert_frame_equal(expected_test_X, test_X)

        # идекса из train нет в
        for i in split(df_agg, random_state)[1].index.values:
            assert i not in split(df_agg, random_state)[0].index
        # правильно np.allclose(split(df_agg, random_state)[1], split(df_agg, random_state)[0])
    except ValueError:
        # если данные не сплитятся или таргет одно значене
        True



