import pandas as pd
import pytest

from src.main.preprocessors import Pipeline

def test_add_col_passes():
    # создаем тестовый пример
    df = pd.DataFrame({
        'col_a': ['a', 'a', 'a'],
        'col_b': ['b', 'b', 'b'],
        'col_c': ['a', 'b', 'c'],
    })

    # вызов функции
    pp = Pipeline(df,
             [('col_a', ), ('col_b',)],
             'h',
             None,
             ['col_a', 'col_b'],
             'col_c')

    ohe_df = pp.get_dummies(df, 'col_c', prefix=None, dropOrigin = True)

    # установка результата
    expected = pd.DataFrame({
        'col_a': ['a', 'a', 'a'],
        'col_b': ['b', 'b', 'b'],
        'a': [1, 0, 0],
        'b': [0, 1, 0],
        'c': [0, 0, 1]
    })
    expected['a'] = expected['a'].astype('uint8')
    expected['b'] = expected['b'].astype('uint8')
    expected['c'] = expected['c'].astype('uint8')

    # assert
    pd.testing.assert_frame_equal(ohe_df, expected)

