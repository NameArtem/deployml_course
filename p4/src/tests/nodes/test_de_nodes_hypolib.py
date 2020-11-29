import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames, column

from bike.pipelines.data_engineering import nodes

# создаем дата сеты в декораторе
@given(
    data_frames(
        [
            column('col_1', dtype=float,
                   elements=st.floats(allow_nan=True,
                                      allow_infinity=False)),
            column('col_2', dtype=int,
                   elements=st.integers(min_value=-100, max_value=100)),
        ]
    )
)
def test_fill_one_hypolib(df):

    for col in df.columns:
        if df[col].isna().sum() > 0:
            # smoke
            assert callable(nodes.fill_one) is True
            # type
            assert type(df[col]) == pd.Series
            # unit
            val = int(0) if df[col].dtype == int else float(0.0)
            assert nodes.fill_one(df[col], val).isna().sum() == 0
            assert type(nodes.fill_one(df[col], val)) == pd.Series
            # unit - все nan заполнены и значение от - до +
            assert nodes.fill_one(df[col], val).between(-np.inf, np.inf).sum() == df.shape[0]