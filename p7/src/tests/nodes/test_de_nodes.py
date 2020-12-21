import numpy as np
import pandas as pd

from bike.pipelines.data_engineering import nodes

def test_fill_one():

    df = pd.DataFrame(data = [np.random.choice([np.nan, "1", 1]) for _ in range(0, 100)],
                      columns = ["col"])

    # smoke
    assert callable(nodes.fill_one) is True
    # type
    assert type(df["col"]) == pd.Series
    assert np.random.choice(["1", 1]) != np.nan and \
           np.random.choice(["1", 1]) is not None
    # unit
    assert nodes.fill_one(df["col"], np.random.choice(["1", 1])).isna().sum() == 0
    assert type(nodes.fill_one(df["col"], np.random.choice(["1", 1]))) == pd.Series