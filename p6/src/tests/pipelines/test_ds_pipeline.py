import numpy as np
import pandas as pd
import pytest

from bike.pipelines.data_science import nodes
from bike.pipelines.data_science import pipeline

from kedro.pipeline import Pipeline, node
from kedro.runner import SequentialRunner

# ["train_x", "train_y", "params:model_params"
# def create_pipeline(inputs = [], outputs = []): <- вот так не делаем!!!
# просто демонстрация pipeline
def create_pipeline(inputs, outputs):
    return Pipeline(
        [
            node(
                nodes.run_training_wo_params,
                inputs = inputs,
                outputs = outputs
            )
        ]
    )


@pytest.fixture(scope='module')
def dataframex():
    return pd.DataFrame({"f1": [np.random.random(1)[0] for _ in range(0,100)],
                         "f2": [np.random.random(1)[0] for _ in range(0,100)]})


@pytest.fixture(scope='module')
def dataframey():
    return pd.DataFrame({"y": [np.random.randint(0, 10) for _ in range(0, 100)],
                         })

@pytest.fixture(scope='module')
def dataframey_bad():
    return pd.DataFrame({"y_none": [np.random.choice(['a', 'b', 'c']) for _ in range(0, 100)],
                         })

@pytest.fixture(scope='module')
def all_catalog(dataframex, dataframey, dataframey_bad):
    #создадим DF, так как они лежат в Кедро, после загрузки в память
    # https://kedro.readthedocs.io/en/stable/05_data/02_kedro_io.html
    from kedro.io import DataCatalog, MemoryDataSet
    catalog = DataCatalog({
        "dataframex": MemoryDataSet(),
        "dataframey": MemoryDataSet(),
        "dataframey_bad": MemoryDataSet()
    })
    catalog.save("dataframex", dataframex)
    catalog.save("dataframey", dataframey)
    catalog.save("dataframey_bad", dataframey_bad)
    return catalog


def test_ds_pipeline(all_catalog):
    runner = SequentialRunner()
    output_name = 'outputs'

    pipeline = create_pipeline(inputs = ["dataframex", "dataframey"],
                               outputs= output_name)
    pipeline_output = runner.run(pipeline, all_catalog)

    assert pipeline_output is not None

    with pytest.raises(ValueError):
        # не учиться на плохоих значениях
        pipeline = create_pipeline(inputs=["dataframex", "dataframey_bad"],
                                   outputs=output_name)
        pipeline_output = runner.run(pipeline, all_catalog)