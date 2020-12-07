from typing import Dict
from kedro.pipeline import Pipeline, node

from src.bike.pipelines.data_engineering import pipeline as de
from src.bike.pipelines.data_science import pipeline as ds
from src.bike.pipelines.predict_pipeline import pipeline as pr

from src.bike.pipelines import qualities as q

def register_pipelines(self) -> Dict[str, Pipeline]:

    de_pipe = de.create_pipeline()
    ds_pipe = ds.create_pipeline()
    pr_pipe = pr.create_pipeline()

    checked_pipe = Pipeline([node(q.data_qual,
                             "trip_data",
                             "trip_data_cheked")
                        ])

    return {
        "de": checked_pipe + de_pipe,
        "ds": ds_pipe,
        "predict": pr_pipe,
        "__default__": checked_pipe + de_pipe + ds_pipe + pr_pipe,
    }



