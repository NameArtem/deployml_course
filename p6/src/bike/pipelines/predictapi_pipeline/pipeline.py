from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                get_api_date,
                "api_data",
                ["predict_date", "row_index", "to_predict"]
            ),
            node(
                make_prediction,
                ["to_predict", "rf_model"],
                "predict"
            ),
            node(
                serve_result,
                ["predict_date", "row_index", "predict"],
                ["predict_date_r", "row_index_r", "predict_r"]
            )
        ]
    )
