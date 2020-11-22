from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                make_prediction,
                ["test_x", "rf_model"],
                "predict"
            ),
            node(
                rmse_cv,
                ["test_x", "test_y", "rf_model"],
                "quality"
            )
        ]
    )
