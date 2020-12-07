from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                run_training,
                ["train_x", "train_y", "params:model_params"],
                "rf_model"
            )
        ]
    )
