from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**kwargs):
    return Pipeline(
        [   node(
                preparator,
                ["station_data", "trip_data", "weather_data",
                 "params:station_nes_names", "params:zip_code",
                 "params:trip_nes_names", "params:map_id"
                 ],
                ["clean_station_data",
                 "clean_trip_data",
                 "clean_weather_data"]
            ),
            node(
                index_creator,
                ["clean_station_data", "params:mind", "params:maxd", "params:freq"],
                "idx"
            ),
            node(
                cleanning,
                ["clean_weather_data",],
                "clean_weather_data_new"
            ),
            node(
                resample,
                ["clean_trip_data", "idx", "params:resample_list", "params:freq"],
                "resample_df"
            ),
            node(
                transform,
                ["resample_df", "clean_station_data", "clean_weather_data_new", "clean_trip_data", "idx"],
                "df_agg"
            ),
            node(
                split,
                ["df_agg", "params:seed"],
                ["train_x", "train_y", "test_x", "test_y"]
            )
        ]
    )
