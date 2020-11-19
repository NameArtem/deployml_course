from __init__ import __version__
import yaml
import os
ROOT = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "src" ))).replace('\\','/')
DATA_PATH = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "src" ), 'data')).replace('\\','/')

# seed
SEED = 28
VERBOSE = 28

# data
DATA_FILE_STATION = "data/data/_station_data.csv"
DATA_FILE_TRIP = "data/data/trip_data.csv"
DATA_FILE_WEATHER = "data/data/weather_data.csv"

CLEAN_DATA = f"/clean_data/cleandata{__version__}.csv"

SAMPLE_NROWS = 10000
SAMPLE_PRC = 0.01

# model name
MODEL_NAME = f"/model/rf_model{__version__}"


STATION_NES_NAMES = ['Station_id', 'Station', 'Lat', 'Long', 'Dock_Count', 'City', 'Inst_date']
TRIP_NES_NAMES = ['Trip_ID', 'Start_Date', 'Start_Station', 'End_Date', 'End_Station', 'Subscriber_Type']

# additional
# sities to zip codes
ZIP_CODE = {'San Francisco': 94107, 'Redwood City': 94063,
            'Palo Alto': 94301, 'Mountain View': 94041,
            'San Jose': 95113}

# station ID mapping
MAP_ID = {85: 23, 86: 25, 87: 49, 88: 69, 89: 72, 90: 72}

MIND = '2014-09-01 00:00:00'
MAXD = '2015-08-31 23:00:00'
FREQ = 'H'

RESAMPLING_PRM = ['End_Station', 'Start_Station']

# Features
FEATURE_NAME = [ 'Hour_cosine', 'Hour_sine', 'Day_of_week_cosine', 'Day_of_week_sine', 'Is_weekday',
                'Is_night', 'Season_cosine', 'Season_sine', 'net_rate_previous_hour', 'Dock_Count',
                'Fog', 'Fog-Rain', 'Clear', 'Rain', 'Rain-Thunderstorm', 'CloudCover', 'PrecipitationIn',
                'WindDirDegrees', 'Max Dew PointF', 'Max Gust SpeedMPH', 'Max Humidity', 'Max Sea Level PressureIn',
                'Max TemperatureF', 'Max VisibilityMiles', 'Max Wind SpeedMPH', 'Mean Humidity',
                'Mean Sea Level PressureIn', 'Mean TemperatureF', 'Mean VisibilityMiles', 'Mean Wind SpeedMPH',
                'MeanDew PointF', 'Min DewpointF', 'Min Humidity', 'Min Sea Level PressureIn', 'Min TemperatureF',
                'Min VisibilityMiles', 'Zip_94041', 'Zip_94063', 'Zip_94107', 'Zip_94301', 'Zip_95113']


MODEL_FEATURES = ['Hour_sine', 'Hour_cosine', 'net_rate_previous_hour', 'Dock_Count',
                'Zip_94107', 'WindDirDegrees', 'Max Gust SpeedMPH', 'Max Sea Level PressureIn',
                'Day_of_week_sine', 'Min Humidity', 'Max TemperatureF', 'Min Sea Level PressureIn',
                'Mean Humidity', 'Max Wind SpeedMPH', 'Max Humidity', 'Min DewpointF',
                'Mean TemperatureF', 'Mean Wind SpeedMPH', 'Is_weekday', 'CloudCover',
                'Min TemperatureF', 'Max Dew PointF', 'Day_of_week_cosine', 'Mean Sea Level PressureIn',
                'MeanDew PointF', 'Min VisibilityMiles', 'Season_cosine', 'PrecipitationIn',
                'Mean VisibilityMiles', 'Season_sine', 'Max VisibilityMiles', 'Fog', 'Clear', 'Is_night',
                'Zip_95113', 'Zip_94041', 'Rain', 'Fog-Rain', 'Rain-Thunderstorm', 'Zip_94063', 'Zip_94301']

TARGET_NAME = 'net_rate'


# Model param
N_ESTIMATORS = int(yaml.safe_load(open(f"{ROOT}/params.yaml"))['main']['n_estimators'])
MAX_DEPTH = int(yaml.safe_load(open(f"{ROOT}/params.yaml"))['main']['max_depth'])
MIN_SAMPLES_LEAF = int(yaml.safe_load(open(f"{ROOT}/params.yaml"))['main']['min_sample_leaf'])
CRITERION = yaml.safe_load(open(f"{ROOT}/params.yaml"))['main']['criterion']



