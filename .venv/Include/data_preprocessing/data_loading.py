#Load raw data and return cleaned, preprocessed data

import pandas as pd
from .data_cleaning import handle_missing_values, standardize_dates, remove_duplicates
from .feature_engineering import create_performance_features, encode_categorical_features

def load_data(filepath):
    return pd.read_csv(filepath)

def prepare_data(filepath):
    data = load_data(filepath)
    data = handle_missing_values(data)
    data = standardize_dates(data, 'date')
    data = remove_duplicates(data, ['player_name', 'date'])
    data = create_performance_features(data)
    data = encode_categorical_features(data, ['country', 'team'])
    return data
