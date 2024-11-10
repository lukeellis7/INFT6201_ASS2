#data_cleaning.py
#Functions for handling missing values, inconsistent data types, duplicates, and outliers

from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def handle_missing_values(data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    numeric_data = data.select_dtypes(include=[np.number])
    imputed_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
    data.update(imputed_data)
    return data

def standardize_dates(data, date_column):
    """
    Converts a specified date column to datetime format if it exists.

    Parameters:
    - data (DataFrame): The dataset.
    - date_column (str): The name of the date column to standardize.

    Returns:
    - DataFrame with standardized date column if it exists.
    """
    if date_column in data.columns:
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
    else:
        print(f"Warning: '{date_column}' column not found in the dataset.")
    return data

def remove_duplicates(data, subset_columns):
    return data.drop_duplicates(subset=subset_columns)

def detect_outliers(data, column, threshold=3):
    mean = data[column].mean()
    std_dev = data[column].std()
    data['is_outlier'] = ((data[column] - mean).abs() > threshold * std_dev)
    return data
