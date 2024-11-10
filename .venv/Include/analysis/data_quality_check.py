#Functions for checking data quality and producing reports

import pandas as pd

def check_missing_values(data):
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    return missing_data

def identify_data_issues(data):
    issues = {
        "missing_values": check_missing_values(data),
        "duplicates": data.duplicated().sum()
    }
    return issues
