import pandas as pd
import numpy as np
from pandas.core.dtypes.api import is_datetime64_any_dtype

def is_alphanumeric(column):
    # Check if value is alphanumeric
    sample_values = column.dropna().sample(min(10, len(column)))
    return any(value for value in sample_values if any(char.isalpha() for char in str(value)))

def replace_missing_with_mean(df):
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Replace ? with mean
            df[column] = df[column].replace('?', np.nan)
            df[column] = pd.to_numeric(df[column], errors='coerce')
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].replace('?', np.nan)
            most_common_date = df[column].mode()[0]
            df[column].fillna(most_common_date, inplace=True)
    return df

def convert_to_categorical(df):
    # Convert alphanumerice/object/data time columns to alphanumeric so that they can be processed properly for algorithms
    for col in df.columns:
        if df[col].dtype == 'object' and is_alphanumeric(df[col]) or is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype('category')
    return df
