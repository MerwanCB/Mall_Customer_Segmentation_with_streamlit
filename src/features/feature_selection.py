import pandas as pd

def select_features(df, feature_names):
    """
    Select specified columns from the DataFrame.
    """
    print(f"Selecting features: {feature_names}")
    # Check if all requested features exist in the DataFrame
    if not all(feature in df.columns for feature in feature_names):
        missing = [f for f in feature_names if f not in df.columns]
        raise ValueError(f"Features not found in DataFrame: {missing}")
    # Return a copy of the selected columns
    return df[feature_names].copy()