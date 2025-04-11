
from typing import List
import pandas as pd


def select_features(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Selects specified feature columns from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_names (List[str]): A list of column names to select.

    Returns:
        pd.DataFrame: A DataFrame containing only the selected features.
    """
    print(f"Selecting features: {feature_names}")
    if not all(feature in df.columns for feature in feature_names):
        missing = [f for f in feature_names if f not in df.columns]
        raise ValueError(f"Features not found in DataFrame: {missing}")
    return df[feature_names].copy()