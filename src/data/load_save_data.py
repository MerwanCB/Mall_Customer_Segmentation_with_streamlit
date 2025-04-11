import pandas as pd
import os

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    print(f"Loading data from: {file_path}")
    return pd.read_csv(file_path)

def save_data(df, file_path):
    """
    Save a DataFrame to a CSV file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    print(f"Saving data to: {file_path}")
    df.to_csv(file_path, index=False)