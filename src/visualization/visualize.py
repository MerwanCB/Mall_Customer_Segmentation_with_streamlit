
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import List, Dict

# Ensure the output directory for figures exists
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def save_pairplot(df: pd.DataFrame, columns: List[str], filename: str) -> None:
    """Generates and saves a pairplot for specified columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        columns (List[str]): The columns to include in the pairplot.
        filename (str): The name for the output image file.
    """
    print(f"Generating pairplot for columns: {columns}")
    plt.figure() # Create a new figure to avoid overlap
    sns.pairplot(df[columns])
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path)
    plt.close() # Close the plot to free memory
    print(f"Pairplot saved to: {save_path}")

def save_cluster_scatterplot(df: pd.DataFrame, x_col: str, y_col: str, cluster_col: str, filename: str) -> None:
    """Generates and saves a scatterplot of clusters.

    Args:
        df (pd.DataFrame): DataFrame with data and cluster labels.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        cluster_col (str): Column name containing cluster labels.
        filename (str): The name for the output image file.
    """
    print(f"Generating cluster scatterplot ({x_col} vs {y_col})")
    plt.figure(figsize=(10, 6)) # Create a new figure
    sns.scatterplot(x=x_col, y=y_col, data=df, hue=cluster_col, palette='colorblind')
    plt.title(f'{y_col} vs {x_col} by Cluster')
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Cluster scatterplot saved to: {save_path}")

def save_elbow_plot(wcss_scores: Dict[int, float], filename: str) -> None:
    """Generates and saves an Elbow plot from WCSS scores.

    Args:
        wcss_scores (Dict[int, float]): Dictionary of k to WCSS score.
        filename (str): The name for the output image file.
    """
    print("Generating Elbow plot...")
    k_values = list(wcss_scores.keys())
    scores = list(wcss_scores.values())
    plt.figure(figsize=(8, 5)) # Create a new figure
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS Score')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(k_values)
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Elbow plot saved to: {save_path}")

def save_silhouette_plot(silhouette_scores: Dict[int, float], filename: str) -> None:
    """Generates and saves a Silhouette plot from silhouette scores.

    Args:
        silhouette_scores (Dict[int, float]): Dictionary of k to Silhouette score.
        filename (str): The name for the output image file.
    """
    print("Generating Silhouette plot...")
    k_values = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    plt.figure(figsize=(8, 5)) # Create a new figure
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method For Optimal k')
    plt.xticks(k_values)
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Silhouette plot saved to: {save_path}")