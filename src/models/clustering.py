
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List, Dict

def train_kmeans(data: pd.DataFrame, n_clusters: int, random_state: int = 42) -> Tuple[KMeans, List[int]]:
    """Trains a KMeans model and returns the model and labels.

    Args:
        data (pd.DataFrame): The data to cluster (features only).
        n_clusters (int): The desired number of clusters.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[KMeans, List[int]]: The fitted KMeans model and the cluster labels.
    """
    print(f"Training KMeans model with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
    kmeans.fit(data)
    labels = kmeans.labels_
    print("KMeans training complete.")
    return kmeans, labels

def calculate_wcss(data: pd.DataFrame, max_k: int, random_state: int = 42) -> Dict[int, float]:
    """Calculates the Within-Cluster Sum of Squares (WCSS) for different k values.

    Args:
        data (pd.DataFrame): The data to cluster (features only).
        max_k (int): The maximum number of clusters to evaluate (inclusive).
        random_state (int): Random state for reproducibility.

    Returns:
        Dict[int, float]: A dictionary mapping k (number of clusters) to WCSS score.
    """
    print(f"Calculating WCSS for k from 3 to {max_k}...")
    wcss_scores = {}
    k_values = range(3, max_k + 1)
    for i in k_values:
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=random_state)
        kmeans.fit(data)
        wcss_scores[i] = kmeans.inertia_ # inertia_ is the WCSS score
    print("WCSS calculation complete.")
    return wcss_scores

def calculate_silhouette_scores(data: pd.DataFrame, max_k: int, random_state: int = 42) -> Dict[int, float]:
    """Calculates the Silhouette Score for different k values.

    Args:
        data (pd.DataFrame): The data to cluster (features only).
        max_k (int): The maximum number of clusters to evaluate (inclusive).
        random_state (int): Random state for reproducibility.

    Returns:
        Dict[int, float]: A dictionary mapping k (number of clusters) to Silhouette score.
    """
    print(f"Calculating Silhouette scores for k from 3 to {max_k}...")
    silhouette_scores = {}
    k_values = range(3, max_k + 1)
    for i in k_values:
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores[i] = score
    print("Silhouette score calculation complete.")
    return silhouette_scores