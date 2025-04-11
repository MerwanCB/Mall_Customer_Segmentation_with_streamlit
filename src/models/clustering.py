import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_kmeans(data, n_clusters, random_state=42):
    """
    Train a KMeans model and return the model and cluster labels.
    """
    print(f"Training KMeans model with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state)
    kmeans.fit(data)
    labels = kmeans.labels_
    print("KMeans training complete.")
    return kmeans, labels

def calculate_wcss(data, max_k, random_state=42):
    """
    Calculate WCSS (Within-Cluster Sum of Squares) for k values from 3 to max_k.
    """
    print(f"Calculating WCSS for k from 3 to {max_k}...")
    wcss_scores = {}
    for i in range(3, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=random_state)
        kmeans.fit(data)
        wcss_scores[i] = kmeans.inertia_  # WCSS score
    print("WCSS calculation complete.")
    return wcss_scores

def calculate_silhouette_scores(data, max_k, random_state=42):
    """
    Calculate Silhouette scores for k values from 3 to max_k.
    """
    print(f"Calculating Silhouette scores for k from 3 to {max_k}...")
    silhouette_scores = {}
    for i in range(3, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(data)
        silhouette_scores[i] = silhouette_score(data, labels)
    print("Silhouette score calculation complete.")
    return silhouette_scores