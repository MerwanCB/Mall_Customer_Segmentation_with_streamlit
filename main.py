import os
import warnings
from src.data.load_save_data import load_data, save_data
from src.features.feature_selection import select_features
from src.models.clustering import (
    train_kmeans,
    calculate_wcss,
    calculate_silhouette_scores,
)
from src.visualization.visualize import (
    save_pairplot,
    save_cluster_scatterplot,
    save_elbow_plot,
    save_silhouette_plot,
)

# --- Configuration ---
RAW_DATA_PATH = "data/raw/mall_customers.csv"
PROCESSED_DATA_PATH = "data/processed/clustered_customers.csv"
INITIAL_K = 5  # Initial number of clusters for 2-feature model
MAX_K_TO_EVALUATE = 8  # Maximum K for Elbow/Silhouette analysis
RANDOM_STATE = 42  # Random seed for reproducibility

# Features for analysis
FEATURES_PAIRPLOT = ["Age", "Annual_Income", "Spending_Score"]
FEATURES_2D_CLUSTERING = ["Annual_Income", "Spending_Score"]
FEATURES_3D_CLUSTERING = ["Age", "Annual_Income", "Spending_Score"]

# --- Main Execution Block ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Suppress warnings

    # Load raw data
    df_raw = load_data(RAW_DATA_PATH)
    print("\nRaw Data Info:")
    print(df_raw.head())
    print(df_raw.shape)
    print(df_raw.describe())

    # Generate pairplot for initial visualization
    save_pairplot(df_raw, FEATURES_PAIRPLOT, "pairplot_features.png")

    # Perform clustering with 2 features
    df_features_2d = select_features(df_raw, FEATURES_2D_CLUSTERING)

    # Train KMeans model with initial K
    model_2d, labels_2d = train_kmeans(
        df_features_2d, n_clusters=INITIAL_K, random_state=RANDOM_STATE
    )

    # Add cluster labels to the dataset
    df_processed = df_raw.copy()
    df_processed["Cluster_2D_k5"] = labels_2d
    print(f"\nCluster counts for k={INITIAL_K} (2 features):")
    print(df_processed["Cluster_2D_k5"].value_counts())

    # Save scatterplot of 2D clusters
    save_cluster_scatterplot(
        df_processed,
        x_col="Annual_Income",
        y_col="Spending_Score",
        cluster_col="Cluster_2D_k5",
        filename="scatter_clusters_2_features.png",
    )

    # Evaluate optimal K for 2 features using Elbow and Silhouette methods
    wcss_2d = calculate_wcss(
        df_features_2d, MAX_K_TO_EVALUATE, random_state=RANDOM_STATE
    )
    save_elbow_plot(wcss_2d, "elbow_plot_2_features.png")

    silhouette_2d = calculate_silhouette_scores(
        df_features_2d, MAX_K_TO_EVALUATE, random_state=RANDOM_STATE
    )
    save_silhouette_plot(silhouette_2d, "silhouette_plot_2_features.png")

    # Evaluate optimal K for 3 features using Silhouette method
    df_features_3d = select_features(df_raw, FEATURES_3D_CLUSTERING)
    silhouette_3d = calculate_silhouette_scores(
        df_features_3d, MAX_K_TO_EVALUATE, random_state=RANDOM_STATE
    )
    save_silhouette_plot(silhouette_3d, "silhouette_plot_3_features.png")

    # Save processed data with cluster labels
    save_data(df_processed, PROCESSED_DATA_PATH)

    print("\nScript finished successfully.")
