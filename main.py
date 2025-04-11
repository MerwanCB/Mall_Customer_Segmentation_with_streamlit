
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
INITIAL_K = 5  # Initial number of clusters for the 2-feature model
MAX_K_TO_EVALUATE = 8  # Max K for Elbow/Silhouette analysis
RANDOM_STATE = 42  # For reproducibility

# Features for different analyses
FEATURES_PAIRPLOT = ["Age", "Annual_Income", "Spending_Score"]
FEATURES_2D_CLUSTERING = ["Annual_Income", "Spending_Score"]
FEATURES_3D_CLUSTERING = ["Age", "Annual_Income", "Spending_Score"]

# --- Main Execution Block ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

    # 1. Load Data
    df_raw = load_data(RAW_DATA_PATH)
    print("\nRaw Data Info:")
    print(df_raw.head())
    print(df_raw.shape)
    print(df_raw.describe())

    # 2. Initial Visualization (Pairplot)
    save_pairplot(df_raw, FEATURES_PAIRPLOT, "pairplot_features.png")

    # 3. Clustering with 2 Features (Annual_Income, Spending_Score)
    df_features_2d = select_features(df_raw, FEATURES_2D_CLUSTERING)

    # Train initial model (k=5)
    model_2d, labels_2d = train_kmeans(
        df_features_2d, n_clusters=INITIAL_K, random_state=RANDOM_STATE
    )

    # Add cluster labels to the main dataframe (create a copy to avoid modifying original)
    df_processed = df_raw.copy()
    df_processed["Cluster_2D_k5"] = labels_2d
    print(f"\nCluster counts for k={INITIAL_K} (2 features):")
    print(df_processed["Cluster_2D_k5"].value_counts())

    # Visualize the 2D clusters
    save_cluster_scatterplot(
        df_processed,
        x_col="Annual_Income",
        y_col="Spending_Score",
        cluster_col="Cluster_2D_k5",
        filename="scatter_clusters_2_features.png",
    )

    # 4. Find Optimal K for 2 Features
    # Elbow Method
    wcss_2d = calculate_wcss(
        df_features_2d, MAX_K_TO_EVALUATE, random_state=RANDOM_STATE
    )
    save_elbow_plot(wcss_2d, "elbow_plot_2_features.png")

    # Silhouette Method
    silhouette_2d = calculate_silhouette_scores(
        df_features_2d, MAX_K_TO_EVALUATE, random_state=RANDOM_STATE
    )
    save_silhouette_plot(silhouette_2d, "silhouette_plot_2_features.png")

    # 5. Find Optimal K for 3 Features (Age, Annual_Income, Spending_Score)
    df_features_3d = select_features(df_raw, FEATURES_3D_CLUSTERING)

    # Silhouette Method (Elbow could also be done, but notebook only did Silhouette here)
    silhouette_3d = calculate_silhouette_scores(
        df_features_3d, MAX_K_TO_EVALUATE, random_state=RANDOM_STATE
    )
    save_silhouette_plot(silhouette_3d, "silhouette_plot_3_features.png")

    # 6. Save Processed Data (with the initial k=5 clusters from 2D analysis)
    save_data(df_processed, PROCESSED_DATA_PATH)

    print("\nScript finished successfully.")
