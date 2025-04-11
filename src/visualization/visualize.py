import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Ensure the output directory for figures exists
FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def save_pairplot(df, columns, filename):
    """
    Generate and save a pairplot for specified columns.
    """
    print(f"Generating pairplot for columns: {columns}")
    plt.figure()  # Avoid overlap with other plots
    sns.pairplot(df[columns])
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path)
    plt.close()  # Free memory
    print(f"Pairplot saved to: {save_path}")

def save_cluster_scatterplot(df, x_col, y_col, cluster_col, filename):
    """
    Generate and save a scatterplot of clusters.
    """
    print(f"Generating cluster scatterplot ({x_col} vs {y_col})")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df, hue=cluster_col, palette='colorblind')
    plt.title(f'{y_col} vs {x_col} by Cluster')
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Cluster scatterplot saved to: {save_path}")

def save_elbow_plot(wcss_scores, filename):
    """
    Generate and save an Elbow plot from WCSS scores.
    """
    print("Generating Elbow plot...")
    k_values = list(wcss_scores.keys())
    scores = list(wcss_scores.values())
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS Score')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(k_values)
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Elbow plot saved to: {save_path}")

def save_silhouette_plot(silhouette_scores, filename):
    """
    Generate and save a Silhouette plot from silhouette scores.
    """
    print("Generating Silhouette plot...")
    k_values = list(silhouette_scores.keys())
    scores = list(silhouette_scores.values())
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method For Optimal k')
    plt.xticks(k_values)
    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Silhouette plot saved to: {save_path}")