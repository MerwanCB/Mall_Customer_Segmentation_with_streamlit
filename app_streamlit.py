import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np

# --- Configuration ---
RAW_DATA_PATH = "data/raw/mall_customers.csv"
PROCESSED_DATA_PATH = "data/processed/clustered_customers.csv"
FIGURES_DIR = "reports/figures"  # Directory for saved plots
MODEL_PATH = "models/kmeans_k5_2features.joblib" # Path to the saved model

# --- Helper Function ---
def check_file(path):
    # Ensure required file exists, otherwise stop execution
    if not os.path.exists(path):
        st.error(f"Error: Required file not found at '{path}'. Please ensure the main script has been run.")
        st.stop()
    return path

# --- Page Setup ---
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("Mall Customer Segmentation Analysis")
st.write(
    """
    This application presents the results of a customer segmentation analysis
    using K-Means clustering on mall customer data.
    """
)

# --- Load and Display Raw Data ---
st.header("1. Data Overview")
try:
    df_raw = pd.read_csv(check_file(RAW_DATA_PATH))
    st.write("Sample of raw customer data:")
    st.dataframe(df_raw.head())
    st.write("Basic statistics of numerical features:")
    st.dataframe(df_raw.describe())
except Exception as e:
    st.error(f"An error occurred while loading the raw data: {e}")
    st.stop()

# --- Exploratory Visualization ---
st.header("2. Exploratory Data Analysis")
st.write("Pairplot showing relationships between key features:")
pairplot_path = check_file(os.path.join(FIGURES_DIR, "pairplot_features.png"))
st.image(pairplot_path, caption="Pairplot of Age, Annual Income, and Spending Score")
st.write(
    """
    Observations:
    - Higher spending scores are concentrated among younger customers.
    - Distinct groups are visible in Annual Income vs Spending Score, suggesting clustering potential.
    """
)

# --- Clustering with 2 Features ---
st.header("3. Clustering with 2 Features (Income & Spending Score)")
st.write("K-Means clustering applied with 'Annual Income' and 'Spending Score' (k=5).")
scatter_2d_path = check_file(os.path.join(FIGURES_DIR, "scatter_clusters_2_features.png"))
st.image(scatter_2d_path, caption="Clusters based on Annual Income and Spending Score (k=5)")
st.write("The scatter plot confirms 5 distinct customer segments.")

# --- Finding Optimal K (2 Features) ---
st.header("4. Finding the Optimal Number of Clusters (k) - 2 Features")
st.write("Elbow and Silhouette methods used to determine the best k value.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Elbow Method")
    elbow_path = check_file(os.path.join(FIGURES_DIR, "elbow_plot_2_features.png"))
    st.image(elbow_path, caption="Elbow Plot for 2 Features")
    st.write("The 'elbow' point suggests k=5.")

with col2:
    st.subheader("Silhouette Method")
    silhouette_2d_path = check_file(os.path.join(FIGURES_DIR, "silhouette_plot_2_features.png"))
    st.image(silhouette_2d_path, caption="Silhouette Plot for 2 Features")
    st.write("The highest Silhouette Score also suggests k=5.")

st.write("**Conclusion:** Optimal k for 2 features is **k=5**.")

# --- Finding Optimal K (3 Features) ---
st.header("5. Finding the Optimal Number of Clusters (k) - 3 Features")
st.write("Adding 'Age' to the analysis and evaluating clusters using the Silhouette Score.")
silhouette_3d_path = check_file(os.path.join(FIGURES_DIR, "silhouette_plot_3_features.png"))
st.image(silhouette_3d_path, caption="Silhouette Plot for 3 Features (Age, Income, Spending Score)")
st.write("**Conclusion:** Optimal k for 3 features is **k=6**, based on Silhouette analysis.")

# --- Display Processed Data ---
st.header("6. Final Clustered Data Sample")
st.write("Processed data includes original information and cluster assignments (k=5, 2 features).")
try:
    df_processed = pd.read_csv(check_file(PROCESSED_DATA_PATH))
    st.write("Sample of processed data with cluster labels:")
    st.dataframe(df_processed.head())
except Exception as e:
    st.error(f"An error occurred while loading the processed data: {e}")
    st.stop()

st.success("Analysis presentation complete!")

# --- Prediction Section ---
st.header("7. Predict Cluster for a New Customer")
st.write("Enter the details for a new customer to predict their cluster (based on the k=5, 2-feature model).")

# Load the trained model
try:
    model = joblib.load(check_file(MODEL_PATH))
except Exception as e:
    st.error(f"An error occurred while loading the clustering model: {e}")
    st.stop()

# Input fields for the two features
income = st.number_input("Annual Income (k$)", min_value=0, value=50, step=1)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# Prediction button
if st.button("Predict Cluster"):
    if model:
        try:
            # Prepare input data in the same format as training data
            input_data = np.array([[income, spending_score]])
            prediction = model.predict(input_data)
            # Map cluster index to a more descriptive name (optional, but good practice)
            # These names are based on typical interpretations of the 2-feature clusters
            cluster_names = {
                0: "Careful Spenders (Low Income, Low Spending)",
                1: "Standard (Average Income, Average Spending)",
                2: "Target Group (High Income, High Spending)",
                3: "Careless (Low Income, High Spending)",
                4: "Sensible Savers (High Income, Low Spending)"
            }
            predicted_cluster_name = cluster_names.get(prediction[0], f"Cluster {prediction[0]}")

            st.success(f"Predicted Cluster: **{predicted_cluster_name}**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model not loaded. Cannot perform prediction.")
