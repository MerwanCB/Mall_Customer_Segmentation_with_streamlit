# [app_streamlit.py]

import streamlit as st
import pandas as pd
import os

# --- Configuration ---
RAW_DATA_PATH = "data/raw/mall_customers.csv"
PROCESSED_DATA_PATH = "data/processed/clustered_customers.csv"
FIGURES_DIR = "reports/figures" # Directory where plots are saved

# --- Helper Function to check if files exist ---
def check_file(path):
    if not os.path.exists(path):
        st.error(f"Error: Required file not found at '{path}'. Please ensure the main script has been run.")
        st.stop()
    return path

# --- Page Setup ---
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("Mall Customer Segmentation Analysis")
st.write(
    """
    Welcome! This application showcases the results of a customer segmentation analysis
    performed on mall customer data. We use unsupervised machine learning (K-Means Clustering)
    to group customers based on their purchasing behavior and demographics.
    """
)

# --- Optional Password Protection ---
# Uncomment the following lines and set a password in Streamlit secrets
# if you want to protect the app.
# password_guess = st.text_input("Please enter your password:", type="password")
# if password_guess != st.secrets["password"]:
#     st.warning("Please enter the correct password to proceed.")
#     st.stop()
# st.success("Password correct!") # Optional feedback

# --- Load Data (for display purposes) ---
st.header("1. Data Overview")
try:
    df_raw = pd.read_csv(check_file(RAW_DATA_PATH))
    st.write("Here's a sample of the raw customer data:")
    st.dataframe(df_raw.head())

    st.write("Basic statistics of the numerical features:")
    st.dataframe(df_raw.describe())

except FileNotFoundError:
    # Error handled by check_file, but added explicit stop just in case
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the raw data: {e}")
    st.stop()


# --- Exploratory Visualization ---
st.header("2. Exploratory Data Analysis")
st.write("A pairplot helps visualize relationships between key features:")
pairplot_path = check_file(os.path.join(FIGURES_DIR, "pairplot_features.png"))
st.image(pairplot_path, caption="Pairplot of Age, Annual Income, and Spending Score")
st.write(
    """
    *   We can observe some potential patterns, like the concentration of higher spending scores
        among younger customers (Age vs Spending_Score).
    *   Annual Income vs Spending Score shows distinct visual groups, suggesting clustering potential.
    """
)


# --- Clustering with 2 Features ---
st.header("3. Clustering with 2 Features (Income & Spending Score)")
st.write(
    """
    We start by applying K-Means clustering using only 'Annual Income' and 'Spending Score'.
    Based on initial visual inspection and common practice, we initially tried k=5 clusters.
    """
)

scatter_2d_path = check_file(os.path.join(FIGURES_DIR, "scatter_clusters_2_features.png"))
st.image(scatter_2d_path, caption="Clusters based on Annual Income and Spending Score (k=5)")

st.write("The scatter plot visually confirms 5 distinct customer segments based on these two features.")


# --- Finding Optimal K (2 Features) ---
st.header("4. Finding the Optimal Number of Clusters (k) - 2 Features")
st.write(
    """
    To formally determine the best number of clusters (k), we use two common methods:
    the Elbow Method (using WCSS - Within-Cluster Sum of Squares) and the Silhouette Score.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Elbow Method")
    elbow_path = check_file(os.path.join(FIGURES_DIR, "elbow_plot_2_features.png"))
    st.image(elbow_path, caption="Elbow Plot for 2 Features")
    st.write("The 'elbow' point, where the rate of WCSS decrease slows down, appears around k=5.")

with col2:
    st.subheader("Silhouette Method")
    silhouette_2d_path = check_file(os.path.join(FIGURES_DIR, "silhouette_plot_2_features.png"))
    st.image(silhouette_2d_path, caption="Silhouette Plot for 2 Features")
    st.write("The Silhouette Score is highest at k=5, indicating the best separation and cohesion of clusters.")

st.write("**Conclusion (2 Features):** Both methods suggest that **k=5** is the optimal number of clusters when using 'Annual Income' and 'Spending Score'.")


# --- Finding Optimal K (3 Features) ---
st.header("5. Finding the Optimal Number of Clusters (k) - 3 Features")
st.write(
    """
    We then explored adding the 'Age' feature to the clustering analysis ('Age', 'Annual Income', 'Spending Score').
    We again used the Silhouette Score to evaluate the optimal number of clusters.
    """
)
silhouette_3d_path = check_file(os.path.join(FIGURES_DIR, "silhouette_plot_3_features.png"))
st.image(silhouette_3d_path, caption="Silhouette Plot for 3 Features (Age, Income, Spending Score)")
st.write(
    """
    **Conclusion (3 Features):** With three features, the Silhouette analysis suggests that **k=6** might provide slightly better-defined clusters compared to k=5.
    Visualizing these clusters directly is harder in 3D, so evaluation relies more heavily on metrics like the Silhouette Score.
    """
)


# --- Show Processed Data ---
st.header("6. Final Clustered Data Sample")
st.write(
    f"""
    The processed data file includes the original customer information along with the
    cluster assignments (based on the initial 2-feature, k=5 analysis shown in Section 3).
    This data can be used for targeted marketing campaigns.
    """
)
try:
    df_processed = pd.read_csv(check_file(PROCESSED_DATA_PATH))
    st.write("Sample of data with added cluster labels (`Cluster_2D_k5`):")
    st.dataframe(df_processed.head())
except FileNotFoundError:
    st.stop() # Error already shown by check_file
except Exception as e:
    st.error(f"An error occurred while loading the processed data: {e}")
    st.stop()

st.success("Analysis presentation complete!")