# Mall Customer Segmentation Analysis

This project helps a mall understand its customers better by grouping them based on how they shop. We look at things like customer age, income, and spending habits to find patterns and create different customer groups using a technique called K-Means clustering.

A web app built with Streamlit shows the results of this analysis, including graphs and explanations.

Visit the Streamlit presentation app here:
[https://merwancb-mall-customer-segmentation-with-s-app-streamlit-u5zwg8.streamlit.app/](https://merwancb-mall-customer-segmentation-with-s-app-streamlit-u5zwg8.streamlit.app/)

## Project Goal

The main aim is to use customer data (Age, Annual Income, Spending Score) to divide mall shoppers into distinct segments. This helps the mall understand who their customers are and how to market to them more effectively.

## Features (Codebase & Presentation App)

*   **Modular Code Structure:** Python code organized into data loading, feature processing, modeling (clustering), and visualization modules.
*   **K-Means Clustering:** Implementation of the K-Means algorithm to segment customers.
*   **Optimal Cluster Selection:** Uses Elbow Method (WCSS) and Silhouette Score analysis to determine the appropriate number of clusters (k).
*   **Data Exploration & Visualization:** Includes scripts to generate pairplots, cluster scatter plots, and evaluation plots (Elbow, Silhouette).
*   **Streamlit Presentation:** A user-friendly web application (`app_streamlit.py`) to showcase the dataset, analysis steps, key visualizations, and conclusions.

## Dataset

The analysis uses the "Mall Customer Dataset," a commonly used dataset for clustering tasks. It contains the following features:

*   **CustomerID:** Unique ID for each customer.
*   **Gender:** Gender of the customer.
*   **Age:** Age of the customer.
*   **Annual_Income:** Customer's annual income (in k$).
*   **Spending_Score:** A score (1-100) assigned by the mall based on spending behavior.

## Technologies Used

*   **Python:** Core programming language.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Scikit-learn:** For K-Means clustering implementation and evaluation metrics (Silhouette Score).
*   **Matplotlib & Seaborn:** For generating static visualizations.
*   **Streamlit:** For building the interactive presentation web application.

## Methodology

1.  **Data Loading & Exploration:** Load the dataset and perform initial exploratory analysis (statistics, correlations, pairplots).
2.  **Feature Selection:** Select relevant features for clustering (initially 'Annual_Income', 'Spending_Score', later adding 'Age').
3.  **K-Means Clustering:** Apply the K-Means algorithm to the selected features.
4.  **Optimal K Determination:** Evaluate different numbers of clusters (k) using the Elbow Method and Silhouette Scores to find the most suitable value.
5.  **Visualization:** Plot the resulting clusters and evaluation metrics.
6.  **Presentation:** Summarize findings and display visualizations using the Streamlit app.

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── mall_customers.csv
│   └── processed/
│       └── clustered_customers.csv
├── main.py                     # Main script to run the analysis pipeline
├── app_streamlit.py            # Streamlit application script
├── reports/
│   └── figures/                # Saved plots and visualizations
│       ├── pairplot_features.png
│       ├── ... (other plots)
├── src/
│   ├── __init__.py
│   ├── data/                   # Data loading/saving module
│   │   └── load_save_data.py
│   ├── features/               # Feature engineering/selection module
│   │   └── feature_selection.py
│   ├── models/                 # Clustering model implementation
│   │   └── clustering.py
│   └── visualization/          # Visualization generation module
│       └── visualize.py
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## Future Enhancements

*   Experiment with other clustering algorithms (e.g., DBSCAN, Hierarchical Clustering).
*   Develop detailed personas for each identified customer segment.
*   Incorporate more features if available (e.g., purchase history, visit frequency).

## Installation & Usage (Local)

To run the analysis pipeline and the Streamlit presentation app locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MerwanCB/Mall_Customer_Segmentation_with_streamlit.git
    cd Mall_Customer_Segmentation_with_streamlit
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
   
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the main analysis script:**
    *(This will process the data and generate the figures needed by the Streamlit app)*
    ```bash
    python main.py
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app_streamlit.py
    ```

This will open the presentation app in your default web browser.

---

Thank you for checking out the Mall Customer Segmentation project!