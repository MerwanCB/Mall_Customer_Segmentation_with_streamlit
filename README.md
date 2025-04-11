
## Future Enhancements

*   Deploy the Streamlit application to a cloud platform (e.g., Streamlit Community Cloud, Heroku).
*   Experiment with other clustering algorithms (e.g., DBSCAN, Hierarchical Clustering).
*   Develop detailed personas for each identified customer segment.
*   Incorporate more features if available (e.g., purchase history, visit frequency).
*   Add interactive elements to the Streamlit app (e.g., selecting features for clustering).

## Installation & Usage (Local)

To run the analysis pipeline and the Streamlit presentation app locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git # Replace with your repo URL
    cd your-repo-name
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
    *(Ensure you have a `requirements.txt` file. If not, create one using `pip freeze > requirements.txt` after installing necessary libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit)*
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