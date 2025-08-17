# AutoRegress: Automated Regression Benchmarking Tool 🚀

**AutoRegress** is a user-friendly web application designed to simplify the process of data cleaning and regression model benchmarking. Upload your dataset, clean it with intuitive tools, and instantly compare the performance of nine different regression models to find the best fit for your data.

## 📋 Key Features

* **Easy Data Upload**: Supports both **CSV** and **XLSX** file formats with a simple drag-and-drop interface.
* **Data Quality Insights**: Automatically generates a report on missing values and potential outliers in your dataset.
* **Intuitive Cleaning Tools**:
    * Handle missing data by imputing with the mean, median, or mode, or by using forward/backward fill.
    * Manage outliers by clipping them to a specified range or removing them entirely.
* **Automated Model Benchmarking**: With a single click, train and evaluate nine different regression models, including:
    * Linear Regression, Ridge, Lasso, and ElasticNet
    * SGD Regressor, Passive Aggressive, and Huber Regressor
    * RANSAC and TheilSen Regressors (robust to outliers)
* **Comprehensive Performance Metrics**: Compares models across eight key metrics: MSE, RMSE, MAE, R-squared, Explained Variance, MSLE, Median Absolute Error, and Max Error.
* **Interactive Results Table**: Highlights the best and worst-performing models for each metric, making it easy to interpret the results.

---

## 🛠️ Built With

* **Backend**: [Flask](https://flask.palletsprojects.com/)
* **Frontend**: [Tailwind CSS](https://tailwindcss.com/)
* **Data Manipulation**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
* **Machine Learning**: [Scikit-learn](https://scikit-learn.org/)

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You need to have Python 3 installed on your system.

### Installation

1.  **Clone the repo**
    ```sh
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Create and activate a virtual environment (recommended)**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required packages**
    ```sh
    pip install -r requirements.txt
    ```
    *Note: If you don't have a `requirements.txt` file, you can create one with the following content:*
    ```
    flask
    pandas
    numpy
    scikit-learn
    openpyxl
    ```

### Running the Application

Execute the main application file from your terminal:

```sh
python app.py
````

This will launch a desktop window with the application running inside.

-----

## 📖 Usage

1.  **Upload Your Data**: Drag and drop your CSV or XLSX file into the upload area on the main page.
2.  **Clean Your Data**:
      * On the "Clean Data" screen, review the missing values and outliers report.
      * Use the provided dropdowns and buttons to impute missing data or handle outliers for each affected column.
3.  **Benchmark Models**:
      * Once your data is clean, select the target variable (the column you want to predict) from the dropdown menu.
      * Click the **"Benchmark"** button to start the analysis. A loading indicator will appear while the models are training.
4.  **Analyze Results**:
      * The results page will display a table comparing the performance of all nine models across various metrics.
      * The best-performing models are highlighted in green, and the worst are in red.
      * You can re-run the benchmark with a different target column at any time.
