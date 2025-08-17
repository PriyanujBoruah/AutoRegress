# app.py

import os
import pandas as pd
import numpy as np
import gc
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, 
    PassiveAggressiveRegressor, HuberRegressor, RANSACRegressor, TheilSenRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, mean_squared_log_error, median_absolute_error, max_error
)

# --- Flask App Setup ---
app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dataframe(filepath):
    """Reads a CSV or XLSX file into a pandas DataFrame."""
    try:
        if filepath.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            return pd.read_excel(filepath)
    except Exception as e:
        flash(f"Error reading file: {e}", "error")
        return None
    return None

def optimize_df(df):
    """Downcasts numerical columns and converts low-cardinality objects to categories."""
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    return df

def get_data_quality(df):
    """Calculates missing values and potential outliers."""
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    quality_report = {}
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_data = missing_percent[missing_percent > 0].to_dict()
    quality_report['missing_values'] = {k: round(v, 2) for k, v in missing_data.items()}
    outliers = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if outlier_count > 0:
            outliers[col] = outlier_count
    quality_report['outliers'] = outliers
    return quality_report

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles file upload and redirects to the cleaning page."""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('clean_data', filename=filename))
    return render_template('index.html')

@app.route('/clean/<filename>')
def clean_data(filename):
    """Displays the data quality report and cleaning tools."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = get_dataframe(filepath)
    if df is None: return redirect(url_for('index'))
    
    df = optimize_df(df)
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    data_quality = get_data_quality(df)
    
    del df; gc.collect()
    
    return render_template('clean.html', filename=filename, data_quality=data_quality, numerical_columns=numerical_columns)

@app.route('/impute/<filename>', methods=['POST'])
def impute(filename):
    """Handles imputation of missing values."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    column = request.form.get('column')
    strategy = request.form.get('strategy')
    df = get_dataframe(filepath)
    if df is None: return redirect(url_for('index'))

    if strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
        df[column] = imputer.fit_transform(df[[column]]).ravel()
    elif strategy in ['ffill', 'bfill']:
        df[column].fillna(method=strategy, inplace=True)
    elif strategy == 'remove_rows':
        df.dropna(subset=[column], inplace=True)
    
    df.to_csv(filepath, index=False)
    flash(f"Successfully applied '{strategy}' to column '{column}'.", "success")
    return redirect(url_for('clean_data', filename=filename))

@app.route('/outliers/<filename>', methods=['POST'])
def handle_outliers(filename):
    """Handles clipping or removing outliers."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    column = request.form.get('column')
    action = request.form.get('action')
    iqr_multiplier = float(request.form.get('iqr_multiplier', 1.5))
    df = get_dataframe(filepath)
    if df is None: return redirect(url_for('index'))

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    if action == 'clip':
        df[column] = np.clip(df[column], lower_bound, upper_bound)
        flash(f"Clipped outliers in '{column}'.", "success")
    elif action == 'remove':
        original_rows = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        flash(f"Removed {original_rows - len(df)} outlier rows.", "success")
    
    df.to_csv(filepath, index=False)
    return redirect(url_for('clean_data', filename=filename))

@app.route('/benchmark/<filename>', methods=['POST'])
def benchmark(filename):
    """Runs regression models and displays a comparison table."""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = get_dataframe(filepath)
    if df is None: return redirect(url_for('index'))
    
    target_column = request.form.get('target_column')
    if not target_column:
        flash("You must select a target column to benchmark.", "error")
        return redirect(url_for('clean_data', filename=filename))

    benchmark_results = {}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "SGD Regressor": SGDRegressor(),
        "Passive Aggressive": PassiveAggressiveRegressor(),
        "Huber Regressor": HuberRegressor(),
        "RANSAC Regressor": RANSACRegressor(),
        "TheilSen Regressor": TheilSenRegressor()
    }
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mse = mean_squared_error(y_test, preds)
            msle = mean_squared_log_error(y_test, preds) if all(y_test >= 0) and all(preds >= 0) else 'N/A'

            benchmark_results[name] = {
                "MSE": round(mse, 4), "RMSE": round(np.sqrt(mse), 4), "MAE": round(mean_absolute_error(y_test, preds), 4),
                "R-squared": round(r2_score(y_test, preds), 4), "Explained Variance": round(explained_variance_score(y_test, preds), 4),
                "MSLE": round(msle, 4) if isinstance(msle, (int, float)) else msle,
                "Median Absolute Error": round(median_absolute_error(y_test, preds), 4), "Max Error": round(max_error(y_test, preds), 4)
            }
        except Exception as e:
            benchmark_results[name] = { "MSE": "Failed", "RMSE": "-", "MAE": "-", "R-squared": "-", "Explained Variance": "-", "MSLE": "-", "Median Absolute Error": "-", "Max Error": "-" }
            print(f"Model {name} failed with error: {e}")

    # --- Find Best and Worst Scores ---
    best_scores = {}
    worst_scores = {}
    error_metrics = ["MSE", "RMSE", "MAE", "MSLE", "Median Absolute Error", "Max Error"]
    performance_metrics = ["R-squared", "Explained Variance"]

    for metric in error_metrics:
        valid_scores = [(model, data[metric]) for model, data in benchmark_results.items() if isinstance(data[metric], (int, float))]
        if valid_scores:
            best_scores[metric] = min(valid_scores, key=lambda item: item[1])[0]
            worst_scores[metric] = max(valid_scores, key=lambda item: item[1])[0]

    for metric in performance_metrics:
        valid_scores = [(model, data[metric]) for model, data in benchmark_results.items() if isinstance(data[metric], (int, float))]
        if valid_scores:
            best_scores[metric] = max(valid_scores, key=lambda item: item[1])[0]
            worst_scores[metric] = min(valid_scores, key=lambda item: item[1])[0]
            
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

    # --- Tooltip Descriptions ---
    model_descriptions = {
        "Linear Regression": "Standard linear model.",
        "Ridge": "Linear model with L2 regularization.",
        "Lasso": "Linear model with L1 regularization, useful for feature selection.",
        "ElasticNet": "Combination of Ridge and Lasso.",
        "SGD Regressor": "Efficient model for very large datasets.",
        "Passive Aggressive": "Good for streaming data.",
        "Huber Regressor": "Robust to outliers.",
        "RANSAC Regressor": "Robust model that fits on subsets of data.",
        "TheilSen Regressor": "Another robust model, good against outliers."
    }
    metric_descriptions = {
        "MSE": "Mean Squared Error: Lower is better.",
        "RMSE": "Root Mean Squared Error: Lower is better.",
        "MAE": "Mean Absolute Error: Lower is better.",
        "R-squared": "Coefficient of determination: Higher is better (1 is perfect).",
        "Explained Variance": "How much of the dataset's variance the model can explain: Higher is better.",
        "MSLE": "Mean Squared Log Error: Good for targets with exponential trends. Lower is better.",
        "Median Absolute Error": "Robust to outliers. Lower is better.",
        "Max Error": "The worst-case error between prediction and true value. Lower is better."
    }

    return render_template('benchmark.html', results=benchmark_results, problem_type="Regression", 
                           best_scores=best_scores, worst_scores=worst_scores, filename=filename, 
                           numerical_columns=numerical_columns, model_descriptions=model_descriptions, 
                           metric_descriptions=metric_descriptions)


if __name__ == '__main__':
    app.run(debug=True)
