# House Price Prediction – End-to-End ML Pipeline

## Project Overview

This project implements an end-to-end machine learning pipeline to predict residential house prices using real-world housing data. The solution covers the full ML lifecycle, starting from data preprocessing, feature engineering, model training to evaluation, and deployment and provides real-time predictions via a Streamlit web application.

The model is trained on the house-prices-advanced-regression-techniques dataset (Kaggle) and uses a tuned XGBoost Regressor for accurate price estimation.

## Problem Statement

Accurately estimating house prices is critical for buyers, sellers, and real estate investors. The challenge lies in handling:

- High-cardinality categorical variables
- Missing and noisy real-world data
- Feature consistency between training and inference

This project addresses these challenges using a robust sklearn Pipeline + ColumnTransformer architecture.

## Solution Architecture
The system follows a production-style machine learning pipeline:

1. Raw housing data ingestion
2. Outlier removal applied during training only
3. Feature preprocessing using sklearn ColumnTransformer:
   - One-Hot Encoding for categorical features
   - Numeric feature pass-through
4. Price prediction using a tuned XGBoost Regressor
5. Real-time inference via Streamlit web application

All preprocessing and model inference are encapsulated inside a single sklearn Pipeline, ensuring feature consistency between training and deployment.

#### Encoding Strategy
Although the original coursework explored target encoding for high-cardinality features, the deployed production pipeline uses One-Hot Encoding to ensure:
- Leakage-free inference
- Robust handling of unseen categories
- Simpler and more reliable deployment


## Technologies Used

**Programming:** Python <br>
**Machine Learning:** XGBoost, scikit-learn <br>
**Data Processing:** Pandas, NumPy <br>
**Model Management:** sklearn Pipeline, ColumnTransformer, joblib <br>
**Web App:** Streamlit <br>
**Deployment:** Streamlit Cloud <br>
**Dataset:** house-prices-advanced-regression-techniques Challenge Dataset (Kaggle) <br>

## Model Details

- **Model:** XGBoost Regressor
- **Hyperparameters:**
n_estimators = 100, max_depth = 3, learning_rate = 0.1, subsample = 0.8

#### Feature Handling
- 36 Numerical Features
- 40 Nominal Categorical Features → One-Hot Encoding

Evaluation Metric: RMSE (Root Mean Squared Error)

The complete preprocessing and model logic is encapsulated in a single sklearn Pipeline, ensuring identical transformations during training and inference.

## Live Application

**Live Demo:** (Add Streamlit Cloud URL here after deployment)

Users can input property characteristics through an interactive UI and receive an estimated house price instantly.

## Running the App Locally

```bash
pip install -r requirements.txt
streamlit run House_Price_Prediction_App.py
