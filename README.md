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

Raw Input Data
      ↓
Outlier Handling (Training Phase)
      ↓
ColumnTransformer
 ├─ One-Hot Encoding (Nominal Features)
 └─ Numeric Feature Pass-through
      ↓
XGBoost Regressor (Tuned)
      ↓
Prediction Output
      ↓
Streamlit Web App

### Encoding Strategy
Although the original coursework explored target encoding for high-cardinality features, 
the deployed production pipeline uses One-Hot Encoding to ensure:
- Leakage-free inference
- Robust handling of unseen categories
- Simpler and more reliable deployment


## Technologies Used

**Programming:** Python
**Machine Learning:** XGBoost, scikit-learn
**Data Processing:** Pandas, NumPy
**Model Management:** sklearn Pipeline, ColumnTransformer, joblib
**Web App:** Streamlit
**Deployment:** Streamlit Cloud
**Dataset:** house-prices-advanced-regression-techniques Challenge Dataset (Kaggle)

## Model Details

- **Model:** XGBoost Regressor
- **Hyperparameters:**
n_estimators = 100
max_depth = 3
learning_rate = 0.1
subsample = 0.8

### Feature Handling
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
