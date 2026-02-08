from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Creating app
app = FastAPI(title="House Price Prediction API")

# Loading pipeline
pipeline = joblib.load("house_price_pipeline.joblib")

# -------------------------------
# Input schema (Only for user inputs)
# -------------------------------
class HouseInput(BaseModel):
    OverallQual: int
    GrLivArea: float
    YearBuilt: int
    GarageCars: int
    TotalBsmtSF: float
    Neighborhood: str
    HouseStyle: str
    KitchenQual: str


# -------------------------------
# Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict_price(data: HouseInput):

    # Full feature template
    NUMERIC_FEATURES = [
        'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
        'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
        'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
        'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
        'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
        'MoSold', 'YrSold'
    ]

    CATEGORICAL_FEATURES = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'LotConfig',
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
        'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
        'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
        'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
    ]

    input_data = {}

    # numeric defaults
    for col in NUMERIC_FEATURES:
        input_data[col] = np.nan

    # categorical defaults
    for col in CATEGORICAL_FEATURES:
        input_data[col] = "None"

    # user inputs
    input_data["OverallQual"] = data.OverallQual
    input_data["GrLivArea"] = data.GrLivArea
    input_data["YearBuilt"] = data.YearBuilt
    input_data["GarageCars"] = data.GarageCars
    input_data["TotalBsmtSF"] = data.TotalBsmtSF
    input_data["Neighborhood"] = data.Neighborhood
    input_data["HouseStyle"] = data.HouseStyle
    input_data["KitchenQual"] = data.KitchenQual

    # safe defaults (same as training assumptions)
    input_data.update({
        "MSZoning": "RL",
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "LotConfig": "Inside",
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "Foundation": "PConc",
        "Heating": "GasA",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "Functional": "Typ",
        "SaleType": "WD",
        "SaleCondition": "Normal",
        "LotFrontage": 60,
        "LotArea": 8000,
        "OverallCond": 5,
        "YearRemodAdd": data.YearBuilt,
        "BsmtFinSF1": 400,
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 400,
        "1stFlrSF": 1000,
        "2ndFlrSF": 500,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 6,
        "Fireplaces": 1,
        "GarageArea": 400,
        "OpenPorchSF": 50,
        "MoSold": 6,
        "YrSold": 2010
    })

    df = pd.DataFrame([input_data])
    prediction = pipeline.predict(df)[0]

    return {"predicted_price": float(prediction)}

@app.get("/")
def read_root():
    return {"message": "House Price Prediction API is running"}
