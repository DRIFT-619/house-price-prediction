import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained pipeline
pipeline = joblib.load("house_price_pipeline.joblib")

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction")
st.write("Enter property details to estimate the house price.")

st.divider()

# ----Assign Null to All Features initially -----
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

# numeric → np.nan
for col in NUMERIC_FEATURES:
    input_data[col] = np.nan

# categorical → string placeholder
for col in CATEGORICAL_FEATURES:
    input_data[col] = "None"

# ---- USER INPUTS (important features only) ----
overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=2000)
garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=800)

neighborhood = st.selectbox(
    "Neighborhood",
    ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"]
)

house_style = st.selectbox(
    "House Style",
    ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"]
)

kitchen_qual = st.selectbox(
    "Kitchen Quality",
    ["TA", "Gd", "Ex", "Fa"]
)

# ---- DEFAULT VALUES FOR REMAINING FEATURES ----
# Categorical Defaults
input_data["OverallQual"] = overall_qual
input_data["GrLivArea"] = gr_liv_area
input_data["YearBuilt"] = year_built
input_data["GarageCars"] = garage_cars
input_data["TotalBsmtSF"] = total_bsmt_sf
input_data["Neighborhood"] = neighborhood
input_data["Neighborhood"] = house_style
input_data["KitchenQual"] = kitchen_qual

input_data["MSZoning"] = "RL"
input_data["Street"] = "Pave"
input_data["LotShape"] = "Reg"
input_data["LandContour"] = "Lvl"
input_data["LotConfig"] = "Inside"
input_data["Condition1"] = "Norm"
input_data["Condition2"] = "Norm"
input_data["BldgType"] = "1Fam"
input_data["RoofStyle"] = "Gable"
input_data["RoofMatl"] = "CompShg"
input_data["Exterior1st"] = "VinylSd"
input_data["Exterior2nd"] = "VinylSd"
input_data["Foundation"] = "PConc"
input_data["Heating"] = "GasA"
input_data["CentralAir"] = "Y"
input_data["Electrical"] = "SBrkr"
input_data["Functional"] = "Typ"
input_data["SaleType"] = "WD"
input_data["SaleCondition"] = "Normal"

# Numeric Defaults
input_data["LotFrontage"] = 60
input_data["LotArea"] = 8000
input_data["OverallCond"] = 5
input_data["YearRemodAdd"] = year_built
input_data["BsmtFinSF1"] = 400
input_data["BsmtFinSF2"] = 0
input_data["BsmtUnfSF"] = 400
input_data["1stFlrSF"] = 1000
input_data["2ndFlrSF"] = 500
input_data["FullBath"] = 2
input_data["HalfBath"] = 1
input_data["BedroomAbvGr"] = 3
input_data["TotRmsAbvGrd"] = 6
input_data["Fireplaces"] = 1
input_data["GarageArea"] = 400
input_data["OpenPorchSF"] = 50
input_data["MoSold"] = 6
input_data["YrSold"] = 2010

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# ---- PREDICTION ----
if st.button("Predict Price"):
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Estimated House Price: ${prediction:,.0f}")
