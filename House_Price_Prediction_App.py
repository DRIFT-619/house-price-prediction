import streamlit as st
import requests

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction")
st.write("Enter property details to estimate the house price.")

st.divider()

# ---- USER INPUTS ----
overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 300, 6000, 1500)
year_built = st.number_input("Year Built", 1870, 2025, 2000)
garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", 0, 3000, 800)

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

# ---- PREDICT ----
if st.button("Predict Price"):

    payload = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
        "KitchenQual": kitchen_qual
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload
    )

    if response.status_code == 200:
        price = response.json()["predicted_price"]
        st.success(f"Estimated House Price: ${price:,.0f}")
    else:
        st.error("Prediction failed. Check API.")
