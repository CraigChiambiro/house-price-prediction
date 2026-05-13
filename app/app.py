import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide"
)

# =========================
# LOAD MODEL + FEATURES
# =========================
model = joblib.load("models/model.pkl")
features = joblib.load("models/features.pkl")

# =========================
# TITLE
# =========================
st.title("🏠 AI House Price Prediction System")
st.caption("Machine Learning Regression Model | Random Forest")

st.write("Predict estimated house prices using property features.")

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns(2)

# =========================
# INPUTS
# =========================
with col1:

    st.subheader("🏡 Property Details")

    overall_qual = st.slider(
        "Overall Quality (1-10)",
        1,
        10,
        5
    )

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=500,
        max_value=5000,
        value=1500
    )

    garage_area = st.number_input(
        "Garage Area",
        min_value=0,
        max_value=1500,
        value=300
    )

    total_bsmt_sf = st.number_input(
        "Basement Area",
        min_value=0,
        max_value=3000,
        value=800
    )

with col2:

    st.subheader("🛏 Additional Features")

    bedrooms = st.slider(
        "Bedrooms",
        1,
        8,
        3
    )

    full_bath = st.slider(
        "Bathrooms",
        1,
        4,
        2
    )

    neighborhood = st.selectbox(
        "Neighborhood",
        [
            "NAmes",
            "CollgCr",
            "OldTown",
            "Edwards"
        ]
    )

    house_style = st.selectbox(
        "House Style",
        [
            "1Story",
            "2Story",
            "1.5Fin"
        ]
    )

# =========================
# BUILD INPUT DATAFRAME
# =========================
input_df = pd.DataFrame([{
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "GarageArea": garage_area,
    "TotalBsmtSF": total_bsmt_sf,
    "BedroomAbvGr": bedrooms,
    "FullBath": full_bath,
    "Neighborhood": neighborhood,
    "HouseStyle": house_style
}])

# =========================
# ONE HOT ENCODING
# =========================
input_df = pd.get_dummies(input_df)

# MATCH TRAINING FEATURES
input_df = input_df.reindex(
    columns=features,
    fill_value=0
)

# =========================
# FEATURE IMPORTANCE
# =========================
def show_feature_importance(model, features):

    importances = model.feature_importances_

    indices = np.argsort(importances)[-10:]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        range(len(indices)),
        importances[indices]
    )

    ax.set_yticks(range(len(indices)))

    ax.set_yticklabels(
        [features[i] for i in indices]
    )

    ax.set_title("Top 10 Important Features")

    st.pyplot(fig)

# =========================
# PREDICTION
# =========================
st.markdown("---")

if st.button("🔍 Predict House Price"):

    prediction = model.predict(input_df)[0]

    st.subheader("📊 Prediction Result")

    st.success(
        f"🏠 Estimated House Price: $ {prediction:,.2f}"
    
    )
    st.caption("Dataset based on US housing market prices.")

    st.info(
        "Prediction generated using Random Forest Regression."
    )

    st.markdown("---")

    st.subheader("📈 Model Feature Importance")

    show_feature_importance(model, features)

# =========================
# FOOTER
# =========================
st.markdown("---")

st.caption(
    "Machine Learning Project | House Price Prediction "
)