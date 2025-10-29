import streamlit as st
import numpy as np
import joblib
import pandas as pd
from nrg_interview_prep.utils import marginal_effects
# ---------------------------
# Load trained OLS model
# ---------------------------
# make sure this file is in your workspace root
model = joblib.load("/workspaces/nrg_interview_prep/data/height_model.pkl")
b = model.params  # coefficients

# ---------------------------
# Define marginal effect function
# ---------------------------


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Height Model Explorer", layout="centered")

st.title("📏 Height Model Explorer")
st.write("""
Explore how **age**, **weight**, and **sex** affect predicted height  
and the marginal height gain for a 10-unit weight increase.
""")

# Inputs
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=0.0,
                          max_value=100.0, value=25.0, step=1.0)
with col2:
    weight = st.number_input("Weight", min_value=20.0,
                             max_value=70.0, value=40.0, step=1.0)
with col3:
    sex = st.selectbox("Sex", ["Female", "Male"])
male = 1 if sex == "Male" else 0

# ---------------------------
# Make predictions
# ---------------------------
# Build design matrix manually to use model.predict
data_point = {
    "weight": weight,
    "I(weight**2)": weight ** 2,
    "age": age,
    "I(age**2)": age ** 2,
    "male": male,
    "weight:male": weight * male,
    "age:male": age * male
}

X = model.model.exog_names
x_row = np.array([data_point.get(var, 0) for var in X])
pred_height = float(np.dot(x_row, model.params))

# Marginal effects
d_w, d_a = marginal_effects(weight, age, male, b)
height_gain_10 = d_w * 10

# ---------------------------
# Display results
# ---------------------------
st.subheader("Predicted Results")
st.metric("Predicted Height", f"{pred_height:.2f} units")
st.metric("Marginal Height Gain (for +10 weight)",
          f"{height_gain_10:.2f} units")
