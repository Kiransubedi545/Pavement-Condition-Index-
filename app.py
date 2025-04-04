import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("xgb_pci_model.pkl")

# PCI category classification
def pci_to_category(pci):
    if pci >= 85:
        return "Good"
    elif pci >= 70:
        return "Satisfactory"
    elif pci >= 55:
        return "Fair"
    elif pci >= 40:
        return "Poor"
    else:
        return "Very Poor"

st.set_page_config(page_title="PCI Prediction App", layout="centered")
st.title("🛣️ Pavement Condition Index (PCI) Prediction")

features = [
    'GATOR_CRACK_A_L', 'GATOR_CRACK_A_M', 'GATOR_CRACK_A_H',
    'BLK_CRACK_A_H', 'TRANS_CRACK_L_M', 'TRANS_CRACK_L_H',
    'POTHOLES_A_L', 'POTHOLES_A_M', 'PATCH_A_H', 'PATCH_A_M'
]

inputs = {}
cols = st.columns(2)
for i, feature in enumerate(features):
    inputs[feature] = cols[i % 2].number_input(
        feature.replace("_", " "), min_value=0.0, value=0.0, step=1.0
    )

if st.button("Predict PCI"):
    x = np.array([list(inputs.values())])
    pci_score = model.predict(x)[0]
    pci_cat = pci_to_category(pci_score)
    st.success(f"✅ Predicted PCI: {pci_score:.2f}")
    st.info(f"🏷️ Condition Category: **{pci_cat}**")
