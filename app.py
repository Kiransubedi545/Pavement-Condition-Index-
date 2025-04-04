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

# Disclaimer
with st.expander("📌 Input Guidelines & Disclaimer"):
    st.markdown("""
    - All distress data entered must represent a **minimum pavement sample unit of 20 m²**, as defined by ASTM D6433.
    - It is recommended to standardize input based on a **49.4 m²** unit.
    - Area-based distresses (e.g., alligator cracking, patching, potholes) must be reported in **square meters (m²)**.
    - Length-based distresses (e.g., transverse cracks) must be recorded in **meters (m)**.
    - This tool is intended for educational and decision-support purposes only.
    """)

st.markdown("""
Enter distress values per sample unit: use **sq.m** for area and **m** for length.
""")

features = [
    'GATOR_CRACK_A_L', 'GATOR_CRACK_A_M', 'GATOR_CRACK_A_H',
    'BLK_CRACK_A_H', 'TRANS_CRACK_L_M', 'TRANS_CRACK_L_H',
    'POTHOLES_A_L', 'POTHOLES_A_M', 'PATCH_A_H', 'PATCH_A_M'
]

# Full guidance for each input
guidance = {
    'GATOR_CRACK_A_L': "Light severity alligator cracking (area in sq.m)",
    'GATOR_CRACK_A_M': "Medium severity alligator cracking (area in sq.m)",
    'GATOR_CRACK_A_H': "High severity alligator cracking (area in sq.m)",
    'BLK_CRACK_A_H': "High severity block cracking (area in sq.m)",
    'TRANS_CRACK_L_M': "Medium severity transverse cracking (length in meters)",
    'TRANS_CRACK_L_H': "High severity transverse cracking (length in meters)",
    'POTHOLES_A_L': "Low severity potholes (area in sq.m)",
    'POTHOLES_A_M': "Medium severity potholes (area in sq.m)",
    'PATCH_A_H': "High severity patching (area in sq.m)",
    'PATCH_A_M': "Medium severity patching (area in sq.m)"
}

inputs = {}
cols = st.columns(2)
for i, feature in enumerate(features):
    with cols[i % 2]:
        inputs[feature] = st.number_input(
            label=feature.replace('_', ' '),
            min_value=0.0,
            value=0.0,
            step=1.0,
            help=guidance[feature],
            key=feature
        )

if st.button("Predict PCI"):
    x = np.array([list(inputs.values())])
    pci_score = model.predict(x)[0]
    pci_cat = pci_to_category(pci_score)
    st.success(f"✅ Predicted PCI: {pci_score:.2f}")
    st.info(f"🏷️ Condition Category: **{pci_cat}**")

# Footer (single line)
st.markdown("""
---
👷‍♂️ Developed by **Kiran Subedi** | 🌐 [kiransubedi545.com.np](https://kiransubedi545.com.np/) | 📧 [Kiransubedi545@gmail.com](mailto:Kiransubedi545@gmail.com)
""")
