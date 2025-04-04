import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
file_path = r'C:\Users\kiran\Desktop\data\Flexible Pavement\Pavement Asset Management\PCI_Result.csv'
df = pd.read_csv(file_path).fillna(0)

# Step 2: Define features and target
features = [
    'GATOR_CRACK_A_L', 'GATOR_CRACK_A_M', 'GATOR_CRACK_A_H',
    'BLK_CRACK_A_H', 'TRANS_CRACK_L_M', 'TRANS_CRACK_L_H',
    'POTHOLES_A_L', 'POTHOLES_A_M', 'PATCH_A_H', 'PATCH_A_M'
]
target = 'PCI_SCORE'

X = df[features]
y = df[target]

# Step 3: Define XGBoost model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Step 4: K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X, y, cv=kf)

# Step 5: Regression Evaluation
rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("📊 K-Fold Regression Metrics (5 folds):")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# Step 6: PCI to Category Conversion
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

y_true_cat = y.apply(pci_to_category)
y_pred_cat = pd.Series(y_pred).apply(pci_to_category)

# Step 7: Classification Report
print("\n📋 PCI Category Classification Report (from K-Fold CV):")
print(classification_report(y_true_cat, y_pred_cat))

# Step 8: Train final model for Feature Importance + Plot
model.fit(X, y)
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importance (XGBoost Full Data)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# Step 9: Actual vs Predicted PCI
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y, y=y_pred)
plt.xlabel("Actual PCI")
plt.ylabel("Predicted PCI")
plt.title("XGBoost (K-Fold): Actual vs Predicted PCI")
plt.grid(True)
plt.tight_layout()
plt.show()

import joblib
joblib.dump(model, "xgb_pci_model.pkl")
print("✅ Model saved as xgb_pci_model.pkl")
