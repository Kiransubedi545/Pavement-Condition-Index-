import pandas as pd
import numpy as np
from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r'C:\Users\kiran\Desktop\data\Flexible Pavement\Pavement Asset Management\PCI_Result.csv'
df = pd.read_csv(file_path).fillna(0)

# Define features and target
features = [
    'GATOR_CRACK_A_L', 'GATOR_CRACK_A_M', 'GATOR_CRACK_A_H',
    'BLK_CRACK_A_H', 'TRANS_CRACK_L_M', 'TRANS_CRACK_L_H',
    'POTHOLES_A_L', 'POTHOLES_A_M', 'PATCH_A_H', 'PATCH_A_M'
]
target = 'PCI_SCORE'

X = df[features]
y = df[target]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Run FLAML AutoML
automl = AutoML()
automl.fit(X_train=X_train, y_train=y_train, task="regression", time_budget=60)

# Predict
y_pred = automl.predict(X_test)

# Regression metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 FLAML AutoML Regression Results:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.3f}")

# PCI Category conversion
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

y_test_cat = y_test.apply(pci_to_category)
y_pred_cat = pd.Series(y_pred).apply(pci_to_category)

# Classification metrics
print("\n📋 Classification Report (PCI Categories):")
print(classification_report(y_test_cat, y_pred_cat))

acc = accuracy_score(y_test_cat, y_pred_cat)
prec = precision_score(y_test_cat, y_pred_cat, average='weighted', zero_division=0)
rec = recall_score(y_test_cat, y_pred_cat, average='weighted')
f1 = f1_score(y_test_cat, y_pred_cat, average='weighted')

print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")

# Plot predictions vs actual
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual PCI")
plt.ylabel("Predicted PCI")
plt.title("FLAML: Actual vs Predicted PCI")
plt.grid(True)
plt.tight_layout()
plt.show()
