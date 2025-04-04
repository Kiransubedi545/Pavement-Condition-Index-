import pandas as pd
import joblib

# Load your dataset
df = pd.read_csv(r'C:\Users\kiran\Desktop\data\Flexible Pavement\Pavement Asset Management\PCI_Result.csv').fillna(0)

# Define features and target
features = [
    'GATOR_CRACK_A_L', 'GATOR_CRACK_A_M', 'GATOR_CRACK_A_H',
    'BLK_CRACK_A_H', 'TRANS_CRACK_L_M', 'TRANS_CRACK_L_H',
    'POTHOLES_A_L', 'POTHOLES_A_M', 'PATCH_A_H', 'PATCH_A_M'
]
target = 'PCI_SCORE'

X = df[features]
y = df[target]

# Train FLAML model
automl = AutoML()
automl.fit(X_train=X, y_train=y, task="regression", time_budget=60)

# Save model to the FLAML app folder
joblib.dump(automl, r'C:\Users\kiran\Desktop\data\Flexible Pavement\Pavement Asset Management\PCI\FLAML\flaml_pci_model.pkl')

print("✅ Model saved successfully.")
