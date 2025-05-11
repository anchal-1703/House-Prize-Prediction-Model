import joblib
import numpy as np
import pandas as pd

# Load the saved model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

new_data = pd.DataFrame([{
    'CRIM': 0.04819,
    'ZN': 80.0,
    'INDUS': 3.64,
    'CHAS': 0,
    'NOX': 0.392,
    'RM': 6.108,
    'AGE': 32.0,
    'DIS': 9.2203,
    'RAD': 1,
    'TAX': 315,
    'PTRATIO': 16.4,
    'B': 392.89,
    'LSTAT': 6.57
}])

# Scale the features using the same scaler used in training
new_data_scaled = scaler.transform(new_data)

# Predict the price
predicted_price = model.predict(new_data_scaled)

print(f"Predicted MEDV (housing price): {predicted_price[0]:.4f}")
