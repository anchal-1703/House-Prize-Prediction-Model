#!/usr/bin/env python
# coding: utf-8

# ## Optimized Real Estate Price Predictor

# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import joblib

# Load the dataset
housing = pd.read_csv("data/data.csv")

# Check dataset integrity
print("Dataset loaded successfully.")
print(housing.info())

# Stratified train-test split based on 'CHAS' column
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Separate features and target variable from the training set
housing = strat_train_set.copy()
X = housing.drop('MEDV', axis=1)
y = housing['MEDV']

# Feature scaling (important for SVR, Ridge, Lasso, etc.)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define regression models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "SVR": SVR(),
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Bayesian Ridge": BayesianRidge()
}

best_model_name = None
best_model = None
best_rmse = float('inf')

# Evaluate models using cross-validation and store results
with open("regression_results.txt", "w") as file:
    for name, model in models.items():
        # Use scaled features for training
        scores = cross_val_score(model, X_scaled, y, scoring='neg_mean_squared_error', cv=10)
        rmse_scores = np.sqrt(-scores)

        output = f"Model: {name}\n"
        output += f"RMSE Scores: {rmse_scores}\n"
        output += f"Mean RMSE: {rmse_scores.mean():.4f}\n"
        output += f"Standard Deviation: {rmse_scores.std():.4f}\n"
        output += "-" * 40 + "\n"

        file.write(output)

        # Check for best model
        if rmse_scores.mean() < best_rmse:
            best_rmse = rmse_scores.mean()
            best_model_name = name
            best_model = model

# Train the best model on the full scaled training set
best_model.fit(X_scaled, y)

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print("Best model saved as 'best_model.pkl'.")

# Save the scaler (for use in predictions later)
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'.")

# Final output
print(f"\nThe Best Fit Model: {best_model_name}")
print(f"Minimum Mean RMSE: {best_rmse:.4f}")
print("All model evaluations written to 'regression_results.txt'.")

#  Visualization
plt.figure(figsize=(10, 6))
rmse_means = [np.sqrt(-cross_val_score(m, X_scaled, y, scoring='neg_mean_squared_error', cv=10)).mean()
              for m in models.values()]
plt.barh(list(models.keys()), rmse_means, color='skyblue')
plt.xlabel("Mean RMSE")
plt.title("Model Comparison - Mean RMSE")
plt.tight_layout()
plt.grid(True)
plt.show()
