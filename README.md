# 🏡 House Price Prediction Model

This project builds a machine learning model to predict housing prices using the Boston Housing dataset. It compares multiple regression algorithms and selects the best-performing model based on cross-validation scores.

---

## 📦 Files

- `best_model.pkl` – Saved trained regression model
- `scaler.pkl` – Scaler used for feature normalization
- `regression_results.txt` – Detailed model performance comparison

---

## 🔥 Features

- Exploratory Data Analysis (EDA)
- Data Preprocessing (Handling missing values, encoding categorical variables, feature scaling)
- Model Training (Linear Regression / Random Forest / XGBoost, etc.)
- Model Evaluation (R² Score, RMSE, MAE)
- Hyperparameter Tuning (Optional)
- Predictions on new/unseen data

---

## 📂 Dataset

- **Source**: Boston Housing Dataset
- **Features**:
  - CRIM: Per capita crime rate
  - ZN: Proportion of residential land zoned for large lots
  - INDUS: Proportion of non-retail business acres
  - CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
  - NOX: Nitric oxide concentration
  - RM: Average number of rooms per dwelling
  - AGE: Proportion of owner-occupied units built before 1940
  - DIS: Distance to employment centers
  - RAD: Accessibility to radial highways
  - TAX: Property tax rate
  - PTRATIO: Pupil–teacher ratio
  - B: Proportion of Black population
  - LSTAT: % lower status of the population
  - **MEDV**: Median house value (target)

  ---
  ## ⚙️ Model Training

- **Algorithms evaluated**:
  - Linear Regression
  - Ridge & Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
  - Gradient Boosting Regressor
  - Bayesian Ridge
  - **Evaluation**:
  - Cross-validated using RMSE (Root Mean Squared Error)
  - Best model is selected and saved as `best_model.pkl`

---

  
## 📈 Workflow

1. **Data Collection**  
   Load the house dataset (CSV, Excel, etc.)

2. **Data Exploration**  
   Understand the data distributions, correlations, outliers.

3. **Data Preprocessing**  
   Clean the data, handle missing values, encode categorical variables, normalize/standardize features.

4. **Model Training**  
   Train ML models like Linear Regression, Decision Tree, Random Forest, etc.

5. **Model Evaluation**  
   Evaluate performance using R² Score, RMSE, MAE.

6. **Prediction**  
   Make predictions on new house data.

---

## 🧠 Requirements

- Python3.6+ 🐍
- Pandas
- NumPy
- Matplotlib 
- Scikit-Learn

---

## 🚀 How to Use

1. Clone the repo and install required libraries (`scikit-learn`, `numpy`, `pandas`, etc.)
2. Run the training script to generate models and performance report.
3. Use the file Model-Usage.py
---

## 📊 Sample Results

| CRIM   | ZN  | INDUS | CHAS | NOX  | RM   | AGE  | DIS  | RAD | TAX | PTRATIO | B      | LSTAT | MEDV |
|--------|-----|-------|------|------|------|------|------|-----|-----|----------|--------|--------|------|
| 0.0063 | 18  | 2.31  | 0    | 0.538| 6.58 | 65.2 | 4.09 | 1   | 296 | 15.3     | 396.9  | 4.98   | 24.0 |
| 0.0273 | 0   | 7.07  | 0    | 0.469| 6.42 | 78.9 | 4.97 | 2   | 242 | 17.8     | 396.9  | 9.14   | 21.6 |
| 0.0273 | 0   | 7.07  | 0    | 0.469| 7.18 | 61.1 | 4.97 | 2   | 242 | 17.8     | 392.8  | 4.03   | 34.7 |
| 0.0324 | 0   | 2.18  | 0    | 0.458| 6.0  | 45.8 | 6.06 | 3   | 222 | 18.7     | 394.6  | 5.21   | 33.4 |
| 0.0690 | 0   | 2.18  | 0    | 0.458| 6.43 | 58.7 | 6.06 | 3   | 222 | 18.7     | 396.9  | 5.91   | 36.2 |


---

## 📈 Sample Output
See regression_results.txt for cross-validation scores of all models and the selected best model.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---
