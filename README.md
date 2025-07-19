# Employee Salary Prediction using Machine Learning

This project aims to predict employee salaries based on various features such as experience, education level, job role, and company type using machine learning models. It helps HR professionals make data-driven decisions on fair compensation.

## Problem Statement

Estimating salaries manually can be inaccurate and inefficient due to multiple influencing factors. A predictive model can automate and standardize this process, ensuring better transparency and decision-making.

## Tech Stack & Tools

- Python 3.10+
- Jupyter Notebook / Google Colab
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- XGBoost

## Features

- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature encoding  
- Model training & comparison (Linear Regression, Decision Tree, XGBoost)  
- Evaluation using R², MAE, RMSE  
- Model saving using `joblib`  

## Dataset

The dataset includes the following features:
- Experience  
- Education Level  
- Job Role  
- Company Type  
- Gender  
- Salary (Target variable)

## Model Training Process

1. **Load Dataset**  
2. **Clean and Prepare Data**  
3. **EDA & Visualization**  
4. **Label Encoding** of categorical features  
5. **Train-Test Split (80-20)**  
6. **Model Training** using Linear Regression, Decision Tree, XGBoost  
7. **Evaluation** based on performance metrics  
8. **Model Saving** for deployment

## Results

- **Best Model**: XGBoost Regressor  
- **Performance**: High R² Score, low MAE and RMSE  
- **Visualization**: Actual vs Predicted Salary plots for interpretation

## Sample Output

Screenshots of:
- Cleaned dataset
- Heatmaps
- Model score outputs
- Final prediction comparison

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
