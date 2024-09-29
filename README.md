# Employee Attrition Prediction

This project aims to predict employee attrition using IBM's HR Analytics dataset. Employee turnover is a critical issue for organizations, leading to increased costs and reduced productivity. By predicting which employees are at risk of leaving, businesses can take preemptive action to retain talent and improve overall workforce management.

## Objectives

1. **Predict Employee Attrition**: Build a machine learning model to classify whether an employee will leave the company (Attrition = Yes or No).
2. **Identify Key Features**: Analyze the most important features that contribute to an employee's decision to leave.

## Dataset

- **Source**: [IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)
- **Description**: The dataset contains 35 columns, including 34 independent variables (features) and 1 target variable (Attrition). It consists of both categorical and numerical features that describe various employee attributes, such as:
    - **Age**
    - **BusinessTravel**
    - **Department**
    - **DistanceFromHome**
    - **MonthlyIncome**
    - **OverTime**
    - **YearsAtCompany**
  
The dataset is clean with no significant missing values.

## Project Structure

```bash
├── data/              # Contains the dataset and any processed data
├── notebooks/         # Jupyter notebooks for EDA, preprocessing, and model training
├── models/            # Saved models
├── src/               # Source code for training and evaluation scripts
├── README.md          # Project README file
├── requirements.txt   # Python dependencies
└── results/           # Results, graphs, and evaluation metrics
