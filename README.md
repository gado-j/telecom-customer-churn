# Telecom Customer Churn Prediction

## Project Overview

This machine learning project predicts whether a telecommunications customer is likely to churn (leave the company). The project demonstrates data exploration, data cleaning, preprocessing, model development, cross-validation, performance evaluation, and model interpretation.

The final model was selected based on its ability to identify customers at risk of churning while maintaining useful overall predictive performance.

## Dataset

The project uses the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

- Original size: 7,043 customers and 21 columns
- Cleaned size: 7,032 customers
- Target variable: `Churn`
- Observed churn rate: 26.6%

The variables describe customer demographics, services, contract types, payment methods, tenure, and charges.

## Project Workflow

1. Defined the churn-prediction problem.
2. Loaded and inspected the dataset.
3. Converted `TotalCharges` from text to numeric values.
4. Removed 11 records with missing `TotalCharges`.
5. Explored churn patterns using summary statistics and visualizations.
6. Removed the customer identifier and encoded the target variable.
7. Created stratified training and test sets.
8. Standardized numerical variables and one-hot encoded categorical variables.
9. Established a Dummy Classifier baseline.
10. Compared Logistic Regression and Decision Tree models using five-fold cross-validation.
11. Evaluated the selected model on the untouched test set.
12. Used permutation importance to interpret the model.

## Model Performance

Logistic Regression was selected because it achieved the strongest cross-validation ROC-AUC while remaining interpretable.

| Metric | Test result |
|---|---:|
| Accuracy | 0.726 |
| Precision | 0.490 |
| Recall | 0.797 |
| F1 score | 0.607 |
| ROC-AUC | 0.835 |

The test-set confusion matrix contained:

- 298 churners correctly identified
- 76 churners missed
- 310 customers incorrectly flagged as churners
- 723 customers correctly predicted to stay

The model prioritized recall, meaning it identified most customers who actually churned, although this resulted in a larger number of false alarms.

## Key Findings

- Customers with shorter tenure generally had higher churn rates.
- Month-to-month contracts were associated with substantially higher churn.
- Higher churn was also observed among customers using fiber-optic internet, electronic-check payments, and no technical support.
- Permutation importance identified `tenure`, `InternetService`, and `Contract` as the most predictively useful variables.

These findings show associations and predictive usefulness, not causal relationships.

## Tools and Libraries

- Google Colab
- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- joblib

## How to Run the Project

1. Download the dataset from Kaggle.
2. Open `telco_customer_churn.ipynb` in Google Colab.
3. Upload the dataset when prompted or update the file path in the loading cell.
4. Select **Runtime > Run all**.

The notebook contains the complete analysis, code, visualizations, model training, and evaluation.

## Limitations

- The dataset represents a fictional telecommunications company.
- The analysis does not establish that any variable causes churn.
- The default classification threshold may not reflect real business costs.
- The model has not been validated using data from another company or time period.
- Fairness should be evaluated before using customer demographic information in practice.

## Conclusion

This project demonstrates an end-to-end supervised machine learning workflow. The final Logistic Regression model provided meaningful predictive value over the baseline and identified approximately 80% of customers who churned. In a practical setting, its predictions could help prioritize customers for retention review, but they should support human decisions rather than operate automatically.
