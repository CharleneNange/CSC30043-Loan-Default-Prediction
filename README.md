# CSC30043-Loan-Default-Prediction
This repository contains the Python implementation for a comparative study of machine learning models used to predict loan defaults. The project focuses on handling class imbalance and evaluating model performance through both traditional metrics and statistical significance testing.

Project Overview
Loan default prediction is a critical financial risk assessment tool that impacts risk management and operational efficiency. This project evaluates four distinct models to identify the most effective approach for uncovering patterns within lending datasets that may not be apparent through conventional analytical methods.

Key Features:
Data Preprocessing: Implementation of median and mode imputation, and the creation of missingness indicator columns for variables suspected to be Missing Not at Random (MNAR).

Feature Engineering: Use of One-Hot encoding for categorical variables and the Robust Scaler to handle right-skewed features and extreme values.

Imbalance Handling: Use of ADASYN (Adaptive Synthetic Sampling) to address the class imbalance by focusing on minority-class instances that are more difficult to learn.

Modeling: Comparison of Logistic Regression, Random Forest, XGBoost, and a Neural Network.

Ensemble Learning: Implementation of a Voting Classifier to capture the strongest characteristics of individual models.

Statistical Evaluation: Use of McNemar’s Test to assess whether the difference in error rates between models is statistically significant.

Results Summary
Top Performer: The Voting Classifier achieved the highest AUC score of 0.767 and a Precision-Recall AUC of 0.906.

Trade-offs: Logistic Regression achieved the highest proportion of True Positives (0.76) but also the highest False Negatives (0.44), while Random Forest recorded the highest False Positives (0.57).

Statistical Insight: McNemar’s test revealed that at a 0.05 significance level, the Voting Classifier's performance was statistically similar to the Neural Network, despite achieving different raw metrics.
