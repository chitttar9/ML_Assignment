# ML Assignment 2: Heart Disease Prediction

## a. Problem Statement
This assignment requires the implementation, evaluation, and deployment of multiple classification models to predict heart disease using a chosen dataset. The goal is to build an interactive Streamlit web application that demonstrates these models and their performance metrics.

## b. Dataset Description
The dataset used is `heart_disease_uci.csv`, sourced from the UCI Machine Learning Repository. It contains various health parameters that can be used to predict the presence of heart disease. The original `num` column, indicating the presence of heart disease (0 for no disease, 1-4 for different stages), has been converted into a binary target variable (0 for no disease, 1 for presence of disease). The dataset has been preprocessed to handle missing values, encode categorical features, and scale numerical features.

**Key characteristics:**
*   **Original Features**: 16 (excluding 'id' and 'dataset')
*   **Instances**: 920
*   **Target Variable**: 'num' (converted to binary 'target')
*   **Dropped columns due to high missing values**: 'ca', 'thal'

## c. Models Used and Comparison
Six classification models were implemented and evaluated on the preprocessed dataset:
1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbor Classifier
4.  Naive Bayes Classifier (GaussianNB)
5.  Random Forest (Ensemble Model)
6.  XGBoost (Ensemble Model)

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | N/A | 0.8779 | 0.8500 | 0.7798 | 0.8137 | 0.5720 |
| Decision Tree Classifier | N/A | 0.7678 | 0.8190 | 0.7890 | 0.8037 | 0.5318 |
| K-Nearest Neighbor (KNN) | 0.8261 | 0.8522 | 0.8532 | 0.8532 | 0.8532 | 0.6399 |
| Naive Bayes (GaussianNB) | 0.7663 | 0.8396 | 0.8511 | 0.7339 | 0.7883 | 0.5380 |
| Random Forest | 0.8207 | 0.8864 | 0.8725 | 0.816 (approx) | 0.8436 | 0.6359 |
| XGBoost | 0.8261 | 0.8850 | 0.8889 | N/A | N/A | 0.6512 |


```

