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
1.	Logistic Regression
2.	Decision Tree Classifier
3.	K-Nearest Neighbor Classifier
4.	Naive Bayes Classifier (GaussianNB)
5.	Random Forest (Ensemble Model)
6.	XGBoost (Ensemble Model)

### Comparison Table
|                     |   Accuracy |   AUC Score |   Precision |   Recall |   F1 Score |   MCC Score |
|:--------------------|-----------:|------------:|------------:|---------:|-----------:|------------:|
| Logistic Regression |     0.788  |      0.8779 |      0.85   |   0.7798 |     0.8134 |      0.572  |
| Decision Tree       |     0.7717 |      0.7678 |      0.819  |   0.789  |     0.8037 |      0.5318 |
| K-Nearest Neighbor  |     0.8261 |      0.8522 |      0.8532 |   0.8532 |     0.8532 |      0.6399 |
| Naive Bayes         |     0.7663 |      0.8396 |      0.8511 |   0.7339 |     0.7882 |      0.538  |
| Random Forest       |     0.8207 |      0.8864 |      0.8725 |   0.8165 |     0.8436 |      0.6359 |
| XGBoost             |     0.8261 |      0.885  |      0.8889 |   0.8073 |     0.8462 |      0.6512 |

### Model Performance Observations

| Model                | Key Strengths                                      | Weaknesses / Notes                                 |
|:---------------------|:---------------------------------------------------|:---------------------------------------------------|
| Logistic Regression  | Good AUC, balanced precision & recall, solid MCC   | Outperformed by ensemble models                    |
| Decision Tree        | Acceptable precision & recall                      | Lowest AUC, prone to overfitting, lower MCC        |
| K-Nearest Neighbor   | Highest accuracy & recall, strong F1 & MCC         | Slightly lower AUC than ensemble models            |
| Naive Bayes          | High precision, respectable AUC                    | Lower recall & MCC, misses positives               |
| Random Forest        | High AUC, F1, precision, balanced metrics, strong MCC | Robust, generalizes well, top performer           |
| XGBoost              | Highest MCC & precision, top accuracy, strong AUC  | Slightly lower AUC than Random Forest, top performer|

**Summary:**  
Ensemble models (Random Forest, XGBoost) consistently deliver the best overall performance. KNN is also strong, especially in recall. Simpler models (Decision Tree, Naive Bayes) lag behind, though Naive Bayes excels in precision. Logistic Regression is a solid baseline but not as strong as ensemble methods.

