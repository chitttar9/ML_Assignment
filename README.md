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
Based on the evaluation metrics presented in the comparison table, here are some observations regarding the performance of each classification model on the heart disease dataset:

*   **Logistic Regression**: This model shows a good balance of performance, with a relatively high AUC score (0.8779) indicating its ability to distinguish between classes. Its precision (0.8500) and recall (0.7798) are solid, suggesting a reasonable trade-off between false positives and false negatives. The MCC score of 0.5720 indicates a decent correlation between actual and predicted values.

*   **Decision Tree**: The Decision Tree model exhibits slightly lower performance across most metrics compared to Logistic Regression and ensemble methods. Its AUC score (0.7678) is the lowest among all models, suggesting less effective class separation. While its precision (0.8190) and recall (0.7890) are acceptable, the lower F1 and MCC scores (0.8037 and 0.5318, respectively) indicate that it might be prone to overfitting or has less generalization capability on this dataset without further tuning.

*   **K-Nearest Neighbor (KNN)**: KNN stands out with the highest accuracy (0.8261) and a very high recall (0.8532) and F1 Score (0.8532), indicating it is quite effective at identifying positive cases. Its AUC score (0.8522) is also strong. The MCC score (0.6399) is competitive, suggesting that the local structure of the data is well-captured by this algorithm.

*   **Naive Bayes**: This model generally performs on the lower side in terms of overall accuracy (0.7663) and recall (0.7339). However, it achieves a high precision (0.8511), implying that when it predicts a positive case, it's often correct, but it misses a significant number of actual positive cases. Its AUC score (0.8396) is respectable, showing its capability in ranking positive instances correctly. The MCC score (0.5380) is among the lowest, indicating moderate agreement with the true classification.

*   **Random Forest**: As an ensemble method, Random Forest demonstrates strong performance with a high AUC score (0.8864), which is among the best, and a robust F1 score (0.8436). Its accuracy (0.8207) and precision (0.8725) are also very good. The MCC score (0.6359) is high, reflecting its overall strong and balanced predictive power across various metrics, suggesting good generalization.

*   **XGBoost**: XGBoost, another powerful ensemble method, performs exceptionally well, achieving the highest MCC score (0.6512) and accuracy (0.8261) tied with KNN. It also boasts the highest precision (0.8889) and a very competitive AUC score (0.8850), slightly trailing Random Forest. This indicates that XGBoost is highly effective in minimizing false positives and providing accurate classifications, making it one of the top performers on this dataset.

**Summary of Trends and Differences:**

The ensemble models (Random Forest and XGBoost) consistently deliver the best overall performance, particularly in terms of AUC score, F1 score, and MCC score, indicating their robustness and ability to handle complex relationships in the data. KNN also shows strong performance, especially in recall, suggesting its effectiveness with the given dataset's structure. Simpler models like Decision Tree and Naive Bayes show comparatively lower overall performance, although Naive Bayes achieved high precision. Logistic Regression offers a good baseline, outperforming the simpler models in most metrics, but falls short of the ensemble methods.

