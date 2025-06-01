
# Credit Card Fraud Detection with Machine Learning

**Group Members:** Immaculate, Joan, Bertha, James, John, Evelyne

## Overview

Credit card fraud is a persistent issue in the financial industry, costing billions globally. With the increase in digital transactions, detecting fraudulent activity in near real-time has become a priority. This project uses machine learning techniques to analyze transaction data and identify potentially fraudulent behavior.

## Objectives

- Analyze and preprocess credit card transaction data.
- Identify patterns and features that indicate fraudulent transactions.
- Develop and compare multiple machine learning models for fraud classification.
- Address class imbalance with techniques like SMOTE and cost-sensitive learning.
- Evaluate model performance using precision, recall, F1-score, and AUC-ROC.
- Ensure the final model is interpretable and efficient for real-time deployment.

## Research Questions

1. What distinguishing patterns exist between fraudulent and legitimate transactions?
2. Which machine learning models are most effective for detecting fraud in imbalanced datasets?
3. How can class imbalance be effectively addressed?
4. How can a balance between precision and recall be achieved to minimize both false positives and false negatives?
5. Is it feasible to deploy the fraud detection model in near real-time?
6. Can the model provide interpretable outputs for compliance and audit purposes?

## Success Criteria

- **Recall (Fraud Class)** ≥ 90%
- **Precision (Fraud Class)** ≥ 70%
- **AUC-ROC Score** ≥ 0.90
- Balanced learning via SMOTE or undersampling.
- Stable model performance across validation folds.
- Potential to reduce financial losses and customer disruptions.

## Dataset

The dataset used consists of **442,171** transaction records with 24 columns, including transaction amounts, locations, merchant details, and fraud labels.

## Exploratory Data Analysis

- Univariate, Bivariate, and Multivariate Analysis.
- Visualizations: Boxplots, Violin Plots, Histograms, Scatterplots, Heatmaps, Pairplots.
- Discovered significant class imbalance: **fraudulent transactions ≈ 0.6%**.

##  Data Preprocessing

- Handled missing values and outliers.
- Feature engineering: time-based features, transaction frequency, z-scores per user, geographic distance.
- Label encoding and one-hot encoding.
- Feature selection via correlation analysis.
- Feature scaling using `StandardScaler` and `MinMaxScaler`.
- Dimensionality reduction with PCA for visualization.

##  Modeling Approach

Models evaluated:

- **Baseline**: Logistic Regression
- **Traditional**: Decision Tree, Random Forest
- **Advanced**: Neural Network (Keras), XGBoost Classifier

Class imbalance was addressed using **SMOTE** and **cost-sensitive learning** via class weights.

##  Model Performance Summary

| Model                | Precision (Fraud) | Recall (Fraud) | AUC-ROC |
|:---------------------|:----------------|:--------------|:--------|
| Logistic Regression  | 0.04             | 0.75          | 0.8691  |
| Decision Tree        | 0.54             | 0.54          | 0.7701  |
| Random Forest        | 0.67             | 0.48          | 0.9232  |
| Neural Network       | 0.48             | 0.60          | 0.8844  |
| XGBoost Classifier   | 0.63             | 0.68          | 0.9306  |

## Key Insights

- Large transaction amounts alone aren’t reliable fraud indicators.
- Fraudulent transactions tend to cluster during late-night hours and weekends.
- SMOTE combined with cost-sensitive models improved fraud recall substantially.


##  Conclusion

The project successfully established a framework for detecting credit card fraud using machine learning, addressing the key objective of analyzing transaction data to identify potential fraud. Two machine learning models — **Random Forest** and **XGBoost** — were developed and evaluated, with techniques like **SMOTE** used to handle class imbalance.

While both models demonstrated strong discriminatory power with **AUC-ROC scores above 0.93** (XGBoost at **0.9306** and Random Forest at **0.9304**), they did not fully meet the ambitious technical success criteria, specifically:

- **Recall (Fraud Class) ≥ 90%**: Neither model achieved this (XGBoost: **0.68**, Random Forest: **0.65**).
- **Precision (Fraud Class) ≥ 70%**: Neither model achieved this (XGBoost: **0.63**, Random Forest: **0.69**).

The **XGBoost Classifier** emerged as the best-performing model among the two evaluated, exhibiting a slightly higher AUC-ROC score and better recall for the fraud class, indicating its superior ability to identify true fraud instances.

In summary, the project achieved its core objective of building and evaluating machine learning models for fraud detection and showcased promising initial results with high AUC-ROC scores. However, further work is required to enhance the models' ability to detect a higher percentage of fraud (recall) while maintaining a low false positive rate (precision) to meet the defined business and technical success criteria.

---

##  Recommendations

To improve the credit card fraud detection system and better meet the project objectives, the following recommendations are proposed:

###  Feature Engineering Enhancement

- **Temporal Features**: Create transaction frequency features over time windows (e.g., last 1 hour, 24 hours, 7 days) per cardholder and merchant.
- **Behavioral Patterns**: Capture deviations from a cardholder's typical behavior (unusual amounts, locations, or merchants).
- **Aggregated Features**: Aggregate transaction amounts, counts, and unique merchants within timeframes.

###  Advanced Imbalance Handling Techniques

- Explore oversampling techniques like **ADASYN** and **Borderline-SMOTE**, or undersampling methods like **NearMiss**.
- Investigate **cost-sensitive learning** where different misclassification penalties are applied to false positives and false negatives.

###  Model Optimization and Hyperparameter Tuning

- Perform exhaustive hyperparameter tuning for XGBoost (e.g., GridSearchCV or RandomizedSearchCV).
- Explore advanced boosting and ensemble methods for potential performance gains.

###  Threshold Adjustment

- Analyze the **precision-recall curve** and **ROC curve** for XGBoost.
- Adjust classification thresholds to optimize for higher recall or desired business trade-offs.

###  Explore Deep Learning Models

- Investigate neural network architectures (MLPs, RNNs for sequential data, Autoencoders for anomaly detection) to learn complex fraud patterns.

### Real-time Implementation Considerations

- Evaluate computational efficiency and latency for potential real-time deployment.
- Consider streaming data frameworks if real-time fraud detection is a business priority.

---

##  Final Model Performance Summary

| Model                | Precision (Fraud) | Recall (Fraud) | AUC-ROC |
|:---------------------|:----------------|:--------------|:--------|
| Random Forest        | 0.69             | 0.65          | 0.9304  |
| XGBoost Classifier   | 0.63             | 0.68          | 0.9306  |

---

##  Performance Analysis and Best Model

Considering the technical success criteria:

- **Recall (Fraud Class) ≥ 90%**: Not achieved by either model.
- **Precision (Fraud Class) ≥ 70%**: Not achieved by either model.
- **AUC-ROC ≥ 0.90**: Successfully met by both models.

The **XGBoost Classifier** slightly outperformed the Random Forest Classifier with a marginally higher AUC-ROC score (**0.9306** vs **0.9304**) and a higher recall (**0.68** vs **0.65**). 

Given that minimizing missed fraudulent transactions (false negatives) is paramount in fraud detection, the **XGBoost Classifier** is the preferred model for this project.

---
