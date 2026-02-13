# ML Assignment 2 (Machine Learning) - Credit Card Fraud Detection

## a. Problem statement
Implement and compare six classification models to detect fraudulent credit card transactions (Class=1) versus legitimate transactions (Class=0). Deploy a Streamlit web app where a user can upload unseen test data (CSV), select a model, and view evaluation metrics and a confusion matrix / classification report.

## b. Dataset description
**Dataset:** Credit Card Fraud Detection (Kaggle)  
**Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
**Instances:** 284,807 transactions  
**Features:** 30 features: `Time`, `Amount`, and `V1` to `V28` (PCA-transformed/anonymized)  
**Target:** `Class` where 0 = legitimate and 1 = fraud  
**Note:** This dataset is highly imbalanced, so AUC, F1, and MCC are important along with accuracy.

## c. Models used

### Metrics comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9755 | 0.9721 | 0.0610 | 0.9184 | 0.1144 | 0.2332 |
| Decision Tree | 0.9907 | 0.8915 | 0.1321 | 0.7857 | 0.2261 | 0.3199 |
| kNN | 0.9995 | 0.9437 | 0.9186 | 0.8061 | 0.8587 | 0.8603 |
| Naive Bayes | 0.9764 | 0.9632 | 0.0588 | 0.8469 | 0.1099 | 0.2195 |
| Random Forest (Ensemble) | 0.9994 | 0.9717 | 0.8404 | 0.8061 | 0.8229 | 0.8228 |
| XGBoost (Ensemble) | 0.9995 | 0.9808 | 0.8632 | 0.8367 | 0.8497 | 0.8496 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | High recall but very low precision, meaning it catches many frauds but raises many false alarms; overall F1 and MCC are low. |
| Decision Tree | Better precision than linear models, but lower AUC than ensembles; prone to overfitting, performance depends on depth constraint. |
| kNN | Very strong precision/F1/MCC here; kNN can work well on scaled PCA features, but is memory-heavy and may be slow on large datasets. |
| Naive Bayes | Fast baseline with high recall but low precision; independence assumptions limit precision on fraud class. |
| Random Forest (Ensemble) | Strong overall balance; high precision and solid recall, giving good F1 and MCC. |
| XGBoost (Ensemble) | Best overall among the ensemble methods here (highest AUC and strong F1/MCC), good balance of precision and recall. |

## Repository structure
