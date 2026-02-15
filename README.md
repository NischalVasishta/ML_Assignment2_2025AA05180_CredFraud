# ML Assignment 2 (Machine Learning) - Credit Card Fraud Detection

## a. Problem statement
Implement and compare six classification models to detect fraudulent credit card transactions (Class = 1) versus legitimate transactions (Class = 0). Deploy a Streamlit web app where a user can upload unseen test data (CSV), select a model, and view evaluation metrics and a confusion matrix / classification report.

## b. Dataset description
**Dataset:** Credit Card Fraud Detection (Kaggle)  
**Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
**Instances:** 284,807 transactions  
**Features:** 30 input features: `Time`, `Amount`, and `V1` to `V28` (PCA-transformed/anonymized)  
**Target:** `Class` where 0 = legitimate and 1 = fraud  
**Notes:** The dataset is highly imbalanced, so metrics like AUC, F1, and MCC are important along with accuracy.

## c. Models used
All models are trained on the same dataset and evaluated using the required metrics.

Models implemented:
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### d. Metrics comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9755 | 0.9721 | 0.0610 | 0.9184 | 0.1144 | 0.2332 |
| Decision Tree | 0.9907 | 0.8915 | 0.1321 | 0.7857 | 0.2261 | 0.3199 |
| kNN | 0.9995 | 0.9437 | 0.9186 | 0.8061 | 0.8587 | 0.8603 |
| Naive Bayes | 0.9764 | 0.9632 | 0.0588 | 0.8469 | 0.1099 | 0.2195 |
| Random Forest (Ensemble) | 0.9994 | 0.9717 | 0.8404 | 0.8061 | 0.8229 | 0.8228 |
| XGBoost (Ensemble) | 0.9995 | 0.9808 | 0.8632 | 0.8367 | 0.8497 | 0.8496 |

### e. Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very high recall but low precision, meaning it catches many frauds but produces many false positives; F1 and MCC remain low. |
| Decision Tree | Better precision than linear baselines but lower AUC than ensembles; can overfit, and performance depends on constraints like max_depth. |
| kNN | Strong precision/F1/MCC on this setup; works well with scaled PCA features, but is memory-heavy and slower on large datasets. |
| Naive Bayes | Fast baseline with good recall but low precision; independence assumptions limit fraud-class precision. |
| Random Forest (Ensemble) | Strong balance between precision and recall; high F1 and MCC, robust performance overall. |
| XGBoost (Ensemble) | Best overall balance here (highest AUC and strong F1/MCC); handles imbalance well and captures complex patterns. |


## f. Conclusion
In this assignment, I implemented an end-to-end machine learning workflow for credit card fraud detection using the Credit Card Fraud dataset (284,807 transactions, 30 features; target Class). Since the dataset is highly imbalanced, I evaluated all six required models using Accuracy, AUC, Precision, Recall, F1-score, and MCC to ensure the comparison reflects real fraud-detection performance.

The results show clear trade-offs: Logistic Regression achieved very high Recall (0.9184) but extremely low Precision (0.0610), meaning it catches most frauds but raises many false alarms (F1 = 0.1144, MCC = 0.2332). The strongest overall performers were the ensemble methods—Random Forest (Precision 0.8404, Recall 0.8061, F1 0.8229, MCC 0.8228) and XGBoost (best AUC 0.9808 with Precision 0.8632, Recall 0.8367, F1 0.8497, MCC 0.8496)—showing a better balance between detecting fraud and limiting false positives.

To complete the deployment requirement, the models and scaler were trained offline, saved as .pkl, and integrated into a Streamlit app that loads the pre-trained models, supports test CSV upload (with a sample option), provides model selection, and displays the required metrics along with a confusion matrix and classification report. The final solution is deployed on Streamlit Community Cloud and fully reproducible via the GitHub repository.

## g. Repository structure
```text
project/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── model_results.csv
├── test_data.csv
└── model/
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
