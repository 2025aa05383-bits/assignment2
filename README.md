# ML Assignment 2: Classification Models and Streamlit Deployment

## a. Problem Statement
This project aims to predict whether an individual's annual income exceeds $50,000 based on demographic and employment data from the 1994 US Census. It involves implementing six machine learning classification models, evaluating their performance using various metrics, and deploying an interactive Streamlit web application to demonstrate the models on user-uploaded test data.

## b. Dataset Description
The Adult Income dataset from the UCI Machine Learning Repository contains 48,842 instances with 14 features (6 numerical and 8 categorical). Features include age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, and native-country. The target variable is binary: income >50K (1) or <=50K (0). Data preprocessing involved removing rows with missing values (marked as '?'), resulting in approximately 45,222 clean instances. The dataset is imbalanced, with about 75% of instances in the <=50K class. Source: https://archive.ics.uci.edu/dataset/2/adult.

## c. Models Used
The following six classification models were implemented using scikit-learn and xgboost on the Adult Income dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

### Comparison Table of Evaluation Metrics

| ML Model Name       | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|---------------------|----------|-------|-----------|--------|-------|-------|
| Logistic Regression | 0.85    | 0.90  | 0.74     | 0.60   | 0.66  | 0.59  |
| Decision Tree       | 0.81    | 0.85  | 0.63     | 0.64   | 0.63  | 0.51  |
| KNN                 | 0.83    | 0.88  | 0.69     | 0.59   | 0.64  | 0.55  |
| Naive Bayes         | 0.81    | 0.88  | 0.59     | 0.75   | 0.66  | 0.53  |
| Random Forest (Ensemble) | 0.85 | 0.91  | 0.74     | 0.61   | 0.67  | 0.60  |
| XGBoost (Ensemble)  | 0.87    | 0.93  | 0.78     | 0.65   | 0.71  | 0.65  |

### Observations on Model Performance

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Performed well with balanced metrics, capturing linear relationships effectively in the demographic features, but slightly limited by class imbalance. |
| Decision Tree       | Decent accuracy but prone to overfitting on noisy features like fnlwgt, resulting in lower MCC and indicating room for pruning. |
| KNN                 | Solid overall performance, but lower recall suggests it misses some high-income cases, possibly due to sensitivity to feature scaling. |
| Naive Bayes         | High recall makes it good for identifying positive cases, but lower precision due to the independence assumption not fully holding with correlated features like education and education-num. |
| Random Forest (Ensemble) | Improved over single Decision Tree by reducing variance through bagging, showing robust performance on imbalanced data. |
| XGBoost (Ensemble)  | Best performer with highest AUC and MCC, benefiting from gradient boosting to handle imbalance and feature interactions optimally. |

## Installation and Setup
1. Clone the repository: `git clone https://github.com/2025aa05383-bits/assignment2`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training script (if needed): `python model/train_models.py` (models are already saved in the `model/` folder).

## Usage
- Upload a CSV file with test data (same columns as the dataset, including 'income' for true labels).
- Select a model from the dropdown to view predictions, metrics, confusion matrix, and classification report.

## Deployment
The app is deployed on Streamlit Community Cloud: [Live App Link](https://assignment2-q6zbawe7hrabklszinrusu.streamlit.app/)
