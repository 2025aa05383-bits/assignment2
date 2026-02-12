import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib
import os

# Create model folder if not exists
os.makedirs('model', exist_ok=True)

columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                   header=None, names=columns, na_values=' ?')

# Clean missing values
data = data.dropna()

# Target: Binary (1 if >50K, 0 otherwise)
y = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
X = data.drop('income', axis=1)

# Identify numerical and categorical features
numerical = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models with pipelines
models = {
    'Logistic Regression': Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000))]),
    'Decision Tree': Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(random_state=42))]),
    'KNN': Pipeline([('prep', preprocessor), ('clf', KNeighborsClassifier())]),
    'Naive Bayes': Pipeline([('prep', preprocessor), ('clf', GaussianNB())]),
    'Random Forest': Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(random_state=42))]),
    'XGBoost': Pipeline([('prep', preprocessor), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))])
}

# Train, evaluate, save
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # All models have predict_proba

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results[name] = {'Accuracy': acc, 'AUC': auc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'MCC': mcc}

    # Save model
    joblib.dump(model, f'model/{name.replace(" ", "_")}.pkl')

# Print results for README table (copy these after running)
print("Results:")
for name, metrics in results.items():
    print(f"{name}: {metrics}")

# Save test data for demo upload in app (includes true 'income' for metrics)
test_data = X_test.copy()
test_data['income'] = y_test
test_data.to_csv('test_data.csv', index=False)
