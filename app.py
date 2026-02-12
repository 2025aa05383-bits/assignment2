import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
model_paths = {
    'Logistic Regression': 'model/Logistic_Regression.pkl',
    'Decision Tree': 'model/Decision_Tree.pkl',
    'KNN': 'model/KNN.pkl',
    'Naive Bayes': 'model/Naive_Bayes.pkl',
    'Random Forest': 'model/Random_Forest.pkl',
    'XGBoost': 'model/XGBoost.pkl'
}
models = {name: joblib.load(path) for name, path in model_paths.items()}

st.title('Machine Learning Classification Demo')

# Dataset upload .
uploaded_file = st.file_uploader('Upload Test CSV (must include "income" column for true labels)', type='csv')

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    if 'income' not in test_df.columns:
        st.error('CSV must have "income" column with true labels (0/1).')
    else:
        y_true = test_df['income']
        X_test = test_df.drop('income', axis=1)
        
        # Model selection
        selected_model = st.selectbox('Select Model', list(models.keys()))
        model = models[selected_model]
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        st.subheader('Evaluation Metrics')
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
            'Value': [acc, auc, prec, rec, f1, mcc]
        })
        st.table(metrics_df)
        
        #Confusion Matrix as Heatmap
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix Heatmap')
        st.pyplot(fig)
        
        #Classification Report as DataFrame Table
        st.subheader('Classification Report')
        report_dict = classification_report(y_true, y_pred, output_dict=True, target_names=['<=50K', '>50K'])
        report_df = pd.DataFrame(report_dict).transpose().round(2)
        st.dataframe(report_df.style.background_gradient(cmap='viridis'))