import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# --- Function to Load Models and Preprocessors ---
def load_assets():
    models = {}
    model_dir = 'model'

    # Load models
    for filename in os.listdir(model_dir):
        if filename.endswith('_model.joblib'):
            model_name = filename.replace('_model.joblib', '').replace('_', ' ')
            filepath = os.path.join(model_dir, filename)
            models[model_name] = joblib.load(filepath)
    
    # Load StandardScaler and OneHotEncoder
    scaler_filepath = os.path.join(model_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_filepath)
    
    encoder_filepath = os.path.join(model_dir, 'encoder.joblib')
    encoder = joblib.load(encoder_filepath)
    
    # Feature lists must match exactly what was used during training
    # Based on the preprocessing output in the notebook:
    # Numerical: ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    # Categorical: ['sex', 'cp', 'restecg', 'slope']
    # Note: fbs and exang were imputed but not included in final feature set
    numerical_features_trained = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_features_trained = ['sex', 'cp', 'restecg', 'slope']
    
    return models, scaler, encoder, numerical_features_trained, categorical_features_trained

# --- Function to Preprocess Data ---
def preprocess_data(df_raw, scaler_obj, encoder_obj, numerical_feats, categorical_feats):
    df_processed = df_raw.copy()

    # 1. Drop columns that were not used in training
    cols_to_drop = ['id', 'dataset', 'ca', 'thal', 'fbs', 'exang']
    for col in cols_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])

    # 2. Separate target variable if present ('num' from raw dataset or already-created 'target')
    if 'num' in df_processed.columns:
        df_processed['target'] = (df_processed['num'] > 0).astype(int)
        df_processed = df_processed.drop(columns=['num'])
        y_true = df_processed['target']
        X_feats = df_processed.drop(columns=['target'])
    elif 'target' in df_processed.columns:
        y_true = df_processed['target']
        X_feats = df_processed.drop(columns=['target'])
        st.info("Found 'target' column in uploaded file — using it as ground truth.")
    else:
        st.info("Target variable 'num' or 'target' not found in uploaded file. Running in prediction-only mode.")
        y_true = None
        X_feats = df_processed.copy()

    # 3. Ensure all expected numerical and categorical features exist in the dataframe.
    #    If missing, create them with sensible defaults so that scaler/encoder can be applied.
    # Numerical: if completely missing, fill with scaler's training mean if available, else 0.0
    for i, col in enumerate(numerical_feats):
        if col not in X_feats.columns:
            fill_val = 0.0
            if hasattr(scaler_obj, 'mean_') and len(getattr(scaler_obj, 'mean_', [])) > i:
                try:
                    fill_val = float(scaler_obj.mean_[i])
                except Exception:
                    fill_val = 0.0
            X_feats[col] = fill_val
        else:
            # Convert to numeric and fill missing values with mean
            X_feats[col] = pd.to_numeric(X_feats[col], errors='coerce').fillna(X_feats[col].mean())

    # Categorical: if missing, create with a placeholder unseen category so encoder will handle it
    for col in categorical_feats:
        if col not in X_feats.columns:
            X_feats[col] = 'MISSING'
        else:
            # Ensure column is treated as string/object type
            X_feats[col] = X_feats[col].astype(str)
            # Fill missing values with mode
            if not X_feats[col].dropna().empty:
                X_feats[col] = X_feats[col].fillna(X_feats[col].mode()[0])
            else:
                X_feats[col] = 'MISSING'

    # 4. Apply StandardScaler to numerical features (use full trained order)
    try:
        scaled_features = scaler_obj.transform(X_feats[numerical_feats])
    except Exception:
        # As a fallback, attempt to coerce to numeric and replace errors with 0
        for col in numerical_feats:
            X_feats[col] = pd.to_numeric(X_feats[col], errors='coerce').fillna(0.0)
        scaled_features = scaler_obj.transform(X_feats[numerical_feats])
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_feats, index=X_feats.index)

    # 5. Apply OneHotEncoder to categorical features (use full trained order)
    #    OneHotEncoder expects the same number of input features as during fit.
    try:
        encoded_features = encoder_obj.transform(X_feats[categorical_feats])
    except Exception:
        # If transform fails, replace categorical columns with a placeholder and retry
        for col in categorical_feats:
            if col not in X_feats.columns:
                X_feats[col] = 'MISSING'
        encoded_features = encoder_obj.transform(X_feats[categorical_feats])

    # Build encoded feature names robustly
    try:
        encoded_feature_names = encoder_obj.get_feature_names_out(categorical_feats)
    except Exception:
        # Fallback using categories_ attribute
        encoded_feature_names = []
        if hasattr(encoder_obj, 'categories_'):
            for feat, cats in zip(categorical_feats, encoder_obj.categories_):
                for cat in cats:
                    encoded_feature_names.append(f"{feat}_{cat}")
    encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X_feats.index)

    # 6. Combine the scaled numerical features and one-hot encoded categorical features
    X_preprocessed = pd.concat([scaled_df, encoded_df], axis=1)

    # 7. Ensure column order matches training data: numerical_feats followed by encoder output columns
    expected_cols = list(numerical_feats) + list(encoded_feature_names)
    # Reindex to expected columns, adding missing columns filled with 0
    X_preprocessed = X_preprocessed.reindex(columns=expected_cols, fill_value=0)

    return X_preprocessed, y_true

# --- Main Streamlit Application ---
st.set_page_config(layout="wide")

st.title("Heart Disease Prediction App")

st.write("Upload a CSV file with patient data to get predictions and evaluate models.")

# Load assets once and cache them
models, scaler, encoder, numerical_features_trained, categorical_features_trained = load_assets()

st.success("Models and preprocessors loaded successfully!")

# Placeholder for file uploader and model selection
st.sidebar.header("Upload Test Data")

# Add download button for test data
if os.path.exists('test_set.csv'):
    with open('test_set.csv', 'rb') as f:
        st.sidebar.download_button(
            label="📥 Download Test Data",
            data=f,
            file_name='test_set.csv',
            mime='text/csv'
        )

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Select a Classification Model",
    list(models.keys())
)

# Display evaluation results section
st.subheader("Evaluation Results")

# Use test_set.csv by default if no file is uploaded
if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
    st.info("Using uploaded file for predictions.")
elif os.path.exists('test_set.csv'):
    df_uploaded = pd.read_csv('test_set.csv')
    st.info("Using default test_set.csv for predictions.")
else:
    df_uploaded = None

if df_uploaded is not None:
    st.write("Uploaded Data Preview:")
    st.dataframe(df_uploaded.head())

    # Preprocess the uploaded data
    X_test_processed, y_test_true = preprocess_data(df_uploaded, scaler, encoder, numerical_features_trained, categorical_features_trained)

    st.write("Preprocessed Features Shape:", X_test_processed.shape)
    if y_test_true is None:
        st.write("No target labels found in upload — prediction-only mode.")
    else:
        st.write("Target Labels Shape:", y_test_true.shape)

    # Store processed data in session state for reuse
    st.session_state['X_test_processed'] = X_test_processed
    st.session_state['y_test_true'] = y_test_true
    st.session_state['data_ready'] = True

    st.success("Data preprocessed successfully!")

    model = models[selected_model_name]
    y_pred = model.predict(st.session_state['X_test_processed'])

    # If no true labels, show predictions only and provide download
    if y_test_true is None:
        st.write(f"Predictions from {selected_model_name} (prediction-only mode):")
        df_out = df_uploaded.copy()
        df_out['prediction'] = y_pred
        st.dataframe(df_out.head())
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download predictions as CSV", data=csv, file_name='predictions.csv', mime='text/csv')
    else:
        # Calculate metrics
        accuracy = accuracy_score(st.session_state['y_test_true'], y_pred)
        precision = precision_score(st.session_state['y_test_true'], y_pred, zero_division=0)
        recall = recall_score(st.session_state['y_test_true'], y_pred, zero_division=0)
        f1 = f1_score(st.session_state['y_test_true'], y_pred, zero_division=0)
        mcc = matthews_corrcoef(st.session_state['y_test_true'], y_pred)

        auc_score = "N/A"
        try:
            y_pred_proba = model.predict_proba(st.session_state['X_test_processed'])[:, 1]
            auc_score = roc_auc_score(st.session_state['y_test_true'], y_pred_proba)
        except AttributeError:
            st.warning(f"Model {selected_model_name} does not support `predict_proba` for AUC calculation.")

        st.write("### Model Performance Metrics")
        metrics_data = {
            "Metric": ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"],
            "Value": [f"{accuracy:.4f}", f"{auc_score:.4f}" if auc_score != "N/A" else auc_score, 
                      f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", f"{mcc:.4f}"]
        }
        st.table(pd.DataFrame(metrics_data))

        st.write("### Classification Report")
        st.text(classification_report(st.session_state['y_test_true'], y_pred, zero_division=0))

        st.write("### Confusion Matrix")
        st.text(confusion_matrix(st.session_state['y_test_true'], y_pred))

else:
    st.warning("No test data available. Please upload a CSV file or ensure test_set.csv exists in the directory.")
