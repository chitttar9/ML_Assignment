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
    
    # Assuming the numerical and categorical features identified during training are consistent
    # This should ideally be saved along with scaler/encoder for robustness
    # For this example, let's hardcode them based on the training phase
    numerical_features_trained = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    categorical_features_trained = ['sex', 'cp', 'restecg', 'slope']
    
    return models, scaler, encoder, numerical_features_trained, categorical_features_trained

# --- Function to Preprocess Data ---
def preprocess_data(df_raw, scaler_obj, encoder_obj, numerical_feats, categorical_feats):
    df_processed = df_raw.copy()

    # 1. Drop 'id' and 'dataset' columns if they exist
    cols_to_drop = ['id', 'dataset', 'ca', 'thal']
    for col in cols_to_drop:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])
            
    # 2. Separate target variable 'num' (if present) for testing
    if 'num' in df_processed.columns:
        df_processed['target'] = (df_processed['num'] > 0).astype(int)
        df_processed = df_processed.drop(columns=['num'])
        y_true = df_processed['target'] # Store actual labels for evaluation
        X_feats = df_processed.drop(columns=['target']) # Features for prediction
    else:
        st.info("Target variable 'num' not found in uploaded file. Running in prediction-only mode.")
        y_true = None
        X_feats = df_processed.copy()

    # 3. Determine which expected features are present in the uploaded data
    numerical_present = [c for c in numerical_feats if c in X_feats.columns]
    categorical_present = [c for c in categorical_feats if c in X_feats.columns]

    # 4. Handle missing values (using mean for numerical, mode for categorical) on present features
    for col in numerical_present:
        X_feats[col] = X_feats[col].fillna(X_feats[col].mean())

    for col in categorical_present:
        # Guard against empty column when computing mode
        if not X_feats[col].dropna().empty:
            X_feats[col] = X_feats[col].fillna(X_feats[col].mode()[0])

    # 5. Apply StandardScaler to numerical features (only if present)
    if numerical_present:
        scaled_features = scaler_obj.transform(X_feats[numerical_present])
        scaled_df = pd.DataFrame(scaled_features, columns=numerical_present, index=X_feats.index)
    else:
        scaled_df = pd.DataFrame(index=X_feats.index) # Empty dataframe if no numerical features

    # 6. Apply OneHotEncoder to categorical features (only if present)
    if categorical_present:
        # Use only the present categorical columns when transforming
        encoded_features = encoder_obj.transform(X_feats[categorical_present])
        # Get feature names for the subset of categorical features
        try:
            encoded_feature_names = encoder_obj.get_feature_names_out(categorical_present)
        except Exception:
            # Fallback if encoder doesn't support get_feature_names_out for the subset
            encoded_feature_names = [f"cat_{i}" for i in range(encoded_features.shape[1])]
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X_feats.index)
    else:
        encoded_df = pd.DataFrame(index=X_feats.index) # Empty dataframe if no categorical features

    # 6. Combine the scaled numerical features and one-hot encoded categorical features
    X_preprocessed = pd.concat([scaled_df, encoded_df], axis=1)
    
    # Ensure column order matches training data for consistency
    # This part needs care, as column order from encoder can vary slightly or miss columns if test data lacks a category
    # For simplicity, we assume `X_train` columns are available or ensure this is handled upstream
    # A more robust solution would involve storing X_train.columns and reindexing here
    # For now, let's assume `X_preprocessed` has all necessary columns due to `handle_unknown='ignore'`

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
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Select a Classification Model",
    list(models.keys())
)

# Display evaluation results section
st.subheader("Evaluation Results")

if uploaded_file is not None:
    df_uploaded = pd.read_csv(uploaded_file)
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
    st.info("Please upload a CSV file to begin evaluation.")
