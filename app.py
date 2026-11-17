import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

# Page configuration
st.set_page_config(
    page_title="ML MODELING  Studio with PCA & Neural Networks",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖  Machine Learning Studio (PCA + Neural Networks)")
st.markdown("Upload your dataset, apply PCA for dimensionality reduction, train a Neural Network or other models, and run predictions.")

# -------------------------------------------------------------------------
# 1. File Upload Section
# -------------------------------------------------------------------------
st.sidebar.header("1. Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())

    # -------------------------------------------------------------------------
    # 2. Feature & Target Selection
    # -------------------------------------------------------------------------
    st.sidebar.header("2. Model Configuration")
    
    target_column = st.sidebar.selectbox("Select Target Column", options=df.columns)
    default_features = [col for col in df.columns if col != target_column]
    feature_columns = st.sidebar.multiselect("Select Feature Columns", options=df.columns, default=default_features)
    
    if not feature_columns:
        st.warning("Please select at least one feature column.")
        st.stop()

    is_numeric_target = pd.api.types.is_numeric_dtype(df[target_column])
    unique_target_values = df[target_column].nunique()
    default_task = "Regression" if (is_numeric_target and unique_target_values > 10) else "Classification"
    
    task_type = st.sidebar.selectbox("Select Task Type", options=["Classification", "Regression"], index=0 if default_task=="Classification" else 1)

    # -------------------------------------------------------------------------
    # 3. Dimensionality Reduction (PCA) Configuration
    # -------------------------------------------------------------------------
    st.sidebar.header("3. Dimensionality Reduction (PCA)")
    use_pca = st.sidebar.checkbox("Apply PCA (Principal Component Analysis)", value=False)
    
    n_components = 2
    if use_pca:
        max_comp = len(feature_columns)
        n_components = st.sidebar.slider("Number of PCA Components", min_value=1, max_value=max_comp, value=min(2, max_comp))

    # Train/Test Split parameters
    test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
    random_state = st.sidebar.number_input("Random Seed", value=42, step=1)

    # -------------------------------------------------------------------------
    # 4. Algorithm Selection (Including Neural Network)
    # -------------------------------------------------------------------------
    st.sidebar.header("4. Algorithm Selection")
    if task_type == "Classification":
        model_name = st.sidebar.selectbox("Choose Classifier", [
            "Neural Network (MLP)", 
            "Random Forest", 
            "Logistic Regression", 
            "Support Vector Machine (SVM)"
        ])
    else:
        model_name = st.sidebar.selectbox("Choose Regressor", [
            "Neural Network (MLP)", 
            "Random Forest", 
            "Linear Regression", 
            "Support Vector Machine (SVM)"
        ])

    # -------------------------------------------------------------------------
    # 5. Training & Testing
    # -------------------------------------------------------------------------
    if st.sidebar.button("🚀 Train Model", type="primary"):
        with st.spinner("Processing data, applying PCA, and training model..."):
            model_df = df[feature_columns + [target_column]].dropna()
            
            X = model_df[feature_columns]
            y = model_df[target_column]
            
            # Encode categorical features if any
            X = pd.get_dummies(X, drop_first=True)
            
            # Split Data first
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
            
            # Standardize features (Crucial for PCA and Neural Networks)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply PCA if enabled
            if use_pca:
                pca = PCA(n_components=n_components, random_state=int(random_state))
                X_train_final = pca.fit_transform(X_train_scaled)
                X_test_final = pca.transform(X_test_scaled)
                st.session_state['pca_transformer'] = pca
            else:
                X_train_final = X_train_scaled
                X_test_final = X_test_scaled
                st.session_state.pop('pca_transformer', None)
                
            st.session_state['scaler'] = scaler
            st.session_state['use_pca'] = use_pca

            # Initialize Model
            if task_type == "Classification":
                if model_name == "Neural Network (MLP)":
                    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=int(random_state))
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(random_state=int(random_state))
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                else:
                    model = SVC()
            else:
                if model_name == "Neural Network (MLP)":
                    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=int(random_state))
                elif model_name == "Random Forest":
                    model = RandomForestRegressor(random_state=int(random_state))
                elif model_name == "Linear Regression":
                    model = LinearRegression()
                else:
                    model = SVR()
            
            # Train model
            model.fit(X_train_final, y_train)
            y_pred = model.predict(X_test_final)
            
            # Store session artifacts
            st.session_state['trained_model'] = model
            st.session_state['feature_columns'] = list(X.columns)
            st.session_state['original_features'] = feature_columns
            st.session_state['task_type'] = task_type
            
            st.success("Model trained successfully with PCA & Neural Network pipeline!")

            # -----------------------------------------------------------------
            # 6. Evaluation Metrics & Visuals
            # -----------------------------------------------------------------
            st.header("📈 Model Evaluation Results")
            
            if task_type == "Classification":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{acc:.4f}")
                m2.metric("Precision", f"{prec:.4f}")
                m3.metric("Recall", f"{rec:.4f}")
                m4.metric("F1 Score", f"{f1:.4f}")
                
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, title="Confusion Matrix",
                                labels=dict(x="Predicted", y="Actual", color="Count"))
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", f"{r2:.4f}")
                m2.metric("RMSE", f"{rmse:.4f}")
                m3.metric("MSE", f"{mse:.4f}")
                m4.metric("MAE", f"{mae:.4f}")
                
                eval_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                fig = px.scatter(eval_df, x="Actual", y="Predicted", opacity=0.7, 
                                 title="Actual vs Predicted Values", trendline="ols")
                st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # 7. Live Prediction Section
    # -------------------------------------------------------------------------
    if 'trained_model' in st.session_state:
        st.markdown("---")
        st.header("🔮 Make Predictions on New Data")
        st.markdown("Provide feature values below; the app will automatically scale and apply PCA before feeding data into the Neural Network.")
        
        orig_features = st.session_state['original_features']
        
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(orig_features):
            col_idx = i % 3
            with cols[col_idx]:
                if pd.api.types.is_numeric_dtype(df[feat]):
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].mean())
                    input_data[feat] = st.number_input(f"{feat}", value=mean_val, min_val=min_val, max_val=max_val)
                else:
                    options = df[feat].dropna().unique().tolist()
                    input_data[feat] = st.selectbox(f"{feat}", options=options)
                    
        if st.button("Generate Prediction", type="primary"):
            input_df = pd.DataFrame([input_data])
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=st.session_state['feature_columns'], fill_value=0)
            
            # Apply preprocessing pipeline (Scaler -> PCA -> Predict)
            input_scaled = st.session_state['scaler'].transform(input_encoded)
            if st.session_state.get('use_pca', False):
                input_final = st.session_state['pca_transformer'].transform(input_scaled)
            else:
                input_final = input_scaled
                
            prediction = st.session_state['trained_model'].predict(input_final)
            
            st.success(f"### Predicted {target_column}: **{prediction[0]}**")

else:
    st.info("👈 Please upload a CSV or Excel dataset using the sidebar to begin.")