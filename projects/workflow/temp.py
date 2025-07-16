import streamlit as st
import pandas as pd
import openai
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import numpy as np
import shap
import matplotlib.pyplot as plt
import pickle

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="ML Assistant", layout="wide")
st.title("🤖 Smart ML Workflow Assistant")

# Session state init
if "step" not in st.session_state:
    st.session_state.step = 1
if "df" not in st.session_state:
    st.session_state.df = None
if "suggestions" not in st.session_state:
    st.session_state.suggestions = ""
if "transformed_df" not in st.session_state:
    st.session_state.transformed_df = None
if "selected_algos" not in st.session_state:
    st.session_state.selected_algos = []
if "target_col" not in st.session_state:
    st.session_state.target_col = None
if "input_cols" not in st.session_state:
    st.session_state.input_cols = []
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None

# Helper to get fallback suggestions from OpenAI on errors
def fetch_ai_fix(error_msg):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a debugging assistant for ML workflows."},
                {"role": "user", "content": f"The following error occurred during data processing or model training: {error_msg}. Please suggest a fix or workaround."}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Failed to fetch AI fix: {e}"

# Steps 1-5: Upload, EDA, Transformations, Feature Selection, Model Suggestion
# Assume these are implemented above

# Step 6: Model Training + Explainability
if st.session_state.step == 6:
    st.subheader("🏋️‍♂️ Model Training & Explainability")

    df = st.session_state.transformed_df
    X = df[st.session_state.input_cols]
    y = df[st.session_state.target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.session_state.X_train = X_train
    st.session_state.y_train = y_train

    for algo in st.session_state.selected_algos:
        model = None
        if algo == "Logistic Regression":
            model = LogisticRegression()
        elif algo == "Decision Tree":
            model = DecisionTreeClassifier()
        elif algo == "Random Forest":
            model = RandomForestClassifier()
        elif algo == "XGBoost":
            model = xgb.XGBClassifier()
        elif algo == "SVM":
            model = SVC()
        elif algo == "k-NN":
            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()
        elif algo == "Linear Regression":
            model = LinearRegression()
        elif algo == "Decision Tree Regressor":
            model = DecisionTreeRegressor()
        elif algo == "Random Forest Regressor":
            model = RandomForestRegressor()
        elif algo == "XGBoost Regressor":
            model = xgb.XGBRegressor()
        elif algo == "SVR":
            model = SVR()

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if st.session_state.problem_type == "Classification":
                score = accuracy_score(y_test, preds)
                st.success(f"{algo} Accuracy: {score:.3f}")
            else:
                score = mean_squared_error(y_test, preds, squared=False)
                st.success(f"{algo} RMSE: {score:.3f}")

            st.session_state.trained_models[algo] = model

            # Explainability
            try:
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_train[:100])
                st.write(f"Feature Importance for {algo}:")
                shap.plots.beeswarm(shap_values, max_display=10, show=False)
                st.pyplot(plt.gcf())
                plt.clf()
            except Exception as e:
                st.warning(f"SHAP not available for {algo}: {e}")

        except Exception as e:
            st.error(f"Training error in {algo}: {e}")
            st.info(fetch_ai_fix(str(e)))

    if st.button("Proceed to Export & Test"):
        st.session_state.step = 7

# Step 7: Save Model & Custom Test
if st.session_state.step == 7:
    st.subheader("📦 Export Model & Test on Custom Input")

    selected_model_name = st.selectbox("Select Trained Model to Export & Test", list(st.session_state.trained_models.keys()))
    model = st.session_state.trained_models[selected_model_name]

    # Download link
    model_filename = f"{selected_model_name.replace(' ', '_')}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    with open(model_filename, "rb") as f:
        st.download_button(
            label=f"📥 Download {selected_model_name} Model",
            data=f,
            file_name=model_filename,
            mime="application/octet-stream"
        )

    st.markdown("---")
    st.markdown("### 🔬 Test Model with Custom Input")
    custom_input = {}
    for col in st.session_state.input_cols:
        val = st.text_input(f"Input for {col}")
        try:
            custom_input[col] = float(val)
        except:
            custom_input[col] = 0.0

    if st.button("Predict with Custom Input"):
        input_df = pd.DataFrame([custom_input])
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info(fetch_ai_fix(str(e)))