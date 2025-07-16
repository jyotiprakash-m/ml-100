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

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") or ""

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

# Helper to get fallback suggestions from OpenAI on errors
def fetch_ai_fix(error_msg):
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a debugging assistant for ML workflows."},
                {"role": "user", "content": f"The following error occurred during data processing or model training: {error_msg}. Please suggest a fix or workaround."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to fetch AI fix: {e}"

# Step 1: Upload dataset
if st.session_state.step == 1:
    st.subheader("📁 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df.head(500)
            st.write("### Preview of Uploaded Data")
            st.dataframe(st.session_state.df)

            if st.button("Next: Analyze with AI"):
                st.session_state.step = 2

        except Exception as e:
            st.error(f"Failed to read file: {e}")

# Step 2: AI Suggestions
if st.session_state.step == 2 and st.session_state.df is not None:
    st.subheader("🤔 AI Suggestions for Preprocessing")

    df = st.session_state.df
    summary = {
        col: {
            "dtype": str(df[col].dtype),
            "missing_pct": round(df[col].isnull().mean() * 100, 2),
            "unique_vals": df[col].nunique()
        }
        for col in df.columns
    }

    prompt = f"""
    You are a data scientist assistant. A user uploaded a dataset with the following schema:

    {summary}

    Based on this metadata, suggest preprocessing steps like imputation, encoding, scaling, outlier detection, or feature engineering. Keep suggestions readable, short, and actionable. Explain WHY each step is useful.
    """

    if st.button("Get AI Suggestions"):
        with st.spinner("Thinking..."):
            try:
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful ML assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                suggestions = response.choices[0].message.content
                st.session_state.suggestions = suggestions
            except Exception as e:
                suggestions = f"⚠️ Failed to fetch suggestions: {e}"
                st.session_state.suggestions = suggestions

    if st.session_state.suggestions:
        st.markdown("### ✨ AI Suggestions")
        st.markdown(st.session_state.suggestions)
        if st.button("Continue to Apply Transformations"):
            st.session_state.step = 3

# Step 3: Apply Transformations
if st.session_state.step == 3 and st.session_state.df is not None:
    st.subheader("🛠️ Apply Suggested Transformations")

    df = st.session_state.df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    st.sidebar.header("🔧 Transformation Settings")
    impute_strategy = st.sidebar.selectbox("Imputation Strategy", ["mean", "median", "most_frequent"])
    scale_strategy = st.sidebar.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler"])
    encoding_method = st.sidebar.selectbox("Encoding Method", ["One-Hot", "Label Encoding"])
    additional_steps = st.sidebar.multiselect("Additional Techniques", [
        "Binning/Discretization",
        "Log Transformation",
        "Outlier Detection",
        "Correlation-Based Feature Selection"
    ])

    try:
        if not numeric_cols.empty:
            imputer = SimpleImputer(strategy=impute_strategy)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        if not cat_cols.empty:
            if encoding_method == "One-Hot":
                df = pd.get_dummies(df, columns=list(cat_cols), drop_first=True)
            else:
                for col in cat_cols:
                    df[col] = df[col].astype('category').cat.codes

        if not numeric_cols.empty:
            scaler = StandardScaler() if scale_strategy == "StandardScaler" else MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        st.session_state.transformed_df = df

        st.write("### ✅ Transformed Data Preview")
        st.dataframe(df.head())

        if st.button("Analyze Results and Suggest Next"):
            st.session_state.step = 4
    except Exception as e:
        fix = fetch_ai_fix(str(e))
        st.error(f"Error during transformation: {e}")
        st.info(fix)

# Step 4: Re-analyze and suggest next steps
if st.session_state.step == 4 and st.session_state.transformed_df is not None:
    st.subheader("🔁 AI Feedback on Transformed Data")

    df = st.session_state.transformed_df
    summary = f"Shape: {df.shape}, Columns: {list(df.columns)[:10]}..., Total Columns: {len(df.columns)}"

    prompt = f"""
    The user has applied preprocessing steps. Here's the transformed data summary:

    {summary}

    Suggest any next steps for modeling or further cleaning. Be concise, and prioritize actions.
    """

    with st.spinner("Thinking..."):
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful ML assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            feedback = response.choices[0].message.content
            st.markdown("### 🔮 AI Suggests:")
            st.markdown(feedback)
        except Exception as e:
            st.error(f"Error generating feedback: {e}")

# Step 5: Target Selection & Model Suggestions
if st.session_state.step == 4 and st.session_state.transformed_df is not None:
    st.subheader("🔢 Select Target Column and Get Model Suggestions")

    df = st.session_state.transformed_df
    target_col = st.selectbox("Choose the target column", df.columns)
    st.session_state.target_col = target_col

    if target_col:
        target_dtype = df[target_col].dtype
        unique_vals = df[target_col].nunique()

        if target_dtype == 'object' or unique_vals <= 10:
            problem_type = "Classification"
            algorithms = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVM", "k-NN"]
        elif np.issubdtype(type(target_dtype), np.number):
            problem_type = "Regression"
            algorithms = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "XGBoost Regressor", "SVR"]
        else:
            problem_type = "Unknown"
            algorithms = []

        st.markdown(f"**Detected Problem Type:** `{problem_type}`")
        if algorithms:
            selected_algos = st.multiselect("Recommended Algorithms", algorithms, default=algorithms[:2])
            st.session_state.selected_algos = selected_algos
            if st.button("Proceed to Training"):
                st.session_state.step = 6

# Step 6: Model Training & Evaluation
if st.session_state.step == 6 and st.session_state.transformed_df is not None and st.session_state.target_col:
    st.subheader("🧪 Training & Evaluation")

    df = st.session_state.transformed_df
    target = st.session_state.target_col
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("### Training Results")
    for algo in st.session_state.selected_algos:
        try:
            if algo == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
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
            else:
                continue

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if y.dtype == 'int' or y.nunique() <= 10:
                score = accuracy_score(y_test, y_pred)
                st.write(f"**{algo} Accuracy:** {score:.4f}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"**{algo} MSE:** {mse:.4f}")
        except Exception as e:
            fix = fetch_ai_fix(str(e))
            st.error(f"Error in {algo}: {e}")
            st.info(fix)
