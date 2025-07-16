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
import joblib

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") or ""

st.set_page_config(page_title="ML Assistant", layout="wide")
st.title("🤖 Smart ML Workflow Assistant")

# Session state init
for key in ["step", "df", "suggestions", "transformed_df", "selected_algos", "target_col", "results"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "step" else 1

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
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
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
                st.session_state.suggestions = response.choices[0].message.content
            except Exception as e:
                st.session_state.suggestions = f"⚠️ Failed to fetch suggestions: {e}"

    if st.session_state.suggestions:
        st.markdown("### ✨ AI Suggestions")
        st.markdown(st.session_state.suggestions)
        if st.button("Continue to Apply Transformations"):
            st.session_state.step = 3

# Step 3: Apply Transformations
if st.session_state.step == 3 and st.session_state.df is not None:
    st.subheader("🛠️ Apply Suggested Transformations")
    df = st.session_state.df.copy()

    # Column selection for further processing
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Select columns to include for further processing:",
        all_columns,
    )
    df = df[selected_columns]

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.sidebar.header("🔧 Transformation Settings")

    # NEW: Select columns for each transformation
    impute_cols = st.sidebar.multiselect("Columns to Impute", numeric_cols, default=numeric_cols)
    scale_cols = st.sidebar.multiselect("Columns to Scale", numeric_cols, default=numeric_cols)
    encode_cols = st.sidebar.multiselect("Columns to Encode", cat_cols, default=cat_cols)

    impute_strategy = st.sidebar.selectbox("Imputation Strategy", ["mean", "median", "most_frequent"])
    scale_strategy = st.sidebar.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler"])
    encoding_method = st.sidebar.selectbox("Encoding Method", ["One-Hot", "Label Encoding"])
    additional_steps = st.sidebar.multiselect("Additional Techniques", [
        "Binning/Discretization",
        "Log Transformation",
        "Outlier Detection",
        "Correlation-Based Feature Selection"
    ])

    # Feature Selection
    st.sidebar.subheader("🧠 Feature Selection")
    fs_method = st.sidebar.selectbox("Feature Selection Method", ["None", "Variance Threshold", "Correlation Threshold", "Select K Best (Univariate)", "Tree-based Importance"])

    try:
        # Impute only selected columns
        if impute_cols:
            imputer = SimpleImputer(strategy=impute_strategy)
            df[impute_cols] = imputer.fit_transform(df[impute_cols])

        # Encode only selected columns
        if encode_cols:
            if encoding_method == "One-Hot":
                df = pd.get_dummies(df, columns=encode_cols, drop_first=True)
            else:
                for col in encode_cols:
                    df[col] = df[col].astype('category').cat.codes

        # Scale only selected columns
        if scale_cols:
            scaler = StandardScaler() if scale_strategy == "StandardScaler" else MinMaxScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])

        # --- Feature Engineering: Combine Multiple Columns ---
        st.markdown("#### ➕ Feature Engineering: Combine Columns")
        combine_cols = st.multiselect("Select columns to combine", all_columns, key="fe_multi_cols")
        operation = st.selectbox("Operation", ["Sum", "Product", "Mean", "Custom Formula"], key="fe_multi_op")
        new_col_name = st.text_input("New column name", value="_".join(combine_cols) + f"_{operation.lower()}" if combine_cols else "")

        custom_formula = ""
        if operation == "Custom Formula":
            st.info("Use Python syntax. Example: col1 + col2 * col3")
            custom_formula = st.text_input("Custom formula", value=" + ".join(combine_cols))

        if st.button("Create Combined Feature"):
            if len(combine_cols) < 2:
                st.warning("Please select at least two columns.")
            elif new_col_name.strip() == "":
                st.warning("Please provide a name for the new column.")
            else:
                try:
                    if operation == "Sum":
                        df[new_col_name] = df[combine_cols].sum(axis=1)
                    elif operation == "Product":
                        df[new_col_name] = df[combine_cols].prod(axis=1)
                    elif operation == "Mean":
                        df[new_col_name] = df[combine_cols].mean(axis=1)
                    elif operation == "Custom Formula":
                        local_dict = {col: df[col] for col in combine_cols}
                        df[new_col_name] = eval(custom_formula, {}, local_dict)
                    st.success(f"Feature '{new_col_name}' created!")
                    st.dataframe(df[[*combine_cols, new_col_name]].head())

                    # Automatically add new feature to selected columns
                    if new_col_name not in selected_columns:
                        selected_columns.append(new_col_name)
                        st.session_state['selected_columns'] = selected_columns
                except Exception as e:
                    st.error(f"Could not create feature: {e}")

        # Apply Feature Selection
        if fs_method != "None":
            from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            target_col = st.session_state.target_col
            if fs_method == "Variance Threshold":
                threshold = st.sidebar.slider("Variance Threshold", 0.0, 0.2, 0.01)
                selector = VarianceThreshold(threshold=threshold)
                selector.fit(df)
                df = df[df.columns[selector.get_support()]]

            elif fs_method == "Correlation Threshold":
                corr_matrix = df.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                corr_thresh = st.sidebar.slider("Correlation Threshold", 0.7, 1.0, 0.9)
                to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
                df = df.drop(columns=to_drop)

            elif fs_method == "Select K Best (Univariate)" and target_col:
                X = df.drop(columns=[target_col])
                y = df[target_col]
                k = st.sidebar.slider("Top K Features", 1, min(20, df.shape[1]-1), 5)
                score_func = f_classif if y.nunique() <= 10 else f_regression
                selector = SelectKBest(score_func=score_func, k=k)
                selector.fit(X, y)
                selected = X.columns[selector.get_support()].tolist()
                selected.append(target_col)
                df = df[selected]

            elif fs_method == "Tree-based Importance" and target_col:
                X = df.drop(columns=[target_col])
                y = df[target_col]
                model = RandomForestClassifier() if y.nunique() <= 10 else RandomForestRegressor()
                model.fit(X, y)
                importances = pd.Series(model.feature_importances_, index=X.columns)
                top_k = st.sidebar.slider("Top K Important Features", 1, min(20, df.shape[1]-1), 5)
                selected = importances.sort_values(ascending=False).head(top_k).index.tolist()
                selected.append(target_col)
                df = df[selected]

        st.session_state.transformed_df = df
        st.write("### ✅ Transformed Data Preview")
        st.dataframe(df.head())
        if st.button("Analyze Results and Suggest Next"):
            st.session_state.step = 4
    except Exception as e:
        fix = fetch_ai_fix(str(e))
        st.error(f"Error during transformation: {e}")
        st.info(fix)

# Step 4: Re-analyze and AutoML Suggestions
if st.session_state.step == 4 and st.session_state.transformed_df is not None:
    st.subheader("🔀 AI Feedback on Transformed Data")
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
                messages=[{"role": "system", "content": "You are a helpful ML assistant."}, {"role": "user", "content": prompt}]
            )
            st.markdown("### 🔮 AI Suggests:")
            st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error generating feedback: {e}")

# Step 5: Target Selection & Model Suggestions
if st.session_state.step == 4 and st.session_state.transformed_df is not None:
    st.subheader("🔢 Select Target Column and Get Model Suggestions")
    df = st.session_state.transformed_df
    target_col = st.selectbox("Choose the target column", df.columns)
    st.session_state.target_col = target_col

    if target_col:
        y = df[target_col]
        problem_type = "Classification" if y.nunique() <= 10 else "Regression"
        st.markdown(f"**Detected Problem Type:** `{problem_type}`")

        algorithms = {
            "Classification": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVM", "k-NN"],
            "Regression": ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor", "XGBoost Regressor", "SVR"]
        }

        selected_algos = st.multiselect("Recommended Algorithms", algorithms[problem_type], default=algorithms[problem_type][:2])
        st.session_state.selected_algos = selected_algos

        if st.button("Proceed to Training"):
            st.session_state.step = 6

# Step 6: Training & Evaluation
if st.session_state.step == 6 and st.session_state.transformed_df is not None and st.session_state.target_col:
    st.subheader("🧪 Training & Evaluation")
    df = st.session_state.transformed_df
    target = st.session_state.target_col
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Show train and test tables
    st.markdown("#### 📝 Training Data (X_train, y_train)")
    st.dataframe(pd.concat([X_train, y_train], axis=1).head())
    st.markdown("#### 📝 Test Data (X_test, y_test)")
    st.dataframe(pd.concat([X_test, y_test], axis=1).head())
    st.markdown(f"**Training Data Shape:** {X_train.shape}, **Test Data Shape:** {X_test.shape}")

    results = []

    for algo in st.session_state.selected_algos:
        try:
            # Model init
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

            # Evaluate
            if y.nunique() <= 10:
                score = accuracy_score(y_test, y_pred)
                results.append([algo, "Accuracy", score])
            else:
                mse = mean_squared_error(y_test, y_pred)
                results.append([algo, "MSE", mse])

            # Save model
            os.makedirs("models", exist_ok=True)
            file_name = f"models/{algo.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, file_name)
            st.download_button("Download Model: " + algo, open(file_name, "rb"), file_name=os.path.basename(file_name))

        except Exception as e:
            fix = fetch_ai_fix(str(e))
            st.error(f"Error in {algo}: {e}")
            st.info(fix)

    # Show metrics table
    if results:
        st.markdown("### 🏋️ Model Comparison")
        st.dataframe(pd.DataFrame(results, columns=["Algorithm", "Metric", "Score"]))
        
        st.session_state.results = pd.DataFrame(results, columns=["Algorithm", "Metric", "Score"])
        # Show the pression recall f1-score for each model
        st.markdown("### 📊 Model Performance Metrics")
        for algo in st.session_state.selected_algos:
            try:
                model_file = f"models/{algo.replace(' ', '_').lower()}.pkl"
                model = joblib.load(model_file)
                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_train)

                st.markdown(f"#### SHAP Values for {algo}")
                shap.summary_plot(shap_values, X_train, show=False)
                st.pyplot(bbox_inches='tight')
            except Exception as e:
                st.error(f"Error generating SHAP values for {algo}: {e}")
                st.info(fetch_ai_fix(str(e)))
    
    if st.button("Proceed to Export & Test"):
        st.session_state.step = 7

# Step 7: Save Model & Custom Test
if st.session_state.step == 7:
    st.markdown("### 🔬 Test Model with Custom Input")

    # Automatically detect input columns from trained data
    if "transformed_df" in st.session_state and st.session_state.transformed_df is not None:
        input_cols = [col for col in st.session_state.transformed_df.columns if col != st.session_state.target_col]
        st.session_state.input_cols = input_cols

    # NEW: Model selection dropdown with "None" option
    model_options = ["None"] + (st.session_state.selected_algos if st.session_state.selected_algos else [])
    selected_model = st.selectbox("Select trained model for prediction", model_options)

    custom_input = {}
    for col in st.session_state.input_cols:
        val = st.text_input(f"Input for {col}")
        try:
            custom_input[col] = float(val)
        except:
            custom_input[col] = 0.0

    if st.button("Predict with Custom Input"):
        if selected_model == "None":
            st.warning("Please select a trained model for prediction.")
        else:
            input_df = pd.DataFrame([custom_input])
            try:
                # Load selected model
                model_file = f"models/{selected_model.replace(' ', '_').lower()}.pkl"
                model = joblib.load(model_file)
                prediction = model.predict(input_df)[0]
                st.success(f"Prediction: {prediction}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info(fetch_ai_fix(str(e)))
