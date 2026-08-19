import streamlit as st
import sys
sys.path.append("src")

from data_loader import load_data, get_feature_summary
from eda_stats import churn_by_category, chi_square_test, t_test_numeric
from model_trainer import preprocess_data, train_and_evaluate
from explainer import get_shap_values, get_feature_importance_summary, explain_single_customer
from retention_copilot import generate_retention_strategy

st.set_page_config(page_title="Churn Retention Copilot", layout="wide")

st.title("Customer Churn Prediction & AI Retention Copilot")

@st.cache_data
def load_and_prep():
    df = load_data()
    X, y, encoders = preprocess_data(df)
    return df, X, y

@st.cache_resource
def train_models(X, y):
    results, models, X_train, X_test, y_train, y_test, scaler = train_and_evaluate(X, y)
    return results, models, X_train, X_test, y_train, y_test, scaler

df, X, y = load_and_prep()
results, models, X_train, X_test, y_train, y_test, scaler = train_models(X, y)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Statistical Analysis", "Model Comparison", "AI Retention Copilot"])

with tab1:
    summary = get_feature_summary(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", summary["rows"])
    col2.metric("Churn Rate", str(summary["churn_rate"]) + "%")
    col3.metric("Features", summary["columns"])

    st.subheader("Churn Rate by Contract Type")
    st.dataframe(churn_by_category(df, "Contract"))

with tab2:
    st.subheader("Statistical Significance Tests")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Tenure vs Churn (T-Test)**")
        st.json(t_test_numeric(df, "tenure"))
    with col2:
        st.write("**Monthly Charges vs Churn (T-Test)**")
        st.json(t_test_numeric(df, "MonthlyCharges"))

    st.write("**Contract Type vs Churn (Chi-Square)**")
    st.json(chi_square_test(df, "Contract"))

with tab3:
    st.subheader("Model Performance Comparison")
    results_df = st.session_state.get("results_df")
    import pandas as pd
    results_table = pd.DataFrame(results).T
    st.dataframe(results_table)

    st.subheader("Feature Importance (SHAP - XGBoost)")
    xgb_model = models["XGBoost"]
    explainer, shap_values = get_shap_values(xgb_model, X_train, X_test)
    st.dataframe(get_feature_importance_summary(shap_values, X_test))

with tab4:
    st.subheader("AI-Generated Retention Strategy")

    xgb_model = models["XGBoost"]
    explainer, shap_values = get_shap_values(xgb_model, X_train, X_test)
    churn_probs = xgb_model.predict_proba(X_test)[:, 1]

    customer_idx = st.slider("Select customer index", 0, len(X_test) - 1, int(churn_probs.argmax()))

    prob = churn_probs[customer_idx]
    st.metric("Churn Probability", str(round(prob * 100, 1)) + "%")

    explanation = explain_single_customer(explainer, shap_values, X_test, customer_idx)
    st.write("**Top Contributing Factors:**")
    st.dataframe(explanation)

    if st.button("Generate Retention Strategy"):
        with st.spinner("Generating..."):
            strategy = generate_retention_strategy(explanation, prob)
        st.success(strategy)