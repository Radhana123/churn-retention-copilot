import shap
import pandas as pd


def get_shap_values(model, X_train, X_test, model_name="XGBoost"):
    """
    Compute SHAP values for tree-based models (Random Forest, XGBoost).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    return explainer, shap_values


def get_feature_importance_summary(shap_values, X_test, top_n=10):
    """
    Rank features by mean absolute SHAP value (overall importance).
    """
    mean_abs_shap = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    return mean_abs_shap.head(top_n)


def explain_single_customer(explainer, shap_values, X_test, customer_index):
    """
    Get SHAP explanation for a single customer's prediction.
    Returns top contributing features (positive = pushes toward churn).
    """
    customer_shap = shap_values[customer_index]

    explanation = pd.DataFrame({
        "feature": X_test.columns,
        "value": X_test.iloc[customer_index].values,
        "shap_contribution": customer_shap
    }).sort_values("shap_contribution", key=abs, ascending=False)

    return explanation.head(5)


if __name__ == "__main__":
    from data_loader import load_data
    from model_trainer import preprocess_data, train_and_evaluate

    df = load_data()
    X, y, encoders = preprocess_data(df)
    results, models, X_train, X_test, y_train, y_test, scaler = train_and_evaluate(X, y)

    xgb_model = models["XGBoost"]
    explainer, shap_values = get_shap_values(xgb_model, X_train, X_test)

    print("=== Top 10 Most Important Features (Overall) ===")
    print(get_feature_importance_summary(shap_values, X_test))

    print("\n=== Explanation for Customer #0 ===")
    print(explain_single_customer(explainer, shap_values, X_test, 0))