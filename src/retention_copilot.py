import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_retention_strategy(customer_explanation_df, churn_probability):
    """
    Use SHAP explanation + churn probability to generate a personalized
    retention strategy via LLM.
    """
    top_factors = ""
    for _, row in customer_explanation_df.iterrows():
        direction = "increases" if row["shap_contribution"] > 0 else "decreases"
        top_factors += "- " + row["feature"] + " (value: " + str(row["value"]) + ") " + direction + " churn risk\n"

    prompt = (
        "You are a customer retention strategist for a telecom company.\n\n"
        "A customer has a churn probability of " + str(round(churn_probability * 100, 1)) + "%.\n\n"
        "The top factors driving this prediction are:\n" + top_factors + "\n"
        "Write a short, specific, actionable retention strategy for this customer "
        "(2-3 sentences). Suggest a concrete offer or action, not generic advice."
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    from data_loader import load_data
    from model_trainer import preprocess_data, train_and_evaluate
    from explainer import get_shap_values, explain_single_customer

    df = load_data()
    X, y, encoders = preprocess_data(df)
    results, models, X_train, X_test, y_train, y_test, scaler = train_and_evaluate(X, y)

    xgb_model = models["XGBoost"]
    explainer, shap_values = get_shap_values(xgb_model, X_train, X_test)

    # Pick a high-risk customer
    churn_probs = xgb_model.predict_proba(X_test)[:, 1]
    high_risk_idx = churn_probs.argmax()

    explanation = explain_single_customer(explainer, shap_values, X_test, high_risk_idx)

    print("Churn probability: " + str(round(churn_probs[high_risk_idx] * 100, 1)) + "%")
    print("\nTop factors:")
    print(explanation)

    print("\nGenerating retention strategy...")
    strategy = generate_retention_strategy(explanation, churn_probs[high_risk_idx])
    print("\n=== AI Retention Strategy ===")
    print(strategy)