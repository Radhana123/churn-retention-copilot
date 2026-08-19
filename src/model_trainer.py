import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix


def preprocess_data(df):
    """
    Encode categorical variables and scale numeric ones.
    """
    df = df.drop(columns=["customerID"])

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y, label_encoders


def train_and_evaluate(X, y):
    """
    Train Logistic Regression, Random Forest, and XGBoost.
    Return a comparison of their performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "precision": round(precision_score(y_test, preds), 3),
            "recall": round(recall_score(y_test, preds), 3),
            "roc_auc": round(roc_auc_score(y_test, probs), 3)
        }
        trained_models[name] = model

    return results, trained_models, X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    X, y, encoders = preprocess_data(df)

    print("Training models...")
    results, models, X_train, X_test, y_train, y_test, scaler = train_and_evaluate(X, y)

    print("\n=== Model Comparison ===")
    for name, metrics in results.items():
        print(name + ": " + str(metrics))