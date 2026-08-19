import pandas as pd


def load_data(path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Load and do basic cleaning of the Telco Customer Churn dataset.
    """
    df = pd.read_csv(path)

    # TotalCharges has some blank strings instead of NaN - fix that
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def get_feature_summary(df):
    """
    Quick summary of the dataset - shape, churn rate, column types.
    """
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "churn_rate": round(df["Churn"].mean() * 100, 2),
        "numeric_cols": df.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        "categorical_cols": df.select_dtypes(include=["object"]).columns.tolist()
    }
    return summary


if __name__ == "__main__":
    df = load_data()
    summary = get_feature_summary(df)

    print("Dataset shape: " + str(summary["rows"]) + " rows, " + str(summary["columns"]) + " columns")
    print("Churn rate: " + str(summary["churn_rate"]) + "%")
    print("\nNumeric columns: " + str(summary["numeric_cols"]))
    print("\nCategorical columns: " + str(summary["categorical_cols"]))
    print("\nFirst few rows:")
    print(df.head())