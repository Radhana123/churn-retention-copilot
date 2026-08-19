import pandas as pd
from scipy import stats


def churn_by_category(df, column):
    """
    Compute churn rate breakdown by a categorical column.
    """
    result = df.groupby(column)["Churn"].agg(["mean", "count"])
    result["mean"] = (result["mean"] * 100).round(2)
    result.columns = ["churn_rate_pct", "count"]
    return result.sort_values("churn_rate_pct", ascending=False)


def chi_square_test(df, column):
    """
    Chi-square test: is there a statistically significant relationship
    between this categorical column and churn?
    """
    contingency = pd.crosstab(df[column], df["Churn"])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    return {
        "column": column,
        "chi2_statistic": round(chi2, 2),
        "p_value": round(p_value, 5),
        "significant": p_value < 0.05
    }


def t_test_numeric(df, column):
    """
    Independent t-test: do churned and non-churned customers differ
    significantly on this numeric column?
    """
    churned = df[df["Churn"] == 1][column]
    not_churned = df[df["Churn"] == 0][column]

    t_stat, p_value = stats.ttest_ind(churned, not_churned)

    return {
        "column": column,
        "churned_mean": round(churned.mean(), 2),
        "not_churned_mean": round(not_churned.mean(), 2),
        "t_statistic": round(t_stat, 2),
        "p_value": round(p_value, 5),
        "significant": p_value < 0.05
    }


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()

    print("=== Churn Rate by Contract Type ===")
    print(churn_by_category(df, "Contract"))

    print("\n=== Churn Rate by Internet Service ===")
    print(churn_by_category(df, "InternetService"))

    print("\n=== Chi-Square Test: Contract vs Churn ===")
    print(chi_square_test(df, "Contract"))

    print("\n=== T-Test: Tenure (months) vs Churn ===")
    print(t_test_numeric(df, "tenure"))

    print("\n=== T-Test: Monthly Charges vs Churn ===")
    print(t_test_numeric(df, "MonthlyCharges"))