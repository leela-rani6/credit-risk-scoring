"""Shared cleaning + feature engineering, used by BOTH train.py and app.py
so the model always sees identical features."""
import numpy as np
import pandas as pd

TARGET = "loan_status"  # 1 = borrower defaulted
HOME = ["RENT", "MORTGAGE", "OWN", "OTHER"]
INTENT = ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
GRADES = ["A", "B", "C", "D", "E", "F", "G"]


def clean_training_data(df: pd.DataFrame):
    """Clean the raw CSV. Returns (clean_df, stats) - stats are reused by the app."""
    df = df.drop_duplicates().copy()

    # Impossible values (e.g. age 144, 123 years of employment)
    df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)]
    df.loc[df["person_emp_length"] > 60, "person_emp_length"] = np.nan

    # Missing flags: "missing" can itself be a risk signal
    df["EmpLengthMissing"] = df["person_emp_length"].isna().astype(int)
    df["IntRateMissing"] = df["loan_int_rate"].isna().astype(int)

    # Impute: interest rate by loan grade (rate is driven by grade), employment by median
    rate_by_grade = df.groupby("loan_grade")["loan_int_rate"].median().to_dict()
    emp_median = float(df["person_emp_length"].median())
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_grade"].map(rate_by_grade))
    df["person_emp_length"] = df["person_emp_length"].fillna(emp_median)

    # Cap extreme incomes (max in raw data is 6,000,000)
    income_cap = float(df["person_income"].quantile(0.99))
    df["person_income"] = df["person_income"].clip(upper=income_cap)

    stats = {"rate_by_grade": rate_by_grade, "emp_median": emp_median, "income_cap": income_cap}
    return df.reset_index(drop=True), stats


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw (already cleaned) columns -> model feature matrix with a fixed column order."""
    X = pd.DataFrame(index=df.index)
    X["person_age"] = df["person_age"]
    X["log_income"] = np.log1p(df["person_income"])
    X["person_emp_length"] = df["person_emp_length"]
    X["loan_amnt"] = df["loan_amnt"]
    X["loan_int_rate"] = df["loan_int_rate"]
    X["loan_percent_income"] = df["loan_percent_income"]
    X["cb_person_cred_hist_length"] = df["cb_person_cred_hist_length"]
    X["grade_num"] = df["loan_grade"].map({g: i for i, g in enumerate(GRADES)}).astype(int)
    X["prior_default"] = (df["cb_person_default_on_file"] == "Y").astype(int)
    X["high_loan_burden"] = (df["loan_percent_income"] > 0.30).astype(int)
    X["EmpLengthMissing"] = df["EmpLengthMissing"]
    X["IntRateMissing"] = df["IntRateMissing"]
    X["rate_x_burden"] = df["loan_int_rate"] * df["loan_percent_income"]
    X["age_at_first_credit"] = df["person_age"] - df["cb_person_cred_hist_length"]
    for h in HOME:
        X[f"home_{h}"] = (df["person_home_ownership"] == h).astype(int)
    for i in INTENT:
        X[f"intent_{i}"] = (df["loan_intent"] == i).astype(int)
    return X
