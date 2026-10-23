"""Streamlit demo: enter loan applicant details -> probability of default + risk band.
Run locally (after running train.py once):  python -m streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

from features import GRADES, HOME, INTENT, engineer

st.set_page_config(page_title="Credit Risk Scorer", page_icon="🏦")
st.title("🏦 Credit Risk Scorer")
st.caption("Estimates the probability that a loan applicant will default. "
           "Demo model trained on the Kaggle 'Credit Risk Dataset' - not for real lending decisions.")


@st.cache_resource
def load_artifacts():
    return joblib.load("outputs/model.joblib")


art = load_artifacts()
model, features, stats = art["model"], art["features"], art["stats"]

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 28)
    income = st.number_input("Annual income", 4000, 500000, 55000, step=1000)
    emp_len = st.number_input("Employment length (years)", 0.0, 50.0, 4.0, step=1.0)
    home = st.selectbox("Home ownership", HOME)
    hist = st.number_input("Credit history length (years)", 2, 40, 4)
with col2:
    amount = st.number_input("Loan amount", 500, 35000, 8000, step=500)
    intent = st.selectbox("Loan purpose", INTENT)
    grade = st.selectbox("Loan grade", GRADES, index=1)
    default_rate = float(stats["rate_by_grade"].get(grade, 11.0))
    rate = st.number_input("Interest rate (%)", 5.0, 25.0, round(default_rate, 2), step=0.25)
    prior = st.selectbox("Previous default on file?", ["N", "Y"])

if st.button("Score applicant", type="primary"):
    row = pd.DataFrame([{
        "person_age": age,
        "person_income": min(income, stats["income_cap"]),
        "person_home_ownership": home,
        "person_emp_length": emp_len,
        "loan_intent": intent,
        "loan_grade": grade,
        "loan_amnt": amount,
        "loan_int_rate": rate,
        "loan_percent_income": round(amount / income, 2),
        "cb_person_default_on_file": prior,
        "cb_person_cred_hist_length": hist,
        "EmpLengthMissing": 0,
        "IntRateMissing": 0,
    }])
    X = engineer(row)[features]
    p = float(model.predict_proba(X)[0, 1])
    band = "Low" if p < art["band_low"] else "Medium" if p < art["band_high"] else "High"

    st.metric("Probability of default", f"{p:.1%}")
    {"Low": st.success, "Medium": st.warning, "High": st.error}[band](f"Risk band: {band}")
    st.progress(min(p, 1.0))
    st.caption(f"Loan is {amount / income:.0%} of annual income.")
