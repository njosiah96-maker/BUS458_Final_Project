
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn # Needed to load sklearn objects from pickle

# Load the trained model and scaler
with open("deployment_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# getting feature names
model_features = model.feature_names_in_

# app title
st.markdown(
    "<h1 style='text-align: center; background-color: #e0f7fa; padding: 15px; color: #00796b;'><b>Loan Approval Decision Predictor</b></h1>",
    unsafe_allow_html=True
)

st.write("\n")
st.markdown("<h2 style='text-align: center;'>Enter Applicant's Details</h2>", unsafe_allow_html=True)
st.write("\n")

#fields for numeric values
Requested_Loan_Amount = st.number_input("Requested Loan Amount", min_value=5000.0, max_value=2500000.0, value=50000.0, step=1000.0)
FICO_score = st.number_input("FICO Score", min_value=300.0, max_value=850.0, value=650.0, step=1.0)
Monthly_Gross_Income = st.number_input("Monthly Gross Income", min_value=0.0, max_value=20000.0, value=5000.0, step=100.0)
Monthly_Housing_Payment = st.number_input("Monthly Housing Payment", min_value=300.0, max_value=50000.0, value=1500.0, step=50.0)

# categorical inputs
Reason_options = ['cover_an_unexpected_cost', 'credit_card_refinancing', 'debt_conslidation', 'home_improvement', 'major_purchase', 'other']
Reason = st.selectbox("Reason for Loan", Reason_options)

Fico_Score_group_options = ['fair', 'poor', 'good', 'excellent', 'very_good']
Fico_Score_group = st.selectbox("FICO Score Group", Fico_Score_group_options)

Employment_Status_options = ['full_time', 'part_time', 'unemployed']
Employment_Status = st.selectbox("Employment Status", Employment_Status_options)

Employment_Sector_options = ['consumer_discretionary', 'information_technology', 'energy', 'consumer_staples', 'communication_services', 'materials', 'utilities', 'real_estate', 'health_care', 'industrials', 'financials', 'Unknown']
Employment_Sector = st.selectbox("Employment Sector", Employment_Sector_options)

Lender_options = ['A', 'B', 'C']
Lender = st.selectbox("Lender", Lender_options)

Ever_Bankrupt_or_Foreclose_options = {0: 'No', 1: 'Yes'}
Ever_Bankrupt_or_Foreclose = st.selectbox("Ever Bankrupt or Foreclose?", options=list(Ever_Bankrupt_or_Foreclose_options.keys()), format_func=lambda x: Ever_Bankrupt_or_Foreclose_options[x])

# Prepping Data for Prediction
if st.button("Predict Loan Approval"):    
    # creating dictionary from inputs
    input_dict = {
        'Requested_Loan_Amount': Requested_Loan_Amount,
        'FICO_score': FICO_score,
        'Monthly_Gross_Income': Monthly_Gross_Income,
        'Monthly_Housing_Payment': Monthly_Housing_Payment,
        'Ever_Bankrupt_or_Foreclose': Ever_Bankrupt_or_Foreclose,
        'Reason': Reason,
        'Fico_Score_group': Fico_Score_group,
        'Employment_Status': Employment_Status,
        'Employment_Sector': Employment_Sector,
        'Lender': Lender
    }

    # creating df
    input_df = pd.DataFrame([input_dict])

    # Recreating additional features
    input_df['Loan_to_Income_Ratio'] = input_df['Requested_Loan_Amount'] / input_df['Monthly_Gross_Income']
    input_df['Housing_to_Income_Ratio'] = input_df['Monthly_Housing_Payment'] / input_df['Monthly_Gross_Income']

    # One-hot encode categorical features
    categorical_cols_for_ohe = [
        'Reason', 'Fico_Score_group', 'Employment_Status', 
        'Employment_Sector', 'Lender'
    ]
    
    # Converting for consistent dummy creation
    for col in categorical_cols_for_ohe:
        input_df[col] = pd.Categorical(input_df[col], categories=eval(f"{col}_options")) # Use eval to get original options

    input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols_for_ohe, drop_first=True)

    # Aligning columns with model's expected features
    # Df with all expected features, filled with zeros
    final_input_df = pd.DataFrame(columns=model_features, data=np.zeros((1, len(model_features))))

    # Adding current input data
    for col in input_df_encoded.columns:
        if col in final_input_df.columns:
            final_input_df[col] = input_df_encoded[col]

    # numerical columns that need scaling
    numerical_cols_for_scaling = [
        'Requested_Loan_Amount',
        'FICO_score',
        'Monthly_Gross_Income',
        'Monthly_Housing_Payment',
        'Loan_to_Income_Ratio',
        'Housing_to_Income_Ratio'
    ]

    # applying scaling
    final_input_df[numerical_cols_for_scaling] = scaler.transform(final_input_df[numerical_cols_for_scaling])

    # prediction
    prediction_proba = model.predict_proba(final_input_df)[:, 1]
    custom_threshold = 0.7
    prediction = (prediction_proba >= custom_threshold).astype(int)[0]

    if prediction == 1:
        st.success(f"Prediction: **APPROVED** (Probability: {prediction_proba[0]:.2f})")
        st.balloons()
    else:
        st.error(f"Prediction: **DENIED** (Probability: {prediction_proba[0]:.2f})")
