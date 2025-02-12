import streamlit as st
import pandas as pd
import pickle
from transformers import pipeline

# Load trained model
with open("RF_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load column names used during training
with open("columns.pkl", "rb") as file:
    training_columns = pickle.load(file)

# Load Hugging Face model
generator = pipeline('text2text-generation', model='google/flan-t5-large')

# Streamlit UI
st.title("Welcome to Loan Eligibility")

# User Inputs
no_of_dependents = st.number_input("No of Dependents", min_value=0, max_value=5)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (Months)", min_value=2, max_value=20)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Prepare input DataFrame
input_data = pd.DataFrame([[no_of_dependents, education, self_employed, income_annum, 
                            loan_amount, loan_term, cibil_score, residential_assets_value, 
                            commercial_assets_value, luxury_assets_value, bank_asset_value]],
                          columns=["no_of_dependents", "education", "self_employed", "income_annum", 
                                   "loan_amount", "loan_term", "cibil_score", "residential_assets_value", 
                                   "commercial_assets_value", "luxury_assets_value", "bank_asset_value"])

# One-hot encode categorical columns
loan_dummies = pd.get_dummies(input_data)

# Ensure all expected columns are present
for col in training_columns:
    if col not in loan_dummies.columns:
        loan_dummies[col] = 0

# Reorder columns to match the training data
loan_dummies = loan_dummies[training_columns]

# Prediction
if st.button("Predict Eligibility"):
    prediction = model.predict(loan_dummies)
    result = "Approved" if prediction[0] == 1 else "Rejected"
    st.write(f"Loan Eligibility Prediction: **{result}**")
    
    # Hugging Face model to generate a response
    prompt = f"""
    A person with the following details applied for a loan:
    - Number of Dependents: {no_of_dependents}
    - Education: {education}
    - Self Employed: {self_employed}
    - Annual Income: {income_annum}
    - Loan Amount: {loan_amount}
    - Loan Term: {loan_term} months
    - CIBIL Score: {cibil_score}
    - Residential Assets Value: {residential_assets_value}
    - Commercial Assets Value: {commercial_assets_value}
    - Luxury Assets Value: {luxury_assets_value}
    - Bank Asset Value: {bank_asset_value}
    
    The loan application was **{'approved' if prediction[0] == 1 else 'rejected'}. 

    Please explain step by step why the loan was {'approved' if prediction[0] == 1 else 'rejected'}:

    If rejected: Provide 3 practical steps the applicant can take to improve their chances for approval next time.
    """
    generated_text = generator(prompt, max_new_tokens=400, num_return_sequences=1, temperature=0.7, top_p=0.9)

    # Extract only the generated text after the prompt
    explanation = generated_text[0]['generated_text'].replace(prompt, "").strip()
    st.write("wanna know why ?? Here you go:")
    st.write(explanation)