import streamlit as st
import numpy as np
import pickle
import pandas as pd
import sklearn

# Load the saved model (forest_model.pkl)
model = pickle.load(open('forest_model.pkl', 'rb'))


import streamlit as st

# Load and display the banner image
st.image('image.png', use_column_width=True, caption="")

# Streamlit app
st.title("Customer Churn Prediction")





# Sidebar to explain the project

st.sidebar.title("Project Overview")
st.sidebar.markdown(
    """
    This project aims to predict **customer churn** using a machine learning model (Random Forest).
    
    ### Key Features:
    - **Age**: Age of the customer.
    - **Gender**: Male or Female.
    - **Tenure**: How long the customer has been with the company (in months).
    - **Usage Frequency**: How often the customer used the service last month.
    - **Support Calls**: Number of support calls made last month.
    - **Payment Delay**: Number of days the customer delayed their payment last month.
    - **Subscription Type**: Type of subscription (Standard, Premium, Basic).
    - **Contract Length**: Length of the contract (Monthly, Annual, Quarterly).
    - **Total Spend**: Total amount the customer has spent.
    - **Last Interaction**: Number of days since the customer's last interaction with the company.

    ### Prediction:
    This model predicts whether a customer is likely to churn or stay, along with a probability score.
    """
)

# Input fields for the user to enter data
st.header("Enter Customer Details:")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
usage_frequency = st.number_input("Usage Frequency (Last Month)", min_value=0, max_value=100, value=10)
support_calls = st.number_input("Support Calls (Last Month)", min_value=0, max_value=50, value=3)
payment_delay = st.number_input("Payment Delay (Days)", min_value=0, max_value=100, value=5)
subscription_type = st.selectbox("Subscription Type", ["Standard", "Premium", "Basic"])
contract_length = st.selectbox("Contract Length", ["Monthly", "Annual", "Quarterly"])
total_spend = st.number_input("Total Spend", min_value=0.0, value=500.0, step=0.1)
last_interaction = st.number_input("Last Interaction (Days)", min_value=0, max_value=100, value=20)

# Convert categorical features to numerical
gender_dict = {"Male": 1, "Female": 0}
subscription_dict = {"Standard": 2, "Premium": 1, "Basic": 0}
contract_dict = {"Monthly": 1, "Annual": 0, "Quarterly": 2}

gender_val = gender_dict[gender]
subscription_val = subscription_dict[subscription_type]
contract_val = contract_dict[contract_length]

# Prepare the input data
input_data = np.array([age, gender_val, tenure, usage_frequency, support_calls, 
                       payment_delay, subscription_val, contract_val, 
                       total_spend, last_interaction]).reshape(1, -1)

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)
    churn_probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"The customer is likely to churn! Probability: {churn_probability:.2f}%")
    else:
        st.success(f"The customer is likely to stay! Probability: {100 - churn_probability:.2f}%")

# Footer with custom animation
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-family: 'Arial', sans-serif;
    }
    </style>
    <div class="footer">
    <p>Powered by Streamlit & Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)