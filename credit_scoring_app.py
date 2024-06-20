import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import data_preprocessing, encoder_Credit_Mix, encoder_Payment_Behaviour, encoder_Payment_of_Min_Amount
from prediction import prediction

col1, col2 = st.columns([1, 5])
with col1:
    st.image("data:image/png;base64,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", width=130)
with col2:
    st.header('Credit Scoring App (Prototype) :sparkles:')

data = pd.DataFrame()

col1, col2, col3 = st.columns(3)

with col1:
    Credit_Mix = st.selectbox(label='Credit_Mix', options=encoder_Credit_Mix.classes_, index=1)
    data["Credit_Mix"] = [Credit_Mix]

with col2:
    Payment_of_Min_Amount = st.selectbox(label='Payment_of_Min_Amount', options=encoder_Payment_of_Min_Amount.classes_, index=1)
    data["Payment_of_Min_Amount"] = [Payment_of_Min_Amount]

with col3:
    Payment_Behaviour = st.selectbox(label='Payment_Behaviour', options=encoder_Payment_Behaviour.classes_, index=5)
    data["Payment_Behaviour"] = Payment_Behaviour

col1, col2, col3, col4 = st.columns(4)

with col1:
    # st.header("Kolom 1")
    Age = int(st.number_input(label='Age', value=23))
    data["Age"] = Age

with col2:
    Num_Bank_Accounts = int(st.number_input(label='Num_Bank_Accounts', value=3))
    data["Num_Bank_Accounts"] = Num_Bank_Accounts

with col3:
    Num_Credit_Card = int(st.number_input(label='Num_Credit_Card', value=4))
    data["Num_Credit_Card"] = Num_Credit_Card

with col4:
    Interest_Rate = float(st.number_input(label='Interest_Rate', value=3))
    data["Interest_Rate"] = Interest_Rate


col1, col2, col3, col4 = st.columns(4)

with col1:
    Num_of_Loan = int(st.number_input(label='Num_of_Loan', value=4))
    data["Num_of_Loan"] = Num_of_Loan

with col2:
    # st.header("Kolom 1")
    Delay_from_due_date = int(st.number_input(label='Delay_from_due_date', value=3))
    data["Delay_from_due_date"] = Delay_from_due_date

with col3:
    Num_of_Delayed_Payment = int(st.number_input(label='Num_of_Delayed_Payment', value=7))
    data["Num_of_Delayed_Payment"] = Num_of_Delayed_Payment

with col4:
    Changed_Credit_Limit = float(st.number_input(label='Changed_Credit_Limit', value=11.27))
    data["Changed_Credit_Limit"] = Changed_Credit_Limit

col1, col2, col3, col4 = st.columns(4)

with col1:
    Num_Credit_Inquiries = float(st.number_input(label='Num_Credit_Inquiries', value=5))
    data["Num_Credit_Inquiries"] = Num_Credit_Inquiries

with col2:
    Outstanding_Debt = float(st.number_input(label='Outstanding_Debt', value=809.98))
    data["Outstanding_Debt"] = Outstanding_Debt

with col3:
    Monthly_Inhand_Salary = float(st.number_input(label='Monthly_Inhand_Salary', value=1824.8))
    data["Monthly_Inhand_Salary"] = Monthly_Inhand_Salary

with col4:
    Monthly_Balance = float(st.number_input(label='Monthly_Balance', value=186.26))
    data["Monthly_Balance"] = Monthly_Balance

col1, col2, col3 = st.columns(3)

with col1:
    Amount_invested_monthly = float(st.number_input(label='Amount_invested_monthly', value=236.64))
    data["Amount_invested_monthly"] = Amount_invested_monthly

with col2:
    Total_EMI_per_month = float(st.number_input(label='Total_EMI_per_month', value=49.5))
    data["Total_EMI_per_month"] = Total_EMI_per_month

with col3:
    Credit_History_Age = float(st.number_input(label='Credit_History_Age', value=216))
    data["Credit_History_Age"] = Credit_History_Age

with st.expander("View the Raw Data"):
    st.dataframe(data=data, width=800, height=10)

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Credit Scoring: {}".format(prediction(new_data)))