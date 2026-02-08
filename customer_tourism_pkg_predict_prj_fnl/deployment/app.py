import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="ravikmrg6/Tourism-pkg-prediction", filename="Tourismpkg_predict.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism package Prediction")
st.write("""
This applicationpredicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them
""")

# User input

TypeofContact = st.selectbox("Company Invited or Self Enquiry", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("The City category", ["Tier1", "Tier2", "Tier3"])
Occupation = st.selectbox("Customer Occupation", ["Salaried", "Freelancer"])
Gender = st.selectbox("Gender of the customer", ["Male", "Female"])
MartialStatus = st.selectbox("Martial status of the customer", ["Single", "Married","Divorced"])
Designation = st.selectbox("Designation of the customer", ["Executive", "Managerial","Professional","Self-Employed"])
ProductPitched = st.selectbox("The type of the product pitched to the Customer",["Basic", "Standard","King","Deluxe","Super Deluxe"] )

Age = st.number_input("Customer Age ", min_value=1.0, max_value=120.0, value=18.0, step=0.1)
NumberOfPersonVisiting = st.number_input("Total num of people accompanying the customer", min_value=1, max_value=1000, value=1, step=1)
PreferredPropertyStar = st.number_input("Preferred hotel rating y the customer",min_value=1, max_value=10, value=1, step=1)
NumberOfTrips = st.number_input("Avg num of trips the customer takes annually",min_value=1, max_value=10, value=1, step=1)
Passport = st.number_input("Passport - Yes-1 or No-0",value=0)
OwnCar = st.number_input("Owncar - Yes-1 or No-0",value=0)
NumberOfChildrenVisiting = st.number_input("Total num of childern (<5 years) accompanying the customer", min_value=0, max_value=5, value=0, step=1)
MonthlyIncome = st.number_input("Gross monthly income of the customer", value=10000)
PitchSatisfactionScore = st.number_input("Customer satisfaction score", min_value=1, max_value=10, value=1, step=1)
NumberOfFollowups = st.number_input("The total number of follow ups by salesperson", value=1)
DurationOfPitch = st.number_input("Duration of sales pitch delivered to the customer", value=1)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'MartialStatus': MartialStatus,
    'Designation': Designation,
    'Age': Age,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome
}])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Yes" if prediction ==1 else "No"
    st.subheader("Prediction Result:")
    st.success(f": Modle predict :: Customer Prod taken **{result}**")
