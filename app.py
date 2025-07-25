import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="EMPLOYEE SALARY PREDICTION", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App 💼")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("👨‍💻 EMPLOYEE DETAILS")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Work", [
    "Private", "Self-emp-not-inc", "Local-gov", "State-gov",
    "Federal-gov", "Self-emp-inc"
])
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
education_num = st.sidebar.slider("Years of Experience", 1, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed",
    "Married-spouse-absent"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried","Other-relative"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country",[
    "United-States", "Cuba", "Jamaica", "India", "Mexico", "Puerto-Rico", "Honduras",
    "England", "Canada", "Germany", "Iran", "Philippines", "Poland", "Columbia", "Cambodia",
])

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame ({
    'Age': [age],
    'workclass' : [workclass],
    'education' : [education] ,
    'education_num' : [education_num] ,
    'marital_status' : [marital_status] ,
    'occupation' : [occupation] ,
    'relationship' : [relationship] ,
    'hours_per_week' : [hours_per_week] ,
    'native_country' : [native_country]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

