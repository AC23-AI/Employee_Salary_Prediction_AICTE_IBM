import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Load model and encoder mappings
model = joblib.load("best_model.pkl")
mappings = joblib.load("encoders.pkl")

st.set_page_config(page_title="EMPLOYEE SALARY PREDICTION", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App 💼")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")
st.sidebar.header("👨‍💻 EMPLOYEE DETAILS")


age = st.sidebar.slider("Age", 18, 65, 30)
education_num = st.sidebar.slider("Years of Experience", 1, 16, 10)
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
workclass_label = st.sidebar.selectbox("Work", list(mappings['workclass'].keys()))
education_label = st.sidebar.selectbox("Education Level", list(mappings['education'].keys()))
marital_status_label = st.sidebar.selectbox("Marital Status", list(mappings['marital_status'].keys()))
occupation_label = st.sidebar.selectbox("Job Role", list(mappings['occupation'].keys()))
relationship_label = st.sidebar.selectbox("Relationship", list(mappings['relationship'].keys()))
native_country_label = st.sidebar.selectbox("Native Country", list(mappings['native_country'].keys()))

# Convert labels to encoded numeric values
def encode(label, mapping):
    return mapping.get(label, mapping.get("Others", np.nan))

input_df = pd.DataFrame({
    'age': [age],
    'workclass': [encode(workclass_label, mappings['workclass'])],
    'education': [encode(education_label, mappings['education'])],
    'educational_num': [education_num],
    'marital_status': [encode(marital_status_label, mappings['marital_status'])],
    'occupation': [encode(occupation_label, mappings['occupation'])],
    'relationship': [encode(relationship_label, mappings['relationship'])],
    'hours_per_week': [hours_per_week],
    'native_country': [encode(native_country_label, mappings['native_country'])]
})

# Display input before prediction
st.write("### 🔎 Input Data")
st.write(input_df)

# Prediction
if st.button("Predict Salary Class"):
    if input_df.isnull().any().any():
        st.error("❌ Some inputs could not be mapped. Please select valid values.")
    else:
        prediction = model.predict(input_df)
        st.success(f"✅ Prediction: {'>50K' if prediction[0] == 1 else '<=50K'}")

# ---------------------- Batch Prediction ------------------------
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    for col in mappings:
        if col in batch_data.columns:
            batch_data[col] = batch_data[col].apply(lambda x: mappings[col].get(x, mappings[col].get("Others", np.nan)))

    if batch_data.isnull().any().any():
        st.warning("⚠️ Some values were not recognized and converted to NaN.")
    else:
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = ['>50K' if p == 1 else '<=50K' for p in batch_preds]
        st.write("✅ Predictions:")
        st.write(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

