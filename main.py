import pandas as pd                              # import pandas library for loadin and processing data
import matplotlib.pyplot as plt                  # matplotlib is used for data visualisation
import joblib 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st

# -----------------------------------------LOADING DATASET---------------------------------------------------------------------------------
data=pd.read_csv(r"C:\Users\Dell\Downloads\Telegram Desktop\adult 3_.csv")     # Read adult3_.csv file
data.head(10)                                    # head(n) used to get initial n rows
data.tail(3)                                     # tail(m) is used to get m rowa from end

#null values
data.isna().sum()                                #mean median mode arbitrary
data.age.value_counts()                          # The .value_counts() method  is used to get a count of the unique values in a column
data.workclass.value_counts()

data.workclass.replace({'?':'Others'},inplace=True) # .replace() is used to replace certain values in a column
data=data[data['workclass']!='Without-pay']      # removing unqiue values of a column which are not contribute in prediction made by model
data=data[data['workclass']!='Never-worked']
data['workclass'].value_counts()

data['occupation'].value_counts()
data.occupation.replace({'?':'Others'},inplace=True)
data=data[data['occupation']!='Armed-Forces']
data['occupation'].value_counts()
data.educational_num.value_counts()
data.relationship.value_counts()
data.education.value_counts()
data.marital_status.value_counts()
data['hours_per_week'].value_counts()
data['native_country'].value_counts()
data.native_country.replace({'?':'Others'},inplace=True)
data['native_country'].value_counts()



#---------------------------------------------OUTLIER DETECTION-------------------------------------------

plt.boxplot(data['age'])                            # .boxplot() provides a quick way to understand the central tendency, spread, and potential outliers of the data through a box and whiskers plot.
plt.show()                                          #.show() displays the plots that have been created.

data=data[(data['age']<=75)&(data['age']>=17)]      # To remove outliers
plt.boxplot(data['age'])
plt.show()

plt.boxplot(data['hours_per_week'])
plt.show()

plt.boxplot(data['educational_num'])
plt.show()

data=data[(data['educational_num']<=16)&(data['educational_num']>=6)]         # To remove outliers
plt.boxplot(data['educational_num'])
plt.show()



#----------------------------------------------LABEL ENCODING -----------------------------------------------------

 
encoder=LabelEncoder()                      
data['workclass']=encoder.fit_transform(data['workclass'])
data['marital_status']=encoder.fit_transform(data['marital_status'])   
data['occupation']=encoder.fit_transform(data['occupation'])
data['relationship']=encoder.fit_transform(data['relationship'])      
data['education'] = encoder.fit_transform(data['education'])
data['native_country']=encoder.fit_transform(data['native_country'])

encoder = LabelEncoder()

def encode_and_store(column):
    encoder.fit(data[column])
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    data[column] = data[column].map(mapping)
    return mapping

# Store encodings
mappings = {
    'workclass': encode_and_store('workclass'),
    'marital_status': encode_and_store('marital_status'),
    'occupation': encode_and_store('occupation'),
    'relationship': encode_and_store('relationship'),
    'education': encode_and_store('education'),
    'native_country': encode_and_store('native_country')
}

# Save mappings
joblib.dump(mappings, "encoders.pkl")


x=data.drop(columns=['income'])
y=data['income']

#-------------------------IMPORTING AND COMPARING MACHINE LEARNING ALGORITHMS----------------------------

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "DecisionTree": DecisionTreeClassifier(),
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

plt.bar(results.keys(), results.values(), color='lime')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison', color ='blue')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

# Save the best model
joblib.dump(best_model, "best_model.pkl")
 


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
    'age': [age],
    'workclass' : [workclass],
    'education' : [education] ,
    'educational_num' : [education_num] ,
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


