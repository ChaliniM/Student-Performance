import streamlit as st
st.set_page_config(page_title="🎯 Student Performance Predictor", layout="centered")
import pandas as pd
import joblib
import numpy as np
# Inject custom CSS
with open(r"C:\Users\chali\OneDrive\Desktop\New folder\c++\StudentPerfo\style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model pipeline
model_pipeline = joblib.load(r'C:\Users\chali\OneDrive\Desktop\New folder\c++\StudentPerfo\model_pipeline.pkl')


st.title("🎓 Predict Student Performance")

# Inputs (matching training columns)
school = st.selectbox("School", ['GP', 'MS'])
sex = st.selectbox("Gender", ['F', 'M'])
age = st.slider("Age", 15, 22, 17)
address = st.selectbox("Address", ['U', 'R'])
famsize = st.selectbox("Family Size", ['LE3', 'GT3'])
Pstatus = st.selectbox("Parental Cohabitation Status", ['T', 'A'])
Medu = st.slider("Mother's Education", 0, 4, 2)
Fedu = st.slider("Father's Education", 0, 4, 2)
Mjob = st.selectbox("Mother's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
Fjob = st.selectbox("Father's Job", ['teacher', 'health', 'services', 'at_home', 'other'])
reason = st.selectbox("Reason to choose school", ['home', 'reputation', 'course', 'other'])
guardian = st.selectbox("Guardian", ['mother', 'father', 'other'])
traveltime = st.slider("Travel Time (1–4)", 1, 4, 1)
studytime = st.slider("Study Time (1–4)", 1, 4, 2)
failures = st.slider("Past Class Failures", 0, 3, 0)
schoolsup = st.selectbox("School Support", ['yes', 'no'])
famsup = st.selectbox("Family Support", ['yes', 'no'])
paid = st.selectbox("Paid Classes", ['yes', 'no'])
activities = st.selectbox("Extracurricular Activities", ['yes', 'no'])
nursery = st.selectbox("Attended Nursery School", ['yes', 'no'])
higher = st.selectbox("Wants Higher Education", ['yes', 'no'])
internet = st.selectbox("Internet Access", ['yes', 'no'])
romantic = st.selectbox("In a Romantic Relationship", ['yes', 'no'])
famrel = st.slider("Family Relationship Quality", 1, 5, 4)
freetime = st.slider("Free Time After School", 1, 5, 3)
goout = st.slider("Going Out with Friends", 1, 5, 3)
Dalc = st.slider("Workday Alcohol Use", 1, 5, 1)
Walc = st.slider("Weekend Alcohol Use", 1, 5, 2)
health = st.slider("Current Health Status", 1, 5, 3)
absences = st.slider("Absences", 0, 93, 4)
G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
G3 = st.slider("Final Grade (G3)", 0, 20, 10)

# Derived features (same as training preprocessing)
LogAbsences = np.log1p(absences)
TotalAlc = Dalc + Walc
AgeGroup = "Teen" if age <= 17 else "Adult"
StudytimeXFailures = studytime * failures
AbsenceCategory = "Low" if absences <= 5 else "High"

# Create input DataFrame
input_dict = {
    'school': [school],
    'sex': [sex],
    'age': [age],
    'address': [address],
    'famsize': [famsize],
    'Pstatus': [Pstatus],
    'Medu': [Medu],
    'Fedu': [Fedu],
    'Mjob': [Mjob],
    'Fjob': [Fjob],
    'reason': [reason],
    'guardian': [guardian],
    'traveltime': [traveltime],
    'studytime': [studytime],
    'failures': [failures],
    'schoolsup': [schoolsup],
    'famsup': [famsup],
    'paid': [paid],
    'activities': [activities],
    'nursery': [nursery],
    'higher': [higher],
    'internet': [internet],
    'romantic': [romantic],
    'famrel': [famrel],
    'freetime': [freetime],
    'goout': [goout],
    'Dalc': [Dalc],
    'Walc': [Walc],
    'health': [health],
    'absences': [absences],
    'G1': [G1],
    'G2': [G2],
    'G3': [G3],
    'LogAbsences': [LogAbsences],
    'TotalAlc': [TotalAlc],
    'AgeGroup': [AgeGroup],
    'StudytimeXFailures': [StudytimeXFailures],
    'AbsenceCategory': [AbsenceCategory]
}

input_df = pd.DataFrame(input_dict)

# Predict
if st.button("Predict Performance"):
    try:
        prediction = model_pipeline.predict(input_df)[0]
        st.success(f"📊 Predicted Student Performance: **{prediction}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
