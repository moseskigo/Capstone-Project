import streamlit as st
import joblib
import numpy as np

# Load your models and scaler
clustering_model = joblib.load('models/patient_clustering_model.pkl')
predictive_model = joblib.load('models/stacking_classifier_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Cluster profiles dictionary
cluster_profiles = {
    0: {
        'name': 'Wise Old Elephant',
        'image': 'images/elephant.jpg',
        'description': 'This cluster represents the oldest demographic, requiring close medical attention due to elevated cardiovascular risks.',
        'tips': [
            'Regular check-ups with your cardiologist.',
            'Monitor your cholesterol and blood pressure levels closely.',
            'Consider a low-sodium, heart-healthy diet.'
        ], 
    },
    1: {
        'name': 'Agile Gazelle',
        'image': 'images/gazelle.jpg',
        'description': 'This cluster features the youngest, healthiest individuals with a low current cardiovascular risk.',
        'tips': [
            'Maintain regular physical activity.',
            'Eat a balanced diet rich in fruits and vegetables.',
            'Regular screening for preventative care.'
        ],
    },
    2: {
        'name': 'Burdened Bear',
        'image': 'images/bear.jpg',
        'description': 'Members of this cluster have moderate health risks and might be dealing with weight and stress management issues.',
        'tips': [
            'Focus on stress reduction techniques.',
            'Moderate-intensity exercise most days of the week.',
            'Consult a dietitian to help manage diet and weight.'
        ],   
    },
    3: {
        'name': 'Speedy Cheetah',
        'image': 'images/cheetah.jpg',
        'description': 'Characterized by their rapid response to lifestyle changes, this group has potential for quick health improvements.',
        'tips': [
            'Engage in high-intensity interval training.',
            'Eat more protein to support muscle recovery.',
            'Get adequate sleep to enhance metabolic rate.'
        ],
    },
    4: {
        'name': 'Steady Horse',
        'image': 'images/horse.jpg',
        'description': 'Steady and resilient, this group benefits from consistent and balanced lifestyle habits.',
        'tips': [
            'Maintain a regular exercise routine.',
            'Include whole grains and fiber in your diet for digestive health.',
            'Regular mental health check-ins.'
        ],
    }
}


# Function to categorize blood pressure
def categorize_blood_pressure(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif (120 <= systolic < 130) and diastolic < 80:
        return "Elevated"
    elif (130 <= systolic < 140) or (80 <= diastolic < 90):
        return "High Blood Pressure (Hypertension) Stage 1"
    elif (140 <= systolic < 180) or (90 <= diastolic < 120):
        return "High Blood Pressure (Hypertension) Stage 2"
    elif systolic >= 180 or diastolic >= 120:
        return "Hypertensive Crisis (Consult your doctor immediately)"
    else:
        return "Unclassified"

# BMI Categorization Function
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obesity"

def categorize_pulse_pressure(pulse_pressure):
    if pulse_pressure < 40:
        return "Low (may indicate poor heart function)"
    elif 40 <= pulse_pressure <= 60:
        return "Normal"
    else:
        return "High (may indicate arterial stiffness or other cardiovascular conditions)"

# Streamlit interface
st.title("Health Risk Assessment Form")

with st.form("user_input_form"):
    st.write("Please fill out the following details to assess your health risk.")
    age = st.number_input('Age (years)', min_value=18, max_value=120)
    cholesterol = st.number_input('Total Cholesterol (mg/dL)', min_value=100.0, max_value=400.0)
    systolic_bp = st.number_input('Systolic Blood Pressure (mmHg)', min_value=80, max_value=250)
    diastolic_bp = st.number_input('Diastolic Blood Pressure (mmHg)', min_value=40, max_value=150)
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=50, max_value=300)
    weight = st.number_input('Weight (kg)', min_value=1.0, max_value=200.0)
    height = st.number_input('Height (meters)', min_value=0.1, max_value=2.5)
    submit_button = st.form_submit_button("Submit")

if submit_button:
    bmi = weight / (height ** 2)
    bmi_category = categorize_bmi(bmi)
    pulse_pressure = systolic_bp - diastolic_bp
    bp_status = categorize_blood_pressure(systolic_bp, diastolic_bp)
    features = [age, cholesterol, systolic_bp, diastolic_bp, glucose, bmi]
    cluster = clustering_model.predict(scaler.transform([features]))[0]
    risk_level = predictive_model.predict_proba(scaler.transform([features]))[0][1]*100
    profile = cluster_profiles[cluster]

    st.success("Submitted successfully!")
    st.image(profile['image'], caption=f"{profile['name']}")
    st.subheader(f"{profile['name']}")
    st.write(profile['description'])
    st.write("Health Tips:")
    for tip in profile['tips']:
        st.write(f"- {tip}")
    st.write(f"Your BMI: {bmi:.2f} ({bmi_category})")
    st.write(f"Pulse Pressure: {pulse_pressure}")
    st.write(f"Blood Pressure Status: {categorize_blood_pressure(systolic_bp, diastolic_bp)}")
    st.write(f"Risk Level: {risk_level:.2f}")
