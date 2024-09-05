import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import sklearn.metrics
import base64

# Function to get base64 string for an image
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Base64 image string for background
background_image = get_base64_encoded_image('images/background.jpg')


# Set page config with an appropriate icon
st.set_page_config(
    page_title="Cardiovascular Risk Checker",
    layout="wide",  # Change layout to wide
    page_icon="💓"
)


# Embedding the CSS to style the tabs and the background
st.markdown(
    f"""
    <style>
    /* General styling for the background */
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:image/jpeg;base64,{background_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white !important; /* Force text color to be white */
    }}

    /* Remove padding around the tabs container */
    .stTabs {{
        margin-top: -5px !important; /* Adjust this value based on how much the tabs are pushed down */
    }}

    /* Custom styling for tabs */
    .stTabs [role="tablist"] > div {{
        border-radius: 8px;
        padding: 10px;
        font-size: 1000px !important;
        font-weight: bold !important;
        color: #333 !important; /* Default text color */
        background-color: rgba(240, 240, 240, 0.08) !important; /* Make background more transparent */
        border: 2px solid rgba(204, 204, 204, 0.08); /* Make border more transparent */
        margin-right: 5px;
        margin-bottom: 10px;
        text-align: center;
        width: 70px;  /* Increase the tab width */
        transition: background-color 0.3s ease, color 0.3s ease; /* Smooth transition for color */
        z-index: 1;  /* Ensure the tab stays on top */
    }}

    /* Hover effect for tabs */
    .stTabs [role="tablist"] > div:hover {{
        background-color: rgba(230, 230, 230, 0.8) !important;
    }}

    /* Active tab styling */
    .stTabs [role="tablist"] > div[aria-selected="true"] {{
        background-color: rgba(220, 220, 220, 0.9) !important; /* Slightly transparent background */
        color: black !important; /* Change text color to black when active */
        border-color: #888;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1); /* Softer shadow */
        z-index: 2; /* Ensure the active tab is on top */
        position: relative;
    }}

    /* Disclaimer box styling */
    .disclaimer-box {{
        background-color: rgba(248, 215, 218, 0.8);
        padding: 20px;
        margin-top: 10px;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 800px;
        text-align: left;
    }}

    .disclaimer-box h4 {{
        color: black;
        margin-bottom: 10px;
    }}

    .disclaimer-box p {{
        color: black;
        text-align: justify;
    }}
    </style>

    <!-- Disclaimer box -->
    <div class='disclaimer-box'>
        <h4>Disclaimer</h4>
        <p>This tool is a student data science project and is <strong>not</strong> a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Create navigation tabs
tabs = st.tabs(["🏠 HOME", "📄 ABOUT", "👥 TEAM"])

with tabs[0]:
    st.title("Welcome to the Cardiovascular Risk Checker")
    st.write("This tool helps you assess your health risk and provides personalized health tips based on your profile.")
    st.write("Please enter your information in the fields on the left and click 'Check Profile' to get started.")

    # Load the models and scaler
    try:
        clustering_model = joblib.load('models/patient_clustering_model.pkl')
        predictive_model = joblib.load('models/stacking_classifier_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except Exception as e:
        st.error(f"Failed to load models or scaler: {e}")
        st.stop()

    # Function to scale features
    def scale_features(features, scaler):
        return scaler.transform(features)

    # BMI Category Descriptions
    category_descriptions = {
        1: "Underweight",
        2: "Normal weight",
        3: "Overweight",
        4: "Obese"
    }

    # Define categorization functions
    def categorize_age_years(age_years):
        if age_years < 40:
            return 1
        elif 40 <= age_years < 45:
            return 2
        elif 45 <= age_years < 50:
            return 3
        elif 50 <= age_years < 55:
            return 4
        elif 55 <= age_years < 60:
            return 5
        elif 60 <= age_years < 65:
            return 6
        else:
            return 7

    def categorize_bmi(weight, height):
        bmi = round(weight / ((height / 100) ** 2), 2)
        if bmi < 18.5:
            return 1
        elif 18.5 <= bmi < 25:
            return 2
        elif 25 <= bmi < 30:
            return 3
        else:
            return 4

    def categorize_blood_pressure(systolic, diastolic):
        if systolic < 120 and diastolic < 80:
            return 1, "Normal"
        elif 120 <= systolic <= 129 and diastolic < 80:
            return 2, "Elevated"
        elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
            return 3, "High Blood Pressure (Stage 1)"
        elif systolic >= 140 or diastolic >= 90:
            return 4, "High Blood Pressure (Stage 2)"
        elif systolic > 180 or diastolic > 120:
            return 5, "Hypertensive Crisis"
        else:
            return 0, "Uncategorized"

    def categorize_pulse_pressure(pulse_pressure):
        if pulse_pressure <= 40:
            return 1, "Normal"
        elif pulse_pressure <= 60:
            return 2, "Elevated"
        else:
            return 3, "High"

    # Cluster profiles dictionary
    cluster_profiles = {
        0: {'name': 'Wise Old Elephant', 'image': 'images/elephant.jpg', 'description': 'You might require close medical attention due to the elevated cardiovascular risks.', 'tips': ['Regular check-ups with your cardiologist.', 'Monitor your cholesterol and blood pressure levels closely.', 'Consider a low-sodium, heart-healthy diet.']},
        1: {'name': 'Agile Gazelle', 'image': 'images/gazelle.jpg', 'description': 'You are the youngest and healthiest with a low current cardiovascular health risk.', 'tips': ['Maintain regular physical activity.', 'Eat a balanced diet rich in fruits and vegetables.', 'Regular screening for preventative care.']},
        2: {'name': 'Burdened Bear', 'image': 'images/bear.jpg', 'description': 'You have moderate cardio health risks and might be dealing with weight and stress management issues.', 'tips': ['Focus on stress reduction techniques.', 'Moderate-intensity exercise most days of the week.', 'Consult a dietitian to help manage diet and weight.']},
        3: {'name': 'Speedy Cheetah', 'image': 'images/cheetah.jpg', 'description': 'Characterized by your rapid response to lifestyle changes, you have potential for quick health improvements.', 'tips': ['Engage in high-intensity interval training.', 'Eat more protein to support muscle recovery.', 'Get adequate sleep to enhance metabolic rate.']},
        4: {'name': 'Steady Horse', 'image': 'images/horse.jpg', 'description': 'Steady and resilient, you benefit from consistent and balanced lifestyle habits.', 'tips': ['Maintain a regular exercise routine.', 'Include whole grains and fiber in your diet for digestive health.', 'Regular mental health check-ins.']}
    }

    # Organize input fields into two columns
    col1, col2 = st.columns(2)

    with col1:
        age_years = st.number_input("Enter your age in years", min_value=0, max_value=120, value=30, step=1, help="Hover over the input field for help")
        weight = st.number_input("Enter your weight in kilograms", min_value=1.0, max_value=200.0, step=0.1, value=70.0, help="Hover over the input field for help")
        height = st.number_input("Enter your height in centimeters", min_value=50.0, max_value=250.0, step=0.1, value=175.0, help="Hover over the input field for help")

    with col2:
        systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=40, max_value=250, step=1, value=110, help="Hover over the input field for help")
        diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, step=1, value=70, help="Hover over the input field for help")
        cholesterol = st.selectbox("Cholesterol Level", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x], help="Hover over the input field for help")
        gluc = st.selectbox("Glucose Level", options=[1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x], help="Hover over the input field for help")

    submit = st.button('Check Profile')

    if submit:
        try:
            # Categorizations
            age_category = categorize_age_years(age_years)
            bmi_category = categorize_bmi(weight, height)
            bp_category, bp_description = categorize_blood_pressure(systolic_bp, diastolic_bp)
            pulse_pressure = systolic_bp - diastolic_bp
            pulse_pressure_category, pulse_pressure_description = categorize_pulse_pressure(pulse_pressure)

            # Prepare features for prediction
            features = np.array([[age_category, bmi_category, bp_category, pulse_pressure_category, cholesterol, gluc]])

            # Scale features
            scaled_features = scale_features(features, scaler)

            # Predict health risk
            health_risk_prediction = predictive_model.predict(features)

            # Predict cluster
            cluster = clustering_model.predict(scaled_features)
            cluster_info = cluster_profiles[cluster[0]]

            # Display results in two columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("<div class='prediction-column'>", unsafe_allow_html=True)
                st.subheader("Health Risk Prediction")
                health_risk_messages = {0: "Your health risk is low. Keep up the good work!", 
                                        1: "Your health risk is moderate. It's important to take proactive steps to improve your health.", 
                                        2: "Your health risk is high. It's crucial to take immediate action to improve your health."}
                st.write(health_risk_messages.get(health_risk_prediction[0], "Risk level unknown."))
                
                # Display categorized results
                st.write(f"Your BMI category is: {bmi_category}, indicating {category_descriptions[bmi_category]}.")
                st.write(f"Your Blood Pressure Category is: {bp_category}, indicating {bp_description}.")
                st.write(f"Your Pulse Pressure Category is: {pulse_pressure_category}, indicating {pulse_pressure_description}.")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='health-risk-column'>", unsafe_allow_html=True)
                st.subheader("Health Risk Profile")
                # Display cluster information in this column
                st.image(cluster_info['image'], caption=cluster_info['name'], use_column_width=True)
                st.write(cluster_info['description'])
                st.write("Here are some health tips based on your health profile:")
                for tip in cluster_info['tips']:
                    st.write(f"- {tip}")
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while processing your request: {e}")

with tabs[1]:
    st.title("About the App")
    st.write("This app is designed to help users assess their health risk and receive personalized health tips")
    st.write("based on their profile. Here's what the app can currently do:")
    st.write("- Accept user input for age, weight, height, blood pressure, cholesterol, and glucose levels.")
    st.write("- Categorize input data and provide feedback on BMI, blood pressure, and pulse pressure.")
    st.write("- Predict the user's health risk and cluster based on machine learning models.")
    st.write("- Display personalized health tips based on the predicted cluster.")

with tabs[2]:
    st.title("Meet the Team")
    team_members = [
        ("Moses Kigo", "moses.wanja@student.moringaschool.com"),
        ("Erik Lekishon", "eric.lekishon@student.moringaschool.com"),
        ("Josephine Gathenya", "josephine.wanjiru@student.moringaschool.com"),
        ("Chepkemoi Chepkemoi", "chepkemoi.chepkemoi@student.moringaschool.com"),
        ("Eunita Nyengo", "eunita.nyengo@student.moringaschool.com")
    ]
    for member, email in team_members:
        st.write(f"- **{member}**: [{email}](mailto:{email})")
