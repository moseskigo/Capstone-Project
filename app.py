from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///health.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.config['SECRET_KEY'] = 'your_secret_key'  # Needed for session management and flash messages

# Load the models
clustering_model = joblib.load('patient_clustering_model.pkl')
predictive_model = joblib.load('stacking_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    cholesterol = db.Column(db.Float, nullable=False)
    blood_pressure_systolic = db.Column(db.Float, nullable=False)
    blood_pressure_diastolic = db.Column(db.Float, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    cluster = db.Column(db.Integer, nullable=False)
    pulse_pressure = db.Column(db.Float, nullable=True)
    risk_level = db.Column(db.Integer, nullable=True)

@app.before_request
def create_tables():
    db.create_all()

animal_profiles = {
    0: {'name': 'Wise Old Elephant', 'image': 'elephant.png', 'description': 'This cluster represents the oldest demographic, requiring close medical attention due to elevated cardiovascular risks.', 'tips': ['Regular check-ups with your cardiologist.', 'Monitor your cholesterol and blood pressure levels closely.', 'Consider a low-sodium, heart-healthy diet.']},
    1: {'name': 'Agile Gazelle', 'image': 'gazelle.png', 'description': 'This cluster features the youngest, healthiest individuals with a low current cardiovascular risk.', 'tips': ['Maintain regular physical activity.', 'Eat a balanced diet rich in fruits and vegetables.', 'Regular screening for preventative care.']},
    2: {'name': 'Burdened Bear', 'image': 'bear.png', 'description': 'Individuals in this cluster face significant metabolic challenges, highlighting the need for lifestyle changes.', 'tips': ['Adopt weight management strategies.', 'Increase physical activity to at least 150 minutes a week.', 'Consult with a dietitian to manage dietary needs.']},
    3: {'name': 'Cheetah Cub', 'image': 'cheetah.png', 'description': 'Despite being young, this cluster is at an emerging risk of cardiovascular issues, requiring preventive measures.', 'tips': ['Focus on maintaining a healthy weight.', 'Avoid smoking and limit alcohol consumption.', 'Monitor your blood pressure regularly.']},
    4: {'name': 'Seasoned Horse', 'image': 'horse.png', 'description': 'This cluster reflects an older demographic that manages their health proactively, often maintaining good health despite their age.', 'tips': ['Continue adherence to prescribed medications.', 'Stay engaged with regular physical activity tailored to your abilities.', 'Maintain regular health screenings and check-ups.']}
}

def calculate_bmi(weight, height):
    return weight / (height / 100)**2

def calculate_pulse_pressure(systolic, diastolic):
    return (systolic * 7.5) - (diastolic * 7.5)

def categorize_age(age):
    if age < 40:
        return 1
    elif 40 <= age < 45:
        return 2
    elif 45 <= age < 50:
        return 3
    elif 50 <= age < 55:
        return 4
    elif 55 <= age < 60:
        return 5
    elif 60 <= age < 65:
        return 6
    else:
        return 7

def categorize_bmi(bmi):
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
        return 1
    elif 120 <= systolic <= 129 and diastolic < 80:
        return 2
    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
        return 3
    elif systolic >= 140 or diastolic >= 90:
        return 4
    elif systolic > 180 or diastolic > 120:
        return 5
    else:
        return 0

def categorize_pulse_pressure(systolic, diastolic):
    pulse_pressure = systolic - diastolic
    if pulse_pressure <= 40:
        return 1
    elif pulse_pressure <= 60:
        return 2
    else:
        return 3


def scale_features(features):
    return scaler.transform(np.array(features).reshape(1, -1))

def assign_cluster(features):
    features_scaled = scale_features(features)
    return clustering_model.predict(features_scaled)[0]

def predict_risk_level(features):
    features_scaled = scale_features(features)
    return predictive_model.predict(features_scaled)[0]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        data = request.form
        try:
            age = int(data['age'])
            cholesterol = float(data['cholesterol'])
            systolic_bp = float(data['blood_pressure_systolic'])
            diastolic_bp = float(data['blood_pressure_diastolic'])
            glucose = float(data['glucose'])
            weight = float(data['weight'])
            height = float(data['height'])

            # Validate input data
        if not (age and weight and height and cholesterol and glucose):
            flash('All fields are required.')
            return render_template('signup.html')

        age = int(age)
        weight = float(weight)
        height = float(height)
        cholesterol = float(cholesterol)
        glucose = float(glucose)

        if not (18 <= age <= 120):
            flash('Invalid age. Must be between 18 and 120.')
            return render_template('signup.html')

        if not (30 <= weight <= 300):
            flash('Invalid weight. Must be between 30 kg and 300 kg.')
            return render_template('signup.html')

        if not (100 <= height <= 250):
            flash('Invalid height. Must be between 100 cm and 250 cm.')
            return render_template('signup.html')

            # Calculate categories
            age_category = categorize_age(age)
            bmi = calculate_bmi(weight, height)
            bmi_category = categorize_bmi(bmi)
            bp_category = categorize_blood_pressure(systolic_bp, diastolic_bp)
            pulse_pressure = calculate_pulse_pressure(systolic_bp, diastolic_bp)
            pulse_pressure_category = categorize_pulse_pressure(systolic_bp, diastolic_bp)

            # Prepare features for clustering and risk prediction
            features = [age_category, bmi_category, bp_category, pulse_pressure_category, cholesterol, glucose]
            cluster = assign_cluster(features)
            risk_level = predict_risk_level(features)

            # Save user data with categories
            user = User(
                age=age,
                cholesterol=cholesterol,
                blood_pressure_systolic=systolic_bp,
                blood_pressure_diastolic=diastolic_bp,
                glucose=glucose,
                bmi=bmi,
                pulse_pressure=pulse_pressure,
                cluster=cluster,
                risk_level=risk_level
            )
            db.session.add(user)
            db.session.commit()

            return redirect(url_for('dashboard', user_id=user.id))
        except ValueError:
            return "Invalid input", 400

    return render_template('signup.html')


@app.route('/dashboard/<int:user_id>')
def dashboard(user_id):
    user = User.query.filter_by(id=user_id).first_or_404()
    profile = animal_profiles[user.cluster]
    return render_template('dashboard.html', user=user, profile=profile)


if __name__ == '__main__':
    app.run(debug=True)