import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Title
st.title("AI-Powered Personalized Fitness, Wellness, and Health Tracker")

# User input form
with st.form(key='user_data'):
    st.header("Input Your Daily Data")

    # Physical Activity
    steps = st.number_input("Steps Taken", min_value=0)
    workout_type = st.selectbox("Workout Type", ["None", "Cardio", "Strength", "Flexibility"])
    workout_duration = st.number_input("Workout Duration (in minutes)", min_value=0)

    # Sleep and Rest
    sleep_hours = st.number_input("Hours of Sleep", min_value=0, max_value=24)

    # Diet and Nutrition
    meals = st.text_input("Meals Consumed (comma separated)")
    sugar_intake = st.number_input("Sugar Intake (in grams)", min_value=0)
    water_intake = st.number_input("Water Intake (in liters)", min_value=0.0)

    # Diabetes Indicators
    glucose_level = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300)
    insulin_sensitivity = st.selectbox("Insulin Sensitivity", ["Normal", "Impaired", "Diabetic"])
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0)

    # Heart Health
    resting_heart_rate = st.number_input("Resting Heart Rate (bpm)", min_value=30, max_value=120)
    blood_pressure_sys = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200)
    blood_pressure_dia = st.number_input("Diastolic Blood Pressure", min_value=50, max_value=130)

    # Mental Health
    stress_level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Stressed"])

    # Submit button
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        meals_list = meals.split(',')
        st.write("Thank you for submitting your data!")
        st.write(f"Steps: {steps}, Workout: {workout_type}, Duration: {workout_duration} min, Meals: {meals_list}, "
                 f"Sugar Intake: {sugar_intake}g, Water Intake: {water_intake}L, Sleep: {sleep_hours} hours")
        st.write(f"Glucose Level: {glucose_level} mg/dL, Insulin Sensitivity: {insulin_sensitivity}, BMI: {bmi}")
        st.write(f"Resting Heart Rate: {resting_heart_rate} bpm, Blood Pressure: {blood_pressure_sys}/{blood_pressure_dia}")
        st.write(f"Stress Level: {stress_level}, Mood: {mood}")

# Feature Engineering Function
def preprocess_data(steps, workout_type, workout_duration, sleep_hours, sugar_intake, water_intake, glucose_level,
                    insulin_sensitivity, bmi, resting_heart_rate, blood_pressure_sys, blood_pressure_dia,
                    stress_level, mood):
    # Example feature engineering
    features = {
        'steps': steps,
        'workout_duration': workout_duration,
        'sleep_hours': sleep_hours,
        'sugar_intake': sugar_intake,
        'water_intake': water_intake,
        'glucose_level': glucose_level,
        'insulin_sensitivity': 1 if insulin_sensitivity == "Diabetic" else 0,
        'bmi': bmi,
        'resting_heart_rate': resting_heart_rate,
        'blood_pressure_sys': blood_pressure_sys,
        'blood_pressure_dia': blood_pressure_dia,
        'stress_level': 2 if stress_level == "High" else 1 if stress_level == "Moderate" else 0,
        'mood': 2 if mood == "Stressed" else 1 if mood == "Neutral" else 0,
        'workout_type': 1 if workout_type != "None" else 0  # Binary encoding for workout
    }
    return pd.DataFrame([features])

# Dummy Dataset Creation (replace with real data later)
data = pd.DataFrame({
    'steps': np.random.randint(1000, 20000, 100),
    'workout_duration': np.random.randint(0, 120, 100),
    'sleep_hours': np.random.uniform(4, 10, 100),
    'sugar_intake': np.random.uniform(10, 50, 100),
    'water_intake': np.random.uniform(1, 5, 100),
    'glucose_level': np.random.uniform(70, 200, 100),
    'insulin_sensitivity': np.random.randint(0, 2, 100),
    'bmi': np.random.uniform(18, 35, 100),
    'resting_heart_rate': np.random.uniform(50, 100, 100),
    'blood_pressure_sys': np.random.uniform(90, 150, 100),
    'blood_pressure_dia': np.random.uniform(60, 100, 100),
    'stress_level': np.random.randint(0, 3, 100),
    'mood': np.random.randint(0, 3, 100),
    'weight_loss': np.random.uniform(-5, 5, 100),  # Target variable
    'diabetes_risk': np.random.randint(0, 2, 100),  # Binary target for diabetes risk
    'heart_health': np.random.randint(0, 2, 100),   # Binary target for heart health
    'workout_type': np.random.randint(0, 2, 100),   # Added workout_type to match features
})

# Splitting data for training and testing
X = data.drop(['weight_loss', 'diabetes_risk', 'heart_health'], axis=1)
y_weight = data['weight_loss']
y_diabetes = data['diabetes_risk']
y_heart = data['heart_health']

# Train the models
weight_loss_model = RandomForestRegressor()
diabetes_model = RandomForestClassifier()
heart_health_model = RandomForestClassifier()

weight_loss_model.fit(X, y_weight)
diabetes_model.fit(X, y_diabetes)
heart_health_model.fit(X, y_heart)

# Prediction Functions
def predict_weight_loss(features):
    return weight_loss_model.predict(features)

def predict_diabetes_risk(features):
    return diabetes_model.predict(features)

def predict_heart_health(features):
    return heart_health_model.predict(features)

# Make Predictions after form submission
if submit_button:
    features = preprocess_data(steps, workout_type, workout_duration, sleep_hours, sugar_intake, water_intake, glucose_level,
                               insulin_sensitivity, bmi, resting_heart_rate, blood_pressure_sys, blood_pressure_dia,
                               stress_level, mood)

    weight_loss_prediction = predict_weight_loss(features)
    diabetes_risk_prediction = predict_diabetes_risk(features)
    heart_health_prediction = predict_heart_health(features)

    # Display predictions
    st.write(f"Predicted Weight Loss: {weight_loss_prediction[0]:.2f} kg")
    if diabetes_risk_prediction[0] == 1:
        st.warning("High Risk of Diabetes. Consult a healthcare provider.")
    else:
        st.success("Low Risk of Diabetes.")

    if heart_health_prediction[0] == 1:
        st.warning("Heart Health Concern. Monitor your health.")
    else:
        st.success("Heart Health is in good condition.")
