import streamlit as st
import joblib
import numpy as np
import json
import os

# Load model and encoders
model = joblib.load("animal_disease_model.pkl")
encoder = joblib.load("symptom_animal_encoder.pkl")
label_encoder = joblib.load("disease_encoder.pkl")

# File to store user credentials
CREDENTIALS_FILE = "users.json"

# Load credentials
def load_credentials():
    if os.path.exists(CREDENTIALS_FILE):
        with open(CREDENTIALS_FILE, "r") as file:
            return json.load(file)
    return {}

# Save credentials
def save_credentials(credentials):
    with open(CREDENTIALS_FILE, "w") as file:
        json.dump(credentials, file)

# Authenticate user
def authenticate(email, password, credentials):
    return credentials.get(email) == password

# Register user
def register(email, password, credentials):
    if email in credentials:
        return False
    credentials[email] = password
    save_credentials(credentials)
    return True

# Set Page Config
st.set_page_config(page_title="Animal Disease Predictor", layout="centered", page_icon="🐾")

# Sidebar
def sidebar():
    st.sidebar.image("1.jpg", caption="Animal Health Matters 🐾", use_column_width=True)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict Disease", "Logout"])

    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About Project:**  
    🐶 Predict diseases based on animal symptoms.  
    🧑‍⚕️ Early detection can save lives.
    """)
    return page

# Home/Login/Register Page
def login_register():
    credentials = load_credentials()
    st.title("🐾 Animal Disease Prediction System")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        auth_choice = st.radio("Select Action", ["Login", "Register"])

        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if auth_choice == "Login":
            if st.button("Login"):
                if authenticate(email, password, credentials):
                    st.session_state.logged_in = True
                    st.success("Login successful!")
                    st.rerun()
                    st.experimental_rerun()
                else:
                    st.error(" Invalid email or password.")

        elif auth_choice == "Register":
            if st.button("Register"):
                if register(email, password, credentials):
                    st.success(" Registration successful! Please log in.")
                else:
                    st.warning(" Email already registered.")

# Prediction Page
def prediction_page():
    st.image("1.jpg", use_column_width=True)
    st.title("🔎 Predict Animal Disease")

    animal = st.selectbox("Select Animal", [
        "Dog", "Cat", "Cow", "Goat", "Horse", "Rabbit", "Chicken", "Cattle", "Turtle", "Hamster",
        "Lion", "Fox", "Monkey", "Birds", "Sheep", "Pig", "Duck", "Snake", "Donkey", "Elephant",
        "Moose", "Tiger", "Buffalo", "Wolves", "Hyenas", "Others"
    ])

    s1 = st.text_input("Symptom 1")
    s2 = st.text_input("Symptom 2")
    s3 = st.text_input("Symptom 3")
    s4 = st.text_input("Symptom 4")
    s5 = st.text_input("Symptom 5")

    if st.button("Predict Disease"):
        if not all([s1, s2, s3, s4, s5]):
            st.warning("⚠️ Please enter all 5 symptoms.")
        else:
            features = [[s1, s2, s3, s4, s5, animal]]
            X = encoder.transform(features)
            pred_label = model.predict(X)[0]
            disease = label_encoder.inverse_transform([pred_label])[0]
            st.success(f"🩺 Predicted Disease: **{disease}**")

# Main App Controller
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login_register()
else:
    page = sidebar()

    if page == "Home":
        st.image("image(1).jpg", use_column_width=True)
        st.title("🐕 Welcome to Animal Disease Predictor!")
        st.markdown("""
        ### 🐾 Features:
        - Predict animal diseases based on 5 symptoms.
        - Fast and accurate detection.
        - Immediate veterinary recommendation.
        """)
    elif page == "Predict Disease":
        prediction_page()
    elif page == "Logout":
        st.session_state.logged_in = False
        st.success("✅ Logged out successfully!")
        
