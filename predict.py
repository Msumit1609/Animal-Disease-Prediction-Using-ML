import joblib

# Load model and encoders
model = joblib.load("animal_disease_model.pkl")
symptom_animal_encoder = joblib.load("symptom_animal_encoder.pkl")
disease_encoder = joblib.load("disease_encoder.pkl")

# New input
sample_symptoms = ["High fever","Cough with phlegm","Confusion","Chills","Fatigue"]
animal = "Lion"

# Prepare input
input_data = symptom_animal_encoder.transform([sample_symptoms + [animal]])
predicted_class = model.predict(input_data)

# Decode predicted label
predicted_disease = disease_encoder.inverse_transform(predicted_class)

print("🔮 Predicted Disease:", predicted_disease[0])
