import pandas as pd
import random

# Define disease to symptoms mapping
disease_symptoms = {
    "Flu": [
        "Fever", "Chills", "Muscle or body aches", "Fatigue", "Dry Cough",
        "Sore throat", "Runny nose", "Headache", "Vomiting", "Diarrhea"
    ],
    "Cold": [
        "Runny nose", "Sneezing", "Sore throat", "Mild cough",
        "Mild fatigue", "Mild fever"
    ],
    "Chikungunya": [
        "High fever", "Severe joint pain", "Muscle pain", "Headache", "Rash", "Fatigue", "Nausea"
    ],
    "Malaria": [
        "High fever with chills", "Sweating", "Headache", "Nausea", "Vomiting",
        "Muscle pain", "Fatigue", "Anemia", "Jaundice"
    ],
    "Dengue": [
        "Sudden high fever", "Severe headache", "Pain behind the eyes", "Muscle and joint pain",
        "Rash", "Mild bleeding", "Fatigue", "Dengue Hemorrhagic Fever"
    ],
    "COVID-19": [
        "Fever", "Chills", "Dry cough", "Shortness of breath", "Fatigue",
        "Muscle aches", "Loss of taste or smell", "Sore throat", "Congestion", "Nausea", "Diarrhea"
    ],
    "Typhoid": [
        "Prolonged high fever", "Weakness", "Fatigue", "Headache", "Abdominal pain",
        "Diarrhea", "Constipation", "Rose-colored spots"
    ],
    "Food Poisoning": [
        "Nausea", "Vomiting", "Diarrhea", "Bloody diarrhea", "Stomach cramps",
        "Fever", "Weakness", "Dehydration"
    ],
    "Pneumonia": [
        "High fever", "Chills", "Cough with phlegm", "Shortness of breath",
        "Chest pain", "Fatigue", "Confusion"
    ],
    "Asthma": [
        "Wheezing", "Shortness of breath", "Chest tightness", "Coughing"
    ],
    "Tuberculosis": [
        "Persistent cough", "Blood in cough", "Fever (evening)", "Night sweats",
        "Weight loss", "Fatigue", "Loss of appetite"
    ],
    "Allergy": [
        "Sneezing", "Runny nose", "Itchy eyes", "Skin rash", "Swelling"
    ],
    "Bronchitis": [
        "Persistent cough with mucus", "Wheezing", "Chest discomfort", "Mild fever", "Chills", "Fatigue"
    ]
}

# List of animals
animals = [
    "Buffaloes", "Sheep", "Pig", "Fowl", "Elephant", "Duck", "Birds", "Cat", "Dog",
    "Donkey", "Deer", "Goat", "Monkey", "Cattle", "Hamster", "Lion", "Horse",
    "Chicken", "Rabbit", "Fox", "Tiger", "Turtle", "Cow", "Mammal", "Snake",
    "Moos", "Wolves", "Hyaenas"
]

# Generate dataset
data = []
for _ in range(3000):
    disease = random.choice(list(disease_symptoms.keys()))
    symptoms = random.sample(disease_symptoms[disease], min(5, len(disease_symptoms[disease])))  # Pick up to 5 symptoms
    animal = random.choice(animals)
    # Ensure always 5 symptoms (fill with 'None' if less)
    while len(symptoms) < 5:
        symptoms.append('None')
    data.append(symptoms + [disease, animal])

# Create DataFrame
df = pd.DataFrame(data, columns=["Symptom1", "Symptom2", "Symptom3", "Symptom4", "Symptom5", "Disease", "AnimalName"])

# Save
df.to_csv('realistic_animal_disease_dataset.csv', index=False)

print("✅ Realistic animal disease dataset created!")
