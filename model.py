import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

# Load the dataset
df = pd.read_csv("realistic_animal_disease_dataset.csv")

# Combine symptoms into one list column
df["Symptoms"] = df[["Symptom1", "Symptom2", "Symptom3", "Symptom4", "Symptom5"]].values.tolist()

# Prepare features and label
X = df[["Symptoms", "AnimalName"]]
y = df["Disease"]

# Combine symptoms and animal into single list for each sample
combined_features = [symptoms + [animal] for symptoms, animal in zip(X["Symptoms"], X["AnimalName"])]

# One-hot encode Symptoms + AnimalName
symptom_animal_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_encoded = symptom_animal_encoder.fit_transform(combined_features)

# Label encode Diseases
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Train XGBoost Classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy: {accuracy:.4f}")

# Save model and encoders
joblib.dump(model, "animal_disease_model.pkl")
joblib.dump(symptom_animal_encoder, "symptom_animal_encoder.pkl")
joblib.dump(disease_encoder, "disease_encoder.pkl")
print(" Model and encoders saved successfully.")
