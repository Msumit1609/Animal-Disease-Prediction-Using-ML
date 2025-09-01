import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("realistic_animal_disease_dataset.csv")

# Combine symptoms and prepare features
df["Symptoms"] = df[["Symptom1", "Symptom2", "Symptom3", "Symptom4", "Symptom5"]].values.tolist()
X = df[["Symptoms", "AnimalName"]]
y = df["Disease"]
combined_features = [symptoms + [animal] for symptoms, animal in zip(X["Symptoms"], X["AnimalName"])]

# Encode features
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_encoded = onehot_encoder.fit_transform(combined_features)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Models to test
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Evaluate each model
for name, model in models.items():
    print(f"\n🔍 Evaluating: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"✅ Accuracy       : {acc:.4f}")
    print(f"🎯 Precision      : {prec:.4f}")
    print(f"📈 Recall         : {rec:.4f}")
    print(f"📊 F1 Score       : {f1:.4f}")
    print("\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("🧮 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
