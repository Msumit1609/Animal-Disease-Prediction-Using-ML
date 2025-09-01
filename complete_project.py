import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import END

# ================= Load and Preprocess Dataset ==================
df = pd.read_csv("disease_symptoms.csv")  # Ensure the CSV file exists

# Encode categorical data (Convert symptoms and diseases to numbers)
encoder = LabelEncoder()
df_encoded = df.apply(encoder.fit_transform)

# Define features (X) and target (y)
X = df_encoded.drop(columns=["Disease"])  # Symptoms as features
y = df_encoded["Disease"]  # Disease as the target variable

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to NumPy arrays
X, y = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Define symptom names and disease list
l1 = list(df.columns[:-1])  # List of symptom names
disease = encoder.classes_  # List of disease names
l2 = [0] * len(l1)  # Initialize symptom presence array

# ================= Define Machine Learning Models ==================
def DecisionTree():
    clf3 = tree.DecisionTreeClassifier()
    clf3 = clf3.fit(X, y)

    # Calculate accuracy
    y_pred = clf3.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
    print("Correct Predictions:", accuracy_score(y_test, y_pred, normalize=False))

    predict_disease(clf3, t1)

def randomforest():
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X, np.ravel(y))

    # Calculate accuracy
    y_pred = clf4.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print("Correct Predictions:", accuracy_score(y_test, y_pred, normalize=False))

    predict_disease(clf4, t2)

def NaiveBayes():
    gnb = GaussianNB()
    gnb = gnb.fit(X, np.ravel(y))

    # Calculate accuracy
    y_pred = gnb.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
    print("Correct Predictions:", accuracy_score(y_test, y_pred, normalize=False))

    predict_disease(gnb, t3)

# ================= Helper Function for Prediction ==================
def predict_disease(model, text_widget):
    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    
    # Reset symptom presence array
    l2 = [0] * len(l1)
    
    for k in range(len(l1)):
        if l1[k] in psymptoms:
            l2[k] = 1

    inputtest = [l2]
    predict = model.predict(inputtest)
    predicted = predict[0]

    # Display result
    text_widget.delete("1.0", END)
    if predicted in range(len(disease)):
        text_widget.insert(END, disease[predicted])
    else:
        text_widget.insert(END, "Not Found")

# ================= GUI Setup ==================
root = tk.Tk()
root.title("Disease Prediction System")

# Symptom Dropdowns
tk.Label(root, text="Symptom 1").grid(row=0, column=0)
Symptom1 = tk.StringVar()
S1 = tk.Entry(root, textvariable=Symptom1)
S1.grid(row=0, column=1)

tk.Label(root, text="Symptom 2").grid(row=1, column=0)
Symptom2 = tk.StringVar()
S2 = tk.Entry(root, textvariable=Symptom2)
S2.grid(row=1, column=1)

tk.Label(root, text="Symptom 3").grid(row=2, column=0)
Symptom3 = tk.StringVar()
S3 = tk.Entry(root, textvariable=Symptom3)
S3.grid(row=2, column=1)

tk.Label(root, text="Symptom 4").grid(row=3, column=0)
Symptom4 = tk.StringVar()
S4 = tk.Entry(root, textvariable=Symptom4)
S4.grid(row=3, column=1)

tk.Label(root, text="Symptom 5").grid(row=4, column=0)
Symptom5 = tk.StringVar()
S5 = tk.Entry(root, textvariable=Symptom5)
S5.grid(row=4, column=1)

# Prediction Buttons
tk.Button(root, text="Decision Tree", command=DecisionTree).grid(row=5, column=0)
tk.Button(root, text="Random Forest", command=randomforest).grid(row=5, column=1)
tk.Button(root, text="Naive Bayes", command=NaiveBayes).grid(row=5, column=2)

# Result Text Fields
tk.Label(root, text="Decision Tree Result").grid(row=6, column=0)
t1 = tk.Text(root, height=1, width=20)
t1.grid(row=6, column=1)

tk.Label(root, text="Random Forest Result").grid(row=7, column=0)
t2 = tk.Text(root, height=1, width=20)
t2.grid(row=7, column=1)

tk.Label(root, text="Naive Bayes Result").grid(row=8, column=0)
t3 = tk.Text(root, height=1, width=20)
t3.grid(row=8, column=1)

# Run the GUI
root.mainloop()
