import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

# comment 
# new comment

# 📂 Charger le dataset CSV
file_path = "data/customer_churn.csv" 
df = pd.read_csv(file_path)


X = df.loc[:,["Age","Account_Manager","Years","Num_Sites"]] 
y = df['Churn']  

# Diviser en ensemble d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardiser les données pour améliorer la performance du modèle
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# SAVE-LOAD using joblib 
# save
joblib.dump(model, "models/model_sklearn.pkl") 
# load
# clf2 = joblib.load("model.pkl")
