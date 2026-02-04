# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SARAVANAN K
RegisterNumber:  25013282
*/

```
# Logistic Regression for Student Placement Prediction

# 1️ Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 2️ Load Dataset
data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

# 3️ Drop Unnecessary Columns
data = data.drop(["sl_no", "salary"], axis=1)

# 4️ Convert Target Variable (status) to Binary
# Placed = 1, Not Placed = 0
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

# 5️ Separate Features and Target
X = data.drop("status", axis=1)
y = data["status"]

# 6️ One-Hot Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

# 7️ Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 8️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 9️ Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 10 Make Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 1️1 Model Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()
```

## Output:

Dataset Preview:
<img width="704" height="328" alt="image" src="https://github.com/user-attachments/assets/eb10f3ca-5c10-440e-9c5e-c7e38ae22e93" />

After Encoding:
<img width="679" height="515" alt="image" src="https://github.com/user-attachments/assets/842d637e-7545-40d2-ab0f-1c4176954ba4" />






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
