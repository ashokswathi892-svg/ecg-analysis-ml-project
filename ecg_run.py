import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load Dataset
# -------------------------------
train = pd.read_csv(r"C:\Users\ashok\Downloads\archive (2)\mitbih_train.csv", header=None)
test  = pd.read_csv(r"C:\Users\ashok\Downloads\archive (2)\mitbih_test.csv", header=None)

X_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1]

X_test = test.iloc[:, :-1]
y_test = test.iloc[:, -1]

# Binary classification
y_train = y_train.apply(lambda x: 0 if x == 0 else 1)
y_test  = y_test.apply(lambda x: 0 if x == 0 else 1)

# -------------------------------
# 2. Model Pipeline (Deployment ready)
# -------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

# -------------------------------
# 3. Train Model
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# 4. Predict
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 5. Output Results
# -------------------------------
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred,
      target_names=["Normal", "Abnormal"]))

# -------------------------------
# 6. Plot Sample ECG with Prediction
# -------------------------------
print("\nShowing Abnormal ECG samples...\n")

count = 0
for i in range(len(X_test)):

    # Check true abnormal sample
    if y_test.iloc[i] == 1:   # 1 = Abnormal

        signal = X_test.iloc[i].values
        pred = y_pred[i]

        label = "Normal" if pred == 0 else "Abnormal"
        print(f"Abnormal Sample {count+1} Prediction: {label}")

        plt.figure(figsize=(8,3))
        plt.plot(signal)
        plt.title(f"ECG Signal - Predicted: {label}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.show()

        count += 1
        if count == 2:   # show only 2 abnormal samples
            break


