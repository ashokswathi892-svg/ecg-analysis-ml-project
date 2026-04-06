import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Loading dataset...")

# Load MIT-BIH training dataset
data = pd.read_csv("mitbih_train.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Convert to binary classification
# 0 = Normal, 1 = Abnormal
y = y.apply(lambda x: 0 if x == 0 else 1)

print("Dataset loaded")

# Create pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

print("Training model...")
model.fit(X, y)
print("Training completed")

# Separate scaler and classifier
scaler = model.named_steps["scaler"]
classifier = model.named_steps["clf"]

# Save model
with open("trained_model.pkl", "wb") as f:
    pickle.dump(classifier, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model saved as trained_model.pkl")
print("✅ Scaler saved as scaler.pkl")

