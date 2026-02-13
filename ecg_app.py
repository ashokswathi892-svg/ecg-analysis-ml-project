import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL & SCALER ----------------
model = pickle.load(open("trained_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="ECG Abnormality Detection", layout="wide")

st.title("❤️ ECG Abnormality Detection")
st.write("MIT-BIH ECG Dataset | Normal vs Abnormal Prediction")

# ---------------- CSV UPLOAD ----------------
uploaded_file = st.file_uploader("Upload ECG CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Assume last column is label
    X = df.iloc[:, :-1]

    # Select row
    row_no = st.number_input(
        "Select ECG sample row number",
        min_value=0,
        max_value=len(X) - 1,
        value=0
    )

    sample = X.iloc[row_no].values.reshape(1, -1)

    # ---------------- SCALING ----------------
    sample_scaled = scaler.transform(sample)

    # ---------------- PREDICTION ----------------
    pred = model.predict(sample_scaled)[0]
    prob = model.predict_proba(sample_scaled)[0]

    confidence = np.max(prob) * 100

    label = "Abnormal" if pred == 1 else "Normal"

    # ---------------- RESULT ----------------
    st.subheader("🧾 Prediction Result")

    if label == "Normal":
        st.success(f"Prediction: **NORMAL**")
    else:
        st.error(f"Prediction: **ABNORMAL**")

    st.info(f"Confidence Level: **{confidence:.2f}%**")

    # ---------------- ECG PLOT ----------------
    st.subheader("📈 ECG Signal Visualization")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sample.flatten())
    ax.set_title(f"ECG Waveform ({label})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

else:
    st.warning("Please upload an ECG CSV file to continue.")
