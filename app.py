import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model
model = load('model.pkl')

st.title("Disease Prediction Dashboard 🧠")
st.write("Select symptoms to predict the likely disease.")

# Symptom inputs (match your dataset columns)
symptoms = [
    'COUGH', 'MUSCLE_ACHES', 'TIREDNESS', 'SORE_THROAT', 'RUNNY_NOSE',
    'STUFFY_NOSE', 'FEVER', 'NAUSEA', 'VOMITING', 'DIARRHEA',
    'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING', 'LOSS_OF_TASTE',
    'LOSS_OF_SMELL', 'ITCHY_NOSE', 'ITCHY_EYES', 'ITCHY_MOUTH',
    'ITCHY_INNER_EAR', 'SNEEZING', 'PINK_EYE'
]

# Create checkboxes for each symptom
user_input = []
for symptom in symptoms:
    user_input.append(st.checkbox(symptom.replace("_", " ").title()))

# Predict button
if st.button("Predict Disease"):
    # Create input dataframe
    input_df = pd.DataFrame([user_input], columns=symptoms)

    # Make prediction
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]

    # Show result
    label_map = {0: 'COVID', 1: 'FLU', 2: 'COLD', 3: 'ALLERGY'}
    st.success(f"Predicted Disease: **{label_map[prediction]}**")

    probs = model.predict_proba(input_df)[0]
    for i, prob in enumerate(probs):
        st.write(f"{label_map[i]}: {prob:.2f}")

    # Confidence chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.barh(list(label_map.values()), probs, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Model Confidence by Disease")
    st.pyplot(fig)

    import csv
    from datetime import datetime

    # Prepare log entry
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symptoms': ', '.join([symptom for symptom, value in zip(symptoms, user_input) if value]),
        'predicted_label': label_map[prediction],
        'confidence_scores': ', '.join([f"{label_map[i]}: {prob:.2f}" for i, prob in enumerate(probs)])
    }

    # Append to CSV
    with open('prediction_log.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=log_entry.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(log_entry)
st.markdown("---")
st.markdown("🔗 [GitHub Profile](https://github.com/datadreamsbyasghar) — Built by Ali Asghar, the Coding Monarch 👑")