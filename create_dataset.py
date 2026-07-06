import pandas as pd
import numpy as np
import os

np.random.seed(42)

os.makedirs("data", exist_ok=True)

n = 1000

age = np.random.randint(25, 90, n)
gender = np.random.choice(["Male", "Female"], n)
diabetes = np.random.choice(["Yes", "No"], n, p=[0.35, 0.65])
hypertension = np.random.choice(["Yes", "No"], n, p=[0.45, 0.55])
heart_disease = np.random.choice(["Yes", "No"], n, p=[0.25, 0.75])
previous_admissions = np.random.randint(0, 7, n)
length_of_stay = np.random.randint(1, 15, n)
medications = np.random.randint(1, 20, n)
lab_abnormal = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])

readmission_risk = (
    (age > 65).astype(int)
    + (diabetes == "Yes").astype(int)
    + (hypertension == "Yes").astype(int)
    + (heart_disease == "Yes").astype(int)
    + (previous_admissions > 2).astype(int)
    + (length_of_stay > 7).astype(int)
    + (medications > 10).astype(int)
    + (lab_abnormal == "Yes").astype(int)
)

readmitted = np.where(readmission_risk >= 4, 1, 0)

df = pd.DataFrame({
    "age": age,
    "gender": gender,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "previous_admissions": previous_admissions,
    "length_of_stay": length_of_stay,
    "medications": medications,
    "lab_abnormal": lab_abnormal,
    "readmitted": readmitted
})

df.to_csv("data/patient_readmission.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nReadmission distribution:")
print(df["readmitted"].value_counts())