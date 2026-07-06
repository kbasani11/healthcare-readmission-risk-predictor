import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("data/patient_readmission.csv")
print("=" * 50)
print("Healthcare Readmission Dataset")
print("=" * 50)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nFirst Five Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

plt.figure(figsize=(6,4))

df["readmitted"].value_counts().plot(
    kind="bar"
)

plt.title("Readmission Distribution")
plt.xlabel("Readmitted")
plt.ylabel("Number of Patients")

os.makedirs("screenshots", exist_ok=True)

plt.tight_layout()
plt.savefig("screenshots/readmission_distribution.png")
plt.show()

plt.figure(figsize=(8,4))

df["age"].hist(bins=20)

plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Patients")

plt.tight_layout()
plt.savefig("screenshots/age_distribution.png")
plt.show()

diabetes = (
    df.groupby("diabetes")["readmitted"]
    .mean()
)

plt.figure(figsize=(6,4))

diabetes.plot(kind="bar")

plt.title("Readmission Rate by Diabetes")
plt.ylabel("Average Readmission")

plt.tight_layout()
plt.savefig("screenshots/diabetes_vs_readmission.png")
plt.show()

heart = (
    df.groupby("heart_disease")["readmitted"]
    .mean()
)

plt.figure(figsize=(6,4))

heart.plot(kind="bar")

plt.title("Readmission Rate by Heart Disease")
plt.ylabel("Average Readmission")

plt.tight_layout()
plt.savefig("screenshots/heart_disease_vs_readmission.png")
plt.show()

plt.figure(figsize=(8,4))

df["length_of_stay"].hist(bins=15)

plt.title("Length of Hospital Stay")
plt.xlabel("Days")
plt.ylabel("Patients")

plt.tight_layout()
plt.savefig("screenshots/length_of_stay.png")
plt.show()