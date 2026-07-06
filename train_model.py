import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("data/patient_readmission.csv")

print("Dataset Loaded Successfully")
print(df.head())

print("\nMissing Values")
print(df.isnull().sum())

encoder = LabelEncoder()

categorical_columns = [
    "gender",
    "diabetes",
    "hypertension",
    "heart_disease",
    "lab_abnormal"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("\nEncoded Dataset")
print(df.head())

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

processed = pd.concat([X, y], axis=1)

processed.to_csv(
    "data/processed_patient_data.csv",
    index=False
)

print("\nProcessed dataset saved successfully.")


# Logistic Regression
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy :", accuracy_score(y_test, log_predictions))
print("Precision:", precision_score(y_test, log_predictions))
print("Recall   :", recall_score(y_test, log_predictions))
print("F1 Score :", f1_score(y_test, log_predictions))

tree_model = DecisionTreeClassifier(random_state=42)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

print("\n===== Decision Tree =====")
print("Accuracy :", accuracy_score(y_test, tree_predictions))
print("Precision:", precision_score(y_test, tree_predictions))
print("Recall   :", recall_score(y_test, tree_predictions))
print("F1 Score :", f1_score(y_test, tree_predictions))

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train, y_train)

forest_predictions = forest_model.predict(X_test)

print("\n===== Random Forest =====")
print("Accuracy :", accuracy_score(y_test, forest_predictions))
print("Precision:", precision_score(y_test, forest_predictions))
print("Recall   :", recall_score(y_test, forest_predictions))
print("F1 Score :", f1_score(y_test, forest_predictions))

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy_score(y_test, log_predictions),
        accuracy_score(y_test, tree_predictions),
        accuracy_score(y_test, forest_predictions)
    ],
    "Precision": [
        precision_score(y_test, log_predictions),
        precision_score(y_test, tree_predictions),
        precision_score(y_test, forest_predictions)
    ],
    "Recall": [
        recall_score(y_test, log_predictions),
        recall_score(y_test, tree_predictions),
        recall_score(y_test, forest_predictions)
    ],
    "F1 Score": [
        f1_score(y_test, log_predictions),
        f1_score(y_test, tree_predictions),
        f1_score(y_test, forest_predictions)
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(results)

os.makedirs("models", exist_ok=True)

joblib.dump(
    forest_model,
    "models/readmission_model.pkl"
)

print("\nRandom Forest model saved successfully!")

print("\nCreating Confusion Matrix...")

ConfusionMatrixDisplay.from_estimator(
    forest_model,
    X_test,
    y_test,
    cmap="Blues"
)

plt.title("Random Forest - Confusion Matrix")
plt.savefig("screenshots/confusion_matrix.png")
plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": forest_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(10,6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("screenshots/feature_importance.png")

plt.show()

results.to_csv(
    "model_results.csv",
    index=False
)

print("\nModel evaluation results saved.")