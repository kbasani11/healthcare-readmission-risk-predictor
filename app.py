import streamlit as st
import joblib
import pandas as pd

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------
st.set_page_config(
    page_title="Healthcare Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------
# LOAD MODEL
# ---------------------------------------
model = joblib.load("models/readmission_model.pkl")

# ---------------------------------------
# SIDEBAR
# ---------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/hospital-3.png", width=80)

st.sidebar.title("Healthcare AI")
st.sidebar.caption("Clinical Decision Support System")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Prediction",
        "Analytics",
        "Model Performance",
        "About"
    ]
)


# ---------------------------------------
# PREDICTION PAGE
# ---------------------------------------
if page == "Prediction":

    st.title("🏥 Healthcare Readmission Risk Predictor")

    st.markdown("""
Predict whether a patient is likely to be readmitted within **30 days**
using a trained Machine Learning model.
""")

    st.info(
        "Complete the patient information and click **Predict Readmission Risk**."
    )

    st.subheader("📊 Prediction Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns([1,1,1.4,1,1])
    
    col1.metric("👨‍⚕️ Patients", "1,000")
    col2.metric("🎯 Accuracy", "93.5%")
    col3.metric("🤖 Model" ,"Random Forest")
    col4.metric("📈 F1 Score", "91.6%")
    col5.metric("📊 Dataset", "1K")

    st.divider()

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 90, 50)

        gender = st.selectbox("Gender", ["Male", "Female"])

        diabetes = st.selectbox("Diabetes", ["No", "Yes"])

        hypertension = st.selectbox("Hypertension", ["No", "Yes"])

        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    with col2:
        previous_admissions = st.slider("Previous Admissions", 0, 10, 1)

        length_of_stay = st.slider("Length of Stay (Days)", 1, 20, 5)

        medications = st.slider("Number of Medications", 1, 25, 5)

        lab_abnormal = st.selectbox("Abnormal Lab Results", ["No", "Yes"])

    predict = st.button(
        "🔍 Predict Readmission Risk",
        use_container_width=True
    )

    if predict:
        gender_encoded = 1 if gender == "Male" else 0
        diabetes_encoded = 1 if diabetes == "Yes" else 0
        hypertension_encoded = 1 if hypertension == "Yes" else 0
        heart_disease_encoded = 1 if heart_disease == "Yes" else 0
        lab_abnormal_encoded = 1 if lab_abnormal == "Yes" else 0

        patient = pd.DataFrame({
            "age": [age],
            "gender": [gender_encoded],
            "diabetes": [diabetes_encoded],
            "hypertension": [hypertension_encoded],
            "heart_disease": [heart_disease_encoded],
            "previous_admissions": [previous_admissions],
            "length_of_stay": [length_of_stay],
            "medications": [medications],
            "lab_abnormal": [lab_abnormal_encoded]
        })

        prediction = model.predict(patient)[0]
        probability = model.predict_proba(patient)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🔴 High Readmission Risk")
            st.metric("Prediction Confidence", f"{probability[1] * 100:.1f}%")
        else:
            st.success("🟢 Low Readmission Risk")
            st.metric("Prediction Confidence", f"{probability[0] * 100:.1f}%")

        st.subheader("Clinical Recommendation")

        if prediction == 1:
            st.success("""
### Recommended Actions

✅ Schedule follow-up within 7 days

✅ Review medications

✅ Monitor patient

✅ Educate patient before discharge
""")
        else:
            st.info("""
- Continue routine follow-up.
- Encourage medication compliance.
- Maintain regular health monitoring.
""")

# ---------------------------------------
# MODEL PAGE
# ---------------------------------------
elif page == "Model Performance":

    st.title("📊 Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "93.5%")
    c2.metric("Precision", "95.9%")
    c3.metric("Recall", "87.6%")
    c4.metric("F1 Score", "91.6%")
    st.image("screenshots/confusion_matrix.png")
    st.image("screenshots/feature_importance.png")


    
# ---------------------------------------
# ANALYTICS PAGE
# ---------------------------------------

elif page == "Analytics":

    st.title("📈 Healthcare Data Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.image("screenshots/readmission_distribution.png")
        st.image("screenshots/diabetes_vs_readmission.png")
        st.image("screenshots/length_of_stay.png")

    with col2:
        st.image("screenshots/age_distribution.png")
        st.image("screenshots/heart_disease_vs_readmission.png")
        st.image("screenshots/feature_importance.png")

# ---------------------------------------
# ABOUT PAGE
# ---------------------------------------
else:

    st.title("ℹ️ About the Project")
    
    st.header("Project Overview")
    
    st.write(
        "This application predicts patient readmission risk using Machine Learning."
    )
    
    st.header("Technologies")
    
    st.write("""
- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- Joblib
""")
    
    st.success("Kavya Basani")

st.markdown("---")

st.caption(
    "Healthcare Readmission Risk Predictor • Machine Learning Powered • © 2026 Kavya Basani"
)