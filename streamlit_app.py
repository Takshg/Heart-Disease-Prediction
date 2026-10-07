import streamlit as st
from joblib import load
import pandas as pd
import time 

st.title(":red[Heart Disease Prediction System]", text_alignment='center')
st.write("Made by Taksh Girdhar")

st.divider()
st.caption(
    "⚠️ **Disclaimer:** This app is an educational project and is not a medical "
    "device or diagnostic tool. Predictions come from a machine learning model "
    "trained on a small historical dataset (the UCI Cleveland Heart Disease "
    "dataset, 303 patients from 1988) and may be inaccurate for any individual. "
    "The results are estimates only and should not be used to make health "
    "decisions. If you have symptoms such as chest pain, shortness of breath, "
    "or concerns about your heart health, please consult a qualified healthcare "
    "professional. In an emergency, call 911 or your local emergency number."
)

st.divider()


st.header("Patient Information")


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Enter your age", min_value = 1, max_value=100)
    sex = st.selectbox(
        "Select gender (1 = male, 0 = female)", 
        [1, 0]
    )
    cp =  st.selectbox(
        "Chest Pain Type (0: asymptomatic, 1: atypical angina, 2: non-anginal pain, 3: typical angina)",
        [0,1,2,3]
    )
    trestbps = st.number_input(
        "Resting Blood Pressure in mm/hg", 
        min_value = 1, max_value= 200
    )
    chol = st.number_input(
        "Serum Cholesterol in mg/dl", 
        min_value = 1, max_value= 700
    )
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl (0 = no, 1 = yes)", 
        [0,1]
    )
    restecg = st.selectbox(
        "Resting ECG (0: showing probable or definite left ventricular hypertrophy by Estes’ criteria, 1: normal, 2: having ST-T wave abnormality)", 
        [0,1,2]
    )


with col2:
    thalach = st.number_input(
        "Maximum heart rate achieved", 
        min_value=1, max_value=600
    )

    exang = st.selectbox(
        "Exercise Induced Angina (1 = yes, 0 = no)", 
        [1,0]
    )

    oldpeak = st.number_input(
        "ST depression induced by exercise relative to rest", 
        min_value = 0.0, max_value = 10.0, step=0.1, format="%.1f"
    )
    slope = st.selectbox(
        "Slope of the peak exercise ST segment (0: downsloping; 1: flat; 2: upsloping)", 
        [0,1,2]
    )

    ca = st.number_input(
        "Number of major vessels [0-3]", 
        min_value=0, max_value=3
    )

    thal = st.selectbox(
            "Thalassemia (Blood Disorder) (1 = normal, 2 = fixed defect, 3 = reversible defect)", 
            [1,2,3]
    )


cp1, cp2, cp3, cp4 = (int(cp == i) for i in range(4))
restecg_0, restecg_1, restecg_2 = (int(restecg==i) for i in range(3))
slope_1, slope_2, slope_3 = (int(slope==i) for i in range(1,4))
thal_3, thal_6, thal_7 = (int(thal==i) for i in [1,2,3])


preprocess = load("models/preprocessor.joblib")
model = load("models/logistic.joblib")

raw = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": float(thal),
}])

pred_data = pd.DataFrame(preprocess.transform(raw), columns=preprocess.get_feature_names_out())

model = load('models/logistic.joblib')
prediction = model.predict(pred_data)[0]
probability = model.predict_proba(pred_data)[0, 1]

st.divider()
st.header("Prediction")

if st.button("Predict Heart Disease Risk", type="primary", width="stretch"): 
    with st.status("Loading data", type="step"):
        time.sleep(1)

    with st.status("Analyzing data", type="step"):
        time.sleep(1)

    if probability < 0.25:
        st.success(f"Low estimated risk ({probability:.0%})")
    elif probability < 0.50:
        st.warning(f"Moderate estimated risk ({probability:.0%})")
    else:
        st.error(f"High estimated risk ({probability:.0%})")

with open("models/logistic.joblib", "rb") as file:
    st.download_button(label="Save model info", file_name="models/logistic.joblib", data=file)


st.divider()

