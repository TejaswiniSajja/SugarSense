
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="SugarSense - Diabetes Risk Screening",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 SugarSense")
st.subheader("Early Diabetes Risk Screening")

st.write(
    "Enter the measurements below to obtain a machine-learning "
    "screening estimate."
)

st.info(
    "Educational screening tool only. It cannot diagnose or rule "
    "out diabetes. Consult a qualified healthcare professional "
    "for medical advice."
)

# ==========================================
# 2. LOAD THE SAVED MODEL
# ==========================================
model_path = BASE_DIR / "sugarsense_model.joblib"


@st.cache_resource
def load_model(path):
    saved_object = joblib.load(path)

    if isinstance(saved_object, dict):
        model = None

        for key in [
            "model",
            "pipeline",
            "estimator",
            "best_estimator_"
        ]:
            if key in saved_object and hasattr(
                saved_object[key], "predict_proba"
            ):
                model = saved_object[key]
                break

        if model is None:
            for value in saved_object.values():
                if hasattr(value, "predict_proba"):
                    model = value
                    break

        threshold = saved_object.get("threshold", 0.35)
        return model, float(threshold)

    return saved_object, 0.35


if not model_path.exists():
    st.error(
        "Model file not found. Place sugarsense_model.joblib "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model, threshold = load_model(str(model_path))

    if model is None:
        st.error("No compatible classification model was found.")
        st.stop()

except Exception as error:
    st.error(f"Error loading model: {error}")
    st.stop()

# ==========================================
# 3. INPUT FORM
# ==========================================
st.markdown("### Enter Patient Measurements")

with st.form("screening_form"):

    pregnancies = st.number_input(
        "Number of Pregnancies",
        min_value=0, max_value=20, value=0
    )

    glucose = st.number_input(
        "Glucose Level",
        min_value=0, max_value=300, value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0, max_value=200, value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0, max_value=100, value=20
    )

    insulin = st.number_input(
        "Insulin Level",
        min_value=0, max_value=1000, value=80
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0, max_value=80.0, value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=18, max_value=120, value=30
    )

    submitted = st.form_submit_button(
        "Estimate Risk",
        use_container_width=True
    )

# ==========================================
# 4. PREPARE FEATURES AND PREDICT
# ==========================================
if submitted:

    input_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
        "Obesity_Flag": int(bmi >= 30),
        "High_Glucose_Flag": int(glucose >= 140),
        "Age_BMI_Interaction": age * bmi
    }])

    # Match the notebook's convention for impossible zero values.
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for col in zero_columns:
        input_df[col] = input_df[col].replace(0, np.nan)

    try:
        # Match feature names and order when available.
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)

            missing_features = [
                col for col in expected_features
                if col not in input_df.columns
            ]

            if missing_features:
                st.error(
                    "Input features do not match the saved model. "
                    f"Missing features: {missing_features}"
                )
                st.stop()

            input_df = input_df[expected_features]

        # Check probability prediction support.
        if not hasattr(model, "predict_proba"):
            st.error(
                "The saved model cannot produce probability estimates."
            )
            st.stop()

        probabilities = model.predict_proba(input_df)

        if probabilities.shape[1] != 2:
            st.error(
                "Expected a binary classification model. "
                "Please check the saved estimator."
            )
            st.stop()

        # Probability for class 1 (diabetes-positive in this dataset).
        classes = list(model.classes_) if hasattr(model, "classes_") else [0, 1]

        if 1 not in classes:
            st.error(
                "The saved model does not identify class 1. "
                "Check its training labels."
            )
            st.stop()

        positive_index = classes.index(1)
        probability = float(probabilities[0, positive_index])

        # ==========================================
        # 5. DISPLAY RISK CATEGORY
        # ==========================================
        st.divider()
        st.markdown("## Screening Result")

        # Illustrative display bands only; not clinical cutoffs.
        if probability < 0.25:
            risk_level = "LOW"
            st.success("🟢 LOW RISK")

            message = (
                "The model estimates a relatively lower probability "
                "for the positive class. This does not rule out diabetes."
            )

        elif probability < 0.50:
            risk_level = "MEDIUM"
            st.warning("🟡 MEDIUM RISK")

            message = (
                "The model estimates an intermediate probability. "
                "Discuss your health concerns and risk factors with "
                "a healthcare professional."
            )

        else:
            risk_level = "HIGH"
            st.error("🔴 HIGH RISK")

            message = (
                "The model estimates a relatively higher probability "
                "for the positive class. Please seek professional "
                "medical assessment."
            )

        st.metric(
            label="Model-estimated probability",
            value=f"{probability:.1%}"
        )

        st.write(message)

        st.caption(
            "The Low/Medium/High categories use illustrative software "
            "bands: below 25%, 25% to below 50%, and 50% or above. "
            "These bands are not clinically validated risk categories."
        )

        st.caption(
            "This estimate is not a medical diagnosis. Do not use it "
            "to start, stop, or change treatment."
        )

    except Exception as error:
        st.error(
            "Prediction could not be completed. The saved model's "
            "expected features or preprocessing may differ from the app."
        )
        st.code(str(error))

# ==========================================
# 6. FOOTER
# ==========================================
st.divider()
st.caption("SugarSense | Machine Learning Mini-Project")