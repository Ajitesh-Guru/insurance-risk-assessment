import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st
from keras.models import load_model

from insurance_risk_assessment.data import load_data
from insurance_risk_assessment.preprocessing import build_preprocessor


@st.cache_resource
def load_prediction_assets():
    data = load_data()
    preprocessor = build_preprocessor()
    preprocessor.fit(data.drop(columns=["charges"]))
    model = load_model("artifacts/models/neural_network.h5")
    return data, preprocessor, model


def bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "healthy weight"
    if bmi < 30:
        return "overweight"
    return "obese"


def main() -> None:
    st.set_page_config(page_title="Insurance Risk Assessment", layout="wide")
    st.title("Insurance Risk Assessment")

    data, preprocessor, model = load_prediction_assets()

    age = st.number_input("Age", min_value=18, max_value=120, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=180)
    weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region", ["southwest", "southeast", "northwest", "northeast"]
    )

    bmi = weight_kg / ((height_cm / 100) ** 2)

    if st.button("Predict insurance charges"):
        applicant = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "region": [region],
            }
        )
        prediction = float(model.predict(preprocessor.transform(applicant), verbose=0)[0][0])
        st.metric("Predicted charges", f"${prediction:,.2f}")

    st.subheader("Risk factors")
    st.write(f"BMI: **{bmi:.2f}** ({bmi_category(bmi)})")
    st.bar_chart(data.groupby("smoker")["charges"].mean())
    st.bar_chart(data.groupby("region")["charges"].mean())
    st.bar_chart(data.groupby("children")["charges"].mean())


if __name__ == "__main__":
    main()
