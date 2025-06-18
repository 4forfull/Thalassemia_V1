# Import Important Libraries
import streamlit as st
from PIL import Image
import pandas as pd
import joblib

# Set Page Title
st.set_page_config(page_title="Thalassemia")

@st.cache_resource
def load_models():
    model_main = joblib.load('./catboost_model_smote_thalassemia.pkl')
    model_sub = joblib.load('./catboost_model_smote_ab.pkl')
    return model_main, model_sub

@st.cache_data
def load_data():
    return pd.read_csv('./test.csv')

@st.cache_data
def load_image():
    return Image.open('img.jpg')

def predict_thalassemia(model_main, model_sub, input_data):
    predict = model_main.predict([input_data])
    if predict == 0:
        return "Diagnosis: Non-thalassemia"
    else:
        predict_sub = model_sub.predict([input_data])
        return "Diagnosis: α-Thalassemia" if predict_sub == 0 else "Diagnosis: β-Thalassemia"

def main():
    st.image(load_image())

    df_main = load_data()
    model_main, model_sub = load_models()

    gender = st.selectbox("Gender (Male:0, Female:1)", df_main['gender'].unique())
    age = st.number_input('Age (0-100)', min_value=0, max_value=100, value=20)
    RBC = st.number_input('Red Blood Cell (0-15)', min_value=0.0, max_value=15.0, value=5.0)
    Hb = st.number_input('Hemoglobin (0-200)', min_value=0.0, max_value=200.0, value=14.0)
    HCT = st.number_input('Hematocrit (0-1)', min_value=0.0, max_value=1.0, value=0.4)
    MCV = st.number_input('Mean Corpuscular Volume (50-200)', min_value=50.0, max_value=200.0, value=85.0)
    MCH = st.number_input('Mean Corpuscular Hemoglobin (10-100)', min_value=10.0, max_value=100.0, value=28.0)
    MCHC = st.number_input('Mean Corpuscular Hemoglobin Concentration (100-1000)', min_value=100.0, max_value=1000.0, value=320.0)
    RDW_CV = st.number_input('RDW-CV (0-0.5)', min_value=0.0, max_value=0.5, value=0.13)
    RDW_SD = st.number_input('RDW-SD (20-150)', min_value=20.0, max_value=150.0, value=42.0)

    input_data = [gender, age, RBC, Hb, HCT, MCV, MCH, MCHC, RDW_CV, RDW_SD]

    if st.button('Predict'):
        result = predict_thalassemia(model_main, model_sub, input_data)
        st.markdown(
            f"<div style='background-color:grey; padding:8px'><h1 style='color: white; text-align: center;'>{result}</h1></div>",
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()
