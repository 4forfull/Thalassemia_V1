# Import Important Library.
import streamlit as st
from PIL import Image
import pandas as pd
from catboost import CatBoostClassifier

# Set Page Title
st.set_page_config(page_title="Thalassemia")

# Load Model
model = CatBoostClassifier()
model.load_model('./catboost_model_smote_thalassemia.cbm')

model1 = CatBoostClassifier()
model1.load_model('./catboost_model_smote_ab.cbm')

# Load Dataset
df_main = pd.read_csv('./test.csv')

# Load Image
image = Image.open('img.jpg')


# Streamlit Function For Building Button & app.
def main():
    st.image(image)

    gender = st.selectbox("Gender (Male:0, Female:1).", df_main['gender'].unique())
    age = st.number_input('Age: 0-100.', value=None)
    RBC = st.number_input('Red Blood Cell\n(RBC: 0-15).', value=None)
    Hb = st.number_input('Hemoglobin (Hb: 0-200).', value=None)
    HCT = st.number_input('Hematocrit (HCT: 0-1).', value=None)
    MCV = st.number_input('Mean Corpuscular Volume (MCV: 50-200).', value=None)
    MCH = st.number_input('Mean Corpuscular Hemoglobin (MCH: 10-100).', value=None)
    MCHC = st.number_input('Mean Corpuscular Hemoglobin Concentration (MCHC: 100-1000).', value=None)
    RDW_CV = st.number_input('Red Cell Distribution Width-Coefficient of Variation (RDW_CV: 0-0.5).', value=None)
    RDW_SD = st.number_input('Red Cell Distribution Width-Standard Deviation (RDW_SD: 20-150).', value=None)
    input = [gender, age, RBC, Hb, HCT, MCV, MCH, MCHC, RDW_CV, RDW_SD]

    if st.button('Predict'):
        result = prediction(input)
        st.markdown(
            f"<div style='background-color:grey; padding:8px'><h1 style='color: white; text-align: center;'>{result}</h1></div>",
            unsafe_allow_html=True)


# Prediction Function to predict from model.
def prediction(input):
    test = [input]
    predict = model.predict(test)
    predict1 = model1.predict(test)
    if predict == 0:
        return "Diagnosis: Non-thalassemia"
    else:
        if predict1 == 0:
            return "Diagnosis: α-Thalassemia"
        else:
            return "Diagnosis: β-Thalassemia"


if __name__ == '__main__':
    main()


