import streamlit as st
import pandas as pd
import numpy as np
import joblib
import locale

# Set locale for Indian currency formatting
locale.setlocale(locale.LC_ALL, 'en_IN')

def format_inr(value):
    return locale.format_string("%d", int(value), grouping=True)

# 🎨 Custom Pink + Dark Theme Styling
st.markdown("""
    <style>
    html, body, .stApp {
        background-color: #fff0f5 !important;
        color: #212529 !important;
    }

    .stButton>button {
        background-color: #ff69b4 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 0.75em !important;
    }

    /* Fix multiselect skill tags */
    div[data-baseweb="tag"], span[data-baseweb="tag"] {
        color: white !important;
        background-color: #ff69b4 !important;
        font-weight: 500 !important;
        border-radius: 6px !important;
    }

    /* Force label + text color to white */
    label, .css-1cpxqw2, .css-1y4p8pa, .stRadio, .stMultiSelect {
        color: white !important;
    }

    .stSlider span, .stSlider div {
        color: #212529 !important;
    }

    /* 🔧 Uniform dropdown styling */
    div[data-baseweb="select"] {
        background-color: #2b2b2b !important;
        border-radius: 8px !important;
        color: white !important;
    }

    div[data-baseweb="select"] * {
        color: white !important;
        font-weight: 500 !important;
    }

    .stSelectbox div[role="combobox"] > div,
    .stMultiSelect div[role="combobox"] > div {
        color: white !important;
    }

    div[data-baseweb="popover"] {
        background-color: #2b2b2b !important;
        color: white !important;
    }
            /* Force all form label and widget title texts to black */
    label, .stSlider label, .stRadio label, .css-1cpxqw2, .css-1y4p8pa, .css-qrbaxs,
    .css-16huue1, .stSelectbox label, .stMultiSelect label {
    color: #212529 !important;
    font-weight: 600 !important;
    }

    /* Fix radio button labels (e.g., Gender) */
    div[data-baseweb="radio"] label {
    color: #212529 !important;
    font-weight: 600 !important;
    }

    /* Fix placeholder text inside dropdowns */
    .stSelectbox div[role="combobox"] > div,
    .stMultiSelect div[role="combobox"] > div {
    color: #f1f1f1 !important;
    }

    /* Force "Male" and "Female" radio labels to black */
    .stRadio div[role="radiogroup"] > label > div {
    color: #212529 !important;
    font-weight: 600 !important;
    }


    </style>
""", unsafe_allow_html=True)

# Load model and features
model = joblib.load("salary_model1.pkl")
feature_list = joblib.load("model_features1.pkl")

# App Title
st.markdown("<h1 style='color:#d63384;'>💼 Salary Prediction App (INR)</h1>", unsafe_allow_html=True)
st.markdown("Use the sliders and dropdowns below to predict a realistic salary in <b>Indian Rupees (₹)</b>.", unsafe_allow_html=True)

# --- Inputs Section ---
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 30)
        hours_per_week = st.slider("Hours per Week", 10, 80, 40)
        years_exp = st.slider("Years of Experience", 0, 50, 5)
        gender = st.radio("Gender", ['Male', 'Female'])
        race = st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])

    with col2:
        workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                                               'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
        education = st.selectbox("Education", ['Bachelors', 'HS-grad', 'Some-college', 'Masters', 'Assoc-acdm',
                                               'Assoc-voc', 'Doctorate', 'Prof-school'])
        occupation = st.selectbox("Occupation", ['Exec-managerial', 'Prof-specialty', 'Tech-support', 'Sales',
                                                 'Adm-clerical', 'Craft-repair', 'Machine-op-inspct', 'Other-service'])
        skills = st.multiselect("Skills", ['python', 'sql', 'excel', 'data analysis', 'leadership',
                                           'typing', 'customer service', 'welding'])

# --- Feature Encoding ---
input_dict = {
    'age': age,
    'hours-per-week': hours_per_week,
    'capital-gain': 0,
    'capital-loss': 0,
    'years_experience': years_exp
}

# One-hot encode categorical fields
for col, options in {
    "workclass": ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                  'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    "education": ['Bachelors', 'HS-grad', 'Some-college', 'Masters', 'Assoc-acdm',
                  'Assoc-voc', 'Doctorate', 'Prof-school'],
    "occupation": ['Exec-managerial', 'Prof-specialty', 'Tech-support', 'Sales',
                   'Adm-clerical', 'Craft-repair', 'Machine-op-inspct', 'Other-service'],
    "gender": ['Male', 'Female'],
    "race": ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
}.items():
    for option in options:
        input_dict[f"{col}_{option}"] = 1 if eval(col) == option else 0

# Skills one-hot
skills_list = ['python', 'sql', 'excel', 'data analysis', 'leadership',
               'typing', 'customer service', 'welding']
for s in skills_list:
    input_dict[f"skill_{s}"] = 1 if s in skills else 0

# Create input DataFrame
input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_list, fill_value=0)

# --- Predict Salary ---
if st.button("🔮 Predict Salary"):
    salary = int(model.predict(input_df)[0])
    salary_formatted = format_inr(salary)
    st.markdown(f"<h2 style='color:#dc3545;'>💰 Predicted Salary: ₹ {salary_formatted}</h2>", unsafe_allow_html=True)
