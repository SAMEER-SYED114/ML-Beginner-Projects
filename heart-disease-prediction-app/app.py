import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Risk Predictor", page_icon="❤️", layout="centered")

# ---------- Style ----------
st.markdown("""
<style>
.stApp {background-color: #F4F7FA;}
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 860px; padding-top: 1.5rem;}
.stApp, .stApp p, .stApp label, .stApp span, .stApp div {color: #1b2b34;}
.topbar {background: #0F766E; padding: 14px 20px; border-radius: 12px; display: flex; align-items: center; margin-bottom: 18px;}
.topbar .name {color: #ffffff !important; font-size: 20px; font-weight: 600;}
.topbar .by {color: #d7f0ec !important; font-size: 13px; margin-left: auto;}
.card {background: #ffffff; border: 1px solid #dde5eb; border-radius: 10px; padding: 12px 14px;}
.card .k {font-size: 12px; color: #6b7c88 !important;}
.card .v {font-size: 26px; font-weight: 600;}
.panel {background: #ffffff; border: 1px solid #dde5eb; border-radius: 10px; padding: 14px 16px; margin-top: 12px;}
.stButton>button {background-color: #0F766E; color: #ffffff; border: none; border-radius: 999px; height: 3.2em; font-size: 16px; font-weight: 600;}
.stButton>button:hover {background-color: #0b5f58; color: #ffffff;}
.note {font-size: 12px; color: #6b7c88 !important; margin-top: 18px;}
</style>
""", unsafe_allow_html=True)

# ---------- Load files ----------
model = joblib.load("Logistic_Regression_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# ---------- Header ----------
st.markdown(
    '<div class="topbar"><span class="name">❤️ CardioCheck</span>'
    '<span class="by">Built by Sameer</span></div>',
    unsafe_allow_html=True,
)

st.subheader("Patient details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------- Predict ----------
if st.button("❤️ Check my heart risk", use_container_width=True):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    prob = float(model.predict_proba(scaled_input)[0][1])

    high = prediction == 1
    color = "#E5484D" if high else "#1E9E63"
    label = "High risk of heart disease" if high else "Low risk of heart disease"
    pos = min(max(prob * 100, 2), 98)

    st.markdown("### Result")

    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
          <div class="card"><div class="k">Risk</div><div class="v" style="color:{color} !important;">{prob:.0%}</div></div>
          <div class="card"><div class="k">Age</div><div class="v">{age}</div></div>
          <div class="card"><div class="k">Resting BP</div><div class="v">{resting_bp}</div></div>
          <div class="card"><div class="k">Cholesterol</div><div class="v">{cholesterol}</div></div>
        </div>

        <div class="panel">
          <div style="font-weight:600;margin-bottom:10px;">Risk meter</div>
          <div style="display:flex;height:10px;border-radius:5px;overflow:hidden;">
            <div style="flex:1;background:#22C08A;"></div>
            <div style="flex:1;background:#E5484D;"></div>
          </div>
          <div style="position:relative;height:14px;">
            <div style="position:absolute;left:{pos}%;top:2px;transform:translateX(-50%);width:0;height:0;
                        border-left:6px solid transparent;border-right:6px solid transparent;
                        border-bottom:8px solid #1b2b34;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:12px;color:#6b7c88;">
            <span>Low</span><span>High</span>
          </div>
          <div style="margin-top:12px;padding:10px 12px;border-radius:8px;font-weight:600;
                      background:{color}22;color:{color} !important;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Disclaimer ----------
st.markdown(
    '<div class="note">This is a student project built for learning. '
    'It is not medical advice. Please consult a doctor for any health concern.</div>',
    unsafe_allow_html=True,
)
