import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF
from io import BytesIO

model = joblib.load("diabetes_model.pkl")

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Risk Prediction System")


# Feature names + model coefficients (for risk factor breakdown)

feature_names = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
                  "HeartDiseaseorAttack", "PhysActivity", "NoDocbcCost", "GenHlth",
                  "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

coefficients = model.coef_[0]

# Mode selection

mode = st.radio(
    "Choose assessment type:",
    ["⚡ Quick Assessment (6 questions)", "📋 Full Assessment (17 questions)"],
    horizontal=True
)

st.divider()

defaults = {
    "cholcheck": 1, "smoker": 0, "stroke": 0, "heartdisease": 0,
    "nodoc": 0, "menthlth": 0, "physhlth": 0, "diffwalk": 0,
    "sex": 0, "education": 4, "income": 5
}

# Age category mapping - readable label -> BRFSS code
age_map = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
    "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
    "70-74": 11, "75-79": 12, "80+": 13
}


# BMI Calculator (shared by both modes)

st.subheader("Step 1: BMI")
bmi_method = st.radio("How would you like to provide BMI?", ["I know my BMI", "Calculate from height & weight"], horizontal=True)

if bmi_method == "I know my BMI":
    bmi = st.number_input("BMI", min_value=12.0, max_value=100.0, value=28.0, step=0.1)
else:
    col_h, col_w = st.columns(2)
    with col_h:
        height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    with col_w:
        weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    st.info(f"Calculated BMI: **{bmi:.1f}**")

# BMI category indicator (WHO standard ranges)
if bmi < 18.5:
    bmi_category, bmi_color = "Underweight", "🔵"
elif bmi < 25:
    bmi_category, bmi_color = "Normal", "🟢"
elif bmi < 30:
    bmi_category, bmi_color = "Overweight", "🟡"
else:
    bmi_category, bmi_color = "Obese", "🔴"

st.write(f"{bmi_color} BMI Category: **{bmi_category}** (WHO standard range)")

st.divider()


# QUICK ASSESSMENT MODE

if mode.startswith("⚡"):
    st.subheader("Step 2: Quick Assessment")
    st.write("Answer these 5 key questions for a fast risk estimate.")

    col1, col2 = st.columns(2)
    with col1:
        highbp = st.selectbox("High Blood Pressure?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        highchol = st.selectbox("High Cholesterol?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with col2:
        genhlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
        age_label = st.selectbox("Age Range", list(age_map.keys()), index=6)
        age = age_map[age_label]
    physactivity = st.selectbox("Physical Activity (last 30 days)?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    input_values = {
        "highbp": highbp, "highchol": highchol, "cholcheck": defaults["cholcheck"],
        "bmi": bmi, "smoker": defaults["smoker"], "stroke": defaults["stroke"],
        "heartdisease": defaults["heartdisease"], "physactivity": physactivity,
        "nodoc": defaults["nodoc"], "genhlth": genhlth, "menthlth": defaults["menthlth"],
        "physhlth": defaults["physhlth"], "diffwalk": defaults["diffwalk"],
        "sex": defaults["sex"], "age": age, "age_label": age_label,
        "education": defaults["education"], "income": defaults["income"]
    }

    st.caption("ℹ️ Quick mode uses average values for fields not asked above. For a more precise result, use Full Assessment.")


# FULL ASSESSMENT MODE

else:
    st.subheader("Step 2: Full Assessment")
    st.write("Answer all remaining questions for the most accurate estimate.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Medical History**")
        highbp = st.selectbox("High Blood Pressure?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        highchol = st.selectbox("High Cholesterol?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        cholcheck = st.selectbox("Cholesterol Check (5yr)?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        stroke = st.selectbox("Ever Had a Stroke?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        heartdisease = st.selectbox("Heart Disease/Attack?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        st.markdown("**Lifestyle & Wellbeing**")
        smoker = st.selectbox("Smoker?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        physactivity = st.selectbox("Physical Activity?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diffwalk = st.selectbox("Difficulty Walking?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        nodoc = st.selectbox("Couldn't See Doctor (Cost)?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    with col3:
        st.markdown("**Health & Demographics**")
        genhlth = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
        menthlth = st.slider("Poor Mental Health Days", 0, 30, 0)
        physhlth = st.slider("Poor Physical Health Days", 0, 30, 0)
        sex = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        age_label = st.selectbox("Age Range", list(age_map.keys()), index=6)
        age = age_map[age_label]
        education = st.slider("Education Level (1-6)", 1, 6, 4)
        income = st.slider("Income Level (1-8)", 1, 8, 4)

    input_values = {
        "highbp": highbp, "highchol": highchol, "cholcheck": cholcheck,
        "bmi": bmi, "smoker": smoker, "stroke": stroke,
        "heartdisease": heartdisease, "physactivity": physactivity,
        "nodoc": nodoc, "genhlth": genhlth, "menthlth": menthlth,
        "physhlth": physhlth, "diffwalk": diffwalk, "sex": sex,
        "age": age, "age_label": age_label, "education": education, "income": income
    }


# LIVE PREVIEW (updates as you move sliders, before clicking Predict)

st.divider()
st.subheader("👁️ Live Preview")
st.caption("This updates instantly as you change inputs above - click Predict below for the full detailed result.")

preview_values = [input_values["highbp"], input_values["highchol"], input_values["cholcheck"],
                  input_values["bmi"], input_values["smoker"], input_values["stroke"],
                  input_values["heartdisease"], input_values["physactivity"], input_values["nodoc"],
                  input_values["genhlth"], input_values["menthlth"], input_values["physhlth"],
                  input_values["diffwalk"], input_values["sex"], input_values["age"],
                  input_values["education"], input_values["income"]]

preview_probability = model.predict_proba(np.array([preview_values]))[0][1]
preview_color = "#2ecc71" if preview_probability < 0.4 else ("#f1c40f" if preview_probability < 0.7 else "#e74c3c")

st.markdown(
    f"<h2 style='color:{preview_color};'>{preview_probability*100:.1f}% estimated risk</h2>",
    unsafe_allow_html=True
)
st.progress(float(preview_probability))

# PREDICTION

st.divider()

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):
    with st.spinner("Analyzing your health profile..."):
        import time
        time.sleep(1.2)

        v = input_values
        values_list = [v["highbp"], v["highchol"], v["cholcheck"], v["bmi"], v["smoker"],
                       v["stroke"], v["heartdisease"], v["physactivity"], v["nodoc"],
                       v["genhlth"], v["menthlth"], v["physhlth"], v["diffwalk"],
                       v["sex"], v["age"], v["education"], v["income"]]

        input_data = np.array([values_list])

        probability = model.predict_proba(input_data)[0][1]
        prediction = 1 if probability >= 0.4 else 0

    st.success("Analysis complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Result")
        if prediction == 1:
            st.error("⚠️ Higher Risk of Diabetes")
        else:
            st.success("✅ Lower Risk of Diabetes")

       
        # GAUGE CHART
       
        gauge_color = "#2ecc71" if probability < 0.4 else ("#f1c40f" if probability < 0.7 else "#e74c3c")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 40], 'color': "#d5f5e3"},
                    {'range': [40, 70], 'color': "#fdebd0"},
                    {'range': [70, 100], 'color': "#fadbd8"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.8,
                    'value': probability * 100
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("What this means")
        if probability >= 0.7:
            st.write("**High** estimated risk. Consulting a doctor for proper testing is strongly recommended.")
        elif probability >= 0.4:
            st.write("**Moderate** estimated risk. Consider a check-up and lifestyle review.")
        else:
            st.write("**Lower** estimated risk based on current inputs.")

       
        # POPULATION COMPARISON CHART
        
        st.write("**How you compare to your age group** (based on survey data):")

        # Approximate diabetes rate (%) by age group, from EDA findings
        age_group_rates = {
            1: 2, 2: 3, 3: 5, 4: 8, 5: 11, 6: 15,
            7: 18, 8: 21, 9: 23, 10: 24, 11: 25, 12: 24, 13: 20
        }
        population_avg = age_group_rates.get(input_values["age"], 15)

        fig_compare = go.Figure(data=[
            go.Bar(name="Your Risk", x=["Comparison"], y=[probability * 100], marker_color=gauge_color),
            go.Bar(name="Age Group Average", x=["Comparison"], y=[population_avg], marker_color="#95a5a6")
        ])
        fig_compare.update_layout(
            barmode='group', height=280, yaxis_title="Diabetes Rate (%)",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    st.divider()

    
    # RISK FACTOR BREAKDOWN
    
    st.subheader("📊 What's driving this result")

    # Har feature ka contribution = coefficient * value (simple linear approximation)
    contributions = []
    for name, coef, val in zip(feature_names, coefficients, values_list):
        contributions.append((name, coef * val))

    # Sabse zyada positive (risk badhane wale) factors, top 3
    contributions.sort(key=lambda x: x[1], reverse=True)
    top_risk_factors = [c for c in contributions if c[1] > 0][:3]

    readable_names = {
        "HighBP": "High Blood Pressure", "HighChol": "High Cholesterol",
        "CholCheck": "Cholesterol Check", "BMI": "BMI", "Smoker": "Smoking",
        "Stroke": "Stroke History", "HeartDiseaseorAttack": "Heart Disease/Attack",
        "PhysActivity": "Physical Activity", "NoDocbcCost": "Cost Barrier to Care",
        "GenHlth": "General Health Rating", "MentHlth": "Mental Health Days",
        "PhysHlth": "Physical Health Days", "DiffWalk": "Difficulty Walking",
        "Sex": "Gender", "Age": "Age Group", "Education": "Education Level",
        "Income": "Income Level"
    }

    if top_risk_factors:
        st.write("The following factors contributed most to a **higher** risk estimate in this profile:")
        for name, _ in top_risk_factors:
            st.write(f"- {readable_names.get(name, name)}")
    else:
        st.write("No strong risk-increasing factors stood out in this profile.")

    st.divider()

   
    # ACTIONABLE HEALTH TIPS
   
    st.subheader("💡 General Health Tips")
    tips = []

    if bmi >= 25:
        tips.append("Your BMI is above the healthy range (18.5–24.9). Gradual weight management through diet and activity can meaningfully reduce diabetes risk.")
    if input_values["physactivity"] == 0:
        tips.append("Regular physical activity (even 30 min/day of walking) is strongly associated with lower diabetes risk.")
    if input_values["genhlth"] >= 4:
        tips.append("Your self-rated general health is on the lower end — a check-up can help identify and address underlying issues early.")
    if input_values["highbp"] == 1:
        tips.append("Managing blood pressure through diet, activity, and medication (if prescribed) also helps reduce diabetes-related complications.")
    if input_values["smoker"] == 1:
        tips.append("Quitting smoking improves overall metabolic health and reduces multiple chronic disease risks, including diabetes.")

    if tips:
        for tip in tips:
            st.write(f"- {tip}")
    else:
        st.write("Your profile doesn't show major lifestyle risk factors based on the inputs provided. Keep up the healthy habits!")

    st.info(
        "This tool provides an estimate based on a machine learning model trained on survey data. "
        "It is for educational purposes only and is not a medical diagnosis. "
        "Please consult a healthcare professional for accurate testing and advice."
    )

    # EMERGENCY GUIDANCE (shows only for high risk)
   
    if probability >= 0.7:
        st.divider()
        st.error(
            "🚨 **Please consult a healthcare professional soon.**\n\n"
            "This estimate suggests a meaningfully elevated risk. A doctor can run proper blood glucose "
            "tests (fasting glucose, HbA1c) to confirm your actual status and recommend next steps."
        )
        st.caption("Nearest hospital/doctor lookup — coming in a future version.")

    st.divider()

    # RADAR CHART - Health Profile Overview

    st.subheader("🕸️ Your Health Profile Overview")
    st.caption("Higher values indicate more risk-associated factors (scaled 0-100 for comparison).")

    radar_categories = ["BMI", "Blood Pressure", "Cholesterol", "General Health", "Physical Activity", "Age Factor"]
    radar_values = [
        min(bmi / 50 * 100, 100),
        100 if input_values["highbp"] == 1 else 10,
        100 if input_values["highchol"] == 1 else 10,
        (input_values["genhlth"] / 5) * 100,
        10 if input_values["physactivity"] == 1 else 100,
        (input_values["age"] / 13) * 100
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=radar_categories,
        fill='toself',
        name='Your Profile',
        line_color=gauge_color
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

  
    # DOWNLOADABLE REPORT (PDF)
   
    st.subheader("📄 Download Your Report")

    risk_level = "High" if probability >= 0.7 else ("Moderate" if probability >= 0.4 else "Lower")

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Diabetes Risk Assessment Report", ln=True, align="C")
    pdf.ln(4)

    # Result section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Result", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Probability of Diabetes: {probability*100:.1f}%", ln=True)
    pdf.cell(0, 8, f"Risk Level: {risk_level}", ln=True)
    pdf.ln(4)

    # Key inputs section
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Key Inputs", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"BMI: {bmi:.1f}", ln=True)
    pdf.cell(0, 8, f"High Blood Pressure: {'Yes' if input_values['highbp'] == 1 else 'No'}", ln=True)
    pdf.cell(0, 8, f"High Cholesterol: {'Yes' if input_values['highchol'] == 1 else 'No'}", ln=True)
    pdf.cell(0, 8, f"General Health Rating (1=Excellent, 5=Poor): {input_values['genhlth']}", ln=True)
    pdf.cell(0, 8, f"Physical Activity: {'Yes' if input_values['physactivity'] == 1 else 'No'}", ln=True)
    pdf.cell(0, 8, f"Age Range: {input_values['age_label']}", ln=True)
    pdf.ln(4)

    # Top contributing factors
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Top Contributing Factors", ln=True)
    pdf.set_font("Helvetica", "", 12)
    if top_risk_factors:
        for name, _ in top_risk_factors:
            pdf.cell(0, 8, f"- {readable_names.get(name, name)}", ln=True)
    else:
        pdf.cell(0, 8, "None identified", ln=True)
    pdf.ln(4)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(
        0, 6,
        "Disclaimer: This report is generated by a machine learning model trained on survey data "
        "(CDC BRFSS). It is for educational purposes only and is NOT a medical diagnosis. "
        "Please consult a healthcare professional for accurate testing and personalized advice."
    )

    pdf_bytes = bytes(pdf.output())

    st.download_button(
        label="⬇️ Download Report (.pdf)",
        data=pdf_bytes,
        file_name="diabetes_risk_report.pdf",
        mime="application/pdf",
        use_container_width=True
    )