import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Employee Attrition Predictor", page_icon="📊", layout="wide"
)


# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("attrition_model_pipeline.pkl")


model = load_model()

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: black;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- HEADER ---
st.title("📊 HR Intelligence: Attrition Risk Analysis")
st.markdown(
    "Predict the likelihood of employee turnover and identify key risk factors using advanced Machine Learning."
)
st.divider()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Employee Profile")
st.sidebar.markdown("Enter details to calculate risk score.")

with st.sidebar:
    # Personal Info
    with st.expander("Personal & Demographics", expanded=True):
        age = st.slider("Age", 18, 60, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        )
        distance = st.slider("Distance From Home (km)", 1, 30, 5)
        education = st.select_slider(
            "Education Level",
            options=[1, 2, 3, 4, 5],
            help="1: Below College, 5: Doctor",
        )

    # Professional Info
    with st.expander("Job & Career", expanded=True):
        dept = st.selectbox(
            "Department", ["Sales", "Research & Development", "Human Resources"]
        )
        job_role = st.selectbox(
            "Job Role",
            [
                "Sales Executive",
                "Research Scientist",
                "Laboratory Technician",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources",
            ],
        )
        job_level = st.slider("Job Level", 1, 5, 2)
        travel = st.selectbox(
            "Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
        )
        overtime = st.selectbox("Overtime", ["Yes", "No"])

    # Financial & Satisfaction
    with st.expander("Compensation & Satisfaction", expanded=False):
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
        stock_options = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        env_sat = st.slider("Environment Satisfaction", 1, 4, 3)
        job_sat = st.slider("Job Satisfaction", 1, 4, 3)
        work_life = st.slider("Work Life Balance", 1, 4, 3)

    # History
    with st.expander("Employment History", expanded=False):
        years_at_co = st.slider("Years At Company", 0, 40, 5)
        years_in_role = st.slider("Years In Current Role", 0, 20, 2)
        total_work_years = st.slider("Total Working Years", 0, 40, 10)
        num_cos = st.slider("Num Companies Worked", 0, 10, 1)
        last_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
        curr_mgr = st.slider("Years With Current Manager", 0, 20, 2)

# --- PREDICTION LOGIC ---
# Create input dictionary mapping to original column names
input_data = {
    "Age": age,
    "BusinessTravel": travel,
    "DailyRate": 800,  # Using median for missing rates
    "Department": dept,
    "DistanceFromHome": distance,
    "Education": education,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": env_sat,
    "Gender": gender,
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": job_level,
    "JobRole": job_role,
    "JobSatisfaction": job_sat,
    "MaritalStatus": marital_status,
    "MonthlyIncome": monthly_income,
    "MonthlyRate": 10000,
    "NumCompaniesWorked": num_cos,
    "OverTime": 1 if overtime == "Yes" else 0,
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": stock_options,
    "TotalWorkingYears": total_work_years,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": work_life,
    "YearsAtCompany": years_at_co,
    "YearsInCurrentRole": years_in_role,
    "YearsSinceLastPromotion": last_promo,
    "YearsWithCurrManager": curr_mgr,
}

input_df = pd.DataFrame([input_data])

# Button Trigger
if st.button("Analyze Attrition Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Result")
        if prediction == 1:
            st.error("🚨 HIGH RISK")
            st.markdown("This employee is **likely** to leave.")
        else:
            st.success("✅ LOW RISK")
            st.markdown("This employee is **likely** to stay.")

        # Confidence Gauge
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Retention Risk (%)"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "#ff4b4b" if prediction == 1 else "#00cc96"},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "salmon"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("Key Metrics Comparison")
        # Creating a spider/radar chart for satisfaction scores
        categories = [
            "Env Satisfaction",
            "Job Satisfaction",
            "Work-Life Balance",
            "Job Involvement",
        ]
        values = [env_sat, job_sat, work_life, 3]  # Hardcoded 3 for Involvement

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values, theta=categories, fill="toself", name="Employee Profile"
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 4])),
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # Recommendation Section
    st.subheader("💡 Strategic Recommendations")
    if prediction == 1:
        if overtime == "Yes":
            st.info(
                "- **Reduce Overtime:** High workload detected. Consider resource redistribution."
            )
        if monthly_income < 4000:
            st.warning(
                "- **Compensation Review:** Monthly income is below benchmark for high-retention groups."
            )
        if last_promo > 3:
            st.info(
                "- **Career Progression:** Stagnation detected. Discuss growth path or promotion."
            )
    else:
        st.info(
            "Employee appears stable. Maintain engagement through regular 1-on-1 feedback sessions."
        )

else:
    # Default State (Landing Page)
    st.info(
        "Adjust the parameters in the sidebar and click 'Analyze Attrition Risk' to see the prediction."
    )

    # Show some global insights (Optional)
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg. Accuracy", "88%", "Model Metric")
    c2.metric("Recall", "82%", "Minority Class")
    c3.metric("F1 Score", "0.85", "Balanced Performance")
