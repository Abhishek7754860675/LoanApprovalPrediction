"""
Loan Approval Predictor
AIML Summer Internship 2026 - Capstone Project (Project 3: Loan Approval Prediction System)

A multi-step Streamlit app that walks the user through applicant + loan
details and returns a model-driven approval decision, an approval score
gauge, an EMI breakdown and personalised tips.
"""

import os
import io
import math
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from fpdf import FPDF

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "Model", "loan_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "Model", "scaler.pkl")

STEP_LABELS = ["Applicant Info", "Financial Details", "Review", "Result"]
TOTAL_STEPS = len(STEP_LABELS)

# --------------------------------------------------------------------------
# Theme ("Navy" - clean white/light background, navy blue, corporate bank style)
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    font-size: 15px;
}

.stApp {
    background: #eef2f7;
    background-image:
        radial-gradient(circle at 10% 0%, rgba(37, 99, 235, 0.05) 0%, transparent 45%),
        radial-gradient(circle at 90% 8%, rgba(20, 49, 92, 0.05) 0%, transparent 45%),
        linear-gradient(180deg, #f5f7fb 0%, #eef2f7 100%);
    background-attachment: fixed;
    color: #1e293b;
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 720px;
}

/* ---- Hero header ---- */
.hero-card {
    text-align: center;
    padding: 1.3rem 1rem 1.1rem 1rem;
    background: linear-gradient(120deg, #14315c, #1e4d7b);
    border-radius: 20px;
    margin-bottom: 1rem;
    box-shadow: 0 10px 26px rgba(20, 49, 92, 0.18);
}
.hero-title {
    font-size: clamp(1.4rem, 5vw, 2rem);
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.01em;
    margin-bottom: 0.3rem;
}
.hero-sub {
    color: #cfe0f2;
    font-size: clamp(0.78rem, 2.2vw, 0.92rem);
    margin-bottom: 0;
}

/* ---- Progress ---- */
.progress-wrap {
    background: rgba(20,49,92,0.08);
    border: 1px solid rgba(20,49,92,0.08);
    border-radius: 999px;
    height: 8px;
    width: 100%;
    overflow: hidden;
    margin-bottom: 0.6rem;
}
.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #1e4d7b, #2563eb);
    transition: width 0.4s ease;
}
.progress-caption {
    text-align: center;
    font-size: 0.76rem;
    color: #64748b;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 1.3rem;
}
.dots-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 0.6rem;
}
.dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: rgba(20,49,92,0.18);
}
.dot.active {
    background: #2563eb;
    box-shadow: 0 0 10px rgba(37,99,235,0.5);
    transform: scale(1.3);
}
.dot.done {
    background: #1e4d7b;
}

/* ---- Card ---- */
.section-card {
    background: #ffffff;
    border: 1px solid rgba(20,49,92,0.08);
    border-radius: 18px;
    padding: 1.4rem 1.5rem 0.5rem 1.5rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 4px 16px rgba(20,49,92,0.06);
}
.section-heading {
    font-size: clamp(1rem, 3.6vw, 1.15rem);
    font-weight: 700;
    color: #14315c;
    margin-bottom: 0.9rem;
}

/* ---- Streamlit widget restyling ---- */
div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid rgba(20,49,92,0.08);
    border-radius: 18px;
    padding: 1.5rem 1.5rem 1rem 1.5rem;
    box-shadow: 0 4px 20px rgba(20,49,92,0.07);
}

label[data-testid="stWidgetLabel"] p {
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    color: #1e4d7b !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input,
.stTextInput input {
    background-color: #f1f5f9 !important;
    color: #1e293b !important;
    border-radius: 10px !important;
    border: 1px solid rgba(20,49,92,0.10) !important;
    font-weight: 500;
    font-size: 0.9rem !important;
}
.stNumberInput input, .stTextInput input {
    padding: 0.5rem 0.75rem !important;
}
div[data-baseweb="select"] svg { color: #1e4d7b !important; }

.stButton button, button[kind="formSubmit"], .stDownloadButton button {
    background: #14315c !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.01em;
    width: 100%;
    box-shadow: 0 6px 16px rgba(20,49,92,0.22);
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.stButton button:hover, button[kind="formSubmit"]:hover, .stDownloadButton button:hover {
    background: #1e4d7b !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(20,49,92,0.3);
}

button[kind="secondary"] {
    background: #ffffff !important;
    color: #14315c !important;
    box-shadow: none !important;
    border: 1.5px solid rgba(20,49,92,0.25) !important;
}
button[kind="secondary"]:hover {
    background: #f1f5f9 !important;
    border-color: rgba(20,49,92,0.4) !important;
}

/* ---- Result page pieces ---- */
.status-banner {
    text-align: center;
    padding: 1rem;
    border-radius: 16px;
    font-size: clamp(1rem, 3.8vw, 1.15rem);
    font-weight: 800;
    margin-bottom: 0.9rem;
}
.status-approved {
    background: rgba(22,163,74,0.08);
    border: 1px solid rgba(22,163,74,0.35);
    color: #15803d;
}
.status-rejected {
    background: rgba(220,38,38,0.07);
    border: 1px solid rgba(220,38,38,0.3);
    color: #b91c1c;
}

div[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    display: flex;
    flex-direction: column;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
    flex: 1;
    display: flex;
    flex-direction: column;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] div[data-testid="stMarkdownContainer"] {
    height: 100%;
}

.congrats-card, .sorry-card, .emi-card {
    background: #ffffff;
    border: 1px solid rgba(20,49,92,0.08);
    border-radius: 16px;
    padding: 1.2rem;
    height: 100%;
    min-height: 220px;
    box-sizing: border-box;
    box-shadow: 0 4px 14px rgba(20,49,92,0.06);
}
.congrats-card { border-color: rgba(22,163,74,0.25); }
.sorry-card { border-color: rgba(220,38,38,0.25); }

.emi-card-title {
    font-weight: 700;
    font-size: 0.98rem;
    color: #14315c;
    margin-bottom: 0.6rem;
}
.emi-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 0.6rem;
    padding: 0.42rem 0;
    border-bottom: 1px dashed rgba(20,49,92,0.14);
    font-size: 0.85rem;
    line-height: 1.3;
}
.emi-row:last-child { border-bottom: none; }
.emi-label { color: #64748b; flex-shrink: 0; }
.emi-value { font-weight: 700; color: #1e293b; text-align: right; word-break: break-word; }
.emi-highlight .emi-value { color: #14315c; font-size: 1.05rem; }
.emi-note {
    font-size: 0.72rem;
    color: #2563eb;
    padding: 0.3rem 0 0.5rem 0;
    border-bottom: 1px dashed rgba(20,49,92,0.14);
}

.tips-card {
    background: rgba(37, 99, 235, 0.05);
    border: 1px solid rgba(37, 99, 235, 0.18);
    border-radius: 16px;
    padding: 0.6rem 0.9rem;
    margin-top: 0.8rem;
}
.tip-item {
    display: flex;
    gap: 0.6rem;
    padding: 0.32rem 0;
    font-size: 0.85rem;
    color: #334155;
    line-height: 1.4;
}

div[data-testid="stExpander"] {
    background: rgba(37, 99, 235, 0.04);
    border: 1px solid rgba(37, 99, 235, 0.18);
    border-radius: 16px;
    margin-top: 0.9rem;
}
div[data-testid="stExpander"] summary {
    font-weight: 700;
    color: #14315c;
    font-size: 0.88rem;
}

.review-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem 1.4rem;
}
.review-item {
    padding: 0.45rem 0;
    border-bottom: 1px dashed rgba(20,49,92,0.10);
    overflow-wrap: break-word;
}
.review-key {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: #1e4d7b;
    font-weight: 700;
}
.review-val {
    font-size: 0.88rem;
    color: #1e293b;
    font-weight: 500;
}

.rate-note {
    background: rgba(37, 99, 235, 0.05);
    border: 1px solid rgba(37, 99, 235, 0.2);
    border-radius: 12px;
    padding: 0.65rem 0.9rem;
    font-size: 0.8rem;
    color: #1e4d7b;
    margin: 0.2rem 0 0.9rem 0;
}

@media (max-width: 480px) {
    .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
    .review-grid { grid-template-columns: 1fr; }
    .section-card, div[data-testid="stForm"] { padding-left: 1.1rem; padding-right: 1.1rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Model / scaler loading
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1
if "data" not in st.session_state:
    st.session_state.data = {}
if "result" not in st.session_state:
    st.session_state.result = None
if "confetti_shown" not in st.session_state:
    st.session_state.confetti_shown = False


def go_to(step):
    st.session_state.step = step
    st.rerun()


def restart():
    st.session_state.step = 1
    st.session_state.data = {}
    st.session_state.result = None
    st.session_state.confetti_shown = False
    st.rerun()


# --------------------------------------------------------------------------
# Header + progress
# --------------------------------------------------------------------------
def render_header():
    st.markdown(
        '<div class="hero-card">'
        '<div class="hero-title">🏦 Loan Approval Predictor</div>'
        '<div class="hero-sub">Check your loan approval chances — powered by your trained ML model</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    step = st.session_state.step
    pct = int((step - 1) / (TOTAL_STEPS - 1) * 100) if TOTAL_STEPS > 1 else 100

    dots_html = '<div class="dots-row">'
    for i in range(1, TOTAL_STEPS + 1):
        cls = "dot"
        if i < step:
            cls += " done"
        elif i == step:
            cls += " active"
        dots_html += f'<div class="{cls}"></div>'
    dots_html += "</div>"

    st.markdown(dots_html, unsafe_allow_html=True)
    st.markdown(
        f'<div class="progress-wrap"><div class="progress-fill" style="width:{pct}%;"></div></div>'
        f'<div class="progress-caption">Step {step} of {TOTAL_STEPS} · {STEP_LABELS[step - 1]}</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Profession -> typical/"normal" interest rate (used only for the EMI
# calculator, not fed into the ML model since it wasn't trained on it)
# --------------------------------------------------------------------------
PROFESSION_RATES = {
    "Salaried - Government": 8.5,
    "Salaried - Private": 9.5,
    "Self-Employed - Business": 11.5,
    "Self-Employed - Professional": 10.5,
    "Freelancer / Gig Worker": 13.0,
    "Farmer / Agriculture": 9.0,
    "Retired / Pensioner": 10.0,
    "Other": 11.0,
}


# --------------------------------------------------------------------------
# Step 1 - Applicant Information
# --------------------------------------------------------------------------
def render_step1():
    st.markdown('<div class="section-heading">👤 Applicant Information</div>', unsafe_allow_html=True)
    with st.form("step1_form"):
        d = st.session_state.data
        name = st.text_input("Applicant Name", value=d.get("Name", ""), placeholder="Enter your full name")
        gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(d.get("Gender", "Male")))
        married = st.selectbox("Married", ["Yes", "No"], index=["Yes", "No"].index(d.get("Married", "Yes")))
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=["0", "1", "2", "3+"].index(d.get("Dependents", "0")))
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=["Graduate", "Not Graduate"].index(d.get("Education", "Graduate")))
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=["No", "Yes"].index(d.get("Self_Employed", "No")))
        profession = st.selectbox(
            "Profession", list(PROFESSION_RATES.keys()),
            index=list(PROFESSION_RATES.keys()).index(d.get("Profession", "Salaried - Private")),
        )
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=["Urban", "Semiurban", "Rural"].index(d.get("Property_Area", "Urban")))

        submitted = st.form_submit_button("Next →")
        if submitted:
            st.session_state.data.update({
                "Name": name.strip() if name.strip() else "Applicant",
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "Profession": profession,
                "Property_Area": property_area,
            })
            go_to(2)


# --------------------------------------------------------------------------
# Step 2 - Financial & Loan Details
# --------------------------------------------------------------------------
LOAN_TERMS = [12, 36, 60, 84, 120, 180, 240, 300, 360, 480]


def render_step2():
    st.markdown('<div class="section-heading">💰 Financial &amp; Loan Details</div>', unsafe_allow_html=True)
    with st.form("step2_form"):
        d = st.session_state.data
        applicant_income = st.number_input(
            "Applicant's Monthly Income (₹)", min_value=0, step=500,
            value=int(d.get("ApplicantIncome", 5000)),
        )
        coapplicant_income = st.number_input(
            "Co-applicant's Monthly Income (₹)", min_value=0, step=500,
            value=int(d.get("CoapplicantIncome", 0)),
        )
        loan_amount = st.number_input(
            "Loan Amount Requested (₹)", min_value=0, step=5000,
            value=int(d.get("LoanAmountActual", 120000)),
        )
        term_default = d.get("Loan_Amount_Term", 360)
        term_labels = [f"{t} months ({t // 12} yrs)" for t in LOAN_TERMS]
        term_index = LOAN_TERMS.index(term_default) if term_default in LOAN_TERMS else LOAN_TERMS.index(360)
        loan_term_label = st.selectbox("Loan Tenure", term_labels, index=term_index)
        loan_term = LOAN_TERMS[term_labels.index(loan_term_label)]

        credit_options = ["Yes (good repayment record)", "No (defaults / no credit history)"]
        credit_default = d.get("Credit_History_Label", credit_options[0])
        credit_history_label = st.selectbox("Good Credit History?", credit_options, index=credit_options.index(credit_default))

        profession = d.get("Profession", "Salaried - Private")
        normal_rate = PROFESSION_RATES.get(profession, 11.0)
        st.markdown(
            f'<div class="rate-note">💼 Normal interest rate for <b>{profession}</b> applicants is around '
            f'<b>{normal_rate}% p.a.</b> — feel free to adjust it on the scale below to match your actual offer.</div>',
            unsafe_allow_html=True,
        )
        interest_rate = st.slider(
            "Interest Rate (% per annum)", min_value=6.0, max_value=20.0,
            value=float(d.get("InterestRate", normal_rate)), step=0.1,
        )

        col1, col2 = st.columns(2)
        with col1:
            back = st.form_submit_button("← Back")
        with col2:
            submitted = st.form_submit_button("Next →")

        if back:
            go_to(1)
        if submitted:
            st.session_state.data.update({
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmountActual": loan_amount,
                "Loan_Amount_Term": loan_term,
                "Credit_History_Label": credit_history_label,
                "InterestRate": interest_rate,
                "NormalRate": normal_rate,
            })
            go_to(3)


# --------------------------------------------------------------------------
# Step 3 - Review
# --------------------------------------------------------------------------
def render_step3():
    d = st.session_state.data
    st.markdown('<div class="section-heading">🧾 Review Your Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    rows = [
        ("Name", d.get("Name")),
        ("Profession", d.get("Profession")),
        ("Gender", d.get("Gender")),
        ("Married", d.get("Married")),
        ("Dependents", d.get("Dependents")),
        ("Education", d.get("Education")),
        ("Self Employed", d.get("Self_Employed")),
        ("Property Area", d.get("Property_Area")),
        ("Applicant Income", f"₹{d.get('ApplicantIncome', 0):,} / month"),
        ("Co-applicant Income", f"₹{d.get('CoapplicantIncome', 0):,} / month"),
        ("Loan Amount", f"₹{d.get('LoanAmountActual', 0):,}"),
        ("Loan Tenure", f"{d.get('Loan_Amount_Term', 0)} months"),
        ("Credit History", d.get("Credit_History_Label")),
        ("Interest Rate", f"{d.get('InterestRate', 0)}% p.a."),
    ]

    grid = '<div class="review-grid">'
    for k, v in rows:
        grid += f'<div class="review-item"><div class="review-key">{k}</div><div class="review-val">{v}</div></div>'
    grid += "</div>"
    st.markdown(grid, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back", key="review_back", use_container_width=True, type="secondary"):
            go_to(2)
    with col2:
        if st.button("🔍 Predict My Loan Approval", key="review_predict", use_container_width=True):
            run_prediction()
            go_to(4)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
GENDER_MAP = {"Male": 1, "Female": 0}
MARRIED_MAP = {"Yes": 1, "No": 0}
DEPENDENTS_MAP = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED_MAP = {"Yes": 1, "No": 0}
PROPERTY_MAP = {"Rural": 0, "Semiurban": 1, "Urban": 2}


def run_prediction():
    d = st.session_state.data
    model, scaler = load_artifacts()

    gender_enc = GENDER_MAP[d["Gender"]]
    married_enc = MARRIED_MAP[d["Married"]]
    dependents_enc = DEPENDENTS_MAP[d["Dependents"]]
    education_enc = EDUCATION_MAP[d["Education"]]
    self_employed_enc = SELF_EMPLOYED_MAP[d["Self_Employed"]]
    property_enc = PROPERTY_MAP[d["Property_Area"]]
    credit_history_enc = 1 if d["Credit_History_Label"].startswith("Yes") else 0

    applicant_income = float(d["ApplicantIncome"])
    coapplicant_income = float(d["CoapplicantIncome"])
    loan_amount_actual = float(d["LoanAmountActual"])
    loan_amount_k = loan_amount_actual / 1000.0  # dataset's LoanAmount is in thousands
    loan_term = float(d["Loan_Amount_Term"])

    total_income = applicant_income + coapplicant_income
    denom = total_income / 1000.0 if total_income > 0 else 1.0
    loan_income_ratio = loan_amount_k / denom

    features = np.array([[
        gender_enc, married_enc, dependents_enc, education_enc, self_employed_enc,
        loan_amount_k, loan_term, credit_history_enc, property_enc,
        total_income, loan_income_ratio,
    ]])

    scaled = scaler.transform(features)
    proba_approved = float(model.predict_proba(scaled)[0][1])
    prediction = int(model.predict(scaled)[0])  # 1 = approved (Y), 0 = not approved (N)

    # EMI calculation - interest rate based on the applicant's profession,
    # adjustable by the user on the slider in Step 2
    annual_rate = float(d.get("InterestRate", d.get("NormalRate", 11.0)))
    normal_rate = float(d.get("NormalRate", annual_rate))
    monthly_rate = annual_rate / 12 / 100
    n = loan_term
    if monthly_rate > 0 and n > 0:
        emi = loan_amount_actual * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)
    elif n > 0:
        emi = loan_amount_actual / n
    else:
        emi = 0
    total_payment = emi * n
    total_interest = total_payment - loan_amount_actual

    st.session_state.result = {
        "approved": prediction == 1,
        "score": round(proba_approved * 100, 1),
        "annual_rate": annual_rate,
        "normal_rate": normal_rate,
        "emi": emi,
        "total_payment": total_payment,
        "total_interest": total_interest,
        "loan_amount": loan_amount_actual,
        "term_months": n,
        "credit_history_enc": credit_history_enc,
        "loan_income_ratio": loan_income_ratio,
        "total_income": total_income,
    }
    # a fresh prediction should be able to trigger the celebration again
    st.session_state.confetti_shown = False


# --------------------------------------------------------------------------
# Gauge
# --------------------------------------------------------------------------
def render_gauge(score):
    if score < 40:
        bar_color = "#f87171"
    elif score < 70:
        bar_color = "#fbbf24"
    else:
        bar_color = "#4ade80"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 38, "color": "#14315c"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8", "tickfont": {"color": "#64748b", "size": 11}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(248,113,113,0.25)"},
                {"range": [40, 70], "color": "rgba(251,191,36,0.25)"},
                {"range": [70, 100], "color": "rgba(74,222,128,0.25)"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 4},
                "thickness": 0.85,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1e293b", "family": "Poppins"},
        height=270,
        margin=dict(l=25, r=25, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# --------------------------------------------------------------------------
# Celebration - full-page confetti burst for a few seconds on approval
# --------------------------------------------------------------------------
def fire_celebration():
    components.html(
        """
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js"></script>
        <script>
        (function() {
            const doc = window.parent.document;
            const old = doc.getElementById('loan-confetti-canvas');
            if (old) { old.remove(); }

            const canvas = doc.createElement('canvas');
            canvas.id = 'loan-confetti-canvas';
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100vw';
            canvas.style.height = '100vh';
            canvas.style.pointerEvents = 'none';
            canvas.style.zIndex = '999999';
            doc.body.appendChild(canvas);

            function launch() {
                const myConfetti = window.parent.confetti.create(canvas, { resize: true, useWorker: true });
                const colors = ['#14315c', '#2563eb', '#60a5fa', '#16a34a'];
                const end = Date.now() + 3500;

                (function frame() {
                    myConfetti({ particleCount: 5, angle: 60, spread: 60, startVelocity: 45, origin: { x: 0, y: 0.7 }, colors: colors });
                    myConfetti({ particleCount: 5, angle: 120, spread: 60, startVelocity: 45, origin: { x: 1, y: 0.7 }, colors: colors });
                    myConfetti({ particleCount: 3, angle: 90, spread: 90, startVelocity: 35, origin: { x: 0.5, y: 0 }, colors: colors });
                    if (Date.now() < end) {
                        requestAnimationFrame(frame);
                    } else {
                        setTimeout(function() { canvas.remove(); }, 800);
                    }
                })();
            }

            if (window.parent.confetti) {
                launch();
            } else {
                const script = doc.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/canvas-confetti@1.9.3/dist/confetti.browser.min.js';
                script.onload = launch;
                doc.body.appendChild(script);
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


# --------------------------------------------------------------------------
# Tips
# --------------------------------------------------------------------------
def get_tips(result, d):
    tips = []
    if result["approved"]:
        tips.append("🎯 Keep your EMI within 30-40% of your monthly income to stay comfortably within budget.")
        tips.append("📅 Set up auto-pay for your EMI so you never miss a due date and protect your credit score.")
        tips.append("💳 Avoid taking on new large loans or credit cards right before/after this loan is disbursed.")
        tips.append("🏦 Consider a small prepayment when you get a bonus — it reduces total interest significantly.")
        if result["loan_income_ratio"] > 15:
            tips.append("⚖️ Your loan-to-income ratio is on the higher side — build an emergency fund of 3-6 EMIs.")
    else:
        tips.append("📈 Improve your credit history by paying existing dues/bills on time for a few months.")
        tips.append("👥 Adding a co-applicant with steady income can improve your combined eligibility.")
        tips.append("💰 Consider requesting a smaller loan amount or a longer tenure to lower the loan-to-income ratio.")
        tips.append("🧾 Keep salary slips / ITRs updated — consistent, documented income improves approval chances.")
        if result["credit_history_enc"] == 0:
            tips.append("🔴 A poor/no credit history was a major factor here — resolving past defaults helps the most.")
    return tips


# --------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------
def build_pdf_report(d, result):
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)
    pdf.add_page()
    left = pdf.l_margin
    page_w = pdf.w - pdf.l_margin - pdf.r_margin

    def new_line(h=8):
        # Always return the cursor to the left margin on a fresh line.
        pdf.set_x(left)
        pdf.ln(h)

    def center_text(text, h, size, bold=True, color=(238, 241, 255)):
        pdf.set_x(left)
        pdf.set_text_color(*color)
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.cell(page_w, h, text, align="C")
        new_line(h)

    pdf.set_fill_color(13, 18, 36)
    pdf.rect(0, 0, 210, 297, "F")

    center_text("Loan Approval Predictor - Report", 12, 20, color=(103, 232, 249))
    center_text(datetime.now().strftime("Generated on %d %b %Y, %H:%M"), 8, 10, bold=False, color=(150, 160, 200))
    pdf.ln(4)

    status_text = "APPROVED" if result["approved"] else "NOT APPROVED"
    status_color = (74, 222, 128) if result["approved"] else (248, 113, 113)
    center_text(f"Decision: {status_text}  |  Approval Score: {result['score']}%", 12, 15, color=status_color)
    pdf.ln(4)

    def section(title):
        pdf.set_x(left)
        pdf.set_text_color(216, 221, 255)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(page_w, 10, title)
        new_line(10)
        pdf.set_draw_color(80, 80, 120)
        pdf.line(left, pdf.get_y(), left + page_w, pdf.get_y())
        pdf.ln(3)

    def row(label, value):
        pdf.set_x(left)
        pdf.set_text_color(154, 164, 196)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(70, 8, str(label))
        pdf.set_text_color(238, 241, 255)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(page_w - 70, 8, str(value))
        new_line(8)

    section("Applicant Information")
    row("Name", d.get("Name"))
    row("Profession", d.get("Profession"))
    row("Gender", d.get("Gender"))
    row("Married", d.get("Married"))
    row("Dependents", d.get("Dependents"))
    row("Education", d.get("Education"))
    row("Self Employed", d.get("Self_Employed"))
    row("Property Area", d.get("Property_Area"))
    pdf.ln(3)

    section("Financial & Loan Details")
    row("Applicant Income", f"Rs. {d.get('ApplicantIncome', 0):,} / month")
    row("Co-applicant Income", f"Rs. {d.get('CoapplicantIncome', 0):,} / month")
    row("Loan Amount Requested", f"Rs. {d.get('LoanAmountActual', 0):,}")
    row("Loan Tenure", f"{d.get('Loan_Amount_Term', 0)} months")
    row("Credit History", d.get("Credit_History_Label"))
    pdf.ln(3)

    section(f"EMI Breakdown (rate used: {result['annual_rate']}% p.a.)")
    row("Normal Rate for Profession", f"{result['normal_rate']}% p.a.")
    row("Monthly EMI", f"Rs. {result['emi']:,.0f}")
    row("Total Payment", f"Rs. {result['total_payment']:,.0f}")
    row("Total Interest", f"Rs. {result['total_interest']:,.0f}")
    pdf.ln(3)

    section("Tips")
    pdf.set_text_color(200, 205, 240)
    pdf.set_font("Helvetica", "", 10)
    for tip in get_tips(result, d):
        clean_tip = tip.encode("latin-1", "ignore").decode("latin-1").strip()
        pdf.set_x(left)
        pdf.multi_cell(page_w, 6, f"- {clean_tip}")
        pdf.set_x(left)
    pdf.ln(2)

    pdf.set_text_color(120, 128, 160)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_x(left)
    pdf.multi_cell(page_w, 5, "This report is generated by a student capstone ML project for educational purposes and is not a real loan sanction.")

    raw = pdf.output()
    return bytes(raw)


# --------------------------------------------------------------------------
# Step 4 - Result
# --------------------------------------------------------------------------
def render_step4():
    result = st.session_state.result
    d = st.session_state.data
    if result is None:
        run_prediction()
        result = st.session_state.result

    name = d.get("Name", "Applicant")

    if result["approved"]:
        st.markdown('<div class="status-banner status-approved">✅ Loan Likely to be APPROVED</div>', unsafe_allow_html=True)
        if not st.session_state.confetti_shown:
            fire_celebration()
            st.session_state.confetti_shown = True
    else:
        st.markdown('<div class="status-banner status-rejected">❌ Loan Likely to be NOT APPROVED</div>', unsafe_allow_html=True)

    render_gauge(result["score"])

    col1, col2 = st.columns(2)
    with col1:
        if result["approved"]:
            st.markdown(
                f'<div class="congrats-card">'
                f'<div style="font-size:clamp(1.05rem,4vw,1.25rem); font-weight:800; color:#15803d; line-height:1.3;">🎉 Congratulations, {name}!</div>'
                f'<div style="color:#475569; margin-top:0.5rem; font-size:0.85rem; line-height:1.5;">'
                f'Based on your profile, your loan application shows a strong approval score of '
                f'<b>{result["score"]}%</b>. Your credit history, income and loan amount align well '
                f'with the trained model\'s approval pattern.'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="sorry-card">'
                f'<div style="font-size:clamp(1rem,3.8vw,1.15rem); font-weight:800; color:#b91c1c; line-height:1.3;">😔 Not Approved This Time, {name}</div>'
                f'<div style="color:#475569; margin-top:0.5rem; font-size:0.85rem; line-height:1.5;">'
                f'Your approval score came out to <b>{result["score"]}%</b>, below the model\'s approval '
                f'threshold. Check the tips below to improve your chances next time.'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    with col2:
        emi_rows = "".join([
            f'<div class="emi-card-title">💳 Loan &amp; EMI Summary</div>',
            f'<div class="emi-row"><span class="emi-label">Loan Amount</span><span class="emi-value">₹{result["loan_amount"]:,.0f}</span></div>',
            f'<div class="emi-row"><span class="emi-label">Interest Rate</span><span class="emi-value">{result["annual_rate"]}% p.a.</span></div>',
            f'<div class="emi-row emi-highlight"><span class="emi-label">Monthly EMI</span><span class="emi-value">₹{result["emi"]:,.0f}</span></div>',
            f'<div class="emi-row"><span class="emi-label">Total Amount to Pay</span><span class="emi-value">₹{result["total_payment"]:,.0f}</span></div>',
        ])
        st.markdown(f'<div class="emi-card">{emi_rows}</div>', unsafe_allow_html=True)

    tips = get_tips(result, d)
    with st.expander("💡 Tips For You — click to view"):
        tips_html = '<div class="tips-card">'
        for tip in tips:
            tips_html += f'<div class="tip-item">{tip}</div>'
        tips_html += "</div>"
        st.markdown(tips_html, unsafe_allow_html=True)

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        pdf_bytes = build_pdf_report(d, result)
        st.download_button(
            "⬇️ Download Report",
            data=pdf_bytes,
            file_name="Loan_Approval_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with col2:
        if st.button("🔄 Start Over", use_container_width=True, type="secondary"):
            restart()


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
render_header()

step = st.session_state.step
if step == 1:
    render_step1()
elif step == 2:
    render_step2()
elif step == 3:
    render_step3()
elif step == 4:
    render_step4()
