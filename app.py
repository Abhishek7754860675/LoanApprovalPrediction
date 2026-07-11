import io
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# =========================================================================
#  PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title='Loan Approval Predictor',
    page_icon=None,
    layout='wide',
    initial_sidebar_state='expanded',
)

# =========================================================================
#  CUSTOM CSS
# =========================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {font-family: 'Poppins', sans-serif;}

    .stApp {background: linear-gradient(180deg, #eef2f9 0%, #ffffff 55%);}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 980px;}

    .header {
        background: linear-gradient(120deg, #10193a 0%, #16305e 55%, #1f8a8a 130%);
        padding: 1.7rem 2rem; border-radius: 20px; color: white;
        box-shadow: 0 14px 30px rgba(16, 25, 58, 0.25); margin-bottom: 1.6rem;
    }
    .header h1 {margin:0; font-size:1.9rem; font-weight:700; letter-spacing: -0.01em;}
    .header p {margin:0.3rem 0 0; font-size:0.98rem; opacity:0.85;}

    /* progress bar */
    .progress-wrap {display:flex; gap:8px; margin: 0 0 1.6rem 0;}
    .progress-seg {flex:1; height:6px; border-radius:999px; background:#dfe6f2;}
    .progress-seg.active {background: linear-gradient(90deg,#16305e,#1f8a8a);}

    .card {
        background: #ffffff; padding: 1.5rem 1.7rem; border-radius: 18px;
        box-shadow: 0 8px 22px rgba(16, 25, 58, 0.06);
        margin-bottom: 1.2rem; border: 1px solid #eef1f7;
    }

    .result-badge-approved, .result-badge-rejected {
        display:inline-block; padding: 0.4rem 1.3rem; border-radius: 999px;
        font-weight:700; letter-spacing: 0.02em; font-size: 0.95rem;
    }
    .result-badge-approved {background: #e4f8ef; color:#12805c; border: 1px solid #b6ecd4;}
    .result-badge-rejected {background: #fdeceb; color:#b8362a; border: 1px solid #f6c6c1;}

    .suggestion-box {
        background: #f4f8ff; border-left: 3px solid #1f8a8a; padding: 0.9rem 1.1rem;
        border-radius: 10px; font-size: 0.93rem; color:#24314f; line-height:1.5;
    }

    div.stButton > button, div.stDownloadButton > button {
        border-radius: 999px; padding: 0.65rem 1.4rem;
        background: linear-gradient(90deg, #16305e, #1f8a8a); color: white;
        border: none; font-weight: 600; transition: all 0.15s ease-in-out;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 8px 18px rgba(16, 48, 94, 0.25);
    }

    [data-testid="stMetricValue"] {font-weight: 700; color:#16305e;}
    section[data-testid="stSidebar"] {background:#10193a;}
    section[data-testid="stSidebar"] * {color:#e8ecf6 !important;}
    section[data-testid="stSidebar"] div.stButton > button {background:#1f8a8a; color:white !important;}
    hr {margin: 0.6rem 0 1.1rem 0; border-color:#eef1f7;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================================
#  FONT HELPER
# =========================================================================
def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


# =========================================================================
#  MODEL LOADING
# =========================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / 'Model' / 'loan_model.pkl'
    scaler_path = base_dir / 'Model' / 'scaler.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


try:
    model, scaler = load_model()
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    LOAD_ERROR = str(e)

# =========================================================================
#  SESSION STATE
# =========================================================================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'history' not in st.session_state:
    st.session_state.history = []
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'result_data' not in st.session_state:
    st.session_state.result_data = None


def go_to(step):
    st.session_state.step = step


def reset_all():
    st.session_state.step = 1
    st.session_state.form_data = {}
    st.session_state.result_data = None


# =========================================================================
#  HELPERS
# =========================================================================
def compute_emi(principal, annual_rate_pct, term_months):
    if principal <= 0 or term_months <= 0:
        return 0.0
    r = (annual_rate_pct / 12) / 100
    if r == 0:
        return principal / term_months
    return principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)


def make_gradient(width, height, top_color, bottom_color):
    base = Image.new('RGB', (width, height), top_color)
    top = Image.new('RGB', (width, height), bottom_color)
    mask = Image.new('L', (width, height))
    mask_data = []
    for y in range(height):
        mask_data.extend([int(255 * (y / height))] * width)
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)
    return base


def draw_gauge(draw, cx, cy, radius, score, thickness=26):
    zones = [(0, 40, (231, 76, 60)), (40, 70, (244, 180, 66)), (70, 100, (31, 138, 138))]
    for start_pct, end_pct, color in zones:
        start_angle = 180 + (start_pct / 100) * 180
        end_angle = 180 + (end_pct / 100) * 180
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        draw.arc(bbox, start_angle, end_angle, fill=color, width=thickness)

    needle_angle_deg = 180 + (score / 100) * 180
    needle_angle_rad = math.radians(needle_angle_deg)
    needle_len = radius - thickness / 2 - 4
    nx = cx + needle_len * math.cos(needle_angle_rad)
    ny = cy + needle_len * math.sin(needle_angle_rad)
    draw.line([cx, cy, nx, ny], fill=(16, 25, 58), width=6)
    draw.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill=(16, 25, 58))


def render_gauge_image(score, width=520, height=310):
    """Transparent-background needle gauge for on-screen display."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy, radius = width // 2, height - 30, min(width // 2 - 20, height - 60)
    draw_gauge(draw, cx, cy, radius, score, thickness=30)
    text = f"{score}%"
    font = get_font(48, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw / 2, cy - radius * 0.55), text, font=font, fill=(16, 25, 58))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def build_scorecard_png(name, score, result, dti_pct, credit_history):
    W, H = 1000, 560
    img = make_gradient(W, H, (247, 251, 255), (223, 238, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, 110], fill=(16, 25, 58))
    draw.text((40, 30), "Loan Approval Report", font=get_font(38, bold=True), fill="white")
    draw.text((40, 78), f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
              font=get_font(16), fill=(200, 210, 230))

    display_name = name.strip() if name and name.strip() else "Applicant"
    draw.text((40, 150), "Applicant", font=get_font(18), fill=(100, 112, 135))
    draw.text((40, 175), display_name, font=get_font(34, bold=True), fill=(20, 28, 50))

    cx, cy, radius = 250, 380, 150
    draw_gauge(draw, cx, cy, radius, score)
    draw.text((cx - 55, cy - 55), f"{score}%", font=get_font(46, bold=True), fill=(16, 25, 58))
    draw.text((cx - 65, cy + 5), "Approval Score", font=get_font(16), fill=(100, 112, 135))

    badge_color = (31, 138, 138) if result == "Approved" else (184, 54, 42)
    draw.rounded_rectangle([560, 150, 940, 210], radius=30, fill=badge_color)
    label = "LOAN APPROVED" if result == "Approved" else "LOAN REJECTED"
    draw.text((610, 165), label, font=get_font(24, bold=True), fill="white")

    stats = [
        ("Debt-to-Income Ratio", f"{dti_pct:.1f}%"),
        ("Credit History", "Good" if credit_history == 1.0 else "Poor / None"),
    ]
    y = 250
    for label_txt, value_txt in stats:
        draw.text((560, y), label_txt, font=get_font(18), fill=(100, 112, 135))
        draw.text((560, y + 24), value_txt, font=get_font(26, bold=True), fill=(20, 28, 50))
        y += 80

    draw.line([40, H - 60, W - 40, H - 60], fill=(205, 218, 235), width=2)
    draw.text((40, H - 45), "Generated by Loan Approval Predictor  -  For informational purposes only",
              font=get_font(14), fill=(120, 132, 155))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def build_suggestion(credit_history, dti_pct, coapplicant_income, total_income):
    if credit_history == 0.0:
        return "Your credit history is marked poor or absent - this weighs heavily on the outcome. A clean repayment record is the fastest way to improve your odds."
    if dti_pct > 45:
        return f"Your estimated EMI is about {dti_pct:.0f}% of monthly income. Lenders prefer this under 40% - consider a smaller amount or a longer term."
    if coapplicant_income == 0:
        return "Adding a co-applicant's income could meaningfully raise your eligibility."
    if total_income == 0:
        return "No income was entered, so this result should not be relied on."
    return "Your profile looks solid across the key factors considered. Keep your credit history clean and debt manageable."


# =========================================================================
#  SIDEBAR
# =========================================================================
with st.sidebar:
    st.markdown("### Loan Approval Predictor")
    st.markdown("---")
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.markdown(f"**History ({len(hist_df)})**")
        st.dataframe(hist_df[['Applicant', 'Prediction', 'Score']], hide_index=True, use_container_width=True)
        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download history", data=csv, file_name="loan_prediction_history.csv", mime="text/csv")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("Checked applications will appear here.")

# =========================================================================
#  HEADER + PROGRESS
# =========================================================================
st.markdown(
    '<div class="header"><h1>Loan Approval Predictor</h1>'
    '<p>A quick, guided check of loan eligibility.</p></div>',
    unsafe_allow_html=True,
)

if not MODEL_READY:
    st.error(f"Could not load the model files.\n\nDetails: {LOAD_ERROR}")
    st.stop()

step = st.session_state.step
st.markdown(
    '<div class="progress-wrap">'
    + ''.join(f'<div class="progress-seg{" active" if i <= step else ""}"></div>' for i in range(1, 4))
    + '</div>',
    unsafe_allow_html=True,
)

# =========================================================================
#  STEP 1 — PERSONAL DETAILS
# =========================================================================
if step == 1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Personal Details")
    fd = st.session_state.form_data
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input('Full Name', value=fd.get('name', ''), placeholder='Enter applicant name')
        gender = st.selectbox('Gender', ['Male', 'Female'], index=['Male', 'Female'].index(fd.get('gender', 'Male')))
    with c2:
        married = st.selectbox('Marital Status', ['Yes', 'No'], index=['Yes', 'No'].index(fd.get('married', 'Yes')))
        dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'], index=['0', '1', '2', '3+'].index(fd.get('dependents', '0')))
    with c3:
        education = st.selectbox('Education', ['Graduate', 'Not Graduate'], index=['Graduate', 'Not Graduate'].index(fd.get('education', 'Graduate')))
        self_employed = st.selectbox('Self Employed', ['Yes', 'No'], index=['Yes', 'No'].index(fd.get('self_employed', 'No')))
    st.markdown('</div>', unsafe_allow_html=True)

    _, right = st.columns([3, 1])
    with right:
        if st.button('Next', use_container_width=True):
            st.session_state.form_data.update({
                'name': name, 'gender': gender, 'married': married,
                'dependents': dependents, 'education': education, 'self_employed': self_employed,
            })
            go_to(2)
            st.rerun()

# =========================================================================
#  STEP 2 — INCOME & LOAN DETAILS
# =========================================================================
elif step == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Income & Loan Details")
    fd = st.session_state.form_data
    col1, col2 = st.columns(2)
    with col1:
        applicant_income = st.number_input('Applicant Monthly Income', min_value=0, step=1000, value=fd.get('applicant_income', 0))
        coapplicant_income = st.number_input('Coapplicant Monthly Income', min_value=0, step=1000, value=fd.get('coapplicant_income', 0))
        credit_history = st.selectbox(
            'Credit History', [1.0, 0.0],
            index=[1.0, 0.0].index(fd.get('credit_history', 1.0)),
            format_func=lambda x: "Good" if x == 1.0 else "Poor / None",
        )
    with col2:
        loan_amount = st.number_input('Loan Amount', min_value=0, step=1000, value=fd.get('loan_amount', 0))
        loan_amount_term = st.number_input('Loan Term (month)', min_value=0, step=12, value=fd.get('loan_amount_term', 0))
        property_area = st.selectbox(
            'Property Area', ['Urban', 'Semiurban', 'Rural'],
            index=['Urban', 'Semiurban', 'Rural'].index(fd.get('property_area', 'Urban')),
        )
    interest_rate = st.slider('Estimated Annual Interest Rate (%)', 5.0, 18.0, fd.get('interest_rate', 9.5), 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        if st.button('Back', use_container_width=True):
            go_to(1)
            st.rerun()
    with right:
        if st.button('Check Eligibility', use_container_width=True):
            if applicant_income + coapplicant_income <= 0:
                st.warning('Enter at least one income value before checking eligibility.')
                st.stop()
            if loan_amount <= 0:
                st.warning('Loan amount must be greater than 0.')
                st.stop()
            if loan_amount_term <= 0:
                st.warning('Loan term must be greater than 0.')
                st.stop()

            st.session_state.form_data.update({
                'applicant_income': applicant_income, 'coapplicant_income': coapplicant_income,
                'credit_history': credit_history, 'loan_amount': loan_amount,
                'loan_amount_term': loan_amount_term, 'property_area': property_area,
                'interest_rate': interest_rate,
            })

            fd = st.session_state.form_data
            gender_val = 1 if fd['gender'] == 'Male' else 0
            married_val = 1 if fd['married'] == 'Yes' else 0
            dependents_val = {'0': 0, '1': 1, '2': 2, '3+': 3}[fd['dependents']]
            education_val = 0 if fd['education'] == 'Graduate' else 1
            self_employed_val = 1 if fd['self_employed'] == 'Yes' else 0
            property_area_val = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}[fd['property_area']]

            loan_amount_thousands = fd['loan_amount'] / 1000  # model was trained with loan amount in thousands
            total_income = fd['applicant_income'] + fd['coapplicant_income']
            loan_income_ratio = loan_amount_thousands / (total_income / 1000) if total_income > 0 else 0.0

            input_data = np.array([[
                gender_val, married_val, dependents_val, education_val, self_employed_val,
                loan_amount_thousands, fd['loan_amount_term'], fd['credit_history'], property_area_val,
                total_income, loan_income_ratio,
            ]])

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1] if hasattr(model, 'predict_proba') else float(prediction)
            score = round(probability * 100, 1)
            result = 'Approved' if prediction == 1 else 'Rejected'

            term_months = fd['loan_amount_term'] if fd['loan_amount_term'] <= 480 else fd['loan_amount_term'] / 30
            emi = compute_emi(fd['loan_amount'], fd['interest_rate'], max(term_months, 1))
            dti_pct = (emi / total_income * 100) if total_income > 0 else 0.0

            st.session_state.result_data = {
                'score': score, 'result': result, 'emi': emi, 'dti_pct': dti_pct,
                'total_income': total_income,
            }
            st.session_state.history.append({
                'Applicant': fd['name'].strip() if fd['name'].strip() else 'Applicant',
                'Prediction': result, 'Score': score,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            })
            go_to(3)
            st.rerun()

# =========================================================================
#  STEP 3 — RESULT
# =========================================================================
elif step == 3:
    fd = st.session_state.form_data
    rd = st.session_state.result_data
    name = fd.get('name', '')

    st.markdown('<div class="card">', unsafe_allow_html=True)
    top_left, top_right = st.columns([1.1, 1])
    with top_left:
        st.image(render_gauge_image(rd['score']), use_container_width=True)
        st.markdown(
            '<p style="text-align:center; color:#5a5f73; margin-top:-0.5rem;">Approval Score</p>',
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown(f"### {name.strip() if name.strip() else 'Applicant'}")
        badge_cls = "result-badge-approved" if rd['result'] == "Approved" else "result-badge-rejected"
        st.markdown(f'<span class="{badge_cls}">{rd["result"].upper()}</span>', unsafe_allow_html=True)
        st.write("")
        m1, m2 = st.columns(2)
        m1.metric('Monthly EMI', f"₹{rd['emi']:,.0f}")
        m2.metric('EMI-to-Income', f"{rd['dti_pct']:.1f}%")
        st.write("")
        suggestion = build_suggestion(fd['credit_history'], rd['dti_pct'], fd['coapplicant_income'], rd['total_income'])
        st.markdown(f'<div class="suggestion-box">{suggestion}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if rd['result'] == 'Approved':
        st.markdown(
            '<div style="text-align:center; font-size:1.3rem; font-weight:600; color:#12805c; margin:0.5rem 0 1rem;">'
            '🎉 Congratulations! Your loan looks eligible for approval. 🎉</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        report_text = (
            f"Loan Approval Report\n"
            f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n\n"
            f"Applicant: {name.strip() if name.strip() else 'Applicant'}\n"
            f"Prediction: {rd['result']}\n"
            f"Approval Score: {rd['score']}%\n"
            f"Estimated Monthly EMI: Rs. {rd['emi']:,.0f}\n"
            f"EMI-to-Income Ratio: {rd['dti_pct']:.1f}%\n"
        )
        st.download_button('Download Text Report', data=report_text, file_name='loan_report.txt', mime='text/plain', use_container_width=True)
    with d2:
        png_buf = build_scorecard_png(name, rd['score'], rd['result'], rd['dti_pct'], fd['credit_history'])
        st.download_button('Download Score Card', data=png_buf, file_name='loan_scorecard.png', mime='image/png', use_container_width=True)
    with d3:
        if st.button('New Application', use_container_width=True):
            reset_all()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
