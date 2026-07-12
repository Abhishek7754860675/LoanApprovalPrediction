import io
import math
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

# =========================================================================
#  PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title='Loan Approval Predictor',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded',
)

# =========================================================================
#  CUSTOM CSS  (IMPROVED MOBILE RESPONSIVENESS)
# =========================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {font-family: 'Poppins', sans-serif;}

    .stApp {background: linear-gradient(180deg, #eef2f9 0%, #ffffff 55%);}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 980px;}

    .header {
        background: linear-gradient(120deg, #0e1630 0%, #163a6e 55%, #17a2a2 130%);
        padding: 1.7rem 2rem; border-radius: 20px; color: white;
        box-shadow: 0 14px 30px rgba(16, 25, 58, 0.28); margin-bottom: 1.6rem;
        position: relative; overflow: hidden;
    }
    .header::after {
        content: ""; position: absolute; right: -40px; top: -40px;
        width: 160px; height: 160px; border-radius: 50%;
        background: rgba(255,255,255,0.08);
    }
    .header h1 {margin:0; font-size:1.9rem; font-weight:700; letter-spacing: -0.01em;}
    .header p {margin:0.3rem 0 0; font-size:0.98rem; opacity:0.88;}

    /* progress bar */
    .progress-wrap {display:flex; gap:8px; margin: 0 0 1.6rem 0;}
    .progress-seg {flex:1; height:6px; border-radius:999px; background:#dfe6f2; transition: background 0.3s ease;}
    .progress-seg.active {background: linear-gradient(90deg,#163a6e,#17a2a2);}

    .card {
        background: #ffffff; padding: 1.5rem 1.7rem; border-radius: 18px;
        box-shadow: 0 8px 22px rgba(16, 25, 58, 0.07);
        margin-bottom: 1.2rem; border: 1px solid #eef1f7;
    }

    .result-badge-approved, .result-badge-rejected {
        display:inline-block; padding: 0.45rem 1.3rem; border-radius: 999px;
        font-weight:700; letter-spacing: 0.02em; font-size: 0.95rem;
    }
    .result-badge-approved {background: #e4f8ef; color:#12805c; border: 1px solid #b6ecd4;}
    .result-badge-rejected {background: #fdeceb; color:#b8362a; border: 1px solid #f6c6c1;}

    .suggestion-box {
        background: #f4f8ff; border-left: 3px solid #17a2a2; padding: 0.9rem 1.1rem;
        border-radius: 10px; font-size: 0.93rem; color:#24314f; line-height:1.5;
    }

    div.stButton > button, div.stDownloadButton > button {
        border-radius: 999px; padding: 0.65rem 1.4rem;
        background: linear-gradient(90deg, #163a6e, #17a2a2); color: white;
        border: none; font-weight: 600; transition: all 0.15s ease-in-out;
        width: 100%;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 8px 18px rgba(16, 48, 94, 0.28);
    }

    [data-testid="stMetricValue"] {font-weight: 700; color:#163a6e;}
    section[data-testid="stSidebar"] {background:#0e1630;}
    section[data-testid="stSidebar"] * {color:#e8ecf6 !important;}
    section[data-testid="stSidebar"] div.stButton > button {background:#17a2a2; color:white !important;}
    hr {margin: 0.6rem 0 1.1rem 0; border-color:#eef1f7;}

    /* Input and Select styling */
    input, select, textarea {
        border-radius: 10px !important;
        border: 1px solid #dfe6f2 !important;
        padding: 0.6rem 0.8rem !important;
        font-size: 0.95rem !important;
    }
    
    input:focus, select:focus, textarea:focus {
        border-color: #17a2a2 !important;
        box-shadow: 0 0 0 2px rgba(23, 162, 162, 0.1) !important;
    }

    /* Improved subheader styling */
    [data-testid="stSubheader"] {
        margin-bottom: 1.2rem !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #0e1630 !important;
    }

    /* ---------- MOBILE RESPONSIVENESS (IMPROVED) ---------- */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem; 
            padding-right: 1rem; 
            padding-top: 1rem;
            max-width: 100% !important;
        }
        
        .header {
            padding: 1.3rem 1.2rem; 
            border-radius: 16px;
            margin-bottom: 1.2rem;
        }
        
        .header h1 {
            font-size: 1.45rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            font-size: 0.88rem;
        }
        
        .card {
            padding: 1.3rem 1.2rem; 
            border-radius: 14px;
            margin-bottom: 1rem;
        }
        
        [data-testid="stColumn"] {
            margin-bottom: 0.8rem !important;
        }
        
        .result-badge-approved, .result-badge-rejected {
            font-size: 0.9rem; 
            padding: 0.4rem 1rem;
            display: block;
            text-align: center;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.85rem;
        }
        
        div.stButton > button, div.stDownloadButton > button {
            width: 100%;
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
        }
        
        [data-testid="stSubheader"] {
            font-size: 1.15rem !important;
        }
        
        input, select, textarea {
            font-size: 1rem !important;
            padding: 0.7rem 0.9rem !important;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding-left: 0.8rem; 
            padding-right: 0.8rem;
            padding-top: 0.8rem;
        }
        
        .header {
            padding: 1.1rem 1rem;
            margin-bottom: 1rem;
        }
        
        .header h1 {
            font-size: 1.25rem;
        }
        
        .header p {
            font-size: 0.80rem;
        }
        
        .card {
            padding: 1rem 1rem;
        }
        
        [data-testid="stSubheader"] {
            font-size: 1.05rem !important;
        }
        
        .suggestion-box {
            padding: 0.8rem 0.9rem;
            font-size: 0.88rem;
        }
        
        div.stButton > button, div.stDownloadButton > button {
            padding: 0.65rem 0.8rem;
            font-size: 0.9rem;
        }
    }

    /* Hide sidebar on very small screens */
    @media (max-width: 480px) {
        section[data-testid="stSidebar"] {
            display: none;
        }
    }
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
#  MODEL LOADING  (searches multiple likely locations instead of assuming
#  one fixed folder structure — works whether Model/ sits next to app.py,
#  one level up, at the repo root, or app.py itself lives at the repo root)
# =========================================================================
def find_model_files():
    here = Path(__file__).resolve().parent
    search_roots = [
        here,                 # same folder as app.py
        here.parent,          # one level up
        here.parent.parent,   # two levels up
        Path.cwd(),           # current working directory (Streamlit Cloud runs from repo root)
    ]
    folder_names = ['Model', 'model', 'models', 'Models']
    tried = []
    for root in search_roots:
        for folder in folder_names:
            model_path = root / folder / 'loan_model.pkl'
            scaler_path = root / folder / 'scaler.pkl'
            tried.append(str(model_path))
            if model_path.exists() and scaler_path.exists():
                return model_path, scaler_path
    # last resort: search recursively from repo root (cwd) for the files
    for base in {here, here.parent, Path.cwd()}:
        for p in base.rglob('loan_model.pkl'):
            s = p.parent / 'scaler.pkl'
            if s.exists():
                return p, s
    raise FileNotFoundError(
        "Could not find 'loan_model.pkl' and 'scaler.pkl'. Looked in:\n"
        + "\n".join(tried)
        + "\n\nMake sure both files are committed to your GitHub repo "
          "(check they aren't excluded by .gitignore) inside a folder "
          "named 'Model' (or 'model')."
    )


@st.cache_resource(show_spinner=False)
def load_model():
    model_path, scaler_path = find_model_files()
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
    zones = [(0, 40, (231, 76, 60)), (40, 70, (244, 180, 66)), (70, 100, (23, 162, 162))]
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
    draw.line([cx, cy, nx, ny], fill=(14, 22, 48), width=6)
    draw.ellipse([cx - 11, cy - 11, cx + 11, cy + 11], fill=(14, 22, 48))


def render_gauge_image(score, width=520, height=310):
    """Transparent-background needle gauge (used only for the static PNG scorecard download)."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy, radius = width // 2, height - 30, min(width // 2 - 20, height - 60)
    draw_gauge(draw, cx, cy, radius, score, thickness=30)
    text = f"{score}%"
    font = get_font(48, bold=True)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((cx - tw / 2, cy - radius * 0.55), text, font=font, fill=(14, 22, 48))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def build_animated_gauge_html(score, width=440):
    """Live, animated needle gauge (sweeps + counts up on every render) shown on the result step."""
    cx = width / 2
    top_pad = 16
    r = min(width / 2 - 34, 150)
    pivot_y = top_pad + r
    gap_below_pivot = 60          # generous clearance so the needle never nears the text
    text_y = pivot_y + gap_below_pivot
    label_y = text_y + 26
    svg_h = label_y + 20
    needle_len = r - 34

    def point_at(t):
        phi = math.radians(180 * (1 - t))
        x = cx + r * math.cos(phi)
        y = pivot_y - r * math.sin(phi)
        return x, y

    zones = [(0, 40, "#e74c3c"), (40, 70, "#f4b442"), (70, 100, "#17a2a2")]
    arcs = ""
    for p1, p2, color in zones:
        x1, y1 = point_at(p1 / 100)
        x2, y2 = point_at(p2 / 100)
        arcs += (
            f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 0 1 {x2:.1f} {y2:.1f}" '
            f'stroke="{color}" stroke-width="18" fill="none" stroke-linecap="round"/>'
        )

    uid = f"g{abs(hash((score, width))) % 10_000_000}"

    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; font-family:'Poppins',sans-serif;">
      <svg id="{uid}-svg" width="{width}" height="{svg_h}" viewBox="0 0 {width} {svg_h}">
        {arcs}
        <circle cx="{cx}" cy="{pivot_y}" r="7" fill="#0e1630"/>
        <g id="{uid}-needle" transform="rotate(-90 {cx} {pivot_y})">
          <line x1="{cx}" y1="{pivot_y}" x2="{cx}" y2="{pivot_y - needle_len}"
                stroke="#0e1630" stroke-width="5" stroke-linecap="round"/>
        </g>
        <text id="{uid}-text" x="{cx}" y="{text_y}" text-anchor="middle"
              font-size="34" font-weight="700" fill="#0e1630" font-family="Poppins, sans-serif">0%</text>
        <text x="{cx}" y="{label_y}" text-anchor="middle"
              font-size="13" fill="#5a5f73" font-family="Poppins, sans-serif">Approval Score</text>
      </svg>
    </div>
    <script>
      (function() {{
        const target = {score};
        const needle = document.getElementById("{uid}-needle");
        const text = document.getElementById("{uid}-text");
        const duration = 1100;
        const start = performance.now();
        function ease(t) {{ return 1 - Math.pow(1 - t, 3); }}
        function frame(now) {{
          const elapsed = now - start;
          const t = Math.min(elapsed / duration, 1);
          const eased = ease(t);
          const current = target * eased;
          const angle = -90 + (current / 100) * 180;
          needle.setAttribute("transform", `rotate(${{angle}} {cx} {pivot_y})`);
          text.textContent = current.toFixed(1) + "%";
          if (t < 1) {{
            requestAnimationFrame(frame);
          }} else {{
            text.textContent = target.toFixed(1) + "%";
          }}
        }}
        requestAnimationFrame(frame);
      }})();
    </script>
    """


def build_confetti_burst_html(count=140):
    """Full-screen confetti burst with 5-second duration — no external libraries.
    Injects into the parent document so it covers the whole viewport."""
    return f"""
    <script>
      (function() {{
        const colors = ["#17a2a2", "#163a6e", "#f4b442", "#e74c3c", "#12805c"];
        const count = {count};

        function spawnInto(doc, win, fixedFull) {{
          const styleEl = doc.createElement('style');
          styleEl.textContent = `
            @keyframes confetti-fall-full {{
              0%   {{ transform: translate(0, 0) rotate(0deg); opacity: 1; }}
              85%  {{ opacity: 1; }}
              100% {{ transform: translate(var(--dx), var(--dy)) rotate(var(--rot)); opacity: 0; }}
            }}
          `;
          doc.head.appendChild(styleEl);

          const wrap = doc.createElement('div');
          wrap.style.position = fixedFull ? 'fixed' : 'absolute';
          wrap.style.top = '0';
          wrap.style.left = '0';
          wrap.style.width = fixedFull ? '100vw' : '100%';
          wrap.style.height = fixedFull ? '100vh' : '1px';
          wrap.style.overflow = 'visible';
          wrap.style.pointerEvents = 'none';
          wrap.style.zIndex = '999999';
          doc.body.appendChild(wrap);

          const viewH = fixedFull ? (win.innerHeight || 800) : 220;
          for (let i = 0; i < count; i++) {{
            const el = doc.createElement('div');
            const size = 6 + Math.random() * 7;
            const color = colors[Math.floor(Math.random() * colors.length)];
            const startX = Math.random() * 100;
            const dx = (Math.random() * 200 - 100);
            const dy = viewH * (0.55 + Math.random() * 0.5);
            const rot = Math.random() * 620 - 310;
            const delay = Math.random() * 150;
            const dur = 3200 + Math.random() * 800;
            el.style.position = 'absolute';
            el.style.left = startX + '%';
            el.style.top = '-20px';
            el.style.width = size + 'px';
            el.style.height = (size * 0.4) + 'px';
            el.style.background = color;
            el.style.borderRadius = '2px';
            el.style.opacity = '1';
            el.style.setProperty('--dx', dx + 'px');
            el.style.setProperty('--dy', dy + 'px');
            el.style.setProperty('--rot', rot + 'deg');
            el.style.animation = `confetti-fall-full ${{dur}}ms linear ${{delay}}ms forwards`;
            wrap.appendChild(el);
          }}
          setTimeout(() => {{ wrap.remove(); styleEl.remove(); }}, 4200);
        }}

        try {{
          const doc = window.parent.document;
          const win = window.parent;
          spawnInto(doc, win, true);
        }} catch (e) {{
          spawnInto(document, window, false);
        }}
      }})();
    </script>
    """


def build_scorecard_png(name, score, result, dti_pct, credit_history):
    W, H = 1000, 560
    img = make_gradient(W, H, (247, 251, 255), (223, 238, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, W, 110], fill=(14, 22, 48))
    draw.text((40, 30), "Loan Approval Report", font=get_font(38, bold=True), fill="white")
    draw.text((40, 78), f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
              font=get_font(16), fill=(200, 210, 230))

    display_name = name.strip() if name and name.strip() else "Applicant"
    draw.text((40, 150), "Applicant", font=get_font(18), fill=(100, 112, 135))
    draw.text((40, 175), display_name, font=get_font(34, bold=True), fill=(20, 28, 50))

    cx, cy, radius = 250, 380, 150
    draw_gauge(draw, cx, cy, radius, score)
    draw.text((cx - 55, cy - 55), f"{score}%", font=get_font(46, bold=True), fill=(14, 22, 48))
    draw.text((cx - 65, cy + 5), "Approval Score", font=get_font(16), fill=(100, 112, 135))

    badge_color = (23, 162, 162) if result == "Approved" else (184, 54, 42)
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
    if dti_pct > 50:
        return "Your EMI-to-income ratio exceeds 50% — lenders typically prefer lower ratios. Consider a lower loan amount or longer term."
    if coapplicant_income > 0:
        return "Your co-applicant's income strengthens your case. Make sure all income documentation is complete and accurate."
    return "Your profile looks solid. Ensure all submitted documents are current and accurate."


def get_profession_interest_rate(profession):
    """
    Returns default interest rate based on profession.
    Based on Indian lending market data for different employment types.
    """
    profession_rates = {
        'Salaried': 7.5,           # Stable income - lowest risk
        'Business Owner': 10.5,     # Business risk - higher rate
        'Self-Employed': 10.0,      # Variable income
        'Freelancer': 11.5,         # Unstable income - higher risk
        'Retired': 8.5,             # Fixed income but age factor
        'Student': 6.5,             # Education loan - subsidized rate
        'Other': 10.0,              # Default average
    }
    return profession_rates.get(profession, 9.5)


# =========================================================================
#  UI: HEADER & PROGRESS
# =========================================================================
if not MODEL_READY:
    st.error(f"⚠️ Model Load Error:\n{LOAD_ERROR}")
    st.stop()

st.markdown(
    '<div class="header">'
    '<h1>🏦 Loan Approval Predictor</h1>'
    '<p>Check your eligibility and get an instant approval score</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Progress indicator
step = st.session_state.step
st.markdown(
    '<div class="progress-wrap">'
    + "".join([
        f'<div class="progress-seg {"active" if i < step else ""}"></div>'
        for i in range(1, 4)
    ])
    + '</div>',
    unsafe_allow_html=True,
)

# =========================================================================
#  STEP 1 — PERSONAL DETAILS
# =========================================================================
if step == 1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👤 Personal Details")
    
    fd = st.session_state.form_data
    
    # Improved responsive layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Basic Information**")
        name = st.text_input('Full Name', value=fd.get('name', ''), placeholder='Enter your full name', key='name_input')
        gender = st.selectbox('Gender', ['Male', 'Female'], index=['Male', 'Female'].index(fd.get('gender', 'Male')), key='gender_input')
        profession = st.selectbox(
            'Profession', 
            ['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer', 'Retired', 'Student', 'Other'],
            index=['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer', 'Retired', 'Student', 'Other'].index(fd.get('profession', 'Salaried')),
            key='profession_input'
        )
        married = st.selectbox('Marital Status', ['Yes', 'No'], index=['Yes', 'No'].index(fd.get('married', 'Yes')), key='married_input')
    
    with col2:
        st.write("**Additional Details**")
        dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'], index=['0', '1', '2', '3+'].index(fd.get('dependents', '0')), key='dependents_input')
        education = st.selectbox('Education', ['Graduate', 'Not Graduate'], index=['Graduate', 'Not Graduate'].index(fd.get('education', 'Graduate')), key='education_input')
        self_employed = st.selectbox('Self Employed', ['Yes', 'No'], index=['Yes', 'No'].index(fd.get('self_employed', 'No')), key='self_employed_input')
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Navigation buttons
    nav_col1, nav_col2 = st.columns([1, 1])
    with nav_col1:
        pass
    with nav_col2:
        if st.button('Next →', use_container_width=True, key='step1_next'):
            if not name.strip():
                st.warning('Please enter your full name')
                st.stop()
            st.session_state.form_data.update({
                'name': name, 'gender': gender, 'profession': profession, 'married': married,
                'dependents': dependents, 'education': education, 'self_employed': self_employed,
            })
            go_to(2)
            st.rerun()

# =========================================================================
#  STEP 2 — INCOME & LOAN DETAILS
# =========================================================================
elif step == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Income & Loan Details")
    fd = st.session_state.form_data
    
    # Responsive layout with better spacing
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Income Information**")
        applicant_income = st.number_input('Applicant Monthly Income (₹)', min_value=0, step=1000, 
                                           value=fd.get('applicant_income', 0), key='applicant_income_input')
        coapplicant_income = st.number_input('Co-applicant Monthly Income (₹)', min_value=0, step=1000, 
                                             value=fd.get('coapplicant_income', 0), key='coapplicant_income_input')
        credit_history = st.selectbox(
            'Credit History', [1.0, 0.0],
            index=[1.0, 0.0].index(fd.get('credit_history', 1.0)),
            format_func=lambda x: "Good ✓" if x == 1.0 else "Poor / None ✗",
            key='credit_history_input'
        )
    
    with col2:
        st.write("**Loan Information**")
        loan_amount = st.number_input('Loan Amount (₹)', min_value=0, step=10000, 
                                      value=fd.get('loan_amount', 0), key='loan_amount_input')
        loan_amount_term = st.number_input('Loan Term (months)', min_value=0, step=12, 
                                           value=fd.get('loan_amount_term', 0), key='loan_term_input')
        property_area = st.selectbox(
            'Property Area', ['Urban', 'Semiurban', 'Rural'],
            index=['Urban', 'Semiurban', 'Rural'].index(fd.get('property_area', 'Urban')),
            key='property_area_input'
        )
    
    st.write("**Interest Rate**")
    # Get profession-based default interest rate
    profession = fd.get('profession', 'Salaried')
    default_interest_rate = get_profession_interest_rate(profession)
    user_selected_rate = fd.get('interest_rate', default_interest_rate)
    st.caption(f"📊 Based on {profession} profession: Starting rate is {default_interest_rate}%")
    interest_rate = st.slider('Estimated Annual Interest Rate (%)', 5.0, 18.0, user_selected_rate, 0.1, key='interest_rate_input')
    st.markdown('</div>', unsafe_allow_html=True)

    # Navigation buttons
    nav_col1, nav_col2 = st.columns([1, 1])
    with nav_col1:
        if st.button('← Back', use_container_width=True, key='step2_back'):
            go_to(1)
            st.rerun()
    with nav_col2:
        if st.button('Check Eligibility ✓', use_container_width=True, key='step2_next'):
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

            loan_amount_thousands = fd['loan_amount'] / 1000
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
    
    # Responsive result layout
    result_col1, result_col2 = st.columns([1, 1.2])
    
    with result_col1:
        components.html(build_animated_gauge_html(rd['score']), height=310)
    
    with result_col2:
        st.markdown(f"### {name.strip() if name.strip() else 'Applicant'}")
        profession_badge = f"**{fd.get('profession', 'N/A')}** | Education: {fd.get('education', 'N/A')}"
        st.markdown(profession_badge)
        st.write("")
        badge_cls = "result-badge-approved" if rd['result'] == "Approved" else "result-badge-rejected"
        st.markdown(f'<span class="{badge_cls}">{rd["result"].upper()}</span>', unsafe_allow_html=True)
        st.write("")
        
        m1, m2 = st.columns(2)
        m1.metric('Monthly EMI', f"₹{rd['emi']:,.0f}")
        m2.metric('EMI/Income', f"{rd['dti_pct']:.1f}%")
        
        st.write("")
        suggestion = build_suggestion(fd['credit_history'], rd['dti_pct'], fd['coapplicant_income'], rd['total_income'])
        st.markdown(f'<div class="suggestion-box">{suggestion}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Confetti and success message
    if rd['result'] == 'Approved':
        components.html(build_confetti_burst_html(), height=0)
        st.markdown(
            '<div style="text-align:center; font-size:1.2rem; font-weight:600; color:#12805c; margin:1rem 0;">'
            '🎉 Congratulations! Your loan looks eligible for approval. 🎉</div>',
            unsafe_allow_html=True,
        )

    # Download options
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("**Download Your Report**")
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        report_text = (
            f"Loan Approval Report\n"
            f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n\n"
            f"Applicant: {name.strip() if name.strip() else 'Applicant'}\n"
            f"Profession: {fd.get('profession', 'N/A')}\n"
            f"Education: {fd.get('education', 'N/A')}\n"
            f"Interest Rate: {fd.get('interest_rate', 'N/A')}% per annum\n"
            f"Prediction: {rd['result']}\n"
            f"Approval Score: {rd['score']}%\n"
            f"Estimated Monthly EMI: ₹ {rd['emi']:,.0f}\n"
            f"EMI-to-Income Ratio: {rd['dti_pct']:.1f}%\n"
        )
        st.download_button('📄 Text Report', data=report_text, file_name='loan_report.txt', 
                          mime='text/plain', use_container_width=True)
    
    with download_col2:
        png_buf = build_scorecard_png(name, rd['score'], rd['result'], rd['dti_pct'], fd['credit_history'])
        st.download_button('🖼️ Score Card', data=png_buf, file_name='loan_scorecard.png', 
                          mime='image/png', use_container_width=True)
    
    with download_col3:
        if st.button('🔄 New Application', use_container_width=True):
            reset_all()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
