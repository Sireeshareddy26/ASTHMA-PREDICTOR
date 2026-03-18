import sklearn
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Asthma Risk Predictor", page_icon="🫁", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .risk-severe   { background:#ffe0e0; border-left:6px solid #e53935; padding:20px; border-radius:8px; }
    .risk-moderate { background:#fff3e0; border-left:6px solid #fb8c00; padding:20px; border-radius:8px; }
    .risk-mild     { background:#fff8e1; border-left:6px solid #fdd835; padding:20px; border-radius:8px; }
    .risk-low      { background:#e8f5e9; border-left:6px solid #43a047; padding:20px; border-radius:8px; }
    .derived-card  { background:white; padding:16px; border-radius:10px;
                     box-shadow:0 2px 6px rgba(0,0,0,.08); text-align:center; margin-bottom:8px; }
    h1 { color: #1a237e; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load("model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

try:
    model, label_encoder = load_model()
except Exception as e:
    st.error(f"❌ Could not load model files: {e}")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫁 Asthma Risk Prediction App")
st.markdown("Enter the **clinical parameters** below. Derived indices will be calculated automatically.")
st.divider()

# ── Clinical parameter inputs (sidebar) ──────────────────────────────────────
st.sidebar.header("🩺 Clinical Parameters")

with st.sidebar:
    pefr   = st.number_input("PEFR (L/min)",                        min_value=80.0,  max_value=680.0, value=350.0, step=1.0)
    rr     = st.number_input("Respiratory Rate (breaths/min)",       min_value=12.0,  max_value=43.0,  value=20.0,  step=0.1)
    hr     = st.number_input("Heart Rate (bpm)",                     min_value=60.0,  max_value=145.0, value=90.0,  step=0.1)
    spo2   = st.number_input("SpO₂ (%)",                             min_value=82.0,  max_value=100.0, value=95.0,  step=0.1)
    height = st.number_input("Height (cm)",                          min_value=138.0, max_value=185.0, value=160.0, step=0.1)
    aec    = st.number_input("Absolute Eosinophil Count (cells/µL)", min_value=50,    max_value=2400,  value=750,   step=1)

    predict_btn = st.button("🔍 Predict Asthma Risk", use_container_width=True, type="primary")

# ── Auto-calculate derived indices ────────────────────────────────────────────
afr             = pefr / height
bsi             = rr * hr
oer             = spo2 / rr
ali             = aec / 100
risk_score      = (oer * 2) - (bsi / 500) + (afr * 3) - (ali * 0.5)
probability     = float(np.clip(1 / (1 + np.exp(-risk_score * 0.3)), 0, 1))
probability_pct = probability * 100

# ── Main panel: show derived indices ─────────────────────────────────────────
st.subheader("📐 Derived Indices (Auto-Calculated)")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="derived-card">
        <h4 style="margin:0;color:#555">AFR</h4>
        <h2 style="margin:4px 0;color:#1565c0">{afr:.3f}</h2>
        <small>PEFR / Height</small></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="derived-card">
        <h4 style="margin:0;color:#555">BSI</h4>
        <h2 style="margin:4px 0;color:#1565c0">{bsi:.1f}</h2>
        <small>RR × HR</small></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="derived-card">
        <h4 style="margin:0;color:#555">OER</h4>
        <h2 style="margin:4px 0;color:#1565c0">{oer:.3f}</h2>
        <small>SpO₂ / RR</small></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="derived-card">
        <h4 style="margin:0;color:#555">ALI</h4>
        <h2 style="margin:4px 0;color:#1565c0">{ali:.2f}</h2>
        <small>AEC / 100</small></div>""", unsafe_allow_html=True)

st.subheader("📊 Risk Scoring (Auto-Calculated)")
r1, r2, r3 = st.columns(3)
with r1:
    st.markdown(f"""<div class="derived-card">
        <h4 style="margin:0;color:#555">Risk Score</h4>
        <h2 style="margin:4px 0;color:#6a1b9a">{risk_score:.4f}</h2>
        <small>Composite score</small></div>""", unsafe_allow_html=True)
with r2:
    st.markdown(f"""<div class="derived-card">
        <h4 style="margin:0;color:#555">Probability</h4>
        <h2 style="margin:4px 0;color:#6a1b9a">{probability:.4f}</h2>
        <small>0 – 1 scale</small></div>""", unsafe_allow_html=True)
with r3:
    st.markdown(f"""<div class="derived-card">
        <h4 style="margin:0;color:#555">Probability (%)</h4>
        <h2 style="margin:4px 0;color:#6a1b9a">{probability_pct:.2f}%</h2>
        <small>Percentage scale</small></div>""", unsafe_allow_html=True)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([[
        pefr, rr, hr, spo2, height, aec,
        afr, bsi, oer, ali,
        risk_score, probability, probability_pct
    ]], columns=[
        'PEFR (L/min)', 'Respiratory Rate (breaths/min)', 'Heart Rate (bpm)',
        'SpO₂ (%)', 'Height (cm)', 'Absolute Eosinophil Count (cells/µL)',
        'AFR (PEFR/Height)', 'BSI (RR × HR)', 'OER (SpO₂/RR)', 'ALI (AEC/100)',
        'Risk Score', 'Probability', 'Probability (%)'
    ])

    try:
        pred_numeric  = model.predict(input_data)
        pred_category = label_encoder.inverse_transform(pred_numeric)[0]

        cat_lower = pred_category.lower()
        if "severe" in cat_lower:
            css, icon, advice = "risk-severe",   "🔴", "Immediate medical attention required. Hospitalisation may be necessary."
        elif "moderate" in cat_lower:
            css, icon, advice = "risk-moderate",  "🟠", "Medical review needed. Inhaler therapy and monitoring recommended."
        elif "mild" in cat_lower:
            css, icon, advice = "risk-mild",      "🟡", "Manage with bronchodilators. Regular follow-up advised."
        else:
            css, icon, advice = "risk-low",       "🟢", "Low risk. Continue preventive care and routine check-ups."

        st.subheader("🎯 Prediction Result")
        st.markdown(f"""<div class="{css}">
            <h2>{icon} {pred_category}</h2>
            <p style="margin:0">{advice}</p>
        </div>""", unsafe_allow_html=True)

        with st.expander("📋 Full Input & Derived Values Summary"):
            summary = {
                "PEFR (L/min)": pefr,
                "Respiratory Rate": rr,
                "Heart Rate": hr,
                "SpO₂ (%)": spo2,
                "Height (cm)": height,
                "AEC (cells/µL)": aec,
                "AFR": round(afr, 4),
                "BSI": round(bsi, 2),
                "OER": round(oer, 4),
                "ALI": round(ali, 4),
                "Risk Score": round(risk_score, 4),
                "Probability": round(probability, 4),
                "Probability (%)": round(probability_pct, 2),
            }
            st.table(pd.DataFrame(summary.items(), columns=["Parameter", "Value"]))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("👈 Enter clinical parameters in the sidebar and click **Predict Asthma Risk**.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
> ⚠️ **Disclaimer:** This tool is for educational and screening purposes only.
> It is **not** a substitute for professional medical advice or clinical diagnosis.
""")
