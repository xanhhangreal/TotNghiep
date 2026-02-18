"""Streamlit demo – Stress Detection System.

Launch:  streamlit run src/app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from pathlib import Path

from config import STRESS_LABELS, RESULTS_DIR, MODELS_DIR
from models import StressModel

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stress Detection", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.stMetric{background:#f0f2f6;padding:10px;border-radius:5px}
</style>""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
def main():
    st.title("🧠 Stress Detection System")
    st.caption("Physiological signal-based stress detection with ML")
    st.divider()

    page = st.sidebar.radio("Navigation", [
        "📊 Dashboard", "🔍 Predictor",
        "📈 Performance", "📚 Docs",
    ])
    {"📊 Dashboard": dashboard,
     "🔍 Predictor": predictor,
     "📈 Performance": performance,
     "📚 Docs": docs}[page]()


# ── pages ─────────────────────────────────────────────────────────────────────

def dashboard():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Models", "4")
    c2.metric("Dataset", "WESAD")
    c3.metric("Signals", "EDA · BVP · TEMP")
    c4.metric("Features", "23")
    st.divider()

    col1, col2 = st.columns(2)
    col1.info("""
    **Physiological Signals**
    - **EDA** – skin conductance (SCR peaks, SCL)
    - **BVP** – heart rate & HRV (SDNN, RMSSD, LF/HF)
    - **TEMP** – skin temperature trend
    """)
    col2.success("""
    **Pipeline**
    1. Preprocessing & resampling
    2. Windowed feature extraction (60 s)
    3. Classification (RF, LR, SVM, DT)
    4. SHAP interpretability
    """)


def predictor():
    st.header("Stress Predictor")

    # Load trained model if available
    model = None
    model_files = list(MODELS_DIR.glob("*.joblib")) if MODELS_DIR.exists() else []
    if model_files:
        sel = st.selectbox("Trained model", [f.stem for f in model_files])
        try:
            model = StressModel.load(str(MODELS_DIR / f"{sel}.joblib"))
            st.success(f"Loaded **{sel}** ({model.model_type})")
        except Exception as e:
            st.warning(f"Load failed: {e}")
    else:
        st.info("No trained models. Run `py -u src/training.py --approach subject_independent` first.")

    st.subheader("Manual input")
    c1, c2, c3 = st.columns(3)
    with c1:
        eda_mean = st.slider("EDA Mean (µS)", 0.0, 20.0, 5.0)
        eda_std = st.slider("EDA Std", 0.0, 10.0, 2.0)
        eda_peaks = st.slider("SCR Peaks", 0, 20, 5)
    with c2:
        hr = st.slider("Heart Rate (bpm)", 40, 150, 80)
        sdnn = st.slider("HRV SDNN (ms)", 10, 200, 50)
        lf_hf = st.slider("LF/HF Ratio", 0.0, 5.0, 1.5)
    with c3:
        temp_m = st.slider("Temp Mean (°C)", 32.0, 36.0, 33.5)
        temp_s = st.slider("Temp Std", 0.0, 2.0, 0.5)
        temp_sl = st.slider("Temp Slope", -2.0, 2.0, 0.0)

    if st.button("Predict", type="primary"):
        vec = np.array([[eda_mean, eda_std, eda_peaks,
                         hr, sdnn, lf_hf, temp_m, temp_s, temp_sl]])
        if model and model.is_fitted:
            try:
                pred = model.predict(vec)[0]
                prob = model.predict_proba(vec)[0]
                label = STRESS_LABELS.get(pred, "?")
                st.success(f"**{('🟢' if pred == 0 else '🔴')} {label}**  "
                           f"(confidence {prob[pred]:.0%})")
            except Exception as e:
                st.error(str(e))
        else:
            st.warning("No model loaded – showing demo result")
            st.info("🔴 **STRESSED** (demo)")

    st.subheader("Upload CSV")
    f = st.file_uploader("Physiological signals CSV", type=["csv"])
    if f:
        st.dataframe(pd.read_csv(f).head(10))


def performance():
    st.header("Model Performance")

    # Load saved results
    files = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    real = None
    if files:
        sel = st.selectbox("Results file", [f.name for f in files])
        try:
            with open(RESULTS_DIR / sel) as fh:
                real = json.load(fh)
            st.success(f"Loaded {sel}")
        except Exception:
            pass

    if real and "models" in real:
        rows = []
        for mn, m in real["models"].items():
            if "error" not in m:
                rows.append({"Model": mn,
                             "Accuracy": m.get("accuracy", 0),
                             "F1": m.get("f1", 0),
                             "ROC-AUC": m.get("roc_auc", 0)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            return

    # Placeholder until real training is done
    st.caption("Sample comparison (train models to see real results)")
    df = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "SVM", "Decision Tree"],
        "Accuracy": [0.88, 0.85, 0.83, 0.82],
        "F1": [0.88, 0.85, 0.83, 0.80],
    })
    st.dataframe(df, use_container_width=True)
    fig = go.Figure()
    for col in ("Accuracy", "F1"):
        fig.add_trace(go.Bar(x=df["Model"], y=df[col], name=col))
    fig.update_layout(barmode="group", height=400)
    st.plotly_chart(fig, use_container_width=True)


def docs():
    st.header("📚 Documentation")
    t1, t2, t3 = st.tabs(["Overview", "Methodology", "Usage"])
    with t1:
        st.markdown("""
**Stress Detection from Wearable Physiological Signals**

Detects binary stress (relaxed / stressed) using EDA, BVP, and skin
temperature signals from an Empatica E4 wrist-worn device.

Evaluated on the **WESAD** dataset (Schmidt et al., 2018) with 15 subjects
undergoing the Trier Social Stress Test.
        """)
    with t2:
        st.markdown("""
1. **Preprocessing** – artifact removal, Butterworth filtering, resampling to 4 Hz
2. **Feature extraction** – 60 s sliding windows (50 % overlap)
   - EDA: mean, std, SCR peaks, SCL
   - BVP: HR stats, HRV time-domain (SDNN, RMSSD), freq-domain (LF, HF)
   - TEMP: mean, std, slope
3. **Classification** – Random Forest, Logistic Regression, SVM, Decision Tree
4. **Evaluation** – subject-dependent CV, subject-independent split, LOSO
5. **Interpretability** – SHAP feature importance
        """)
    with t3:
        st.code("""# 1) Install deps
pip install -r requirements.txt

# 2) Place WESAD data
#    data/public/WESAD/S2/S2.pkl  ...

# 3) Train
py -u src/training.py --approach all

# 4) Demo
streamlit run src/app.py""", language="bash")


if __name__ == "__main__":
    main()
