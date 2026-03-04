"""Streamlit demo – Nghiên cứu phát hiện trạng thái căng thẳng từ tín hiệu sinh lý đa phương thức bằng học sâu.

Launch:  streamlit run src/app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from pathlib import Path

from config import STRESS_LABELS, STRESS_LABELS_3CLASS, RESULTS_DIR, MODELS_DIR
from models import StressModel

# Try importing DL modules (optional – only needed if DL models are trained)
try:
    from dl_models import load_dl_model, MODEL_REGISTRY
    _HAS_DL = True
except ImportError:
    _HAS_DL = False

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Phát hiện căng thẳng", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
.stMetric{background:#f0f2f6;padding:10px;border-radius:5px}
</style>""", unsafe_allow_html=True)


# ── sidebar ───────────────────────────────────────────────────────────────────
def main():
    st.title("🧠 Phát hiện trạng thái căng thẳng")
    st.caption("Nghiên cứu phát hiện trạng thái căng thẳng từ tín hiệu sinh lý đa phương thức bằng học sâu")
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
    c1.metric("ML Models", "4")
    c2.metric("DL Models", "3")
    c3.metric("Dataset", "WESAD (15 subj)")
    c4.metric("Total Features", "~70")
    st.divider()

    col1, col2 = st.columns(2)
    col1.info("""
    **Physiological Signals — All Modalities**

    *Wrist (Empatica E4)*
    - **EDA** – skin conductance (SCR peaks, SCL)
    - **BVP** – heart rate & HRV (SDNN, RMSSD, LF/HF)
    - **TEMP** – skin temperature trend
    - **ACC** – wrist accelerometer

    *Chest (RespiBAN)*
    - **ECG** – heart rate, HRV, QRS, pNN50
    - **EMG** – muscle tension, spectral features
    - **Resp** – breathing rate, depth, I:E ratio
    - **EDA / Temp / ACC** – chest sensor variants
    """)
    col2.success("""
    **Pipeline**
    1. Preprocessing & resampling (all modalities)
    2. Windowed feature extraction (60 s, ~70 features)
    3. **ML**: Random Forest, Logistic Regression, SVM, Decision Tree
    4. **DL**: 1D-CNN, 1D-UNet, 1D-ResNet34 (PyTorch)
    5. Evaluation: Subject-Dependent, Subject-Independent, LOSO
    6. SHAP interpretability
    """)


def predictor():
    st.header("Stress Predictor")

    model_type = st.radio("Model type", ["ML (sklearn)", "DL (PyTorch)"],
                          horizontal=True)

    if model_type == "ML (sklearn)":
        _predictor_ml()
    else:
        _predictor_dl()


def _predictor_ml():
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

    if st.button("Predict (ML)", type="primary"):
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


def _predictor_dl():
    if not _HAS_DL:
        st.warning("PyTorch not installed. Run `pip install torch` first.")
        return

    pt_files = list(MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []
    if pt_files:
        sel = st.selectbox("DL model checkpoint", [f.stem for f in pt_files])
        try:
            model = load_dl_model(str(MODELS_DIR / f"{sel}.pt"))
            st.success(f"Loaded **{sel}** ({type(model).__name__})")
        except Exception as e:
            st.warning(f"Load failed: {e}")
    else:
        st.info("No DL models. Run `py -u src/dl_training.py --arch all --approach loso` first.")

    st.subheader("Upload CSV (feature vectors)")
    f = st.file_uploader("CSV with feature columns", type=["csv"])
    if f:
        df = pd.read_csv(f)
        st.dataframe(df.head(10))


def performance():
    st.header("Model Performance")

    # Load saved results
    files = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    if not files:
        st.info("No results. Train models first.")
        _show_placeholder()
        return

    sel = st.selectbox("Results file", [f.name for f in files])
    try:
        with open(RESULTS_DIR / sel) as fh:
            real = json.load(fh)
        st.success(f"Loaded {sel}")
    except Exception:
        real = None

    if not real:
        _show_placeholder()
        return

    # ── DL LOSO results ──────────────────────────────────────────────────
    if "per_subject" in real:
        st.subheader(f"{real.get('arch', '?')} — "
                     f"{real.get('n_classes', '?')}-class LOSO")
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",
                  f"{real['accuracy_mean']:.4f} ± {real['accuracy_std']:.4f}")
        m2.metric("F1",
                  f"{real['f1_mean']:.4f} ± {real['f1_std']:.4f}")
        rows = []
        for pf in real["per_subject"]:
            rows.append({
                "Subject": f"S{pf['test_subject']}",
                "Accuracy": pf["accuracy"],
                "F1": pf["f1"],
                "AUC": pf.get("roc_auc", float("nan")),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        return

    # ── ML+DL comparison ─────────────────────────────────────────────────
    if "ml" in real and "dl" in real:
        rows = []
        for section in ("ml", "dl"):
            for mn, m in real[section].items():
                rows.append({
                    "Type": section.upper(),
                    "Model": mn,
                    "Accuracy": f"{m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f}",
                    "F1": f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}",
                })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        return

    # ── generic models dict ──────────────────────────────────────────────
    if "models" in real:
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

    st.json(real)


def _show_placeholder():
    st.caption("Placeholder — train models to see real results")
    df = pd.DataFrame({
        "Model": ["Random Forest", "SVM", "CNN-1D", "UNet-1D", "ResNet-1D"],
        "Accuracy": [0.88, 0.83, 0.91, 0.90, 0.92],
        "F1": [0.88, 0.83, 0.91, 0.89, 0.92],
    })
    st.dataframe(df, use_container_width=True)
    fig = go.Figure()
    for col in ("Accuracy", "F1"):
        fig.add_trace(go.Bar(x=df["Model"], y=df[col], name=col))
    fig.update_layout(barmode="group", height=400)
    st.plotly_chart(fig, use_container_width=True)


def docs():
    st.header("📚 Documentation")
    t1, t2, t3, t4 = st.tabs(["Overview", "Methodology", "DL Models", "Usage"])
    with t1:
        st.markdown("""
**Nghiên cứu phát hiện trạng thái căng thẳng từ tín hiệu sinh lý đa phương thức bằng học sâu**

Detects stress using **all modalities** from *wrist* (Empatica E4: EDA, BVP,
TEMP, ACC) and *chest* (RespiBAN: ECG, EMG, EDA, Temp, Resp, ACC).

Supports **binary** (relaxed / stressed) and **3-class** (baseline / stress /
amusement) classification.

Evaluated on the **WESAD** dataset (Schmidt et al., 2018) with 15 subjects
undergoing the Trier Social Stress Test.
        """)
    with t2:
        st.markdown("""
1. **Preprocessing** – modality-specific Butterworth filtering, resampling
   - Wrist: EDA/TEMP → 4 Hz, BVP → 64 Hz, ACC → 32 Hz
   - Chest: ECG → bandpass 0.5–40 Hz, EMG → envelope, Resp → 0.1–0.5 Hz
2. **Feature extraction** – 60 s sliding windows (50 % overlap), ~70 features
   - EDA: mean, std, SCR peaks, SCL (wrist + chest)
   - BVP/HR: heart-rate stats, HRV (SDNN, RMSSD), freq (LF/HF)
   - TEMP: mean, std, slope (wrist + chest)
   - ECG: R-peak HR, HRV time & freq, pNN50, QRS morphology
   - EMG: RMS, median/mean freq, peak count
   - Resp: rate, depth, I:E ratio
   - ACC: magnitude, SMA, energy, spectral entropy
3. **ML classifiers** – Random Forest, Logistic Regression, SVM, Decision Tree
4. **DL classifiers** – CNN-1D, UNet-1D, ResNet-1D (PyTorch)
5. **Evaluation** – subject-dependent, subject-independent, LOSO
6. **Interpretability** – SHAP feature importance
        """)
    with t3:
        st.markdown("""
### Deep-Learning Architectures

| Model | Core idea | Final dim |
|---|---|---|
| **CNN-1D** | 3-block Conv1d → BN → ReLU → Pool, AdaptiveAvgPool | 128 |
| **UNet-1D** | Encoder-only with multi-scale GAP fusion (32+64+128+256=480) | 480 |
| **ResNet-1D** | ResNet-34 adapted to 1D ([3,4,6,3] blocks, up to 512 ch) | 512 |

All models accept input shape `(B, 1, n_features)` and output logits for
*C* classes. Trained with **AdamW**, **ReduceLROnPlateau**, and **early
stopping** (patience=10).

Class imbalance is handled via inverse-frequency **weighted CrossEntropyLoss**.
        """)
    with t4:
        st.code("""# 1) Install deps
pip install -r requirements.txt

# 2) Place WESAD data
#    data/WESAD/S2/S2.pkl  ...

# 3) Train ML models
py -u src/training.py --approach all --device both --n-classes 2

# 4) Train DL models
py -u src/dl_training.py --arch resnet1d --approach loso --classes 2

# 5) Compare all
py -u src/dl_training.py --arch all --approach compare

# 6) Demo
streamlit run src/app.py""", language="bash")


if __name__ == "__main__":
    main()
