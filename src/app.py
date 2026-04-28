"""Streamlit demo for multimodal stress detection.

Launch:
    streamlit run src/app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import MODELS_DIR, RESULTS_DIR, STRESS_LABELS, STRESS_LABELS_3CLASS
from ml_models import StressModel

try:
    import torch
    from dl_models import load_dl_model

    _HAS_DL = True
except ImportError:
    _HAS_DL = False


st.set_page_config(
    page_title="Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("🧠 Stress Detection")
    st.caption("Multimodal WESAD pipeline (ML + DL)")
    st.divider()

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Predictor", "📈 Performance", "📚 Docs"],
    )
    {
        "📊 Dashboard": dashboard,
        "🔍 Predictor": predictor,
        "📈 Performance": performance,
        "📚 Docs": docs,
    }[page]()


def dashboard():
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ML Models", "4")
    c2.metric("DL Models", "3")
    c3.metric("Dataset", "WESAD (15 subjects)")
    c4.metric("Feature Count", "~89")
    st.divider()

    left, right = st.columns(2)
    left.info(
        """
**All Modalities**

Wrist (Empatica E4): EDA, BVP, TEMP, ACC  
Chest (RespiBAN): ECG, EMG, EDA, Temp, Resp, ACC
"""
    )
    right.success(
        """
**Pipeline**
1. Preprocessing & resampling  
2. Window-based feature extraction  
3. ML: RF / LR / SVM / DT  
4. DL: CNN1D / UNet1D / ResNet1D  
5. Evaluation: subject-dependent, subject-independent, LOSO
"""
    )


def predictor():
    st.header("Stress Predictor")
    model_type = st.radio(
        "Model type",
        ["ML (sklearn)", "DL (PyTorch)"],
        horizontal=True,
    )
    if model_type == "ML (sklearn)":
        _predictor_ml()
    else:
        _predictor_dl()


def _predictor_ml():
    model = None
    model_files = sorted(MODELS_DIR.glob("*.joblib")) if MODELS_DIR.exists() else []
    if model_files:
        sel = st.selectbox("Trained model", [f.stem for f in model_files], key="ml_model_sel")
        try:
            model = StressModel.load(str(MODELS_DIR / f"{sel}.joblib"))
            st.success(f"Loaded `{sel}` ({model.model_type})")
        except Exception as e:
            st.warning(f"Load failed: {e}")
    else:
        st.info("No trained ML models found in `models/`.")

    st.subheader("Manual Input")
    c1, c2, c3 = st.columns(3)
    with c1:
        eda_mean = st.slider("EDA Mean (uS)", 0.0, 20.0, 5.0)
        eda_std = st.slider("EDA Std", 0.0, 10.0, 2.0)
        eda_peaks = st.slider("SCR Peaks", 0, 20, 5)
    with c2:
        hr = st.slider("Heart Rate (bpm)", 40, 150, 80)
        sdnn = st.slider("HRV SDNN (ms)", 10, 200, 50)
        lf_hf = st.slider("LF/HF Ratio", 0.0, 5.0, 1.5)
    with c3:
        temp_m = st.slider("Temp Mean (C)", 32.0, 36.0, 33.5)
        temp_s = st.slider("Temp Std", 0.0, 2.0, 0.5)
        temp_sl = st.slider("Temp Slope", -2.0, 2.0, 0.0)

    if st.button("Predict (ML)", type="primary", key="predict_ml_manual"):
        vec = np.array(
            [[eda_mean, eda_std, eda_peaks, hr, sdnn, lf_hf, temp_m, temp_s, temp_sl]],
            dtype=float,
        )
        if model and model.is_fitted:
            expected = int(getattr(model.scaler, "n_features_in_", vec.shape[1]))
            if vec.shape[1] != expected:
                st.warning(
                    f"Model expects {expected} features but manual input has {vec.shape[1]}. "
                    "Use CSV feature prediction below."
                )
            else:
                try:
                    pred = int(model.predict(vec)[0])
                    prob = model.predict_proba(vec)[0]
                    st.success(f"{STRESS_LABELS.get(pred, str(pred))} ({prob[pred]:.0%})")
                except Exception as e:
                    st.error(str(e))
        else:
            st.warning("No fitted model loaded.")

    st.subheader("Predict From Feature CSV")
    f = st.file_uploader("CSV with numeric feature columns", type=["csv"], key="ml_csv")
    if f is not None:
        df = pd.read_csv(f)
        st.dataframe(df.head(10), use_container_width=True)

        if model and model.is_fitted:
            num_df = df.select_dtypes(include=[np.number]).copy()
            if num_df.empty:
                st.warning("CSV has no numeric feature columns.")
                return

            expected = int(getattr(model.scaler, "n_features_in_", num_df.shape[1]))
            if num_df.shape[1] != expected:
                st.warning(
                    f"Feature count mismatch: expected {expected}, got {num_df.shape[1]}."
                )
                return

            if st.button("Run ML CSV Inference", key="predict_ml_csv"):
                try:
                    X = num_df.to_numpy(dtype=float)
                    preds = model.predict(X)
                    probs = model.predict_proba(X)
                    out = pd.DataFrame(
                        {
                            "prediction": [STRESS_LABELS.get(int(p), str(p)) for p in preds],
                            "confidence": [
                                float(probs[i, int(preds[i])]) for i in range(len(preds))
                            ],
                        }
                    )
                    st.dataframe(out.head(50), use_container_width=True)
                except Exception as e:
                    st.error(str(e))


def _predictor_dl():
    if not _HAS_DL:
        st.warning("PyTorch is not installed.")
        return

    pt_files = sorted(MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []
    if not pt_files:
        st.info("No DL checkpoints found in `models/`.")
        return

    sel = st.selectbox("DL checkpoint", [f.stem for f in pt_files], key="dl_model_sel")
    try:
        model, ckpt = load_dl_model(str(MODELS_DIR / f"{sel}.pt"), return_state=True)
        st.success(f"Loaded `{sel}` ({type(model).__name__})")
    except Exception as e:
        st.warning(f"Load failed: {e}")
        return

    n_features = int(ckpt.get("n_features", 0) or 0)
    n_classes = int(ckpt.get("n_classes", 2) or 2)
    scaler = ckpt.get("scaler")
    label_map = STRESS_LABELS if n_classes == 2 else STRESS_LABELS_3CLASS

    st.subheader("Predict From Feature CSV")
    f = st.file_uploader("CSV with numeric feature columns", type=["csv"], key="dl_csv")
    if f is None:
        return

    df = pd.read_csv(f)
    st.dataframe(df.head(10), use_container_width=True)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.empty:
        st.warning("CSV has no numeric feature columns.")
        return

    if n_features > 0 and num_df.shape[1] != n_features:
        st.warning(
            f"Feature count mismatch: checkpoint expects {n_features}, "
            f"CSV has {num_df.shape[1]}."
        )
        return

    if st.button("Run DL CSV Inference", key="predict_dl_csv"):
        try:
            X = num_df.to_numpy(dtype=float)
            X[~np.isfinite(X)] = np.nan
            col_med = np.nanmedian(X, axis=0)
            col_med = np.where(np.isfinite(col_med), col_med, 0.0)
            rr, cc = np.where(np.isnan(X))
            X[rr, cc] = col_med[cc]

            if scaler is not None:
                X = scaler.transform(X)

            model.eval()
            with torch.no_grad():
                xb = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)

            out = pd.DataFrame(
                {
                    "prediction": [label_map.get(int(p), str(p)) for p in preds],
                    "confidence": [float(probs[i, int(preds[i])]) for i in range(len(preds))],
                }
            )
            st.dataframe(out.head(50), use_container_width=True)
        except Exception as e:
            st.error(str(e))


def performance():
    st.header("Model Performance")
    files = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    if not files:
        st.info("No results found in `results/`.")
        _show_placeholder()
        return

    sel = st.selectbox("Results file", [f.name for f in files], key="result_file")
    try:
        with open(RESULTS_DIR / sel, encoding="utf-8") as fh:
            real = json.load(fh)
    except Exception:
        real = None

    if not real:
        _show_placeholder()
        return

    if "per_subject" in real:
        st.subheader(f"{real.get('arch', '?')} ({real.get('n_classes', '?')}-class LOSO)")
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{real['accuracy_mean']:.4f} ± {real['accuracy_std']:.4f}")
        m2.metric("F1", f"{real['f1_mean']:.4f} ± {real['f1_std']:.4f}")
        rows = [
            {
                "Subject": f"S{pf['test_subject']}",
                "Accuracy": pf["accuracy"],
                "F1": pf["f1"],
                "AUC": pf.get("roc_auc", float("nan")),
            }
            for pf in real["per_subject"]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        return

    if "ml" in real and "dl" in real:
        rows = []
        for section in ("ml", "dl"):
            for mn, m in real[section].items():
                rows.append(
                    {
                        "Type": section.upper(),
                        "Model": mn,
                        "Accuracy": f"{m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f}",
                        "F1": f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}",
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        return

    if "models" in real:
        rows = []
        for mn, m in real["models"].items():
            if "error" not in m:
                rows.append(
                    {
                        "Model": mn,
                        "Accuracy": m.get("accuracy", 0),
                        "F1": m.get("f1", 0),
                        "ROC-AUC": m.get("roc_auc", 0),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            return

    st.json(real)


def _show_placeholder():
    st.caption("Placeholder — train models to see real results.")
    df = pd.DataFrame(
        {
            "Model": ["Random Forest", "SVM", "CNN-1D", "UNet-1D", "ResNet-1D"],
            "Accuracy": [0.88, 0.83, 0.91, 0.90, 0.92],
            "F1": [0.88, 0.83, 0.91, 0.89, 0.92],
        }
    )
    st.dataframe(df, use_container_width=True)
    fig = go.Figure()
    for col in ("Accuracy", "F1"):
        fig.add_trace(go.Bar(x=df["Model"], y=df[col], name=col))
    fig.update_layout(barmode="group", height=380)
    st.plotly_chart(fig, use_container_width=True)


def docs():
    st.header("📚 Documentation")
    st.code(
        """# 1) Install deps
pip install -r requirements.txt

# 2) Train ML
py -u src/training.py --approach all --device both --n-classes 2

# 3) Train DL
py -u src/dl_training.py --arch all --classes both --approach loso

# 4) Run app
streamlit run src/app.py""",
        language="bash",
    )


if __name__ == "__main__":
    main()
