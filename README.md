# Stress Detection from Wearable Physiological Signals

Binary stress classification (relaxed / stressed) using EDA, BVP, and skin
temperature from an Empatica E4 wrist-worn device.  Evaluated on the
**WESAD** dataset (Schmidt et al., 2018).

## Project Structure

```
src/
├── config.py          # paths, constants, model hyperparameters
├── preprocessing.py   # signal filtering, resampling, normalization
├── wesad_loader.py    # WESAD pickle loading, label alignment
├── features.py        # EDA / BVP / TEMP feature extraction
├── models.py          # sklearn model wrappers (RF, LR, SVM, DT)
├── training.py        # training pipeline (CLI entry point)
├── shap_analysis.py   # SHAP interpretability
└── app.py             # Streamlit demo

scripts/
└── setup_wesad.py     # download & verify WESAD dataset

notebooks/
└── 01_data_exploration.ipynb

data/public/WESAD/     # WESAD pickle files (S2/ … S17/)
models/                # saved .joblib models
results/               # JSON experiment results
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & extract WESAD into data/public/WESAD/
py -u scripts/setup_wesad.py

# 3. Train all approaches
py -u src/training.py --approach all

# 4. Launch demo
streamlit run src/app.py
```

## Training Approaches

| Approach | Description |
|---|---|
| `subject_dependent` | Per-subject k-fold CV |
| `subject_independent` | Pool subjects, split by subject groups |
| `loso` | Leave-One-Subject-Out CV (most rigorous) |

```bash
py -u src/training.py --approach loso --window 60 --step 30
```

## Key References

- Schmidt, P. et al. (2018). *Introducing WESAD, a Multimodal Dataset for
  Wearable Stress and Affect Detection.* ICMI.
- Ninh, V.-T. (2023). *Stress Detection in Lifelog Data for Improved
  Personalized Lifelog Retrieval System.* PhD thesis, DCU.
