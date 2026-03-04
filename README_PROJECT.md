# Stress Detection System - Thesis Project

**Author**: Anh Ng  
**Timeline**: 8-12 weeks (Feb 2026 - Apr 2026)  
**Objective**: Build a stress detection system using physiological signals from wearable devices, compare subject-dependent vs subject-independent models, and create a demo system.

---

## Project Overview

This thesis project aims to:
1. **Detect stress moments** in lifelog data using physiological signals (EDA/GSR, BVP, Skin Temperature)
2. **Compare approaches**: Subject-dependent vs Subject-independent stress detection models
3. **Explain predictions** using SHAP (SHapley Additive exPlanations) for model interpretability
4. **Evaluate on multiple datasets**: Public datasets (WESAD, AffectiveROAD) + own collected data
5. **Build a demo system** to visualize stress predictions and model explanations

---

## Project Structure

```
src/
├── 1_eda/                      # Exploratory Data Analysis
│   └── 01_data_exploration.ipynb
├── 2_data/
│   ├── preprocessing.py        # Data loading & preprocessing
│   └── dataset.py              # Dataset classes (PyTorch-compatible)
├── 3_features/
│   ├── extraction.py           # EDA, BVP, TEMP feature extraction
│   └── features.ipynb
├── 4_models/
│   ├── baseline.py             # Decision Tree, Random Forest, LR, SVM
│   ├── training.py             # Training pipeline
│   ├── evaluation.py           # Metrics & evaluation
│   └── shap_analysis.py        # SHAP interpretation
├── 5_system/
│   ├── app.py                  # Streamlit demo application
│   └── stress_detector.py      # Main inference module
│
data/
├── public/                     # Public datasets (WESAD, AffectiveROAD, CognitiveLOAD)
│   └── README_DATASETS.md
└── own/                        # Self-collected stress data

models/                         # Saved trained models (.pkl)
results/                        # Experiment outputs (metrics, plots, logs)
thesis/                         # Final thesis document
```

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Jupyter kernels (optional)
python -m ipykernel install --user --name stress_env --display-name "Stress Detection"
```

### 2. Download Datasets

Navigate to `data/public/README_DATASETS.md` for detailed instructions:
- **WESAD**: Lab-based stress dataset with Trier Social Stress Test (TSST)
- **AffectiveROAD**: Driving-based stress dataset from consumer-grade wearables
- **CognitiveLOAD**: Cognitive workload detection dataset

### 3. Explore Data

Start with EDA notebook:
```bash
jupyter notebook src/1_eda/01_data_exploration.ipynb
```

### 4. Train Models

```bash
# Run baseline training on WESAD dataset
python src/4_models/training.py --dataset WESAD --model random_forest --subject_dependent False
```

### 5. Launch Demo

```bash
streamlit run src/5_system/app.py
```

---

## Phases & Milestones

| Phase | Timeline | Tasks |
|-------|----------|-------|
| **1. Prep** | Wk 1-2 | Project setup, EDA on public datasets, literature analysis |
| **2. Data Collection** | Wk 2-3 | Design stress protocol, collect own data |
| **3. Feature Engineering** | Wk 4-5 | Extract physiological features, baseline models |
| **4. Optimization** | Wk 6-7 | Hyperparameter tuning, SHAP analysis, model comparison |
| **5. Validation** | Wk 8 | Test on own dataset, real-world performance |
| **6. Demo & Integration** | Wk 9-10 | Build system, visualization, reportgeneration |
| **7. Thesis Writing** | Wk 11-12 | Documentation, results analysis, final write-up |

---

## Key References

- **Reference Thesis**: `references/Thesis Stress Detection in Lifelog Data for Improved Personalized Lifelog Retrieval system.pdf`
  - Chapter 2: Physiological mechanism, stress detection background
  - Chapter 4: Subject-dependent vs subject-independent models comparison
  - Chapter 5: Real-world stress detection, SHAP analysis

- **Key Datasets**:
  - WESAD (Schmidt et al., 2018)
  - AffectiveROAD (El Haouij et al., 2019)
  - CognitiveLOAD (data source to be determined)

---

## Key Technologies

- **ML Frameworks**: scikit-learn (traditional ML), PyTorch (optional for deep learning)
- **Interpretability**: SHAP library for model explanation
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Web Framework**: Streamlit for interactive demo
- **Signal Processing**: SciPy, NeuroKit2, BioSPPy

---

## Next Steps

1. ✅ **Project initialized** - Folder structure created
2. ⏳ **Run EDA notebook** - Explore WESAD dataset
3. ⏳ **Implement feature extraction** - Create physiological signal features
4. ⏳ **Train baseline models** - Evaluate baseline performance
5. ⏳ **SHAP analysis** - Interpret model decisions
6. ⏳ **Collect own data** - Real-world validation
7. ⏳ **Build demo system** - User-facing application
8. ⏳ **Write thesis** - Finalize documentation

---

## Contact & Notes

- Thesis advisor: [To be added]
- GitHub repo: [To be added - make sure to git commit regularly]
- Data privacy: Ensure all personal data is anonymized and handled ethically

---

**Last Updated**: Feb 18, 2026
