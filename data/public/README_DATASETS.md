# Public Datasets for Stress Detection

This directory contains instructions for downloading and setting up public stress detection datasets.

## Datasets Overview

| Dataset | Source | Sizes | Stressor | Device | Signals |
|---------|--------|-------|----------|--------|---------|
| **WESAD** | UCI ML Repository | ~2 GB | Trier Social Stress Test (TSST) | RespiBAN + Empatica E4 | EDA, BVP, TEMP, ECG, EMG, ACC |
| **AffectiveROAD** | Harvard Dataverse | ~500 MB | Driving (traffic, highways) | Empatica E4 | EDA, BVP, TEMP, HR |
| **CognitiveLOAD** | [Source TBD] | [TBD] | Cognitive tasks | Consumer wearables | HR, HRV, EDA |

---

## WESAD Dataset

### About
- **Paper**: Schmidt et al., 2018 - "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection"
- **Participants**: 15 subjects
- **Duration**: ~2.5 hours per subject
- **Signals**: 7 channels (EDA, BVP, TEMP, ACC, ECG, EMG, respiratory rate)
- **Labels**: Baseline (no stress), stress (TSST), amusement (funny video)

### Download Instructions

1. Visit: https://archive.ics.uci.edu/ml/machine-learning-databases/00465/
2. Download: `WESAD.zip` (~2 GB)
3. Extract to: `data/public/WESAD/`

```bash
cd data/public
# Download using wget or curl
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00465/WESAD.zip
unzip WESAD.zip
```

### Dataset Structure
```
WESAD/
├── S2/      # Subject 2
│   ├── S2.pkl     # Main data file (Python pickle format)
│   ├── S2_E4_Data.csv
│   └── S2_BioHarness_Data.csv
├── S3/
├── S4/
... (15 subjects total)
└── README
```

### Loading WESAD Data

```python
import pickle

with open('data/public/WESAD/S2/S2.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Structure:
# data['signal'] - dict of signal arrays
# data['label'] - labels (0=baseline, 1=stress, 2=amusement)
# data['subject'] - subject ID
```

---

## AffectiveROAD Dataset

### About
- **Paper**: El Haouij et al., 2019 - "AffectiveROAD: A Multimodal Database for Assessing Affective Computing on Real Road Driving"
- **Participants**: 20 subjects
- **Signals**: EDA, BVP, TEMP, HR (from Empatica E4)
- **Duration**: ~1-2 hours per subject
- **Stressor**: Driving through busy roads, highways, traffic

### Download Instructions

1. Visit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/8HH5GB
2. Download files:
   - `AffectiveROAD_PhysiologicalData.zip` (~500 MB)
   - `AffectiveROAD_Documentation.pdf`
3. Extract to: `data/public/AffectiveROAD/`

```bash
cd data/public
# Download and extract (requires manual download from Harvard Dataverse)
unzip AffectiveROAD_PhysiologicalData.zip
```

### Dataset Structure
```
AffectiveROAD/
├── Participant_1/
│   ├── EDA.csv
│   ├── BVP.csv
│   ├── TEMP.csv
│   ├── HR.csv
│   └── labels.csv
├── Participant_2/
...
└── README
```

---

## CognitiveLOAD Dataset (Optional)

### Status: 
- To be added based on availability
- Alternative: Create synthetic cognitive load data or use existing public cognitive workload datasets

### Candidates:
- IEEE StressLVL Dataset
- Public wearable sensor repositories

---

## Data Processing Notes

### Sampling Rates
- **EDA/GSR**: 4 Hz (consumer grade), 256 Hz (lab grade)
- **BVP**: 64 Hz (Empatica E4)
- **TEMP**: 4 Hz
- **HR**: 1 Hz (derived)

### Preprocessing Steps (See `src/2_data/preprocessing.py`)
1. **Resample** signals to common frequency
2. **Remove artifacts** (spikes, dropout)
3. **Normalize** (Z-score, Min-Max)
4. **Segment** into windows (60 sec, 50% overlap)
5. **Align labels** with physiological signals

### Stress Labels
- **0**: Relaxed/Baseline (no stress)
- **1**: Stressed/High-load/Amusement
- **2**: (optional) Amusement/Neutral

---

## Ethical & Privacy Considerations

- ✅ All datasets are **anonymized** (no personal identifiers)
- ✅ Publicly available with proper licensing
- ✅ Use only for research purposes
- ✅ Respect data provider's terms & conditions

---

## Troubleshooting

### Large file sizes
- WESAD is ~2 GB - ensure sufficient disk space
- Consider downloading specific subjects first

### File format issues
- WESAD uses pickle format (Python 2/3 compatible)
- Use `encoding='latin1'` when loading with Python 3

### Missing dependencies
- Install: `pip install pickle pandas scipy`

---

## Dataset Loading Template

```python
import os
import pickle
import pandas as pd
from pathlib import Path

# Load WESAD
def load_wesad(subject_id):
    path = f'data/public/WESAD/S{subject_id}/S{subject_id}.pkl'
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

# Load AffectiveROAD
def load_affectiveroad(participant_id):
    base_path = f'data/public/AffectiveROAD/Participant_{participant_id}/'
    eda = pd.read_csv(base_path + 'EDA.csv')
    bvp = pd.read_csv(base_path + 'BVP.csv')
    temp = pd.read_csv(base_path + 'TEMP.csv')
    hr = pd.read_csv(base_path + 'HR.csv')
    labels = pd.read_csv(base_path + 'labels.csv')
    return {'eda': eda, 'bvp': bvp, 'temp': temp, 'hr': hr, 'labels': labels}
```

---

**Last Updated**: Feb 18, 2026

For more information, visit the thesis reference document: `references/Thesis Stress Detection in Lifelog Data...pdf` - Chapter 4 (Dataset descriptions)
