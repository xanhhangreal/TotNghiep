# Methodology

Tai lieu nay tong hop phuong phap thuc hien trong repo de ban viet bao cao/do an.

## 1) Scope

- Bai toan: phat hien stress tu tin hieu sinh ly da phuong thuc.
- Dataset chinh: WESAD (15 subjects, S2-S17 tru S1/S12).
- Nhiem vu:
  - Nhi phan: `Non-stress` vs `Stress`
  - 3 lop: `Baseline`, `Stress`, `Amusement`

## 2) Input Signals

- Wrist (Empatica E4): `EDA (4Hz)`, `BVP (64Hz)`, `TEMP (4Hz)`, `ACC (32Hz)`
- Chest (RespiBAN): `ECG/EMG/EDA/Temp/Resp/ACC (700Hz)`

Cac che do dau vao:

- `wrist`
- `chest`
- `both` (ket hop wrist + chest)

## 3) Preprocessing

Pipeline chinh trong `src/preprocessing.py`:

1. Chuan hoa key, dong bo sampling-rate.
2. Loai artifact/outlier (z-score) theo tung kieu tin hieu.
3. Loc Butterworth:
   - BVP/ECG/EMG/Resp: band-pass tuong ung
   - EDA/Temp/ACC magnitude: low-pass
4. Dien gia tri thieu (interpolate/forward/mean tuy buoc).
5. Resample ve target rate cho qua trinh trich xuat.
6. Normalize (z-score/minmax tuy context).

## 4) Windowing va Labeling

- Cua so mac dinh: `window=60s`, `step=30s`.
- Label cua moi window duoc gan bang majority vote tren segment label hop le.
- Label map:
  - Binary: baseline+amusement -> non-stress, stress -> stress.
  - Tri-class: baseline/stress/amusement -> 0/1/2.

## 5) Modeling

### 5.1 Feature-based ML

`src/training.py` + `src/ml_models.py`:

- RandomForest
- LogisticRegression
- SVM
- DecisionTree
- AdaBoost
- LDA
- KNN

### 5.2 Feature-based DL

`src/dl_training.py` + `src/dl_models.py`:

- CNN1D
- UNet1D
- ResNet1D

Luu y: nhom DL nay hoc tren **feature vector** (khong hoc truc tiep waveform thô).

### 5.3 Raw-signal DL baseline

`src/raw_dl_training.py` + `src/raw_signal.py`:

- RawCNN1D tren cua so tin hieu da preprocess/resample.
- Muc dich: baseline end-to-end de doi chieu voi feature-based pipeline.

## 6) Evaluation Protocols

- `subject_dependent`
- `subject_independent`
- `LOSO` (Leave-One-Subject-Out)

Trong bao cao nen dung LOSO la ket qua chinh.

## 7) Explainability

- SHAP cho mo hinh ML (dac biet RF) trong `src/shap_analysis.py`.
- Artifact hinh trong `results_summary/figures/shap_top_features.png`.

