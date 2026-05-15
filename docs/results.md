# Results (WESAD, LOSO)

Tai lieu nay tong hop ket qua chinh de dua vao bao cao/trinh bay.
Nguon: `results_summary/final_benchmark_summary.*`, `device_ablation_summary.*`, `subject_error_analysis.*`.

## 1) Cau hinh tong quan

- Dataset: WESAD (15 subjects)
- Device mode chinh: `both`
- Window/step: `60s/30s`
- Protocol chinh: LOSO

## 2) Binary (2-class)

| Family | Model | Accuracy (mean +/- std) | F1 (mean +/- std) |
|---|---|---:|---:|
| ML | AdaBoost | 0.9710 +/- 0.0485 | 0.9715 +/- 0.0474 |
| ML | RandomForest | 0.9691 +/- 0.0491 | 0.9695 +/- 0.0481 |
| ML | SVM | 0.9663 +/- 0.0560 | 0.9669 +/- 0.0546 |
| ML | LogisticRegression | 0.9662 +/- 0.0575 | 0.9647 +/- 0.0620 |
| DL (feature) | UNet1D | 0.9640 +/- 0.0580 | 0.9637 +/- 0.0589 |
| DL (raw) | RawCNN1D | 0.9519 +/- 0.0940 | 0.9448 +/- 0.1157 |

## 3) Tri-class (3-class)

| Family | Model | Accuracy (mean +/- std) | F1 (mean +/- std) |
|---|---|---:|---:|
| ML | AdaBoost | 0.9247 +/- 0.0648 | 0.9211 +/- 0.0674 |
| ML | RandomForest | 0.9205 +/- 0.0688 | 0.9088 +/- 0.0821 |
| DL (feature) | UNet1D | 0.8955 +/- 0.0551 | 0.8839 +/- 0.0703 |
| DL (raw) | RawCNN1D | 0.8217 +/- 0.1358 | 0.8154 +/- 0.1330 |

## 4) Subject kho (error analysis)

Theo trung binh tren 10 mo hinh LOSO:

- Binary: `S2`, `S3`, `S14` la 3 subject kho nhat.
- Tri-class: `S2`, `S10`, `S14` la 3 subject kho nhat.

Chi tiet xem:

- `results_summary/subject_error_analysis.md`
- `results_summary/subject_level_metrics.csv`

## 5) Artifact hinh de dua vao slide/bao cao

- `results_summary/figures/model_comparison_binary.png`
- `results_summary/figures/model_comparison_3class.png`
- `results_summary/figures/confusion_matrix_binary.png`
- `results_summary/figures/confusion_matrix_3class.png`
- `results_summary/figures/shap_top_features.png`

## 6) Ghi chu khi viet bao cao

- Can noi ro su khac nhau giua:
  - DL feature-based (CNN1D/UNet1D/ResNet1D tren feature vector)
  - DL raw-signal baseline (RawCNN1D tren waveform windows)
- Khi so sanh benchmark WESAD cong bo, phai ghi ro khac biet ve preprocessing, windowing, split protocol va tap tin hieu su dung.

