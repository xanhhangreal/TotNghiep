# Device Ablation Summary (LOSO)

## Binary (2-class)
### Device: wrist
| Family | Input | Model | Accuracy (mean +/- std) | F1 (mean +/- std) | Source |
|---|---|---|---:|---:|---|
| DL | feature_based | CNN1D | 0.6781 +/- 0.0496 | 0.6626 +/- 0.0378 | dl_loso_cnn1d_2cls_20260514_002436.json |
| DL | raw_signal | RawCNN1D | 0.8185 +/- 0.0720 | 0.8194 +/- 0.0652 | dl_loso_rawcnn1d_wrist_2cls_20260514_002345.json |
| ML | feature_based | KNN | 0.8158 +/- 0.0271 | 0.8037 +/- 0.0330 | loso_20260514_002354.json |
| ML | feature_based | RandomForest | 0.8045 +/- 0.1003 | 0.7436 +/- 0.1616 | loso_20260514_002354.json |
| ML | feature_based | SVM | 0.7617 +/- 0.0621 | 0.6875 +/- 0.1077 | loso_20260514_002354.json |
| ML | feature_based | LDA | 0.6860 +/- 0.0663 | 0.6659 +/- 0.0407 | loso_20260514_002354.json |
| ML | feature_based | LogisticRegression | 0.6554 +/- 0.0779 | 0.6049 +/- 0.0655 | loso_20260514_002354.json |
| ML | feature_based | AdaBoost | 0.5438 +/- 0.0133 | 0.5141 +/- 0.0259 | loso_20260514_002354.json |
| ML | feature_based | DecisionTree | 0.5521 +/- 0.1521 | 0.5046 +/- 0.0774 | loso_20260514_002354.json |
