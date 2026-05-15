# Final Benchmark Summary (WESAD, LOSO 15 subjects)

## Binary (2-class)
| Family | Model | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---:|---:|
| ML | AdaBoost | 0.9710 +/- 0.0485 | 0.9715 +/- 0.0474 |
| ML | RandomForest | 0.9691 +/- 0.0491 | 0.9695 +/- 0.0481 |
| ML | SVM | 0.9663 +/- 0.0560 | 0.9669 +/- 0.0546 |
| ML | LogisticRegression | 0.9662 +/- 0.0575 | 0.9647 +/- 0.0620 |
| DL | UNet1D | 0.9640 +/- 0.0580 | 0.9637 +/- 0.0589 |
| ML | KNN | 0.9548 +/- 0.0607 | 0.9545 +/- 0.0613 |
| ML | LDA | 0.9512 +/- 0.0690 | 0.9497 +/- 0.0724 |
| DL | CNN1D | 0.9306 +/- 0.0787 | 0.9316 +/- 0.0777 |
| DL | ResNet1D | 0.9205 +/- 0.0803 | 0.9183 +/- 0.0833 |
| ML | DecisionTree | 0.8650 +/- 0.1720 | 0.8612 +/- 0.1831 |

## Tri-class (3-class)
| Family | Model | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---:|---:|
| ML | AdaBoost | 0.9247 +/- 0.0648 | 0.9211 +/- 0.0674 |
| ML | RandomForest | 0.9205 +/- 0.0688 | 0.9088 +/- 0.0821 |
| DL | UNet1D | 0.8955 +/- 0.0551 | 0.8839 +/- 0.0703 |
| ML | SVM | 0.8753 +/- 0.0848 | 0.8540 +/- 0.1023 |
| DL | CNN1D | 0.8556 +/- 0.1226 | 0.8524 +/- 0.1211 |
| ML | LDA | 0.8586 +/- 0.1210 | 0.8422 +/- 0.1270 |
| ML | DecisionTree | 0.8343 +/- 0.1280 | 0.8222 +/- 0.1245 |
| ML | KNN | 0.8373 +/- 0.1055 | 0.8126 +/- 0.1109 |
| ML | LogisticRegression | 0.8245 +/- 0.1256 | 0.8006 +/- 0.1305 |
| DL | ResNet1D | 0.7859 +/- 0.1537 | 0.7671 +/- 0.1628 |

## Source files used

- Binary (2-class) / ML / AdaBoost: loso_20260515_003946.json
- Binary (2-class) / ML / RandomForest: loso_20260515_003946.json
- Binary (2-class) / ML / SVM: loso_20260515_003946.json
- Binary (2-class) / ML / LogisticRegression: loso_20260515_003946.json
- Binary (2-class) / ML / KNN: loso_20260515_003946.json
- Binary (2-class) / ML / LDA: loso_20260515_003946.json
- Binary (2-class) / ML / DecisionTree: loso_20260515_003946.json
- Binary (2-class) / DL / UNet1D: dl_loso_unet1d_2cls_20260515_005433.json
- Binary (2-class) / DL / CNN1D: dl_loso_cnn1d_2cls_20260515_005101.json
- Binary (2-class) / DL / ResNet1D: dl_loso_resnet1d_2cls_20260515_010603.json
- Tri-class (3-class) / ML / AdaBoost: loso_20260515_004532.json
- Tri-class (3-class) / ML / RandomForest: loso_20260515_004532.json
- Tri-class (3-class) / ML / SVM: loso_20260515_004532.json
- Tri-class (3-class) / ML / LDA: loso_20260515_004532.json
- Tri-class (3-class) / ML / DecisionTree: loso_20260515_004532.json
- Tri-class (3-class) / ML / KNN: loso_20260515_004532.json
- Tri-class (3-class) / ML / LogisticRegression: loso_20260515_004532.json
- Tri-class (3-class) / DL / UNet1D: dl_loso_unet1d_3cls_20260515_011411.json
- Tri-class (3-class) / DL / CNN1D: dl_loso_cnn1d_3cls_20260515_011059.json
- Tri-class (3-class) / DL / ResNet1D: dl_loso_resnet1d_3cls_20260515_012605.json
