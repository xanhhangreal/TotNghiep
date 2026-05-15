# Device Ablation Summary (LOSO)

## Binary (2-class)
### Device: both
| Family | Input | Model | Accuracy (mean +/- std) | F1 (mean +/- std) | Source |
|---|---|---|---:|---:|---|
| DL | feature_based | UNet1D | 0.9640 +/- 0.0580 | 0.9637 +/- 0.0589 | dl_loso_unet1d_2cls_20260515_005433.json |
| DL | feature_based | CNN1D | 0.9306 +/- 0.0787 | 0.9316 +/- 0.0777 | dl_loso_cnn1d_2cls_20260515_005101.json |
| DL | feature_based | ResNet1D | 0.9205 +/- 0.0803 | 0.9183 +/- 0.0833 | dl_loso_resnet1d_2cls_20260515_010603.json |
| DL | raw_signal | RawCNN1D | 0.9519 +/- 0.0940 | 0.9448 +/- 0.1157 | dl_loso_rawcnn1d_both_2cls_20260515_190914.json |
| ML | feature_based | AdaBoost | 0.9710 +/- 0.0485 | 0.9715 +/- 0.0474 | loso_20260515_003946.json |
| ML | feature_based | RandomForest | 0.9691 +/- 0.0491 | 0.9695 +/- 0.0481 | loso_20260515_003946.json |
| ML | feature_based | SVM | 0.9663 +/- 0.0560 | 0.9669 +/- 0.0546 | loso_20260515_003946.json |
| ML | feature_based | LogisticRegression | 0.9662 +/- 0.0575 | 0.9647 +/- 0.0620 | loso_20260515_003946.json |
| ML | feature_based | KNN | 0.9548 +/- 0.0607 | 0.9545 +/- 0.0613 | loso_20260515_003946.json |
| ML | feature_based | LDA | 0.9512 +/- 0.0690 | 0.9497 +/- 0.0724 | loso_20260515_003946.json |
| ML | feature_based | DecisionTree | 0.8650 +/- 0.1720 | 0.8612 +/- 0.1831 | loso_20260515_003946.json |

## Tri-class (3-class)
### Device: both
| Family | Input | Model | Accuracy (mean +/- std) | F1 (mean +/- std) | Source |
|---|---|---|---:|---:|---|
| DL | feature_based | UNet1D | 0.8955 +/- 0.0551 | 0.8839 +/- 0.0703 | dl_loso_unet1d_3cls_20260515_011411.json |
| DL | feature_based | CNN1D | 0.8556 +/- 0.1226 | 0.8524 +/- 0.1211 | dl_loso_cnn1d_3cls_20260515_011059.json |
| DL | feature_based | ResNet1D | 0.7859 +/- 0.1537 | 0.7671 +/- 0.1628 | dl_loso_resnet1d_3cls_20260515_012605.json |
| DL | raw_signal | RawCNN1D | 0.8217 +/- 0.1358 | 0.8154 +/- 0.1330 | dl_loso_rawcnn1d_both_3cls_20260515_192405.json |
| ML | feature_based | AdaBoost | 0.9247 +/- 0.0648 | 0.9211 +/- 0.0674 | loso_20260515_004532.json |
| ML | feature_based | RandomForest | 0.9205 +/- 0.0688 | 0.9088 +/- 0.0821 | loso_20260515_004532.json |
| ML | feature_based | SVM | 0.8753 +/- 0.0848 | 0.8540 +/- 0.1023 | loso_20260515_004532.json |
| ML | feature_based | LDA | 0.8586 +/- 0.1210 | 0.8422 +/- 0.1270 | loso_20260515_004532.json |
| ML | feature_based | DecisionTree | 0.8343 +/- 0.1280 | 0.8222 +/- 0.1245 | loso_20260515_004532.json |
| ML | feature_based | KNN | 0.8373 +/- 0.1055 | 0.8126 +/- 0.1109 | loso_20260515_004532.json |
| ML | feature_based | LogisticRegression | 0.8245 +/- 0.1256 | 0.8006 +/- 0.1305 | loso_20260515_004532.json |
