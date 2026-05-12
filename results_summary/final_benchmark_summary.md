# Final Benchmark Summary (WESAD, LOSO 15 subjects)

## Binary (2-class)
| Family | Model | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---:|---:|
| ML | LogisticRegression | 0.9906 ± 0.0271 | 0.9904 ± 0.0278 |
| ML | RandomForest | 0.9860 ± 0.0288 | 0.9862 ± 0.0284 |
| ML | SVM | 0.9759 ± 0.0412 | 0.9762 ± 0.0406 |
| DL | UNet1D | 0.9663 ± 0.0580 | 0.9660 ± 0.0589 |
| DL | ResNet1D | 0.9445 ± 0.0769 | 0.9442 ± 0.0770 |
| DL | CNN1D | 0.9411 ± 0.0842 | 0.9362 ± 0.0993 |
| ML | DecisionTree | 0.8238 ± 0.2125 | 0.8104 ± 0.2413 |

## Tri-class (3-class)
| Family | Model | Accuracy (mean ± std) | F1 (mean ± std) |
|---|---|---:|---:|
| ML | RandomForest | 0.9205 ± 0.0688 | 0.9088 ± 0.0821 |
| DL | UNet1D | 0.8656 ± 0.0781 | 0.8592 ± 0.0776 |
| ML | SVM | 0.8753 ± 0.0848 | 0.8540 ± 0.1023 |
| DL | CNN1D | 0.8303 ± 0.1461 | 0.8301 ± 0.1430 |
| ML | DecisionTree | 0.8343 ± 0.1280 | 0.8222 ± 0.1245 |
| ML | LogisticRegression | 0.8245 ± 0.1256 | 0.8006 ± 0.1305 |
| DL | ResNet1D | 0.7926 ± 0.1298 | 0.7770 ± 0.1299 |

## Source files used

- Binary (2-class) / DL / CNN1D: dl_loso_cnn1d_2cls_20260429_070406.json
- Binary (2-class) / DL / ResNet1D: dl_loso_resnet1d_2cls_20260429_080335.json
- Binary (2-class) / DL / UNet1D: dl_loso_unet1d_2cls_20260429_071522.json
- Binary (2-class) / ML / DecisionTree: loso_20260429_064809.json
- Binary (2-class) / ML / LogisticRegression: loso_20260429_064809.json
- Binary (2-class) / ML / RandomForest: loso_20260429_064809.json
- Binary (2-class) / ML / SVM: loso_20260429_064809.json
- Tri-class (3-class) / DL / CNN1D: dl_loso_cnn1d_3cls_20260429_081640.json
- Tri-class (3-class) / DL / ResNet1D: dl_loso_resnet1d_3cls_20260429_091557.json
- Tri-class (3-class) / DL / UNet1D: dl_loso_unet1d_3cls_20260429_082709.json
- Tri-class (3-class) / ML / DecisionTree: loso_20260429_065330.json
- Tri-class (3-class) / ML / LogisticRegression: loso_20260429_065330.json
- Tri-class (3-class) / ML / RandomForest: loso_20260429_065330.json
- Tri-class (3-class) / ML / SVM: loso_20260429_065330.json
