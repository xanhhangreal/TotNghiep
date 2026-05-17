# Reproduce Experiments (WESAD, LOSO)

Tai lieu nay mo ta cac lenh toi thieu de tai lap ket qua benchmark trong repo.

## 1) Chuan bi moi truong

```bash
git pull origin main
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Neu can chay giao dien Streamlit de demo (khong bat buoc cho bao cao/khoa luan):

```bash
python -m pip install -r requirements-demo.txt
```

Neu may ban dung `py` thay vi `python`, co the thay the tuong ung trong cac lenh.

## 2) Chuan bi du lieu WESAD

```bash
python src/setup_wesad.py
```

Neu da co san `data/WESAD`, co the bo qua buoc nay.

## 3) Chay thuc nghiem LOSO

ML (feature-based):

```bash
python src/training.py --approach loso --device both --n-classes 2
python src/training.py --approach loso --device both --n-classes 3
```

DL feature-based:

```bash
python src/dl_training.py --arch all --classes both --approach loso --device both
```

DL raw-signal baseline:

```bash
python src/raw_dl_training.py --classes both --device both
```

## 4) Tong hop ket qua

```bash
python src/build_results_summary.py
python src/build_device_ablation_summary.py
python src/analyze_loso_errors.py
```

## 5) Artifact can nop/doi chieu

- `results/*.json`: ket qua goc tung lan chay.
- `results_summary/final_benchmark_summary.md`: bang benchmark chinh.
- `results_summary/device_ablation_summary.md`: ablation theo device/input.
- `results_summary/subject_error_analysis.md`: phan tich subject kho.
- `results_summary/figures/*.png`: bieu do va confusion matrix.

## 6) Kiem tra nhanh truoc khi push

```bash
python -m compileall src tests
python -m pytest -q
```

