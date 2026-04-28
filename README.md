# Nghiên cứu phát hiện trạng thái căng thẳng từ tín hiệu sinh lý đa phương thức bằng học sâu

> **Stress Detection from Multimodal Physiological Signals Using Deep Learning**

Phân loại stress nhị phân (Relaxed / Stressed) và 3-class (Baseline / Stress /
Amusement) sử dụng tín hiệu sinh lý đa phương thức từ **cổ tay** (Empatica E4:
EDA, BVP, TEMP, ACC) và **ngực** (RespiBAN: ECG, EMG, EDA, Temp, Resp, ACC).

Đánh giá trên tập dữ liệu **WESAD** (Schmidt et al., 2018) với **mô hình học máy**
(Random Forest, Logistic Regression, SVM, Decision Tree) và **mô hình học sâu**
(CNN-1D, UNet-1D, ResNet-1D).

---

## Mục lục

1. [Cấu trúc dự án](#cấu-trúc-dự-án)
2. [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
3. [Cài đặt môi trường](#cài-đặt-môi-trường)
4. [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
5. [Chạy huấn luyện mô hình](#chạy-huấn-luyện-mô-hình)
6. [Chạy demo (Streamlit)](#chạy-demo-streamlit)
7. [Triển khai (Deploy)](#triển-khai-deploy)
8. [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Cấu trúc dự án

```
├── README.md                 # Hướng dẫn sử dụng (file này)
├── requirements.txt          # Danh sách thư viện Python
├── .gitignore
│
├── src/                      # MÃ NGUỒN
│   ├── config.py             # Cấu hình đường dẫn, hằng số, siêu tham số
│   ├── wesad_loader.py       # Đọc dữ liệu WESAD (pickle), căn chỉnh tín hiệu
│   ├── preprocessing.py      # Tiền xử lý tín hiệu (lọc, resample, chuẩn hóa)
│   ├── features.py           # Trích xuất ~70 đặc trưng từ tất cả phương thức
│   ├── ml_models.py          # Bọc mô hình sklearn (RF, LR, SVM, DT)
│   ├── dl_models.py          # Kiến trúc PyTorch (CNN-1D, UNet-1D, ResNet-1D)
│   ├── training.py           # Pipeline huấn luyện ML (CLI)
│   ├── dl_training.py        # Pipeline huấn luyện DL (CLI)
│   ├── shap_analysis.py      # Phân tích SHAP
│   ├── app.py                # Ứng dụng demo Streamlit
│   ├── setup_wesad.py        # Script tải & xác minh WESAD
│   └── notebooks/
│       └── 01_data_exploration.ipynb
│
├── data/                     # Dữ liệu (gitignored)
│   └── WESAD/
│       ├── S2/S2.pkl
│       ├── S3/S3.pkl
│       └── ...               # S2–S17 (trừ S1, S12)
│
├── models/                   # Mô hình đã lưu (gitignored)
├── results/                  # Kết quả huấn luyện JSON (gitignored)
└── references/               # Bài báo, tài liệu tham khảo (PDF)
```

---

## Yêu cầu hệ thống

| Yêu cầu | Phiên bản tối thiểu |
|----------|---------------------|
| Python   | 3.10+               |
| pip      | 22.0+               |
| RAM      | 8 GB (khuyến nghị 16 GB) |
| GPU      | Không bắt buộc (hỗ trợ CUDA nếu có) |
| OS       | Windows 10/11, Linux, macOS |

---

## Cài đặt môi trường

### 1. Clone repository

```bash
git clone <repository-url>
cd TotNghiep
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

> **Lưu ý GPU**: Nếu máy có NVIDIA GPU và muốn tận dụng CUDA, cài PyTorch theo
> hướng dẫn tại [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
> **trước** khi chạy lệnh trên.

### 4. Kiểm tra cài đặt

```bash
python -c "import torch; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import streamlit; print('Streamlit', streamlit.__version__)"
```

---

## Chuẩn bị dữ liệu

### Tự động (khuyến nghị)

```bash
python src/setup_wesad.py
```

Script sẽ tải WESAD (~2.2 GB) từ server Uni Siegen, giải nén vào `data/WESAD/`,
và kiểm tra tất cả 15 subjects.

### Thủ công

1. Tải WESAD từ: <https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx>
2. Giải nén vào thư mục `data/WESAD/`
3. Đảm bảo cấu trúc:
   ```
   data/WESAD/S2/S2.pkl
   data/WESAD/S3/S3.pkl
   ...
   data/WESAD/S17/S17.pkl
   ```

---

## Chạy huấn luyện mô hình

### Học máy (ML)

```bash
# Tất cả phương pháp đánh giá (subject-dependent, independent, LOSO)
python src/training.py --approach all --device both --n-classes 2

# Chỉ LOSO, dùng tất cả modalities, phân loại 3-class
python src/training.py --approach loso --device both --n-classes 3
```

| Tham số | Mô tả | Mặc định |
|---------|--------|----------|
| `--approach` | `subject_dependent`, `subject_independent`, `loso`, `all` | `all` |
| `--device`   | `wrist`, `chest`, `both` | `both` |
| `--n-classes` | `2` (binary) hoặc `3` (3-class) | `2` |
| `--window`   | Kích thước cửa sổ (giây) | `60` |
| `--step`     | Bước trượt (giây) | `30` |

### Học sâu (DL)

```bash
# Huấn luyện ResNet-1D với LOSO, binary
python src/dl_training.py --arch resnet1d --approach loso --classes binary

# Huấn luyện tất cả kiến trúc DL, cả binary và 3-class
python src/dl_training.py --arch all --classes both --approach loso

# So sánh ML + DL
python src/dl_training.py --arch all --approach compare
```

| Tham số | Mô tả | Mặc định |
|---------|--------|----------|
| `--arch` | `cnn1d`, `unet1d`, `resnet1d`, `all` | `all` |
| `--approach` | `loso`, `subject_independent`, `subject_dependent`, `compare`, `all` | `loso` |
| `--classes` | `binary`, `3class`, `both` | `binary` |
| `--epochs` | Số epoch tối đa | `100` |
| `--batch-size` | Batch size | `64` |
| `--lr` | Learning rate | `0.001` |

Kết quả được lưu tại `results/` dưới dạng JSON.

---

## Chạy demo (Streamlit)

```bash
streamlit run src/app.py
```

Mở trình duyệt tại `http://localhost:8501`. Giao diện gồm 4 trang:

| Trang | Mô tả |
|-------|--------|
| **Dashboard** | Tổng quan dự án, số liệu, kiến trúc |
| **Predictor** | Dự đoán stress từ dữ liệu tải lên (ML hoặc DL) |
| **Performance** | So sánh hiệu suất các mô hình (từ file JSON) |
| **Docs** | Tài liệu kỹ thuật: phương pháp, kiến trúc DL, hướng dẫn |

---

## Triển khai (Deploy)

### Cách 1: Local

```bash
streamlit run src/app.py --server.port 8501 --server.address 0.0.0.0
```

### Cách 2: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY models/ models/
COPY results/ results/
EXPOSE 8501
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t stress-detection .
docker run -p 8501:8501 stress-detection
```

### Cách 3: Streamlit Community Cloud

1. Push repository lên GitHub
2. Truy cập [share.streamlit.io](https://share.streamlit.io)
3. Kết nối repository, chọn `src/app.py` làm main file
4. Deploy

---

## Tài liệu tham khảo

1. Schmidt, P. et al. (2018). *Introducing WESAD, a Multimodal Dataset for
   Wearable Stress and Affect Detection.* ICMI 2018.
   DOI: [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985)
2. Ninh, V.-T. (2023). *Stress Detection in Lifelog Data for Improved
   Personalized Lifelog Retrieval System.* PhD thesis, DCU.
3. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
4. Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for
   Biomedical Image Segmentation.* MICCAI.
