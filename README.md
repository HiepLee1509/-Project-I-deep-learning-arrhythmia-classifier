# 🫀 Mô hình AI phân loại rối loạn nhịp tim ECG

Model này là model AI dùng để **phân tích tín hiệu ECG đơn kênh** và **phát hiện rối loạn nhịp tim** từ các thiết bị đo di động. Repository này chứa mã nguồn xử lý dữ liệu, xây dựng mô hình và huấn luyện mô hình deep learning.

---

## 🚀 Features

* Xử lý và tiền xử lý tín hiệu ECG (lọc nhiễu, chuẩn hóa, chia đoạn 5s).
* Mô hình Deep Learning:

  * CNN-1D
  * LSTM
  * Transformer
* Pipeline huấn luyện đầy đủ (train / validate / test).
* Đánh giá mô hình: Accuracy, F1-score, Confusion Matrix.
* Hướng tới triển khai trên thiết bị edge/embedded.

---

## 📦 Project Structure

```
├── mydata/                 # Dữ liệu ECG đầu vào
├── models/               # Kiến trúc mô hình CNN, LSTM, Transformer
├── frontend.py              # Giao diện trực quan hóa
├── backend.py           # Tiền xử lý & chuẩn hóa dữ liệu và xây dựng model AI
└── README.md
```

---

## 🧠 Model Architectures

* **CNN 1D:** Trích xuất đặc trưng cục bộ.
* **LSTM:** Mô hình hóa quan hệ thời gian.
* **Attention mechanism:**

---

## 🛠️ Preprocessing

* Lọc nhiễu (DWT + R-peaks + Segmentation).
* Chuẩn hóa tín hiệu.
* Chia đoạn 5 giây.

---

## 📥 Cài đặt & Chuẩn bị môi trường

### 1️⃣ Clone repository

### 2️⃣ Tạo môi trường ảo (khuyến nghị)

**Venv (Python built-in):**

```
python -m venv venv
.\venv\Scripts\activate
```

### 3️⃣ Cài đặt dependencies

```
pip install -r requirements.txt
```

---

## 🖥️ Deploy Local
```bash
streamlit run frontend.py
```

## 📘 Requirements

* Python 3.8+
* numpy
* scipy
* pandas
* matplotlib
* torch >= 1.12


## 🎯 Goal

Xây dựng mô hình **nhẹ – chính xác – ổn định**, có thể chạy trên thiết bị đeo hoặc Holter di động để phát hiện rối loạn nhịp tim thời gian thực.

---
