# 🫀AI Model

**Ironman Holter** là mô-đun AI dùng để **phân tích tín hiệu ECG đơn kênh** và **phát hiện rối loạn nhịp tim** từ các thiết bị đo di động. <br> Repository này chứa mã nguồn xử lý dữ liệu, xây dựng mô hình và huấn luyện mô hình deep learning.

## 🚀 Features
- Xử lý và tiền xử lý tín hiệu ECG (lọc nhiễu, chuẩn hóa, chia đoạn 5s).
- Mô hình Deep Learning:
  - CNN-1D
  - LSTM
  - Transformer
- Pipeline huấn luyện đầy đủ (train/validate/test).
- Đánh giá mô hình: Accuracy, F1-score, Confusion Matrix.
- Hướng tới triển khai trên thiết bị edge/embedded.

## 📦 Project Structure
```
├── data/                 # Dữ liệu ECG đầu vào  
├── preprocessing/        # Hàm xử lý và chuẩn hóa tín hiệu  
├── models/               # Kiến trúc mô hình CNN, LSTM, CAT-Net  
├── utils/                # Hàm hỗ trợ (metrics, plotting, ...)  
├── train.py              # Script huấn luyện  
├── evaluate.py           # Script đánh giá  
└── README.md  
```

## 🧠 Model Architectures
- **CNN:** trích xuất đặc trưng cục bộ.
- **LSTM:** mô hình hóa quan hệ thời gian.
- **Transformer:** học quan hệ dài hạn bằng attention.

## 🛠️ Preprocessing
- Lọc nhiễu (baseline wandering, powerline).
- Chuẩn hóa tín hiệu.
- Chia đoạn 5 giây.
- Chuyển đổi sang tensor.

## 🏋️ Training
python train.py --model catnet --epochs 50 --batch_size 32

## 📈 Evaluation
python evaluate.py --model catnet --checkpoint checkpoints/catnet_best.pth

Kết quả đánh giá:
- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix
- Loss & Accuracy curves

## 📘 Requirements
Python 3.8+  
numpy  
scipy  
pandas  
matplotlib  
torch >= 1.12  

Cài đặt nhanh:
pip install -r requirements.txt

## 🎯 Goal
Xây dựng mô hình **nhẹ – chính xác – ổn định**, có thể chạy trên thiết bị đeo hoặc Holter di động để phát hiện rối loạn nhịp tim theo thời gian thực.

## 📄 License
MIT License.
