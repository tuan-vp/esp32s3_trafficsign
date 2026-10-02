# quantization_model

Module train TinyCNN, self-training và export ONNX/TFLite cho bài toán nhận diện biển báo (10 lớp).

## CHÚ Ý: ĐƯA FOLDER TRAIN VÀ TEST VÀO TRONG NÀY ĐỂ TRAIN


## 1) Yêu Cầu

- Python 3.10+
- Khuyến nghị dùng virtual environment
- CHÚ Ý: ĐƯA FOLDER TRAIN VÀ TEST VÀO TRONG NÀY ĐỂ TRAIN
## 2) Cài Đặt

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 3) Chuẩn Bị Dữ Liệu

Dự án cần dữ liệu train/test theo cấu trúc bạn đang dùng nội bộ. Sau khi đặt dữ liệu đúng chỗ, chạy augment:

```powershell
python scripts/utils/arguemted_data.py
```

Mặc định pipeline đọc dữ liệu từ `arguemted_data` (xem `src/settings.py`, biến `COMMON.data_root`).

## 4) Train Và Self-Training

```powershell
python scripts/training/train_tinycnn.py
python scripts/self_training/pseudo_label_from_test.py
python scripts/training/train_tinycnn.py
```

Checkpoint mặc định:

- Best: `models/checkpoints/tinycnn_best.pth`
- Last: `models/checkpoints/tinycnn_last.pth`

## 5) Đánh Giá Và Export

```powershell
python scripts/evaluation/validate_tinycnn.py
python scripts/export/export_tinycnn_onnx.py
python scripts/export/convert_onnx_to_tflite.py
python scripts/evaluation/predict_tflite_float32.py
python scripts/evaluation/predict_tflite_int8.py
```

Output chính:

- ONNX: `models/exports/tinycnn.onnx`
- TFLite float32: `models/exports/tflite_models/tinycnn_float32.tflite`
- TFLite int8: `models/exports/tflite_models/tinycnn_int8.tflite`

## 6) Cấu Hình Nhanh

Sửa trong `src/settings.py`:

- `COMMON.data_root`: đường dẫn dữ liệu train
- `COMMON.apply_exposure`: bật/tắt cân bằng sáng
- `COMMON.num_workers`: `-1` để auto
- `TRAIN.*`: epochs, learning rate, checkpoint path

## 7) Troubleshooting

Nếu convert ONNX -> TFLite lỗi liên quan `ml_dtypes`:

```powershell
python -m pip install --force-reinstall "ml-dtypes==0.2.0" "numpy==1.26.4"
```
