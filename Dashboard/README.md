# Social Listening Sentiment Dashboard

Dashboard Streamlit tương tác cho bài toán phân tích cảm xúc mạng xã hội (F&B Social Listening),
đi kèm với notebook `Midterm_DA.ipynb`. Giao diện lấy cảm hứng từ admin template **Gentelella**
(sidebar tối, thẻ KPI, panel biểu đồ), viết lại hoàn toàn bằng CSS tùy chỉnh cho Streamlit.

## Cấu trúc thư mục

```
Dashboard/
├── app.py                # App Streamlit chính (chạy file này)
├── theme.py              # Design system: màu sắc, CSS, các hàm dựng giao diện (KPI card, panel...)
├── prepare_data.py       # Script tiền xử lý: train model, sinh toàn bộ artifact cho app.py
├── requirements.txt      # Thư viện đã pin version, cài lại được y hệt
├── README.md
├── data/                 # Artifact dữ liệu (sinh ra bởi prepare_data.py)
│   ├── mentions.parquet       # 60,000 mention mẫu (dùng cho trang Tổng quan, Chủ đề)
│   ├── test_eval.parquet      # 12,000 mention tập test (dùng cho trang Hiệu năng mô hình)
│   ├── topic_info.json        # Thông tin 5 chủ đề LDA
│   ├── model_metrics.json     # Bảng so sánh model, ROC/PR, McNemar, Cross-Validation
│   └── shap_importance.json   # Top từ khóa ảnh hưởng đến dự đoán
└── models/               # Model đã huấn luyện sẵn (sinh ra bởi prepare_data.py)
    ├── vectorizer.pkl          # TF-IDF vectorizer đã fit
    ├── lr_model.pkl            # Logistic Regression đã huấn luyện
    └── mean_train.npy          # Vector trung bình TF-IDF trên tập train (giải thích dự đoán)
```

## Cách chạy (dùng virtual environment — khuyến nghị)

Chạy lần lượt từng lệnh sau trong PowerShell/Terminal, tại đúng thư mục `Dashboard/`:

```bash
# 1. Tạo môi trường ảo riêng cho project (chỉ cần làm 1 lần)
python -m venv venv

# 2. Kích hoạt môi trường ảo — PHẢI thấy "(venv)" xuất hiện đầu dòng lệnh sau bước này
venv\Scripts\Activate.ps1        # Windows PowerShell
# source venv/bin/activate       # macOS/Linux

# 3. Cài thư viện (chỉ cần làm lại nếu requirements.txt thay đổi)
pip install -r requirements.txt

# 4. Chạy dashboard
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501` (thường tự mở sẵn).

Nếu bước 2 báo lỗi PowerShell chặn chạy script, chạy lệnh sau rồi thử lại bước 2:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Lần sau mở lại project chỉ cần lặp lại bước 2 và 4 (không cần tạo lại venv hay cài lại thư viện).

> **Lưu ý khi nộp bài / chia sẻ project:** không zip/gửi kèm thư mục `venv/` — nó chỉ là bản sao
> thư viện cài đặt (thường vài trăm MB), tái tạo lại được ngay bằng 3 lệnh ở trên với
> `requirements.txt`. Cũng nên xoá `__pycache__/` nếu có trước khi nén.

## Tái tạo lại data/model từ dataset gốc (tuỳ chọn)

Bộ `data/` và `models/` đã có sẵn trong project, **không bắt buộc phải chạy lại**. Chỉ cần chạy
lại nếu muốn đổi `SAMPLE_SIZE`/`RANDOM_STATE` hoặc train trên máy khác:

```bash
python prepare_data.py
```

File dataset gốc cần có: `training_1600000_processed_noemoticon_csv.zip` (Kaggle:
`kazanova/sentiment140`) — sửa biến `RAW_PATH` trong `prepare_data.py` trỏ đúng đường dẫn trên
máy bạn trước khi chạy.

## 4 trang trong dashboard

1. **Tổng quan** — KPI tổng mention/%tích cực/%tiêu cực, donut chart, xu hướng %tiêu cực theo
   ngày (tự đánh dấu ngày cao điểm), phân phối độ dài văn bản, mention theo khung giờ trong
   ngày. Sidebar có bộ lọc (khoảng ngày, cảm xúc, chủ đề), nút **Đặt lại bộ lọc** và nút
   **Tải dữ liệu đã lọc (CSV)**.
2. **Chủ đề & Từ khóa** — chọn 1 trong 5 chủ đề LDA để xem từ khóa đặc trưng, tỷ lệ cảm xúc theo
   từng chủ đề, và Treemap quy mô 5 chủ đề trên toàn bộ dữ liệu.
3. **Hiệu năng mô hình** — KPI Accuracy/F1/AUC, so sánh 3 mô hình (Naive Bayes / Logistic
   Regression / SVM), slider ngưỡng phân loại tương tác (confusion matrix cập nhật real-time),
   đường cong ROC và Precision-Recall, bảng các trường hợp dự đoán sai, biểu đồ diễn giải mô
   hình. Số liệu trang này cố định trên tập test 12,000 mention, không đổi theo bộ lọc sidebar.
4. **Thử dự đoán trực tiếp** — nhập câu tiếng Anh bất kỳ (hoặc bấm 1 trong 3 câu ví dụ có sẵn),
   mô hình dự đoán ngay Tích cực/Tiêu cực kèm biểu đồ từ khóa ảnh hưởng đến kết quả, và bảng
   lịch sử 5 lượt dự đoán gần nhất trong phiên.

Sidebar có thể thu gọn thành thanh icon hẹp bằng cách bấm vào logo "SL" ở góc trên — bấm lại để
mở rộng về trạng thái đầy đủ.

## Lưu ý khi dùng lại/nộp bài

- Mọi số liệu trong dashboard được tính từ `RANDOM_STATE=42`, `SAMPLE_SIZE=60000` — **khớp
  100% với notebook** `Midterm_DA.ipynb`. Nếu đổi 1 trong 2 giá trị này ở `prepare_data.py`, nhớ
  đổi tương ứng bên notebook để 2 deliverable không lệch số.
- Mô hình chỉ huấn luyện trên tiếng Anh (Sentiment140) — trang "Thử dự đoán trực tiếp" sẽ không
  chính xác với input tiếng Việt.
- Trang Hiệu năng mô hình dùng công thức `đóng góp = hệ số hồi quy × (giá trị TF-IDF − giá trị
  trung bình tập train)` để giải thích dự đoán — đây chính xác là công thức SHAP cho mô hình
  tuyến tính, tính tay để tránh phụ thuộc thư viện `shap` lúc chạy dashboard (đã dùng `shap`
  đầy đủ trong notebook để đối chiếu).
