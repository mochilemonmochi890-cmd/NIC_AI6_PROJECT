> **Lưu ý:** đây là README của bản chạy trên bộ UCI tiếng Anh (đường dẫn tương đối theo thư mục gốc cũ). Bản chính cho dữ liệu của nhóm nằm ở `../README.md`.

# TV2 – Học máy truyền thống (Đề tài 18: phát hiện SMS spam)

Gói bàn giao của TV2. Mọi số liệu dưới đây là kết quả đã chạy thật bằng `train_ml.py`
(chi tiết trong `results/ket_qua_cuoi.json`). Nhãn `y`: **1 = spam, 0 = ham**. Mọi chỉ số tính cho **lớp spam**.

## 1. Cấu trúc thư mục

```
TV2_hoc_may/
├── data/
│   ├── raw/sms.tsv            # bộ UCI gốc (5.574 tin)
│   ├── train.csv  val.csv  test.csv   # cột: id, text, y
├── evaluate.py                # hàm tính điểm DÙNG CHUNG với TV3
├── features_regex.py          # đặc trưng Regex (phải đi cùng model.joblib)
├── train_ml.py                # chạy toàn bộ thí nghiệm từ đầu đến cuối
├── chay_tren_colab.ipynb      # bản chạy trên Google Colab
├── model/
│   ├── model.joblib           # pipeline cuối (cho TV4)
│   └── config.json            # ngưỡng quyết định
├── results/
│   ├── ket_qua_ml.csv (+_VI)              # 8 tổ hợp BoW/TF-IDF x 4 mô hình
│   ├── so_sanh_mat_can_bang.csv (+_VI)    # các cách xử lý mất cân bằng
│   ├── ghep_regex.csv                     # TF-IDF vs TF-IDF + Regex
│   ├── ung_vien_xep_hang_val.csv          # xếp hạng cấu hình theo validation
│   ├── pred_ml_test.csv, pred_ml_val.csv  # id,text,y_true,y_pred,score
│   ├── tin_bi_doan_sai.csv                # phân tích lỗi
│   ├── thong_ke_regex_theo_lop.csv        # trung bình đặc trưng Regex theo lớp
│   ├── ket_qua_cuoi.json                  # số liệu chốt
│   └── figures/                           # biểu đồ cho báo cáo
├── requirements.txt
└── README.md
```

## 2. Dữ liệu và cách chia (TV3 phải dùng đúng bản này)

| | Số tin | Tỉ lệ spam |
|---|---|---|
| UCI gốc | 5.574 | 13,4% |
| Sau khi bỏ 403 tin trùng | 5.171 | |
| train | 3.619 | 12,63% |
| val | 517 | 12,57% |
| test | 1.035 | 12,66% |

- Chia stratify, `random_state=42`: 80% train+val / 20% test, rồi tách val = 12,5% của phần train.
- Đã kiểm tra **không có tin trùng** giữa train, val, test.
- **Quy ước:** train để học; **val để chọn mô hình/tham số/ngưỡng/EarlyStopping**; test chỉ chấm điểm một lần ở cuối.

## 3. Kết quả

### 3.1 Tám tổ hợp (BoW/TF-IDF x 4 mô hình, `class_weight=balanced` nếu có)

| Vector | Mô hình | Precision | Recall | F1 (test) | F1 (val) |
|---|---|---|---|---|---|
| BoW | Naive Bayes | 0,951 | 0,893 | 0,921 | 0,944 |
| BoW | Logistic Regression | 0,929 | 0,893 | 0,910 | 0,939 |
| BoW | SVM | 0,974 | 0,870 | 0,919 | 0,968 |
| BoW | Random Forest | 0,991 | 0,855 | 0,918 | 0,917 |
| TF-IDF | Naive Bayes | 1,000 | 0,687 | 0,815 | 0,850 |
| TF-IDF | Logistic Regression | 0,915 | 0,908 | 0,912 | 0,919 |
| TF-IDF | SVM | 0,944 | 0,901 | 0,922 | 0,938 |
| TF-IDF | Random Forest | 0,926 | 0,863 | 0,893 | 0,902 |

### 3.2 Xử lý mất cân bằng (TF-IDF, F1 spam trên test)

| Cách xử lý | Naive Bayes | Logistic Reg. | SVM | Random Forest |
|---|---|---|---|---|
| Không xử lý (mốc) | 0,815 | 0,825 | 0,921 | 0,872 |
| class_weight=balanced | không hỗ trợ | 0,912 | 0,922 | 0,893 |
| Nhân bản ngẫu nhiên | 0,889 | 0,915 | 0,924 | 0,861 |
| SMOTE | 0,875 | 0,902 | 0,898 | 0,905 |
| Giảm bớt lớp ham | 0,803 | 0,886 | 0,877 | 0,915 |

Nhận xét: với Naive Bayes và Logistic Regression, xử lý mất cân bằng **tăng Recall mạnh** (Logistic: 0,70 lên 0,91). SVM gần như không cần vì đã tốt sẵn. Giảm bớt lớp ham làm Precision tụt (chặn nhầm tin tốt nhiều hơn). Không có cách nào thắng ở mọi mô hình.

### 3.3 Ghép đặc trưng Regex (cải thiện lớn nhất)

| Mô hình | F1 test: TF-IDF | F1 test: TF-IDF + Regex |
|---|---|---|
| Naive Bayes | 0,815 | 0,904 |
| Logistic Regression | 0,912 | 0,945 |
| SVM | 0,922 | 0,952 |
| Random Forest | 0,893 | 0,960 |

Bảng `thong_ke_regex_theo_lop.csv` cho thấy lý do: tin spam có số điện thoại (57% so với 0%), có tiền (50% so với 0,5%), nhiều chữ số và dấu `!` hơn hẳn tin thường.

### 3.4 Mô hình cuối (chọn theo F1 trên validation)

**Random Forest + TF-IDF (1-2 gram) + 9 đặc trưng Regex + `class_weight=balanced`**, ngưỡng 0,5.

| Tập | Precision | Recall | F1 | Ghi chú |
|---|---|---|---|---|
| **Test** (chấm 1 lần) | **1,000** | **0,924** | **0,960** | FP = 0, FN = 10, TP = 121; PR-AUC 0,989 |
| CV 5-fold (train+val) | 0,986 ± 0,012 | 0,918 ± 0,023 | 0,950 ± 0,016 | độ ổn định |

Suy luận: ~9,6 ms/tin (gọi từng tin), 0,07 giây cho 1.035 tin (gọi theo lô). Mô hình nặng 8,8 MB.

## 4. Những điều cần nói thật khi báo cáo

1. **Validation nhỏ** (517 tin, khoảng 65 spam), nên F1 val 0,992 là lạc quan do chọn mô hình trên chính tập này. Con số đáng tin hơn là **F1 test 0,960 và CV 0,950 ± 0,016**. Chênh lệch giữa Random Forest, SVM và Logistic khi có Regex (0,945 đến 0,960) nằm trong khoảng nhiễu: một tin sai khác nhau làm F1 đổi khoảng 0,004.
2. **Precision = 1,000 nhưng Recall = 0,924**: mô hình thận trọng, không chặn nhầm tin tốt nào nhưng còn lọt 10 spam (xem `tin_bi_doan_sai.csv`). Đây là kiểu lỗi thường chấp nhận được cho bộ lọc tin nhắn.
3. Mô hình **chỉ học tiếng Anh**. Thử nhanh một tin spam tiếng Việt cho điểm 0,31 (không bị gắn spam). Đây là lý do cần dữ liệu tiếng Việt của TV1 và thí nghiệm zero-shot/huấn luyện lại của TV5.
4. SVM trong gói là `LinearSVC` bọc `CalibratedClassifierCV` để có xác suất, nên không hoàn toàn là SVM thuần.
5. Số liệu này dựa trên **bản chia do TV2 tự thực hiện** từ bộ UCI. Nếu nhóm chốt bản chia khác (của TV1), chạy lại: `python train_ml.py --train train.csv --test test.csv`. Kết quả sẽ thay đổi nhẹ.
6. Dữ liệu UCI có 403 tin trùng đã bị loại; nếu không loại, điểm test bị thổi phồng.

## 5. Cách chạy lại

```bash
pip install -r requirements.txt
python train_ml.py                                   # dùng bộ UCI trong data/raw
python train_ml.py --train train.csv --test test.csv # dùng file của nhóm (cột text và y/label)
```
Mất khoảng 1 phút. Đọc file UCI phải dùng `quoting=csv.QUOTE_NONE` (nếu không sẽ mất 2 dòng).
Trên Colab: mở `chay_tren_colab.ipynb`.

## 6. Cách nạp mô hình (cho TV4)

```python
import json, joblib   # đặt features_regex.py cùng thư mục với đoạn code này
model = joblib.load("model/model.joblib")
nguong = json.load(open("model/config.json"))["nguong"]      # 0.5

s = model.predict_proba(["Khuyen mai lon nhan ngay tien mat"])[:, 1]
nhan = ["spam" if x >= nguong else "ham" for x in s]
```
- Đầu vào là **list các chuỗi** (không cần tự vector hóa; pipeline đã gồm vectorizer, Regex và bộ phân loại).
- **Bắt buộc** cài cùng phiên bản `scikit-learn` (ghi trong `ket_qua_cuoi.json`: 1.9.1) và để `features_regex.py` cạnh mã nạp. Lệch phiên bản có thể gây lỗi hoặc cảnh báo.
- Mô hình dùng `joblib`/pickle: chỉ nạp file từ nguồn tin cậy.

## 7. Bàn giao cho TV3 (học sâu)

TV3 **không dùng mô hình này để huấn luyện**, mà cần:

| Cần | Lấy ở đâu |
|---|---|
| Dữ liệu chia sẵn, không chia lại | `data/train.csv`, `val.csv`, `test.csv` |
| Cách tính điểm giống hệt | `evaluate.py` (hàm `evaluate(y_true, y_pred, score)`) |
| Mốc ML để so sánh | `results/ket_qua_ml.csv`, `ket_qua_cuoi.json` |
| Dự đoán ML từng dòng | `results/pred_ml_test.csv` |

Quy ước cho TV3:
- Dùng `val` cho EarlyStopping và chọn tham số; `test` chỉ chấm một lần.
- Xử lý mất cân bằng bằng `class_weight` (tỉ lệ ham:spam khoảng 7:1).
- Xuất `results/pred_dl_test.csv` **cùng định dạng** (`id,text,y_true,y_pred,score`) để TV5 ghép so sánh theo từng dòng.
- Tính điểm bằng cùng hàm: `from evaluate import evaluate`.
- Mốc cần so: **F1 spam = 0,960 (ML tốt nhất, có Regex)**; **khoảng 0,92 (tốt nhất khi không có Regex, ví dụ TF-IDF + SVM 0,922)**. So sánh công bằng nên nêu cả hai, vì LSTM/Conv1D tự học từ chuỗi từ chứ không dùng Regex.

## 8. Checklist bàn giao

- [x] `train/val/test.csv` có cột `id,text,y`, tỉ lệ spam ~12,6% ở cả ba, không trùng tin
- [x] `evaluate.py` chạy độc lập
- [x] `ket_qua_ml.csv` đủ 8 tổ hợp; `so_sanh_mat_can_bang.csv` đủ các cách xử lý
- [x] `pred_ml_test.csv` đủ 1.035 dòng, khớp `test.csv` theo `id`
- [x] `model.joblib` nạp lại được trong tiến trình mới và cho kết quả trùng
- [x] Phiên bản thư viện ghi trong README và `ket_qua_cuoi.json`
