# TV2 – Học máy truyền thống (Đề tài 18: phát hiện SMS spam)

Bản này chạy trên **file `train.csv`, `test.csv` của nhóm** (tin nhắn **tiếng Việt**, cột `message,label`, 1 = spam, 0 = ham).
Mọi số liệu do `train_ml.py` sinh ra (xem `results/ket_qua_cuoi.json`). Chỉ số tính cho **lớp spam**.
Kết quả chạy trên bộ UCI tiếng Anh (bản trước) được giữ trong `tham_khao_UCI_tieng_Anh/` để so sánh.

## 1. Tóm tắt (đọc trước)

**Mô hình cuối:** Random Forest + TF-IDF + 13 đặc trưng Regex + `class_weight=balanced`, ngưỡng 0,55.

| Tập | Precision | Recall | F1 spam |
|---|---|---|---|
| **Test** (chấm 1 lần, 566 tin) | **0,788** | **0,918** | **0,848** |
| CV 5-fold (train+val, chia theo nhóm mẫu câu) | 0,940 ± 0,019 | 0,969 ± 0,004 | 0,954 ± 0,011 |

Trên test: bắt đúng 123/134 tin spam, nhưng **chặn nhầm 33 tin tốt** (FP). Với bộ lọc tin nhắn, FP là lỗi đáng ngại hơn FN.

**Ba điều phải nói thật trong báo cáo:**
1. **Điểm validation/CV (0,95–0,97) cao hơn điểm test (0,85) rất nhiều.** Test khó hơn: chỉ 9% tin test có mẫu câu đã gặp ở train. Tách theo nhóm: mẫu câu đã thấy F1 = 0,991 (67 tin), **mẫu câu mới F1 = 0,765 (499 tin)**. Khả năng tổng quát hóa sang mẫu mới mới là con số thực tế.
2. **Đặc trưng Regex giúp trên validation nhưng không giúp trên test** (mục 4.3). Đừng viết "Regex cải thiện mô hình" nếu không nêu rõ điều này.
3. **Một số "FP" có thể là nhãn sai.** 11/33 tin tốt bị chặn nhầm chứa từ khóa casino/vay/link lạ (ví dụ quảng cáo `pkwin`, `fg68888.com` đang mang nhãn ham). Ngoài ra có 3 nhóm tin cùng khuôn nhưng một tin nhãn ham, một tin nhãn spam. Cần người của nhóm rà lại (mục 5.2).

## 2. Cấu trúc thư mục

```
TV2_hoc_may/
├── README.md
├── requirements.txt        thư viện để HUẤN LUYỆN (đầy đủ)
├── requirements-deploy.txt   thư viện để NẠP MÔ HÌNH / chạy API (nhẹ)
├── train_ml.py            chạy toàn bộ thí nghiệm
├── evaluate.py            hàm tính điểm DÙNG CHUNG với TV3
├── features_regex.py      đặc trưng Regex (phải đi cùng model.joblib)
├── chay_tren_colab.ipynb
├── data/
│   ├── raw/train_goc.csv, test_goc.csv   file gốc của nhóm
│   └── train.csv, val.csv, test.csv      bản đã làm sạch + chia (cột id,text,y) <- TV3 dùng
├── model/model.joblib, config.json       cho TV4
├── results/               bảng, dự đoán, hình, danh sách nhãn cần rà (kết quả chính)
├── results_chia_ngau_nhien/   cùng thí nghiệm nhưng val chia ngẫu nhiên (để chứng minh mục 5.1)
└── tham_khao_UCI_tieng_Anh/   kết quả bộ UCI tiếng Anh (bản cũ)
```

## 3. Dữ liệu và cách chia

| Bước | Số liệu |
|---|---|
| File gốc | train 2.394 tin (spam 26,7%), test 597 tin (spam 26,5%) |
| Trùng lặp | 303 tin trùng trong train, 31 trong test, 27 tin có ở cả hai tập |
| Sau làm sạch | train **1806** + val **258** + test **566** |
| Tỉ lệ spam | train 28,0%, val 27,9%, test 23,7% |

- Loại tin trùng; 27 tin trùng giữa hai tập bị **loại khỏi train** để test không bị rò rỉ.
- **Val tách theo nhóm "mẫu câu"** (bỏ thẻ `[..]`, số, lấy 50 ký tự đầu làm khóa; tin cùng khuôn nằm cùng một phía). Val có 0% mẫu câu trùng train. Lý do: tin nhắn này sinh từ các khuôn lặp (OTP, nhà mạng, casino), tách ngẫu nhiên làm val quá dễ (so sánh ở mục 5.1).
- Test tỉ lệ spam thấp hơn train (23,7% so với 28,0%): có lệch phân phối.
- **Quy ước:** train để học; val để chọn mô hình/tham số/ngưỡng; test chỉ chấm một lần.

## 4. Kết quả

### 4.1 Tám tổ hợp BoW/TF-IDF x 4 mô hình (`class_weight=balanced` nếu có)
| Vector | Mô hình | Precision | Recall | F1 (test) | F1 (val) | FP | FN |
|---|---|---|---|---|---|---|---|
| BoW | Naive Bayes | 0,676 | 0,731 | 0,703 | 0,825 | 47 | 36 |
| BoW | Logistic Regression | 0,686 | 0,963 | 0,801 | 0,960 | 59 | 5 |
| BoW | SVM | 0,718 | 0,970 | 0,825 | 0,946 | 51 | 4 |
| BoW | Random Forest | 0,780 | 0,873 | 0,824 | 0,946 | 33 | 17 |
| TFIDF | Naive Bayes | 0,833 | 0,672 | 0,744 | 0,797 | 18 | 44 |
| TFIDF | Logistic Regression | 0,696 | 0,940 | 0,800 | 0,923 | 55 | 8 |
| TFIDF | SVM | 0,768 | 0,940 | 0,846 | 0,947 | 38 | 8 |
| TFIDF | Random Forest | 0,744 | 0,933 | 0,828 | 0,954 | 43 | 9 |

### 4.2 Xử lý mất cân bằng (TF-IDF; F1 spam trên test, trong ngoặc là Recall)
| Cách xử lý | Naive Bayes | Logistic Regression | SVM | Random Forest |
|---|---|---|---|---|
| Không xử lý (mốc) | 0,744 (R 0,672) | 0,824 (R 0,806) | 0,848 (R 0,933) | 0,832 (R 0,851) |
| class_weight=balanced | không hỗ trợ | 0,800 (R 0,940) | 0,846 (R 0,940) | 0,828 (R 0,933) |
| Nhân bản ngẫu nhiên | 0,762 (R 0,895) | 0,814 (R 0,933) | 0,838 (R 0,851) | 0,829 (R 0,925) |
| SMOTE | 0,775 (R 0,910) | 0,806 (R 0,933) | 0,868 (R 0,881) | 0,848 (R 0,918) |
| Giảm bớt lớp ham | 0,755 (R 0,873) | 0,751 (R 0,948) | 0,785 (R 0,955) | 0,751 (R 0,948) |

Nhận xét: xử lý mất cân bằng **tăng Recall** (Naive Bayes, Logistic) nhưng **đổi lại Precision giảm** (nhiều FP hơn). Giảm bớt lớp ham làm Precision tụt mạnh nhất. SVM và Random Forest ít nhạy hơn.

### 4.3 Ghép đặc trưng Regex
| Mô hình | F1 test: TF-IDF | F1 test: TF-IDF + Regex | F1 val: TF-IDF | F1 val: TF-IDF + Regex |
|---|---|---|---|---|
| Naive Bayes | 0,744 | 0,596 | 0,797 | 0,549 |
| Logistic Regression | 0,800 | 0,708 | 0,923 | 0,922 |
| SVM | 0,846 | 0,819 | 0,947 | 0,966 |
| Random Forest | 0,828 | 0,825 | 0,954 | 0,973 |

Regex tăng F1 trên **validation** của SVM và Random Forest, nhưng trên **test** thì không (SVM 0,846 → 0,819; Random Forest gần như không đổi), và làm Naive Bayes/Logistic tệ hơn. Trong dữ liệu này, nhiều đặc trưng Regex không phân biệt tốt (ví dụ tỉ lệ tin có link: ham 55%, spam 60%; có `[MONEY]`: ham 72%, spam 30% vì tin nhà mạng hợp lệ chứa nhiều thông tin tiền). Chi tiết: `results/thong_ke_regex_theo_lop.csv`.

### 4.4 Cấu hình được chọn (theo F1 trên validation)
Random Forest + TF-IDF + Regex + `class_weight=balanced`; val F1 0,973; ngưỡng 0,55.
Lưu ý: **TF-IDF + SVM (không Regex) có F1 test 0,846, ngang mô hình cuối (0,848)**. Chênh lệch này nằm trong nhiễu (test chỉ có 134 tin spam; một tin sai đổi F1 khoảng 0,004). Chọn theo validation là đúng quy ước, nhưng không nên tuyên bố "Random Forest thắng".

Suy luận: 11,48 ms/tin (từng tin), 0,0744 giây cho 566 tin (theo lô). Mô hình khoảng 5,7 MB.

## 5. Phân tích sâu (dùng cho Chương 4)

### 5.1 Chia val ngẫu nhiên và chia theo nhóm mẫu câu
| | Val chia ngẫu nhiên | Val chia theo nhóm (bản chính) |
|---|---|---|
| Tin val có mẫu câu trong train | 31% | 0% |
| F1 test của mô hình chọn được | 0,809 (Precision 0,724, FP 47) | **0,848** (Precision 0,788, FP 33) |

Chia theo nhóm cho F1 test cao hơn (chênh lệch này có thể một phần là nhiễu vì test chỉ có 134 tin spam), nhưng **chưa xóa hết khoảng cách** giữa val và test. Giả thuyết: khóa mẫu câu 50 ký tự còn thô, tin cùng nội dung nhưng khác tiền tố (khác tên thương hiệu) vẫn bị coi là mẫu khác; và test có thể đến từ nguồn khác train. Dữ liệu hiện có chưa đủ để kết luận chắc.

**Về độ "sạch" của tập test:** mã chọn mô hình/tham số/ngưỡng chỉ dùng validation, nhưng quyết định chuyển sang chia validation theo nhóm được đưa ra **sau khi đã thấy** khoảng cách giữa điểm validation và điểm test (lần chạy đầu với validation ngẫu nhiên cho F1 test 0,809). Vì vậy test không hoàn toàn "chưa từng ảnh hưởng" tới quy trình; nên nêu rõ điều này trong báo cáo thay vì nói test được chấm "đúng một lần" theo nghĩa tuyệt đối.

### 5.2 Nhãn cần rà lại (làm tay, ~15 phút)
Mình **không tự sửa nhãn**. Hai file dưới đây giúp nhóm rà nhanh, xếp theo độ đáng ngờ:

- `results/nhan_khong_nhat_quan.csv` (6 tin): **3 nhóm tin cùng khuôn nhưng một tin nhãn ham, một tin nhãn spam**. Đây là bằng chứng khách quan nhất. Ví dụ: hai tin "Thiết bị [NUMBER] đã bị mở khóa..." trong test giống nhau, chỉ khác dấu xuống dòng, nhưng id 426 nhãn ham còn id 542 nhãn spam; tin giả mạo Techcombank id 421 (test) nhãn ham trong khi tin cùng khuôn ở train (id 1304, tên miền giả) nhãn spam; hai tin casino `8qbet` trong train (id 753 nhãn ham, id 1355 nhãn spam).
- `results/tin_can_ra_soat_nhan.csv` (13 trong 44 tin bị đoán sai): cột `ly_do` gồm **A** (thuộc nhóm nhãn ngược nhau), **B** (nhãn ham nhưng chứa từ khóa casino/vay/link), **C** (mô hình rất tự tin: điểm ≥ 0,8 với tin nhãn ham, ≤ 0,2 với tin nhãn spam). Điền nhãn đúng vào cột cuối.
- Trong test có 4 tin `pkwin`: 3 tin nhãn spam, 1 tin nhãn ham (id 427).
- `results/tin_bi_doan_sai.csv`: toàn bộ 44 tin bị đoán sai.

Lưu ý: nhóm B có thể gồm cả tin hợp lệ (Tiki, Ahamove, Vietnamobile, Bách Hóa Xanh gửi link chính hãng). Đừng đổi nhãn hàng loạt, hãy xem từng tin.
Nếu nhóm sửa nhãn, phải sửa trong **file gốc** rồi chạy lại để mọi người dùng cùng một bản.

### 5.3 Giới hạn
- Mô hình dựa trên từ khóa/n-gram nên yếu với tin viết **không dấu** và tin **gõ nhiễu** (`TUYuDU...`); đây là hướng cải tiến (n-gram ký tự, chuẩn hóa không dấu).
- Dữ liệu nhỏ (khoảng 2.100 tin sau làm sạch); số liệu có độ nhiễu vài điểm phần trăm.
- **Dữ liệu chưa ẩn danh hoàn toàn:** kiểm tra bằng Regex đơn giản cho thấy khoảng 183 tin (164 trong train, 19 trong test) còn mã OTP dạng số, ví dụ `Ma OTP xac thuc GD la 066595` (số lượng có thể lệch đôi chút). TV1 nên thay bằng `[NUMBER]` trước khi đưa lên GitHub hoặc nộp; sau khi đổi phải chạy lại mô hình vì mô hình đã học các thẻ này.

## 6. Cách chạy
```bash
pip install -r requirements.txt
python train_ml.py --train data/raw/train_goc.csv --test data/raw/test_goc.csv --regex_set vi
```
Cột `message` (hoặc `text`) và `label` (0/1 hoặc ham/spam) hoặc `y`. Mất khoảng 1 phút. Tùy chọn: `--split random` (val ngẫu nhiên), `--out thu_muc` (xuất ra thư mục khác).
Trên Colab: mở `chay_tren_colab.ipynb`.

## 7. Nạp mô hình (cho TV4)
```python
import json, joblib   # features_regex.py phải nằm cùng thư mục với đoạn code này
model = joblib.load("model/model.joblib")
nguong = json.load(open("model/config.json"))["nguong"]
s = model.predict_proba(["Chuc mung ban trung thuong, bam link nhan qua"])[:, 1]
nhan = ["spam" if x >= nguong else "ham" for x in s]
```
Đầu vào là list chuỗi (pipeline đã gồm vectorizer + Regex + bộ phân loại). **`model.joblib` là Pipeline scikit-learn thuần: API không cần `imbalanced-learn`, `matplotlib`, `seaborn`.** Cài bằng `pip install -r requirements-deploy.txt` (có `scikit-learn==1.9.1`, đúng phiên bản dùng khi huấn luyện; lệch phiên bản có thể gây lỗi hoặc cảnh báo). Chỉ nạp file từ nguồn tin cậy (pickle). Tin nhắn nên ẩn danh giống dữ liệu train (số thành `[NUMBER]`, tiền thành `[MONEY]`...) vì mô hình đã học các thẻ này.

## 8. Bàn giao cho TV3 (học sâu)
| Cần | Lấy ở đâu |
|---|---|
| Dữ liệu chia sẵn, **không chia lại** | `data/train.csv`, `val.csv`, `test.csv` (cột id,text,y) |
| Cách tính điểm | `evaluate.py` (`evaluate(y_true, y_pred, score)`) |
| Mốc ML | `results/ket_qua_ml.csv`, `ket_qua_cuoi.json` |
| Dự đoán ML từng dòng | `results/pred_ml_test.csv` |

- Val cho EarlyStopping/chọn tham số; test chỉ chấm một lần. Dùng `class_weight` (ham:spam khoảng 2,6:1).
- Xuất `results/pred_dl_test.csv` cùng định dạng (`id,text,y_true,y_pred,score`).
- **Mốc so sánh: F1 spam test ≈ 0,85** (mô hình cuối 0,848; TF-IDF + SVM 0,846). Báo cáo thêm F1 theo "mẫu câu mới" (0,765) vì đó là khả năng tổng quát hóa thật.
- Khi dùng `TextVectorization`, nên đặt `standardize="lower"` để giữ ký tự đặc biệt (mặc định xóa dấu câu, trong khi ký tự lạ là dấu hiệu của tin spam gõ nhiễu). Khoảng 47% tin ham và 67% tin spam trong train có dấu tiếng Việt; phần còn lại viết không dấu, nên cần dùng cùng cách chuẩn hóa ở train và test.

## 9. Checklist
- [x] `train/val/test.csv` cột `id,text,y`; không trùng tin giữa các tập; val không có mẫu câu trùng train
- [x] `evaluate.py` chạy độc lập
- [x] 8 tổ hợp, so sánh 5 cách xử lý mất cân bằng, ghép Regex, chỉnh ngưỡng, CV 5-fold
- [x] `pred_ml_test.csv` đủ 566 dòng khớp `test.csv` theo `id`
- [x] `model.joblib` nạp lại được trong tiến trình mới
- [ ] Nhóm rà lại nhãn các tin ở mục 5.2
- [ ] TV1 quét và ẩn danh các mã OTP còn sót (mục 5.3), rồi chạy lại `train_ml.py`
