"""Đặc trưng Regex (số điện thoại, URL, tiền, ký tự đặc biệt, chữ in hoa, độ dài).
FILE NÀY PHẢI ĐI CÙNG model.joblib: pipeline đã lưu gọi regex_features từ module này.
Khi nạp mô hình ở nơi khác (API của TV4), đặt features_regex.py cùng thư mục với code nạp."""
import re
import numpy as np

PHONE = re.compile(r"(\+?\d[\d\s\-]{8,}\d)")
URL = re.compile(r"(https?://\S+|www\.\S+|\b\w+\.(com|net|org|vn|co\.uk)\b)", re.I)
MONEY = re.compile(r"[£$€]|\b\d+\s?(p|usd|vnd|đ|k)\b", re.I)
SPECIAL = re.compile(r"[^\w\s]")

REGEX_COLS = ["log_do_dai", "log_so_tu", "co_sdt", "co_link", "co_tien",
              "log_ky_tu_dac_biet", "log_so_chu_so", "ti_le_chu_hoa", "log_dau_cham_than"]


def regex_features(texts):
    rows = []
    for t in texts:
        letters = [c for c in t if c.isalpha()]
        up = sum(c.isupper() for c in letters)
        rows.append([np.log1p(len(t)), np.log1p(len(t.split())),
                     int(bool(PHONE.search(t))), int(bool(URL.search(t))),
                     int(bool(MONEY.search(t))), np.log1p(len(SPECIAL.findall(t))),
                     np.log1p(sum(c.isdigit() for c in t)),
                     up / len(letters) if letters else 0.0, np.log1p(t.count("!"))])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------------------
# Bộ đặc trưng mở rộng cho dữ liệu tiếng Việt (13 cột = 9 cột trên + 4 cột dưới)
TAG = re.compile(r"\[[A-Za-z_]+\]")                 # thẻ ẩn danh: [MONEY], [NUMBER], [TIME]...
ACCENT = re.compile(r"[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", re.I)

REGEX_COLS_VI = REGEX_COLS + ["log_so_the_an_danh", "co_the_money", "ti_le_ky_tu_dac_biet", "co_dau_tieng_viet"]


def regex_features_vi(texts):
    base = regex_features(texts)
    extra = []
    for t in texts:
        n = max(len(t), 1)
        extra.append([np.log1p(len(TAG.findall(t))),
                      int("[MONEY]" in t.upper()),
                      len(SPECIAL.findall(t)) / n,
                      int(bool(ACCENT.search(t)))])
    return np.hstack([base, np.array(extra, dtype=float)])
