"""
TV2 - HỌC MÁY TRUYỀN THỐNG cho đề tài 18 (phát hiện SMS spam).

Chạy (dữ liệu tiếng Việt của nhóm, cột message/text và label/y):
    python train_ml.py --train data/raw/train_goc.csv --test data/raw/test_goc.csv --regex_set vi
Tùy chọn:
    --split random     tách validation ngẫu nhiên (mặc định: theo nhóm mẫu câu)
    --out THU_MUC      xuất data/, results/, model/ ra thư mục khác
    --regex_set en     9 đặc trưng Regex (dùng cho bộ UCI tiếng Anh; mặc định)
Bộ UCI tiếng Anh (bản tham khảo): python train_ml.py --raw tham_khao_UCI_tieng_Anh/data/raw/sms.tsv --out tham_khao_UCI_tieng_Anh

Quy ước:
  * Nhãn y: 1 = spam, 0 = ham. Mọi chỉ số tính cho LỚP SPAM.
  * Chọn mô hình/tham số/ngưỡng CHỈ bằng tập validation (hoặc CV trên train).
    Tập test chỉ dùng để chấm điểm, không dùng để chọn.
"""
import argparse, csv, json, os, re, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

import re
from sklearn.model_selection import (train_test_split, StratifiedKFold, StratifiedGroupKFold,
                                     GridSearchCV, cross_validate)
from sklearn.base import clone
from sklearn.pipeline import FeatureUnion, Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from evaluate import evaluate
from features_regex import regex_features, regex_features_vi, REGEX_COLS, REGEX_COLS_VI

warnings.filterwarnings("ignore")
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
D = R = F = M = None
SPLIT_MODE = "group"


def template_key(text):
    """Khóa 'mẫu câu': bỏ thẻ ẩn danh, số, khoảng trắng, lấy 50 ký tự đầu.
    Tin cùng khuôn (OTP, nhà mạng, casino...) có cùng khóa -> phải nằm cùng một phía khi chia."""
    t = re.sub(r"\[[A-Za-z_]+\]", "[T]", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", t))[:50]


def cv_obj(n):
    if SPLIT_MODE == "group":
        return StratifiedGroupKFold(n, shuffle=True, random_state=SEED), True
    return StratifiedKFold(n, shuffle=True, random_state=SEED), False

REGEX_FN, REGEX_NAMES = None, None


def init_dirs(base):
    global D, R, F, M
    D, R, F, M = (os.path.join(base, p) for p in ("data", "results", "results/figures", "model"))
    for p in (D, R, F, M):
        os.makedirs(p, exist_ok=True)


TEN_MODEL = {"NaiveBayes": "Naive Bayes", "LogReg": "Hồi quy Logistic",
             "SVM": "SVM (LinearSVC + hiệu chỉnh xác suất)", "RandomForest": "Rừng ngẫu nhiên"}
TEN_VEC = {"BoW": "Túi từ (BoW)", "TFIDF": "TF-IDF"}


# ----------------------------------------------------------------- 1. DỮ LIỆU
def doc_file(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tc = next((cols[c] for c in ("text", "message", "v2", "sms") if c in cols), None)
    if tc is None:
        raise ValueError(f"{path}: không thấy cột văn bản (text/message/v2). Cột có: {list(df.columns)}")
    lc = cols["y"] if "y" in cols else next((cols[c] for c in ("label", "v1", "category") if c in cols), None)
    if lc is None:
        raise ValueError(f"{path}: không thấy cột nhãn (y/label/v1)")
    num = pd.to_numeric(df[lc], errors="coerce")
    if num.notna().all():                       # nhãn dạng số: 1 = spam, 0 = ham
        y = num.astype(int)
    else:                                       # nhãn dạng chữ: spam / ham
        y = (df[lc].astype(str).str.lower().str.strip() == "spam").astype(int)
    assert set(y.unique()) <= {0, 1}, f"{path}: nhãn phải là 0/1 hoặc ham/spam"
    return pd.DataFrame({"text": df[tc].astype(str), "y": y.values})


def chuan_bi_du_lieu(args):
    if args.train and args.test:
        tr_full, te = doc_file(args.train), doc_file(args.test)
        nguon = f"file của nhóm: {os.path.basename(args.train)}, {os.path.basename(args.test)}"
    else:
        raw = args.raw or os.path.join(HERE, "tham_khao_UCI_tieng_Anh", "data", "raw", "sms.tsv")
        df = pd.read_csv(raw, sep="\t", names=["label", "text"], encoding="utf-8", quoting=csv.QUOTE_NONE)
        df["y"] = (df["label"] == "spam").astype(int)
        n0 = len(df)
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
        print(f"Bộ UCI: {n0} tin, bỏ {n0 - len(df)} tin trùng -> còn {len(df)}")
        tr_full, te = train_test_split(df[["text", "y"]], test_size=0.2,
                                       stratify=df["y"], random_state=SEED)
        nguon = "UCI SMS Spam Collection (tự chia 80/20, stratify, seed 42)"

    # bỏ tin trùng trong từng tập và giữa các tập (chống rò rỉ)
    tr_full = tr_full.drop_duplicates(subset="text")
    te = te.drop_duplicates(subset="text")
    trung = set(tr_full["text"]) & set(te["text"])
    if trung:
        print(f"CẢNH BÁO: {len(trung)} tin có ở cả train và test -> loại khỏi train để tránh rò rỉ")
        tr_full = tr_full[~tr_full["text"].isin(trung)]

    tr_full = tr_full.reset_index(drop=True)
    te = te.reset_index(drop=True)
    if SPLIT_MODE == "group":      # val gồm các mẫu câu KHÔNG có trong train (giống tình huống thật/test)
        g = tr_full["text"].map(template_key)
        i_tr, i_va = next(StratifiedGroupKFold(8, shuffle=True, random_state=SEED).split(tr_full, tr_full["y"], g))
        tr, va = tr_full.iloc[i_tr], tr_full.iloc[i_va]
    else:
        tr, va = train_test_split(tr_full, test_size=0.125, stratify=tr_full["y"], random_state=SEED)
    tr = tr.reset_index(drop=True); va = va.reset_index(drop=True)
    for d in (tr, va, te):
        pass
    tr.insert(0, "id", range(len(tr)))
    va.insert(0, "id", range(len(va)))
    te.insert(0, "id", range(len(te)))
    tr.to_csv(f"{D}/train.csv", index=False)
    va.to_csv(f"{D}/val.csv", index=False)
    te.to_csv(f"{D}/test.csv", index=False)

    assert not set(tr["text"]) & set(te["text"]) and not set(va["text"]) & set(te["text"])
    assert not set(tr["text"]) & set(va["text"])
    info = {"nguon": nguon, "cach_chia_val": "theo nhom mau cau" if SPLIT_MODE == "group" else "ngau nhien",
            "val_co_mau_cau_trong_train": round(float(va["text"].map(template_key).isin(set(tr["text"].map(template_key))).mean()), 3),
            "test_co_mau_cau_trong_train": round(float(te["text"].map(template_key).isin(set(tr["text"].map(template_key))).mean()), 3),
            "train": len(tr), "val": len(va), "test": len(te),
            "ti_le_spam_train": round(tr["y"].mean(), 4),
            "ti_le_spam_val": round(va["y"].mean(), 4),
            "ti_le_spam_test": round(te["y"].mean(), 4)}
    print(json.dumps(info, ensure_ascii=False, indent=1))
    return tr, va, te, info


# ----------------------------------------------------------------- 2. EDA NHẸ
def eda(tr):
    plt.figure(figsize=(5, 4))
    c = tr["y"].map({0: "Tin bình thường", 1: "Tin rác (spam)"}).value_counts()
    ax = c.plot(kind="bar", color=["#4c9a6a", "#d9534f"])
    for i, v in enumerate(c.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    plt.title("Phân bố lớp trong tập huấn luyện"); plt.ylabel("Số tin nhắn"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(f"{F}/phan_bo_lop.png", dpi=200); plt.close()

    L = tr["text"].str.len()
    plt.figure(figsize=(6, 4))
    plt.hist(L[tr["y"] == 0], bins=40, alpha=.6, label="Bình thường", color="#4c9a6a", density=True)
    plt.hist(L[tr["y"] == 1], bins=40, alpha=.6, label="Spam", color="#d9534f", density=True)
    plt.xlabel("Độ dài tin nhắn (ký tự)"); plt.ylabel("Mật độ"); plt.legend()
    plt.title("Độ dài tin nhắn: spam và bình thường")
    plt.tight_layout(); plt.savefig(f"{F}/do_dai_tin.png", dpi=200); plt.close()


# (đặc trưng Regex nằm trong features_regex.py để pipeline lưu ra file nạp lại được)


# --------------------------------------------------------- 4. TẠO PIPELINE
def make_vec(kind, regex=False):
    base = CountVectorizer() if kind == "BoW" else TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    if not regex:
        return base
    rg = SkPipeline([("f", FunctionTransformer(REGEX_FN)), ("s", MinMaxScaler())])
    return FeatureUnion([("txt", base), ("rgx", rg)])


def make_clf(name, cw=None):
    if name == "NaiveBayes":
        return MultinomialNB()
    if name == "LogReg":
        return LogisticRegression(max_iter=2000, class_weight=cw)
    if name == "SVM":
        return CalibratedClassifierCV(LinearSVC(class_weight=cw), cv=5)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=SEED, n_jobs=-1)
    raise ValueError(name)


def make_pipe(vec, model, sampler=None, cw=None, regex=False):
    steps = [("vec", make_vec(vec, regex))]
    if sampler is not None:
        steps.append(("samp", sampler))
    steps.append(("clf", make_clf(model, cw)))
    return Pipeline(steps)


def score_of(pipe, X):
    return pipe.predict_proba(X)[:, 1]


def eval_vs(pipe, va, te):
    sv, st = score_of(pipe, va["text"]), score_of(pipe, te["text"])
    v = evaluate(va["y"], (sv >= .5).astype(int), sv)
    t = evaluate(te["y"], (st >= .5).astype(int), st)
    return v, t


MODELS = ["NaiveBayes", "LogReg", "SVM", "RandomForest"]


# ---------------------------------------------- 5. THÍ NGHIỆM A: 8 TỔ HỢP
def thi_nghiem_a(tr, va, te):
    print("\n=== A. BoW/TF-IDF x 4 mô hình (class_weight=balanced nếu có) ===")
    rows = []
    for vec in ("BoW", "TFIDF"):
        for m in MODELS:
            cw = None if m == "NaiveBayes" else "balanced"
            pipe = make_pipe(vec, m, cw=cw)
            t0 = time.time(); pipe.fit(tr["text"], tr["y"]); tfit = time.time() - t0
            t0 = time.time(); pipe.predict(te["text"]); tinf = time.time() - t0
            v, t = eval_vs(pipe, va, te)
            rows.append({"Vector": vec, "Model": m, "Val_P": v["Precision"], "Val_R": v["Recall"],
                         "Val_F1": v["F1"], "Test_P": t["Precision"], "Test_R": t["Recall"],
                         "Test_F1": t["F1"], "PR_AUC": t["PR_AUC"], "FP": t["FP"], "FN": t["FN"],
                         "Fit_s": round(tfit, 2), "Infer_s": round(tinf, 4)})
            print(f"{vec:6s} {m:13s} valF1={v['F1']:.3f} testP={t['Precision']:.3f} "
                  f"testR={t['Recall']:.3f} testF1={t['F1']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------- 6. THÍ NGHIỆM B: XỬ LÝ MẤT CÂN BẰNG
def thi_nghiem_b(tr, va, te):
    print("\n=== B. So sánh cách xử lý mất cân bằng (TF-IDF) ===")
    cach = {"Không xử lý (mốc)": dict(),
            "class_weight=balanced": dict(cw="balanced"),
            "Nhân bản ngẫu nhiên": dict(sampler=lambda: RandomOverSampler(random_state=SEED)),
            "SMOTE": dict(sampler=lambda: SMOTE(random_state=SEED)),
            "Giảm bớt lớp ham": dict(sampler=lambda: RandomUnderSampler(random_state=SEED))}
    rows = []
    for m in MODELS:
        for ten, cfg in cach.items():
            if ten == "class_weight=balanced" and m == "NaiveBayes":
                continue  # MultinomialNB không có class_weight
            smp = cfg["sampler"]() if "sampler" in cfg else None
            pipe = make_pipe("TFIDF", m, sampler=smp, cw=cfg.get("cw"))
            pipe.fit(tr["text"], tr["y"])
            v, t = eval_vs(pipe, va, te)
            rows.append({"Model": m, "Cach_xu_ly": ten, "Val_P": v["Precision"], "Val_R": v["Recall"],
                         "Val_F1": v["F1"], "Test_P": t["Precision"], "Test_R": t["Recall"],
                         "Test_F1": t["F1"], "FP": t["FP"], "FN": t["FN"]})
            print(f"{m:13s} {ten:24s} valF1={v['F1']:.3f} testP={t['Precision']:.3f} testR={t['Recall']:.3f}")
    return pd.DataFrame(rows), cach


# ------------------------------------ 7. THÍ NGHIỆM C: GHÉP ĐẶC TRƯNG REGEX
def thi_nghiem_c(tr, va, te):
    print("\n=== C. TF-IDF + đặc trưng Regex (class_weight=balanced) ===")
    rows = []
    for m in MODELS:
        cw = None if m == "NaiveBayes" else "balanced"
        for regex in (False, True):
            pipe = make_pipe("TFIDF", m, cw=cw, regex=regex)
            pipe.fit(tr["text"], tr["y"])
            v, t = eval_vs(pipe, va, te)
            rows.append({"Model": m, "Dac_trung": "TF-IDF + Regex" if regex else "TF-IDF",
                         "Val_F1": v["F1"], "Test_P": t["Precision"], "Test_R": t["Recall"],
                         "Test_F1": t["F1"], "FP": t["FP"], "FN": t["FN"]})
            print(f"{m:13s} {'+Regex' if regex else '      '} valF1={v['F1']:.3f} testF1={t['F1']:.3f}")
    return pd.DataFrame(rows)


# --------------------------------------------- 8. CHỌN MÔ HÌNH + GRIDSEARCH
GRID = {
    "NaiveBayes": {"vec__ngram_range": [(1, 1), (1, 2)], "clf__alpha": [0.05, 0.1, 0.5, 1.0]},
    "LogReg": {"vec__ngram_range": [(1, 1), (1, 2)], "clf__C": [1, 5, 10, 50]},
    "SVM": {"vec__ngram_range": [(1, 1), (1, 2)], "clf__estimator__C": [0.1, 0.5, 1, 2]},
    "RandomForest": {"vec__ngram_range": [(1, 1), (1, 2)], "clf__max_depth": [None, 60]},
}


def chon_va_tinh_chinh(bang_b, bang_c, cach, tr, va, te):
    print("\n=== D. Chọn cấu hình tốt nhất theo F1 spam trên VALIDATION ===")
    cb = bang_b.assign(Regex=False)[["Model", "Cach_xu_ly", "Regex", "Val_F1", "Val_P"]]
    cc = bang_c[bang_c["Dac_trung"] == "TF-IDF + Regex"].assign(
        Cach_xu_ly=lambda d: np.where(d["Model"] == "NaiveBayes", "Không xử lý (mốc)", "class_weight=balanced"),
        Regex=True, Val_P=np.nan)[["Model", "Cach_xu_ly", "Regex", "Val_F1", "Val_P"]]
    ung_vien = pd.concat([cb, cc], ignore_index=True).sort_values(
        ["Val_F1", "Val_P"], ascending=False, na_position="last").reset_index(drop=True)
    ung_vien.to_csv(f"{R}/ung_vien_xep_hang_val.csv", index=False)
    top = ung_vien.iloc[0]
    m, ten, rg = top["Model"], top["Cach_xu_ly"], bool(top["Regex"])
    print(f"Cấu hình chọn: {m} + TF-IDF{' + Regex' if rg else ''} + {ten} (Val_F1={top['Val_F1']})")
    cfg = cach[ten]

    def tao():
        return make_pipe("TFIDF", m, sampler=(cfg["sampler"]() if "sampler" in cfg else None),
                         cw=cfg.get("cw"), regex=rg)

    grid = {("vec__txt__" + k[5:] if rg and k.startswith("vec__") else k): v for k, v in GRID[m].items()}
    cv3, usg = cv_obj(3)
    gs = GridSearchCV(tao(), grid, scoring="f1", cv=cv3, n_jobs=1, refit=True)
    gs.fit(tr["text"], tr["y"], **({"groups": tr["text"].map(template_key)} if usg else {}))
    print("Tham số tốt nhất (CV 3-fold trên train):", gs.best_params_, "| F1 CV =", round(gs.best_score_, 4))
    base = tao().fit(tr["text"], tr["y"])
    v0, _ = eval_vs(base, va, te)
    v1, _ = eval_vs(gs.best_estimator_, va, te)
    print(f"Val F1 mặc định: {v0['F1']} | sau tinh chỉnh: {v1['F1']}")
    best = gs.best_estimator_ if v1["F1"] >= v0["F1"] else base
    if best is base:
        print("Bản tinh chỉnh không tốt hơn trên validation -> giữ cấu hình mặc định")
    return best, {"model": m, "dac_trung": "TF-IDF + Regex" if rg else "TF-IDF", "xu_ly_mat_can_bang": ten,
                  "best_params": {k: str(v) for k, v in gs.best_params_.items()},
                  "cv_f1_train": round(float(gs.best_score_), 4),
                  "val_f1_mac_dinh": v0["F1"], "val_f1_tinh_chinh": v1["F1"],
                  "dung_ban_tinh_chinh": bool(best is not base)}


# --------------------------------------------------------- 9. CHỈNH NGƯỠNG
def chinh_nguong(best, va):
    s = score_of(best, va["text"])
    p, r, th = precision_recall_curve(va["y"], s)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    top = f1.max()
    cand = th[f1 >= top - 1e-9]
    nguong = float(np.median(cand))
    f_05 = evaluate(va["y"], (s >= .5).astype(int))["F1"]
    f_th = evaluate(va["y"], (s >= nguong).astype(int))["F1"]
    dung = (f_th - f_05) >= 0.005          # chỉ đổi ngưỡng khi lợi rõ ràng, tránh học vẹt validation
    print(f"\n=== E. Ngưỡng: val F1 @0.5={f_05} | @{nguong:.3f}={f_th} -> dùng {'ngưỡng mới' if dung else '0.5'}")
    return (nguong if dung else 0.5), {"nguong_toi_uu_val": round(nguong, 4), "val_f1_o_0.5": f_05,
                                       "val_f1_o_nguong_moi": f_th, "nguong_dung": round(nguong if dung else 0.5, 4)}


# ----------------------------------------------------------------- 10. VẼ HÌNH
def ve_hinh(best, te, nguong, bang_a):
    s = score_of(best, te["text"]); pred = (s >= nguong).astype(int)
    cm = confusion_matrix(te["y"], pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bình thường", "Spam"],
                yticklabels=["Bình thường", "Spam"])
    plt.xlabel("Mô hình dự đoán"); plt.ylabel("Nhãn thực tế"); plt.title("Ma trận nhầm lẫn (tập test)")
    plt.tight_layout(); plt.savefig(f"{F}/confusion_matrix.png", dpi=200); plt.close()

    p, r, _ = precision_recall_curve(te["y"], s)
    plt.figure(figsize=(5, 4)); plt.plot(r, p, color="#d9534f")
    plt.xlabel("Recall (độ phủ spam)"); plt.ylabel("Precision"); plt.title("Đường Precision-Recall (test)")
    plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(f"{F}/pr_curve.png", dpi=200); plt.close()

    a = bang_a.copy(); a["ten"] = a["Vector"] + " + " + a["Model"]
    a = a.sort_values("Test_F1")
    plt.figure(figsize=(7, 4.5)); plt.barh(a["ten"], a["Test_F1"], color="#4a7fb5")
    plt.xlim(max(0, a["Test_F1"].min() - .05), 1); plt.xlabel("F1 lớp spam (test)")
    plt.title("So sánh 8 tổ hợp vector hóa x mô hình")
    plt.tight_layout(); plt.savefig(f"{F}/so_sanh_8_to_hop.png", dpi=200); plt.close()


# ---------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train"); ap.add_argument("--test")
    ap.add_argument("--out", default=HERE, help="thư mục chứa data/, results/, model/")
    ap.add_argument("--raw", help="file sms.tsv của UCI (chỉ dùng khi không có --train/--test)")
    ap.add_argument("--regex_set", choices=["en", "vi"], default="en",
                    help="en: 9 đặc trưng Regex (UCI); vi: 13 đặc trưng, thêm thẻ ẩn danh/ký tự nhiễu")
    ap.add_argument("--split", choices=["group", "random"], default="group",
                    help="group: val tách theo nhóm mẫu câu (khuyên dùng); random: tách ngẫu nhiên")
    args = ap.parse_args()
    global REGEX_FN, REGEX_NAMES, SPLIT_MODE
    SPLIT_MODE = args.split
    REGEX_FN, REGEX_NAMES = (regex_features_vi, REGEX_COLS_VI) if args.regex_set == "vi" else (regex_features, REGEX_COLS)
    init_dirs(args.out)
    t_all = time.time()

    tr, va, te, info = chuan_bi_du_lieu(args)
    eda(tr)

    A = thi_nghiem_a(tr, va, te)
    B, cach = thi_nghiem_b(tr, va, te)
    C = thi_nghiem_c(tr, va, te)

    A.to_csv(f"{R}/ket_qua_ml.csv", index=False)
    B.to_csv(f"{R}/so_sanh_mat_can_bang.csv", index=False)
    C.to_csv(f"{R}/ghep_regex.csv", index=False)
    # bản tiếng Việt để đọc/đưa vào báo cáo
    A_vi = A.copy(); A_vi["Model"] = A_vi["Model"].map(TEN_MODEL); A_vi["Vector"] = A_vi["Vector"].map(TEN_VEC)
    A_vi.to_csv(f"{R}/ket_qua_ml_VI.csv", index=False, encoding="utf-8-sig")
    B.to_csv(f"{R}/so_sanh_mat_can_bang_VI.csv", index=False, encoding="utf-8-sig")

    best, chon = chon_va_tinh_chinh(B, C, cach, tr, va, te)
    nguong, ng_info = chinh_nguong(best, va)

    # ---- chấm test MỘT LẦN cho mô hình cuối
    s_te = score_of(best, te["text"]); pred = (s_te >= nguong).astype(int)
    kq = evaluate(te["y"], pred, s_te)
    print("\n=== KẾT QUẢ CUỐI trên TEST (chấm một lần) ===")
    from evaluate import in_tieng_viet
    in_tieng_viet(kq, "Mô hình cuối")

    # ---- CV 5-fold trên train+val (độ ổn định, không đụng test)
    full = pd.concat([tr, va], ignore_index=True)
    cv5, usg = cv_obj(5)
    cv = cross_validate(clone(best), full["text"], full["y"], cv=cv5, scoring=["precision", "recall", "f1"], n_jobs=1,
                        **({"groups": full["text"].map(template_key)} if usg else {}))
    cv_info = {k: f"{cv['test_'+k].mean():.4f} ± {cv['test_'+k].std():.4f}" for k in ("precision", "recall", "f1")}
    print("CV 5-fold (train+val) lớp spam:", cv_info)

    # ---- test tách theo "mẫu câu đã thấy / mới" và danh sách tin nên rà lại nhãn
    kt = set(full["text"].map(template_key))
    seen = te["text"].map(template_key).isin(kt).values
    nhom = {}
    for ten, mk in (("mau_cau_da_thay_trong_train", seen), ("mau_cau_moi", ~seen)):
        if mk.sum() > 0 and te["y"].values[mk].sum() > 0:
            nhom[ten] = {"so_tin": int(mk.sum()), **evaluate(te["y"].values[mk], pred[mk], s_te[mk])}
    print("Test theo nhóm mẫu câu:", {k: (v["so_tin"], v["F1"]) for k, v in nhom.items()})

    # ---- thời gian suy luận
    t0 = time.time(); best.predict_proba(te["text"]); t_batch = time.time() - t0
    t0 = time.time()
    for x in te["text"].iloc[:100]:
        best.predict_proba([x])
    t_one = (time.time() - t0) / 100

    # ---- xuất dự đoán + lỗi
    out = te[["id", "text"]].copy(); out["y_true"] = te["y"].values
    out["y_pred"] = pred; out["score"] = np.round(s_te, 6)
    out.to_csv(f"{R}/pred_ml_test.csv", index=False)
    sv = score_of(best, va["text"])
    ov = va[["id", "text"]].copy(); ov["y_true"] = va["y"].values
    ov["y_pred"] = (sv >= nguong).astype(int); ov["score"] = np.round(sv, 6)
    ov.to_csv(f"{R}/pred_ml_val.csv", index=False)
    err = out[out["y_true"] != out["y_pred"]].copy()
    err["loai_loi"] = np.where(err["y_true"] == 0, "FP: tin tốt bị chặn nhầm", "FN: spam bị lọt")
    err.sort_values("loai_loi").to_csv(f"{R}/tin_bi_doan_sai.csv", index=False, encoding="utf-8-sig")
    # ---- danh sách nhãn cần người rà lại (xếp theo lý do, KHÔNG tự sửa nhãn)
    allx = pd.concat([tr.assign(tap="train"), va.assign(tap="val"), te.assign(tap="test")], ignore_index=True)
    allx["khoa_mau_cau"] = allx["text"].map(template_key)
    nn = allx.groupby("khoa_mau_cau")["y"].nunique()
    mix_keys = set(nn[nn > 1].index)       # cùng khuôn tin nhưng có cả nhãn ham lẫn spam
    mix = allx[allx["khoa_mau_cau"].isin(mix_keys)].sort_values(["khoa_mau_cau", "y"])
    mix = mix[["khoa_mau_cau", "tap", "id", "text", "y"]].rename(columns={"y": "nhan_hien_tai"})
    mix.to_csv(f"{R}/nhan_khong_nhat_quan.csv", index=False, encoding="utf-8-sig")

    KW = re.compile(r"casino|nap dau|nạp đầu|vay|bit\.ly|https?://|\.vip|\.bet|win", re.I)
    rv = out[out["y_true"] != out["y_pred"]].copy()
    rv["khoa_mau_cau"] = rv["text"].map(template_key)
    co_a = rv["khoa_mau_cau"].isin(mix_keys)
    co_b = (rv["y_true"] == 0) & rv["text"].str.contains(KW)
    co_c = ((rv["y_true"] == 0) & (rv["score"] >= 0.8)) | ((rv["y_true"] == 1) & (rv["score"] <= 0.2))
    rv["ly_do"] = [" + ".join(x for x, f in (("A: cùng khuôn tin nhưng có tin nhãn ngược", a_), ("B: nhãn ham nhưng chứa từ khóa casino/vay/link",
                   b_), ("C: mô hình rất tự tin", c_)) if f) for a_, b_, c_ in zip(co_a, co_b, co_c)]
    rv["so_ly_do"] = co_a.astype(int) + co_b.astype(int) + co_c.astype(int)
    rv = rv[rv["so_ly_do"] > 0].sort_values(["so_ly_do", "score"], ascending=[False, False])
    rv["nhan_hien_tai"] = rv["y_true"].map({0: "ham", 1: "spam"})
    rv["mo_hinh_nghi"] = np.where(rv["y_true"] == 0, "spam", "ham")
    rv["nhan_dung_theo_nhom (dien tay)"] = ""
    rv[["id", "text", "nhan_hien_tai", "mo_hinh_nghi", "score", "ly_do", "nhan_dung_theo_nhom (dien tay)"]].to_csv(
        f"{R}/tin_can_ra_soat_nhan.csv", index=False, encoding="utf-8-sig")
    print(f"Tin sai cần rà nhãn: {len(rv)}/{len(err)} | nhóm mẫu câu có nhãn ngược nhau: {len(mix_keys)} ({len(mix)} tin)")

    # ---- thống kê đặc trưng regex theo lớp (cho báo cáo)
    rf = pd.DataFrame(REGEX_FN(tr["text"]), columns=REGEX_NAMES)
    rf["lop"] = tr["y"].map({0: "ham", 1: "spam"}).values
    rf.groupby("lop").mean().round(3).to_csv(f"{R}/thong_ke_regex_theo_lop.csv", encoding="utf-8-sig")

    ve_hinh(best, te, nguong, A)

    # ---- lưu mô hình + cấu hình
    # Xuất Pipeline scikit-learn THUẦN: bỏ bước lấy mẫu (SMOTE...) vì khi dự đoán không dùng,
    # nhờ vậy nơi nạp mô hình (API của TV4) KHÔNG cần cài imbalanced-learn.
    sk_model = SkPipeline([(n, st) for n, st in best.steps if n != "samp"])
    assert np.allclose(sk_model.predict_proba(te["text"]), best.predict_proba(te["text"])), "pipeline xuất ra khác pipeline gốc"
    joblib.dump(sk_model, f"{M}/model.joblib")
    json.dump({"nguong": nguong, "nhan": {"0": "ham", "1": "spam"}, "input": "list[str]",
               "goi": "model.predict_proba(texts)[:,1] >= nguong -> spam"},
              open(f"{M}/config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    final = {"du_lieu": info, "cau_hinh_chon": chon, "nguong": ng_info,
             "test_cuoi": kq, "test_theo_nhom_mau_cau": nhom, "cv5_train_val": cv_info,
             "thoi_gian_suy_luan": {"batch_%d_tin_s" % len(te): round(t_batch, 4),
                                    "1_tin_ms": round(t_one * 1000, 2)},
             "phien_ban": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                           "pandas": pd.__version__, "numpy": np.__version__}}
    json.dump(final, open(f"{R}/ket_qua_cuoi.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nXong trong {time.time() - t_all:.0f} giây. Kết quả ở thư mục results/, model/")


if __name__ == "__main__":
    main()
