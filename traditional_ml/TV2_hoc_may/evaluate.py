"""Hàm tính điểm DÙNG CHUNG cho ML (TV2) và DL (TV3).
Mọi chỉ số đều tính cho LỚP SPAM (nhãn 1)."""
import numpy as np
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             roc_auc_score, average_precision_score)


def evaluate(y_true, y_pred, score=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    out = {"Precision": round(float(p), 4),
           "Recall": round(float(r), 4),
           "F1": round(float(f1), 4)}
    if score is not None:
        out["ROC_AUC"] = round(float(roc_auc_score(y_true, score)), 4)
        out["PR_AUC"] = round(float(average_precision_score(y_true, score)), 4)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["TN"], out["FP"], out["FN"], out["TP"] = int(tn), int(fp), int(fn), int(tp)
    return out


def in_tieng_viet(kq, ten=""):
    print(f"[{ten}] Precision spam: {kq['Precision']} | Recall spam: {kq['Recall']} | F1 spam: {kq['F1']}")
    print(f"   Chặn nhầm tin tốt (FP): {kq['FP']} | Bỏ sót spam (FN): {kq['FN']} | Bắt đúng spam (TP): {kq['TP']}")
