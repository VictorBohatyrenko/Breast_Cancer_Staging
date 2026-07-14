"""Computes node-level (4-class: negative/itc/micro/macro) QWK for CAMELYON17
patients 100-199, comparing our submission.csv against ground-truth labels."""
import re

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

STAGE2ID = {"negative": 0, "itc": 1, "micro": 2, "macro": 3}

GT_XLSX = "/data/bohatyrenko1/experiments/Camelyon+(4-classes).xlsx"          # or wherever it lives on the server
SUBMISSION_CSV = "/home/root_server/bohatyrenko/experiments/submission.csv"


def load_ground_truth(path: str) -> pd.DataFrame:
    gt = pd.read_excel(path, sheet_name="Sheet1")
    gt = gt[gt["slide"].str.startswith("patient")].copy()
    gt["patient_id"] = gt["slide"].str.extract(r"patient_(\d+)")[0].astype(int)
    gt = gt[(gt["patient_id"] >= 100) & (gt["patient_id"] <= 199)].copy()
    gt["label"] = gt["label"].str.lower().str.strip()
    gt = gt.rename(columns={"slide": "slide_uid"})[["slide_uid", "label"]]
    return gt


def load_predictions(path: str) -> pd.DataFrame:
    sub = pd.read_csv(path)
    sub = sub[sub["patient"].str.endswith(".tif")].copy()  # drop pN (.zip) rows, keep per-node rows
    sub["slide_uid"] = sub["patient"].str.replace(r"\.tif$", "", regex=True)
    sub["stage"] = sub["stage"].str.lower().str.strip()
    return sub[["slide_uid", "stage"]]


def main() -> None:
    gt = load_ground_truth(GT_XLSX)
    pred = load_predictions(SUBMISSION_CSV)

    merged = gt.merge(pred, on="slide_uid", how="inner", suffixes=("_true", "_pred"))
    missing = set(gt["slide_uid"]) - set(merged["slide_uid"])
    if missing:
        print(f"⚠ {len(missing)} slides in ground truth had no matching prediction: {sorted(missing)[:10]}...")

    y_true = merged["label"].map(STAGE2ID).to_numpy()
    y_pred = merged["stage"].map(STAGE2ID).to_numpy()

    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    print(f"Слайдів у порівнянні: {len(merged)}")
    print(f"\nQWK (patients 100-199, node-level 4-class): {qwk:.4f}\n")
    print("Confusion matrix (rows=true, cols=pred) [negative, itc, micro, macro]:")
    print(pd.DataFrame(cm, index=["negative", "itc", "micro", "macro"],
                        columns=["negative", "itc", "micro", "macro"]))


if __name__ == "__main__":
    main()