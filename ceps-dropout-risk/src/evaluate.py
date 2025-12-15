from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, brier_score_loss
)

from utils import load_config, load_joblib, ensure_dir, save_json


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到数据文件：{path}")
    return pd.read_csv(path)


def main(config_path: str) -> None:
    cfg: Dict[str, Any] = load_config(config_path)

    test_path = cfg["data"]["test_path"]
    target_col = cfg["data"]["target_col"]
    id_col = cfg["data"].get("id_col", None)

    df = load_dataset(test_path)

    y_true = df[target_col].astype(int).values
    X = df.drop(columns=[target_col], errors="ignore")
    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col], errors="ignore")

    preprocessor = load_joblib("outputs/models/preprocessor.joblib")
    model = load_joblib("outputs/models/lgbm_model.joblib")

    Xt = preprocessor.transform(X)
    prob = model.predict_proba(Xt)[:, 1]

    auc = roc_auc_score(y_true, prob)
    pr_auc = average_precision_score(y_true, prob)

    y_pred = (prob >= 0.5).astype(int)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, prob)

    metrics = {
        "ROC_AUC": float(auc),
        "PR_AUC": float(pr_auc),
        "Recall@0.5": float(rec),
        "F1@0.5": float(f1),
        "Brier": float(brier),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }

    ensure_dir("outputs/metrics")
    save_json(metrics, "outputs/metrics/test_metrics.json")

    print("[DONE] 测试集指标：")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
