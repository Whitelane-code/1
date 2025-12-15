from __future__ import annotations

import argparse
import os
from typing import Optional, Dict, Any

import pandas as pd

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from utils import (
    load_config, ensure_dir, infer_columns, build_preprocessor,
    save_joblib, set_seed
)


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到数据文件：{path}\n"
            "请把清洗后的 CSV 放到 data/processed/，或修改 configs/config.yaml 的路径。"
        )
    return pd.read_csv(path)


def main(config_path: str) -> None:
    cfg: Dict[str, Any] = load_config(config_path)

    seed = int(cfg["seed"])
    set_seed(seed)

    train_path = cfg["data"]["train_path"]
    target_col = cfg["data"]["target_col"]
    id_col = cfg["data"].get("id_col", None)

    categorical_cols = cfg["data"].get("categorical_cols", []) or []
    numerical_cols = cfg["data"].get("numerical_cols", []) or []

    df = load_dataset(train_path)

    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col], errors="ignore")
    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col], errors="ignore")

    # 若未显式指定，就自动推断数值列/类别列
    if len(categorical_cols) == 0 and len(numerical_cols) == 0:
        categorical_cols, numerical_cols = infer_columns(df, target_col=target_col, id_col=id_col)

    preprocessor = build_preprocessor(categorical_cols, numerical_cols)

    test_size = float(cfg["train"]["test_size"])
    stratify = y if bool(cfg["train"].get("stratify", True)) else None

    Xtr, Xva, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    Xtr_t = preprocessor.fit_transform(Xtr)
    Xva_t = preprocessor.transform(Xva)

    params = cfg["model"]["params"]
    model = LGBMClassifier(**params)

    # ========================
    model.fit(
        Xtr_t, y_train,
        eval_set=[(Xva_t, y_val)],
        eval_metric="auc",
        callbacks=[
            early_stopping(stopping_rounds=int(cfg["early_stopping"]["stopping_rounds"]), verbose=False),
            log_evaluation(period=int(cfg["early_stopping"]["log_period"])),
        ],
    )
    # =========================

    p_val = model.predict_proba(Xva_t)[:, 1]
    auc_val = roc_auc_score(y_val, p_val)
    print(f"[OK] Validation AUC = {auc_val:.4f}")

    ensure_dir("outputs/models")
    save_joblib(preprocessor, "outputs/models/preprocessor.joblib")
    save_joblib(model, "outputs/models/lgbm_model.joblib")

    ensure_dir("outputs/metrics")
    with open("outputs/metrics/val_auc.txt", "w", encoding="utf-8") as f:
        f.write(f"{auc_val:.6f}\n")

    with open("outputs/metrics/columns_used.txt", "w", encoding="utf-8") as f:
        f.write("categorical_cols:\n")
        for c in categorical_cols:
            f.write(f"  - {c}\n")
        f.write("numerical_cols:\n")
        for c in numerical_cols:
            f.write(f"  - {c}\n")

    print("[DONE] 模型与预处理器已保存到 outputs/models/；指标保存到 outputs/metrics/。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
