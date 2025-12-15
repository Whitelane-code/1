from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import shap

from utils import load_config, load_joblib, ensure_dir


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

    X = df.drop(columns=[target_col], errors="ignore")
    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col], errors="ignore")

    preprocessor = load_joblib("outputs/models/preprocessor.joblib")
    model = load_joblib("outputs/models/lgbm_model.joblib")

    Xt = preprocessor.transform(X)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(Xt.shape[1])]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xt)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        sv = shap_values[1]
    else:
        sv = shap_values

    ensure_dir("outputs/figures")

    plt.figure()
    shap.summary_plot(sv, Xt, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_summary.png", dpi=200)
    plt.close()

    print("[DONE] 已保存 outputs/figures/shap_summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
