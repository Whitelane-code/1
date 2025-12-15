from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def infer_columns(df: pd.DataFrame, target_col: str, id_col: Optional[str] = None) -> Tuple[List[str], List[str]]:
    cols = df.columns.tolist()
    drop_cols = {target_col}
    if id_col and id_col in cols:
        drop_cols.add(id_col)

    feature_df = df.drop(columns=list(drop_cols), errors="ignore")
    cat_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in feature_df.columns if c not in cat_cols]
    return cat_cols, num_cols


def build_preprocessor(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numerical_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def save_joblib(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)


def load_joblib(path: str) -> Any:
    return joblib.load(path)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
