"""Feature pipeline shared by the experiments and the submission script.

`build_features` builds ONE feature matrix for the training and the test clients
together, so that both get exactly the same columns, dtypes and category sets.
Client-level aggregates of the large tables are cached on disk, because computing
them takes minutes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bureau import build_bureau_features
from credit_card import build_credit_card_features
from installments import build_installments_features
from pos_cash import build_pos_cash_features
from previous import build_previous_features

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

RATIO_COLS = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_ANNUITY_RATIO",
    "GOODS_PRICE_CREDIT_RATIO",
    "EMPLOYED_BIRTH_RATIO",
]

# (cache name, builder, count columns that are 0 for clients without any records)
AGGREGATES: list[tuple[str, Callable[[Path], pd.DataFrame], list[str]]] = [
    ("bureau", build_bureau_features, ["BUREAU_COUNT"]),
    ("previous", build_previous_features, ["PREV_COUNT"]),
    ("installments", build_installments_features, ["INST_COUNT"]),
    ("pos_cash", build_pos_cash_features, ["POS_COUNT", "POS_LOANS"]),
    ("credit_card", build_credit_card_features, ["CC_COUNT", "CC_CARDS"]),
]


def add_ratio_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features; division by zero becomes NaN instead of inf."""
    out = data.copy()
    out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    out["ANNUITY_INCOME_RATIO"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    out["CREDIT_ANNUITY_RATIO"] = out["AMT_CREDIT"] / out["AMT_ANNUITY"]
    out["GOODS_PRICE_CREDIT_RATIO"] = out["AMT_GOODS_PRICE"] / out["AMT_CREDIT"]
    out["EMPLOYED_BIRTH_RATIO"] = out["DAYS_EMPLOYED"] / out["DAYS_BIRTH"]
    out[RATIO_COLS] = out[RATIO_COLS].replace([np.inf, -np.inf], np.nan)
    return out


def prepare_application(train: pd.DataFrame, test: pd.DataFrame):
    """Clean the main tables together; returns (X, y, ids, n_train).

    Train and test are concatenated first, so that text columns get the same
    category set in both parts.
    """
    y = train[TARGET].copy()
    n_train = len(train)

    full = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
    # 365243 is a placeholder, not a real number of days
    full["DAYS_EMPLOYED"] = full["DAYS_EMPLOYED"].replace(365243, np.nan)

    text_cols = [c for c in full.columns if not pd.api.types.is_numeric_dtype(full[c])]
    for col in text_cols:
        full[col] = full[col].astype("category")

    ids = full[ID_COL]
    X = add_ratio_features(full.drop(columns=[ID_COL]))
    return X, y, ids, n_train


def _cached(name: str, builder, data_dir: Path, cache_dir: Path | None) -> pd.DataFrame:
    """Run `builder(data_dir)` or load its result from the on-disk cache."""
    if cache_dir is not None:
        path = Path(cache_dir) / f"{name}.pkl"
        if path.exists():
            print(f"{name}: loaded from cache")
            return pd.read_pickle(path)
    print(f"{name}: building")
    feats = builder(data_dir)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        feats.to_pickle(path)
    return feats


def build_features(data_dir, cache_dir=None):
    """Build the full feature matrix.

    Returns (X_train, y, X_test, ids_test). Columns are the same, in the same order,
    as in the experiments of `baseline.py`.
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "application_train.csv")
    test = pd.read_csv(data_dir / "application_test.csv")

    X, y, ids, n_train = prepare_application(train, test)
    n_total = len(X)

    parts = [X]
    for name, builder, count_cols in AGGREGATES:
        feats = _cached(name, builder, data_dir, cache_dir)
        aligned = feats.reindex(ids.to_numpy())  # clients without records get NaN
        aligned.index = X.index
        aligned[count_cols] = aligned[count_cols].fillna(0)
        parts.append(aligned)
        del feats

    X_full = pd.concat(parts, axis=1)
    assert len(X_full) == n_total, "a join changed the number of rows"

    X_train = X_full.iloc[:n_train]
    X_test = X_full.iloc[n_train:]
    ids_test = ids.iloc[n_train:].reset_index(drop=True)
    return X_train, y, X_test, ids_test