"""Client-level aggregates from `bureau`.

One row per credit reported by other institutions (~1.7M rows). The table carries
SK_ID_CURR directly, so one aggregation level is enough:
    credit -> client (SK_ID_CURR)

Note: `bureau_balance` (monthly statuses of these credits, ~27M rows) was tested
earlier and removed. The feature-group ablation showed no measurable contribution
(-0.0001 CV AUC), while it was by far the heaviest part of the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BUREAU_COLS = [
    "SK_ID_CURR",
    "SK_ID_BUREAU",
    "CREDIT_ACTIVE",
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "AMT_CREDIT_MAX_OVERDUE",
    "CNT_CREDIT_PROLONG",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_OVERDUE",
]


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credits to one row per client (index: SK_ID_CURR)."""
    b = bureau  # modified in place to save memory; the caller does not reuse it
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype("float32")

    out = b.groupby("SK_ID_CURR").agg(
        BUREAU_COUNT=("SK_ID_BUREAU", "size"),
        BUREAU_ACTIVE_SHARE=("IS_ACTIVE", "mean"),
        BUREAU_DAYS_CREDIT_MIN=("DAYS_CREDIT", "min"),  # oldest credit
        BUREAU_DAYS_CREDIT_MAX=("DAYS_CREDIT", "max"),  # newest credit
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_CREDIT_SUM_SUM=("AMT_CREDIT_SUM", "sum"),
        BUREAU_CREDIT_SUM_MEAN=("AMT_CREDIT_SUM", "mean"),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_OVERDUE_SUM=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_MAX_OVERDUE_MAX=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUREAU_PROLONG_SUM=("CNT_CREDIT_PROLONG", "sum"),
    )
    out["BUREAU_DEBT_CREDIT_RATIO"] = (
        out["BUREAU_DEBT_SUM"] / out["BUREAU_CREDIT_SUM_SUM"]
    ).replace([np.inf, -np.inf], np.nan)
    return out


def build_bureau_features(data_dir: Path) -> pd.DataFrame:
    """Read the table and return client-level features (index: SK_ID_CURR)."""
    bureau = pd.read_csv(
        Path(data_dir) / "bureau.csv",
        usecols=BUREAU_COLS,
        dtype={"SK_ID_CURR": "int32", "SK_ID_BUREAU": "int32"},
    )
    return aggregate_bureau(bureau)