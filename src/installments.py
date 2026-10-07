"""Client-level aggregates from `installments_payments`.

One row per scheduled instalment of a previous Home Credit loan (~13.6M rows).
The table carries SK_ID_CURR directly, so one aggregation level is enough:
    instalment -> client (SK_ID_CURR)

Two derived quantities carry most of the signal:
    DPD (days past due)  = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT   (> 0: paid late)
    PAYMENT_DIFF         = AMT_INSTALMENT - AMT_PAYMENT           (> 0: underpaid)

Rows with a missing DAYS_ENTRY_PAYMENT / AMT_PAYMENT (no payment recorded) keep NaN
in the derived columns, so they do not count as "paid on time".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INST_COLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
]

# Ignore payment differences below this (floating-point noise, tiny roundings)
DIFF_TOLERANCE = 0.01
RECENT_DAYS = 365  # window for the "recent behaviour" features


def aggregate_installments(inst: pd.DataFrame) -> pd.DataFrame:
    """Aggregate instalments to one row per client (index: SK_ID_CURR)."""
    i = inst  # modified in place to save memory; the caller does not reuse it

    dpd = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]  # > 0: paid late
    i["DPD"] = dpd.clip(lower=0)
    i["DBD"] = (-dpd).clip(lower=0)  # days paid early
    i["IS_LATE"] = (dpd > 0).astype("float32").where(dpd.notna())

    i["PAYMENT_DIFF"] = i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]  # > 0: underpaid
    i["PAYMENT_RATIO"] = (i["AMT_PAYMENT"] / i["AMT_INSTALMENT"]).replace(
        [np.inf, -np.inf], np.nan
    )
    i["IS_UNDERPAID"] = (i["PAYMENT_DIFF"] > DIFF_TOLERANCE).astype("float32").where(
        i["PAYMENT_DIFF"].notna()
    )

    # Recent behaviour: instalments due within the last year
    recent = i["DAYS_INSTALMENT"] >= -RECENT_DAYS
    i["IS_LATE_RECENT"] = i["IS_LATE"].where(recent)
    i["DPD_RECENT"] = i["DPD"].where(recent)

    return i.groupby("SK_ID_CURR").agg(
        INST_COUNT=("SK_ID_PREV", "size"),
        INST_DPD_MAX=("DPD", "max"),
        INST_DPD_MEAN=("DPD", "mean"),
        INST_LATE_SHARE=("IS_LATE", "mean"),
        INST_DBD_MEAN=("DBD", "mean"),
        INST_PAYMENT_DIFF_MAX=("PAYMENT_DIFF", "max"),
        INST_PAYMENT_DIFF_MEAN=("PAYMENT_DIFF", "mean"),
        INST_UNDERPAID_SHARE=("IS_UNDERPAID", "mean"),
        INST_PAYMENT_RATIO_MEAN=("PAYMENT_RATIO", "mean"),
        INST_PAYMENT_RATIO_MIN=("PAYMENT_RATIO", "min"),
        INST_LATE_SHARE_RECENT=("IS_LATE_RECENT", "mean"),
        INST_DPD_MEAN_RECENT=("DPD_RECENT", "mean"),
    )


def build_installments_features(data_dir: Path) -> pd.DataFrame:
    """Read the table and return client-level features (index: SK_ID_CURR)."""
    # ~13.6M rows: use compact dtypes to keep memory low
    inst = pd.read_csv(
        Path(data_dir) / "installments_payments.csv",
        usecols=INST_COLS,
        dtype={
            "SK_ID_PREV": "int32",
            "SK_ID_CURR": "int32",
            "DAYS_INSTALMENT": "float32",
            "DAYS_ENTRY_PAYMENT": "float32",
            "AMT_INSTALMENT": "float32",
            "AMT_PAYMENT": "float32",
        },
    )
    return aggregate_installments(inst)