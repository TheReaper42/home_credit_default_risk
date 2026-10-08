"""Client-level aggregates from `POS_CASH_balance`.

One row per month of a point-of-sale / cash loan at Home Credit (~10M rows).
The table carries SK_ID_CURR directly, so one aggregation level is enough:
    monthly record -> client (SK_ID_CURR)

MONTHS_BALANCE is relative to the current application: -1 is the most recent month.
SK_DPD / SK_DPD_DEF are days past due (the second ignores tolerated small debts).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

POS_COLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "CNT_INSTALMENT",
    "CNT_INSTALMENT_FUTURE",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
]

RECENT_MONTHS = 12  # window for the "recent behaviour" features


def aggregate_pos_cash(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly records to one row per client (index: SK_ID_CURR)."""
    p = pos  # modified in place to save memory; the caller does not reuse it

    p["HAS_DPD"] = (p["SK_DPD"] > 0).astype("float32")
    p["IS_ACTIVE"] = (p["NAME_CONTRACT_STATUS"] == "Active").astype("float32")

    # Recent behaviour: the last RECENT_MONTHS months before the application
    recent = p["MONTHS_BALANCE"] >= -RECENT_MONTHS
    p["DPD_RECENT"] = p["SK_DPD"].where(recent)
    p["HAS_DPD_RECENT"] = p["HAS_DPD"].where(recent)

    return p.groupby("SK_ID_CURR").agg(
        POS_COUNT=("SK_ID_PREV", "size"),  # number of monthly records
        POS_LOANS=("SK_ID_PREV", "nunique"),  # number of distinct loans
        POS_MONTHS_MIN=("MONTHS_BALANCE", "min"),  # how far back the history goes
        POS_ACTIVE_SHARE=("IS_ACTIVE", "mean"),
        POS_CNT_INSTALMENT_MEAN=("CNT_INSTALMENT", "mean"),
        POS_CNT_INSTALMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
        POS_DPD_MAX=("SK_DPD", "max"),
        POS_DPD_MEAN=("SK_DPD", "mean"),
        POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        POS_DPD_SHARE=("HAS_DPD", "mean"),
        POS_DPD_MEAN_RECENT=("DPD_RECENT", "mean"),
        POS_DPD_SHARE_RECENT=("HAS_DPD_RECENT", "mean"),
    )


def build_pos_cash_features(data_dir: Path) -> pd.DataFrame:
    """Read the table and return client-level features (index: SK_ID_CURR)."""
    # ~10M rows: use compact dtypes to keep memory low
    pos = pd.read_csv(
        Path(data_dir) / "POS_CASH_balance.csv",
        usecols=POS_COLS,
        dtype={
            "SK_ID_PREV": "int32",
            "SK_ID_CURR": "int32",
            "MONTHS_BALANCE": "int16",
            "CNT_INSTALMENT": "float32",
            "CNT_INSTALMENT_FUTURE": "float32",
            "NAME_CONTRACT_STATUS": "category",
            "SK_DPD": "int32",
            "SK_DPD_DEF": "int32",
        },
    )
    return aggregate_pos_cash(pos)