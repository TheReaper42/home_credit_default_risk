"""Client-level aggregates from `bureau` and `bureau_balance`.

Table hierarchy (one row per ...):
    bureau_balance : credit x month   (key: SK_ID_BUREAU)
    bureau         : credit           (keys: SK_ID_BUREAU, SK_ID_CURR)
    application    : client           (key: SK_ID_CURR)

Aggregation must follow the same chain: months -> credit -> client.
Never join `bureau_balance` straight to clients: it has no SK_ID_CURR, and
skipping a level would mix months of different credits together.
"""

from __future__ import annotations

import gc
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


def aggregate_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    """Level 1: months -> one row per credit (index: SK_ID_BUREAU).

    STATUS values: 'C' closed, 'X' unknown, '0' no delay, '1'..'5' increasing delay.
    """
    # Map the (few) status categories to numbers, then index by category codes;
    # much faster than mapping 27M strings. 'C' and 'X' become NaN.
    cats = bb["STATUS"].cat.categories
    num_by_code = np.array(
        [float(c) if c.isdigit() else np.nan for c in cats], dtype="float32"
    )
    status_num = num_by_code[bb["STATUS"].cat.codes.to_numpy()]

    tmp = pd.DataFrame(
        {
            "SK_ID_BUREAU": bb["SK_ID_BUREAU"].to_numpy(),
            "MONTHS_BALANCE": bb["MONTHS_BALANCE"].to_numpy(),
            "STATUS_NUM": status_num,
        }
    )
    tmp["IS_DPD"] = (tmp["STATUS_NUM"] >= 1).astype("float32")  # NaN >= 1 is False

    return tmp.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS_COUNT=("MONTHS_BALANCE", "size"),
        BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),  # how long ago the record starts
        BB_DPD_SHARE=("IS_DPD", "mean"),  # share of months with any delay
        BB_STATUS_MAX=("STATUS_NUM", "max"),  # worst delay status
    )


def aggregate_bureau(bureau: pd.DataFrame, bb_agg: pd.DataFrame) -> pd.DataFrame:
    """Level 2: credits (+ their monthly summary) -> one row per client."""
    b = bureau.merge(bb_agg, left_on="SK_ID_BUREAU", right_index=True, how="left")
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
        BUREAU_BB_DPD_SHARE_MEAN=("BB_DPD_SHARE", "mean"),
        BUREAU_BB_STATUS_MAX=("BB_STATUS_MAX", "max"),
        BUREAU_BB_MONTHS_MEAN=("BB_MONTHS_COUNT", "mean"),
    )
    out["BUREAU_DEBT_CREDIT_RATIO"] = (
        out["BUREAU_DEBT_SUM"] / out["BUREAU_CREDIT_SUM_SUM"]
    ).replace([np.inf, -np.inf], np.nan)
    return out


def build_bureau_features(data_dir: Path) -> pd.DataFrame:
    """Read both tables and return client-level features (index: SK_ID_CURR)."""
    data_dir = Path(data_dir)

    bureau = pd.read_csv(
        data_dir / "bureau.csv",
        usecols=BUREAU_COLS,
        dtype={"SK_ID_CURR": "int32", "SK_ID_BUREAU": "int32"},
    )
    # bureau_balance has ~27M rows: use compact dtypes to keep memory low
    bb = pd.read_csv(
        data_dir / "bureau_balance.csv",
        dtype={"SK_ID_BUREAU": "int32", "MONTHS_BALANCE": "int16", "STATUS": "category"},
    )

    bb_agg = aggregate_bureau_balance(bb)
    del bb
    gc.collect()

    return aggregate_bureau(bureau, bb_agg)