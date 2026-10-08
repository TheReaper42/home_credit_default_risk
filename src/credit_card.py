"""Client-level aggregates from `credit_card_balance`.

One row per month of a credit card at Home Credit (~3.8M rows). The table carries
SK_ID_CURR directly, so one aggregation level is enough:
    monthly record -> client (SK_ID_CURR)

MONTHS_BALANCE is relative to the current application: -1 is the most recent month.
Note: the source column for the receivable amount is misspelled (AMT_RECIVABLE);
it is not used here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CC_COLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_ATM_CURRENT",
    "AMT_DRAWINGS_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "AMT_PAYMENT_TOTAL_CURRENT",
    "CNT_DRAWINGS_CURRENT",
    "SK_DPD",
    "SK_DPD_DEF",
]

RECENT_MONTHS = 12  # window for the "recent behaviour" features


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division where a zero denominator gives NaN instead of inf."""
    return (num / den).replace([np.inf, -np.inf], np.nan)


def aggregate_credit_card(cc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly records to one row per client (index: SK_ID_CURR)."""
    c = cc  # modified in place to save memory; the caller does not reuse it

    c["UTILIZATION"] = _safe_ratio(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"])
    # Share of drawings made as ATM cash withdrawals
    c["ATM_SHARE"] = _safe_ratio(c["AMT_DRAWINGS_ATM_CURRENT"], c["AMT_DRAWINGS_CURRENT"])
    # How much was paid relative to the minimum required payment
    c["PAYMENT_MIN_RATIO"] = _safe_ratio(
        c["AMT_PAYMENT_TOTAL_CURRENT"], c["AMT_INST_MIN_REGULARITY"]
    )
    c["HAS_DPD"] = (c["SK_DPD"] > 0).astype("float32")

    # Recent behaviour: the last RECENT_MONTHS months before the application
    recent = c["MONTHS_BALANCE"] >= -RECENT_MONTHS
    c["UTILIZATION_RECENT"] = c["UTILIZATION"].where(recent)
    c["HAS_DPD_RECENT"] = c["HAS_DPD"].where(recent)

    return c.groupby("SK_ID_CURR").agg(
        CC_COUNT=("SK_ID_PREV", "size"),  # number of monthly records
        CC_CARDS=("SK_ID_PREV", "nunique"),  # number of distinct cards
        CC_UTILIZATION_MEAN=("UTILIZATION", "mean"),
        CC_UTILIZATION_MAX=("UTILIZATION", "max"),
        CC_UTILIZATION_MEAN_RECENT=("UTILIZATION_RECENT", "mean"),
        CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
        CC_BALANCE_MAX=("AMT_BALANCE", "max"),
        CC_ATM_SHARE_MEAN=("ATM_SHARE", "mean"),
        CC_PAYMENT_MIN_RATIO_MEAN=("PAYMENT_MIN_RATIO", "mean"),
        CC_PAYMENT_MIN_RATIO_MIN=("PAYMENT_MIN_RATIO", "min"),
        CC_DPD_MAX=("SK_DPD", "max"),
        CC_DPD_MEAN=("SK_DPD", "mean"),
        CC_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
        CC_DPD_SHARE=("HAS_DPD", "mean"),
        CC_DPD_SHARE_RECENT=("HAS_DPD_RECENT", "mean"),
        CC_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
        CC_DRAWINGS_CNT_MEAN=("CNT_DRAWINGS_CURRENT", "mean"),
    )


def build_credit_card_features(data_dir: Path) -> pd.DataFrame:
    """Read the table and return client-level features (index: SK_ID_CURR)."""
    amount_cols = [
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "AMT_PAYMENT_TOTAL_CURRENT",
        "CNT_DRAWINGS_CURRENT",
    ]
    dtypes = {col: "float32" for col in amount_cols}
    dtypes.update(
        {
            "SK_ID_PREV": "int32",
            "SK_ID_CURR": "int32",
            "MONTHS_BALANCE": "int16",
            "SK_DPD": "int32",
            "SK_DPD_DEF": "int32",
        }
    )
    cc = pd.read_csv(
        Path(data_dir) / "credit_card_balance.csv", usecols=CC_COLS, dtype=dtypes
    )
    return aggregate_credit_card(cc)