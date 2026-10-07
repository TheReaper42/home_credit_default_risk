"""Client-level aggregates from `previous_application`.

`previous_application` has one row per previous application of a client at Home
Credit and carries SK_ID_CURR directly, so one level of aggregation is enough:
    previous application -> client (SK_ID_CURR)

Note for later: columns such as DAYS_FIRST_DRAWING, DAYS_FIRST_DUE, DAYS_LAST_DUE
and DAYS_TERMINATION use the placeholder 365243 for "not available". Replace it
with NaN (as is done for DAYS_EMPLOYED in the main table) before using them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PREV_COLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "AMT_ANNUITY",
    "AMT_APPLICATION",
    "AMT_CREDIT",
    "RATE_DOWN_PAYMENT",
    "NAME_CONTRACT_STATUS",
    "DAYS_DECISION",
    "CNT_PAYMENT",
]


def aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """Aggregate previous applications to one row per client (index: SK_ID_CURR)."""
    p = prev  # modified in place to save memory; the caller does not reuse it

    status = p["NAME_CONTRACT_STATUS"]
    p["IS_APPROVED"] = (status == "Approved").astype("float32")
    p["IS_REFUSED"] = (status == "Refused").astype("float32")
    p["IS_CANCELED"] = (status == "Canceled").astype("float32")

    # How much was asked for vs. how much was granted (application level)
    p["APP_CREDIT_RATIO"] = (p["AMT_APPLICATION"] / p["AMT_CREDIT"]).replace(
        [np.inf, -np.inf], np.nan
    )
    # Values that exist only for refused / approved applications
    p["DECISION_REFUSED"] = p["DAYS_DECISION"].where(p["IS_REFUSED"] == 1)
    p["CREDIT_APPROVED"] = p["AMT_CREDIT"].where(p["IS_APPROVED"] == 1)

    return p.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "size"),
        PREV_APPROVED_SHARE=("IS_APPROVED", "mean"),
        PREV_REFUSED_SHARE=("IS_REFUSED", "mean"),
        PREV_CANCELED_SHARE=("IS_CANCELED", "mean"),
        PREV_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_APPLICATION_MAX=("AMT_APPLICATION", "max"),
        PREV_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_CREDIT_MAX=("AMT_CREDIT", "max"),
        PREV_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_APP_CREDIT_RATIO_MEAN=("APP_CREDIT_RATIO", "mean"),
        PREV_DAYS_DECISION_MAX=("DAYS_DECISION", "max"),  # most recent decision
        PREV_DAYS_DECISION_MIN=("DAYS_DECISION", "min"),  # oldest decision
        PREV_DAYS_LAST_REFUSAL=("DECISION_REFUSED", "max"),  # most recent refusal
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
        PREV_DOWN_PAYMENT_RATE_MEAN=("RATE_DOWN_PAYMENT", "mean"),
        PREV_APPROVED_CREDIT_MEAN=("CREDIT_APPROVED", "mean"),
    )


def build_previous_features(data_dir: Path) -> pd.DataFrame:
    """Read the table and return client-level features (index: SK_ID_CURR)."""
    prev = pd.read_csv(
        Path(data_dir) / "previous_application.csv",
        usecols=PREV_COLS,
        dtype={"SK_ID_PREV": "int32", "SK_ID_CURR": "int32"},
    )
    return aggregate_previous_application(prev)