# %% [markdown]
# # Baseline: Home Credit Default Risk
# Uses only `application_train.csv`, LightGBM, stratified K-fold, ROC AUC metric.
# Cross-validation lives in `cv.py` (`run_cv`, `compare_folds`), so every
# experiment here is one call and results are directly comparable.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

from cv import compare_folds, run_cv

DATA_DIR = Path("data")  # adjust this path to your repository layout
SEED = 42
TARGET = "TARGET"

# %% [markdown]
# ## 1. Loading and first look

# %%
df = pd.read_csv(DATA_DIR / "application_train.csv")
print("shape:", df.shape)
print(df[TARGET].value_counts(normalize=True))

# %%
missing = df.isna().mean().sort_values(ascending=False)
print(missing.head(15))
print(df.dtypes.value_counts())

# %% [markdown]
# ## 2. Minimal preprocessing
# - `DAYS_EMPLOYED == 365243` is a placeholder, not a real value: replace it with NaN.
# - String columns are cast to `category`, which LightGBM handles natively.

# %%
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = df[col].astype("category")

y = df[TARGET]
X_base = df.drop(columns=[TARGET, "SK_ID_CURR"])

print("features:", X_base.shape[1], "| categorical:", len(cat_cols))

# %% [markdown]
# ## 3. Experiment 1: baseline (raw columns only)

# %%
res_base = run_cv(X_base, y, seed=SEED)
print(res_base.summary_row("Baseline: raw application_train columns"))

# %% [markdown]
# ## 4. Experiment 2: ratio features
# Each feature group is a separate function, so it can be added, removed or
# ablated independently.

# %%
RATIO_COLS = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_ANNUITY_RATIO",
    "GOODS_PRICE_CREDIT_RATIO",
    "EMPLOYED_BIRTH_RATIO",
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


X_ratios = add_ratio_features(X_base)

# %%
res_ratios = run_cv(X_ratios, y, seed=SEED)
print(res_ratios.summary_row("+ 5 ratio features"))

# %%
# Paired per-fold comparison: is the gain stable across folds?
compare_folds(res_ratios, res_base, "with ratios", "baseline")

# %% [markdown]
# ## 5. Experiment 3: combinations of the external scores
# `EXT_SOURCE_1/2/3` are the strongest features, so combining them is the
# most promising next step. Feature groups are added cumulatively, and every
# step is compared fold by fold with the previous one.

# %%
EXT_COLS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
# Hand-picked weights (roughly follow the gain importance ratio); feel free to change.
EXT_WEIGHTS = {"EXT_SOURCE_1": 1.0, "EXT_SOURCE_2": 3.0, "EXT_SOURCE_3": 3.0}

EXT_GROUPS = {
    "mean/min/max/std": [
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_MIN",
        "EXT_SOURCE_MAX",
        "EXT_SOURCE_STD",
    ],
    "product": ["EXT_SOURCE_PROD"],
    "weighted mean": ["EXT_SOURCE_WEIGHTED"],
    "missing count": ["EXT_SOURCE_NA_COUNT"],
}


def ext_source_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return ONLY the new external-score features (same index as `data`)."""
    ext = data[EXT_COLS].astype("float64")
    weights = pd.Series(EXT_WEIGHTS)
    out = pd.DataFrame(index=data.index)
    out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)  # NaNs are skipped
    out["EXT_SOURCE_MIN"] = ext.min(axis=1)
    out["EXT_SOURCE_MAX"] = ext.max(axis=1)
    out["EXT_SOURCE_STD"] = ext.std(axis=1)
    out["EXT_SOURCE_PROD"] = ext.prod(axis=1, skipna=False)  # NaN if any score is missing
    # Weighted mean over the scores that are present for each row
    present = ext.notna()
    out["EXT_SOURCE_WEIGHTED"] = (ext.fillna(0) * weights).sum(axis=1) / (
        present * weights
    ).sum(axis=1).replace(0, np.nan)
    out["EXT_SOURCE_NA_COUNT"] = ext.isna().sum(axis=1)
    return out


ext_feats = ext_source_features(X_ratios)

# %%
# Cumulative experiments: add one group at a time, compare with the previous step.
ext_results = {}
current_cols: list[str] = []
prev_name, prev_res = "+ 5 ratio features", res_ratios

for group_name, cols in EXT_GROUPS.items():
    current_cols += cols
    X_step = pd.concat([X_ratios, ext_feats[current_cols]], axis=1)
    res = run_cv(X_step, y, seed=SEED, verbose=False)
    name = f"+ EXT_SOURCE {group_name}"
    print(f"\n=== {name} ===")
    compare_folds(res, prev_res, name, prev_name)
    ext_results[name] = res
    prev_name, prev_res = name, res

# %%
# Rows ready to paste into the README results table
for name, res in ext_results.items():
    print(res.summary_row(name))

# %% [markdown]
# ## 6. Feature importance (latest experiment)

# %%
print(prev_res.importances["mean"].sort_values(ascending=False).head(20))

# %% [markdown]
# ## 7. Experiment 4: aggregates from `bureau` and `bureau_balance`
# The external-score combinations gave no gain, so this experiment builds on
# `X_ratios`. The aggregation follows the table hierarchy (see `bureau.py`):
# months -> credit (SK_ID_BUREAU) -> client (SK_ID_CURR).

# %%
from bureau import build_bureau_features

bureau_feats = build_bureau_features(DATA_DIR)  # indexed by SK_ID_CURR
print("clients with bureau history:", len(bureau_feats))

# %%
# Align to the rows of the main table; clients without bureau records get NaN.
bureau_aligned = bureau_feats.reindex(df["SK_ID_CURR"].to_numpy())
bureau_aligned.index = df.index
bureau_aligned["BUREAU_COUNT"] = bureau_aligned["BUREAU_COUNT"].fillna(0)

X_bureau = pd.concat([X_ratios, bureau_aligned], axis=1)
assert len(X_bureau) == len(X_ratios), "join changed the number of rows"
print("features:", X_bureau.shape[1])

# %%
res_bureau = run_cv(X_bureau, y, seed=SEED)
print(res_bureau.summary_row("+ bureau / bureau_balance aggregates"))
compare_folds(res_bureau, res_ratios, "with bureau", "ratios only")

# %%
print(res_bureau.importances["mean"].sort_values(ascending=False).head(20))

# %% [markdown]
# ## 8. Experiment 5: aggregates from `previous_application`
# The bureau features gave a clear gain, so this experiment builds on `X_bureau`.
# Unlike `bureau_balance`, this table has SK_ID_CURR directly: one aggregation level.

# %%
from previous import build_previous_features

prev_feats = build_previous_features(DATA_DIR)  # indexed by SK_ID_CURR
print("clients with previous applications:", len(prev_feats))

# %%
prev_aligned = prev_feats.reindex(df["SK_ID_CURR"].to_numpy())
prev_aligned.index = df.index
prev_aligned["PREV_COUNT"] = prev_aligned["PREV_COUNT"].fillna(0)

X_prev = pd.concat([X_bureau, prev_aligned], axis=1)
assert len(X_prev) == len(X_bureau), "join changed the number of rows"
print("features:", X_prev.shape[1])

# %%
res_prev = run_cv(X_prev, y, seed=SEED)
print(res_prev.summary_row("+ previous_application aggregates"))
compare_folds(res_prev, res_bureau, "with previous_application", "bureau")

# %%
print(res_prev.importances["mean"].sort_values(ascending=False).head(20))

# %% [markdown]
# ## 9. Experiment 6: aggregates from `installments_payments`
# Builds on `X_prev`. The table is large (~13.6M rows), so loading and aggregating
# takes a while; the key features are days past due and underpaid amounts.

# %%
from installments import build_installments_features

inst_feats = build_installments_features(DATA_DIR)  # indexed by SK_ID_CURR
print("clients with instalment records:", len(inst_feats))

# %%
inst_aligned = inst_feats.reindex(df["SK_ID_CURR"].to_numpy())
inst_aligned.index = df.index
inst_aligned["INST_COUNT"] = inst_aligned["INST_COUNT"].fillna(0)

X_inst = pd.concat([X_prev, inst_aligned], axis=1)
assert len(X_inst) == len(X_prev), "join changed the number of rows"
print("features:", X_inst.shape[1])

# %%
res_inst = run_cv(X_inst, y, seed=SEED)
print(res_inst.summary_row("+ installments_payments aggregates"))
compare_folds(res_inst, res_prev, "with installments", "previous_application")

# %%
print(res_inst.importances["mean"].sort_values(ascending=False).head(20))

# %% [markdown]
# ## Next steps
# 1. Record every result row in the README table.
# 2. Keep a table only if its gain is stable (better in most folds, mean difference
#    above roughly 0.002-0.003).
# 3. Move on to `POS_CASH_balance` and `credit_card_balance` (they carry SK_ID_CURR
#    directly, so they can be aggregated straight to clients).
# 4. Run an ablation per feature group (for example without `BUREAU_BB_*`).