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
# ## 7. Experiment 4: aggregates from `bureau`
# The external-score combinations gave no gain, so this experiment builds on
# `X_ratios`. (An earlier version also aggregated `bureau_balance`; the ablation
# showed that it added nothing measurable, so it was removed.)

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
# ## 10. Experiment 7: aggregates from `POS_CASH_balance`
# Builds on `X_inst`. Monthly records of point-of-sale and cash loans (~10M rows).

# %%
from pos_cash import build_pos_cash_features

pos_feats = build_pos_cash_features(DATA_DIR)  # indexed by SK_ID_CURR
print("clients with POS_CASH records:", len(pos_feats))

# %%
pos_aligned = pos_feats.reindex(df["SK_ID_CURR"].to_numpy())
pos_aligned.index = df.index
pos_aligned["POS_COUNT"] = pos_aligned["POS_COUNT"].fillna(0)
pos_aligned["POS_LOANS"] = pos_aligned["POS_LOANS"].fillna(0)

X_pos = pd.concat([X_inst, pos_aligned], axis=1)
assert len(X_pos) == len(X_inst), "join changed the number of rows"
print("features:", X_pos.shape[1])

# %%
res_pos = run_cv(X_pos, y, seed=SEED)
print(res_pos.summary_row("+ POS_CASH_balance aggregates"))
compare_folds(res_pos, res_inst, "with POS_CASH", "installments")

# %%
print(res_pos.importances["mean"].sort_values(ascending=False).head(20))

# %% [markdown]
# ## 11. Experiment 8: aggregates from `credit_card_balance`
# Monthly records of credit cards (~3.8M rows). Many clients have no card at all,
# so most rows will have NaN here; that is expected.
# Build on `X_pos` if the POS_CASH gain was stable; otherwise switch the base below
# to `X_inst` / `res_inst`.

# %%
from credit_card import build_credit_card_features

X_cc_base, res_cc_base = X_pos, res_pos

cc_feats = build_credit_card_features(DATA_DIR)  # indexed by SK_ID_CURR
print("clients with credit card records:", len(cc_feats))

# %%
cc_aligned = cc_feats.reindex(df["SK_ID_CURR"].to_numpy())
cc_aligned.index = df.index
cc_aligned["CC_COUNT"] = cc_aligned["CC_COUNT"].fillna(0)
cc_aligned["CC_CARDS"] = cc_aligned["CC_CARDS"].fillna(0)

X_cc = pd.concat([X_cc_base, cc_aligned], axis=1)
assert len(X_cc) == len(X_cc_base), "join changed the number of rows"
print("features:", X_cc.shape[1])

# %%
res_cc = run_cv(X_cc, y, seed=SEED)
print(res_cc.summary_row("+ credit_card_balance aggregates"))
compare_folds(res_cc, res_cc_base, "with credit card", "previous step")

# %%
print(res_cc.importances["mean"].sort_values(ascending=False).head(20))

# %% [markdown]
# ## 12. Ablation: contribution of each feature group
# Each group is dropped in turn from the full model (`X_cc`) and the CV is re-run
# with the same folds. Six extra CV runs, so this takes a while.

# %%
from ablation import columns_with_prefix, run_ablation

cols = X_cc.columns
feature_groups = {
    "ratios": RATIO_COLS,
    "bureau": columns_with_prefix(cols, "BUREAU_"),
    "previous_application": columns_with_prefix(cols, "PREV_"),
    "installments": columns_with_prefix(cols, "INST_"),
    "pos_cash": columns_with_prefix(cols, "POS_"),
    "credit_card": columns_with_prefix(cols, "CC_"),
}
for name, group_cols in feature_groups.items():
    print(f"{name}: {len(group_cols)} features")

# Sanity check: the groups must cover exactly the engineered columns
covered = sum(len(c) for c in feature_groups.values())
assert covered == X_cc.shape[1] - X_base.shape[1], "groups do not match the columns"

# %%
ablation = run_ablation(X_cc, y, feature_groups, reference=res_cc, seed=SEED)
print()
print(ablation.round(5))

# %% [markdown]
# ## 13. Hyperparameter tuning with Optuna
# The search runs on the final feature set `X_cc` with a higher learning rate for
# speed (see `tune.py`). The best CV score over many trials is optimistically biased,
# so the final comparison below uses a different fold split.

# %%
from tune import run_study

study = run_study(X_cc, y, n_trials=40, seed=SEED)
print("best CV AUC in the search:", round(study.best_value, 5))
print(study.best_params)

# %%
# Fair evaluation on a NEW fold split: default vs tuned parameters, same learning rate.
EVAL_SEED = 2024
FINAL_LR = 0.05

res_default = run_cv(X_cc, y, params={"learning_rate": FINAL_LR}, seed=EVAL_SEED)
res_tuned = run_cv(
    X_cc, y, params={**study.best_params, "learning_rate": FINAL_LR}, seed=EVAL_SEED
)
print(res_default.summary_row("Default parameters (new fold split)"))
print(res_tuned.summary_row("Tuned parameters (new fold split)"))
compare_folds(res_tuned, res_default, "tuned", "default")

# %% [markdown]
# ## 14. Model interpretation: SHAP values
# The final model is trained on all rows with the tuned parameters and a fixed
# number of trees. SHAP values come from LightGBM itself (see `explain.py`).
# If the kernel was restarted, reload the study first:
# `study = run_study(X_cc, y, n_trials=0)`.

# %%
from explain import (
    check_additivity,
    fit_final_model,
    plot_beeswarm,
    plot_dependence,
    plot_importance,
    shap_values,
    top_numeric_features,
)

# About 1.1x the mean best iteration of the tuned CV run (~830 trees at lr 0.05)
FINAL_N_ESTIMATORS = 900
final_model = fit_final_model(
    X_cc, y, study.best_params, FINAL_N_ESTIMATORS, learning_rate=FINAL_LR, seed=SEED
)

# %%
# SHAP on a random sample of the training rows (all rows would be slow)
X_shap = X_cc.sample(n=20000, random_state=SEED)
shap_vals, base_value = shap_values(final_model, X_shap)
print("max additivity error:", check_additivity(final_model, X_shap, shap_vals, base_value))

# %%
fig_importance = plot_importance(shap_vals, top=20, path="figures/shap_importance.png")
print(shap_vals.abs().mean().sort_values(ascending=False).head(20))

# %%
fig_beeswarm = plot_beeswarm(shap_vals, X_shap, top=15, path="figures/shap_beeswarm.png")

# %%
fig_dependence = plot_dependence(
    shap_vals,
    X_shap,
    top_numeric_features(shap_vals, X_shap, n=4),
    path="figures/shap_dependence.png",
)

# %% [markdown]
# ## Next steps
# 1. Record the tuning result and the SHAP findings in the README (the figures are
#    saved in `figures/`).
# 2. Optionally lower the learning rate (for example 0.02-0.03) for the final model.
# 3. Build the feature pipeline for `application_test` and submit to Kaggle.