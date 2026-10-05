"""Reusable cross-validation helper for the Home Credit project.

Every experiment becomes a single call to `run_cv`, and results from two
experiments can be compared fold by fold with `compare_folds`.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DEFAULT_PARAMS = dict(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    importance_type="gain",
    n_jobs=-1,
    verbose=-1,
)


@dataclass
class CVResult:
    fold_aucs: list[float]
    oof: np.ndarray
    oof_auc: float
    importances: pd.DataFrame  # one column per fold plus a "mean" column

    @property
    def cv_mean(self) -> float:
        return float(np.mean(self.fold_aucs))

    @property
    def cv_std(self) -> float:
        return float(np.std(self.fold_aucs))

    def summary_row(self, name: str) -> str:
        """Markdown table row, ready to paste into README or results.md."""
        return f"| {name} | {self.cv_mean:.4f} ± {self.cv_std:.4f} | {self.oof_auc:.4f} |"


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: dict | None = None,
    n_folds: int = 5,
    seed: int = 42,
    early_stopping_rounds: int = 100,
    verbose: bool = True,
) -> CVResult:
    """Stratified K-fold CV with LightGBM; returns per-fold AUCs, OOF predictions
    and feature importances.

    The same `seed` always produces the same folds, so results of different
    experiments are directly comparable fold by fold.
    """
    model_params = {**DEFAULT_PARAMS, **(params or {}), "random_state": seed}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    fold_aucs: list[float] = []
    importances = pd.DataFrame(index=X.columns)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, oof[va_idx])
        fold_aucs.append(auc)
        importances[f"fold_{fold}"] = model.feature_importances_
        if verbose:
            print(f"fold {fold}: AUC = {auc:.5f} (iters: {model.best_iteration_})")

    importances["mean"] = importances.mean(axis=1)
    result = CVResult(
        fold_aucs=fold_aucs,
        oof=oof,
        oof_auc=float(roc_auc_score(y, oof)),
        importances=importances,
    )
    if verbose:
        print(f"\nCV AUC: {result.cv_mean:.5f} +/- {result.cv_std:.5f}")
        print(f"OOF AUC: {result.oof_auc:.5f}")
    return result


def compare_folds(
    new: CVResult,
    old: CVResult,
    name_new: str = "new",
    name_old: str = "old",
) -> pd.DataFrame:
    """Per-fold paired comparison of two experiments run with the same seed."""
    table = pd.DataFrame(
        {
            name_old: old.fold_aucs,
            name_new: new.fold_aucs,
        },
        index=[f"fold_{i}" for i in range(1, len(new.fold_aucs) + 1)],
    )
    table["diff"] = table[name_new] - table[name_old]
    wins = int((table["diff"] > 0).sum())
    print(f"{name_new} beats {name_old} in {wins}/{len(table)} folds, "
          f"mean diff = {table['diff'].mean():+.5f}")
    return table