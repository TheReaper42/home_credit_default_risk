"""Hyperparameter search for the LightGBM model with Optuna.

Design choices:
- The search uses a higher learning rate (SEARCH_LR) so that each trial is fast;
  the best parameters are then re-evaluated at the final learning rate.
- The CV of `cv.py` is reused, with the same folds for every trial. After every
  fold the running mean AUC is reported to Optuna, so clearly bad trials are
  pruned early instead of training all folds.
- The first trial is a configuration close to the LightGBM defaults, so the search
  always has a reference point measured under the same conditions.
- The study is stored in SQLite, so it can be interrupted and resumed.

Caveat: the best CV score of many trials is optimistically biased (it is the
maximum over trials on the same folds). Evaluate the final parameters on a
different fold split, as done in `baseline.py`.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from cv import run_cv

SEARCH_LR = 0.1  # higher learning rate during the search, for speed

# A configuration close to the defaults, evaluated first as a reference
DEFAULT_TRIAL = {
    "num_leaves": 31,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_alpha": 1e-3,
    "reg_lambda": 1.0,
    "cat_smooth": 10.0,
}


def suggest_params(trial: optuna.Trial) -> dict:
    """Search space for LGBMClassifier."""
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, 128, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        # Smoothing for categorical splits (e.g. ORGANIZATION_TYPE has ~58 values)
        "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
    }


def make_objective(X: pd.DataFrame, y: pd.Series, seed: int):
    """Build the Optuna objective: mean CV AUC with early pruning after each fold."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        params["learning_rate"] = SEARCH_LR

        def on_fold(fold: int, fold_aucs: list[float]) -> None:
            trial.report(float(np.mean(fold_aucs)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = run_cv(X, y, params=params, seed=seed, verbose=False, on_fold=on_fold)
        return result.cv_mean

    return objective


def run_study(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 40,
    timeout: float | None = None,
    seed: int = 42,
    study_name: str = "home_credit_lgbm",
    storage: str = "sqlite:///optuna_home_credit.db",
) -> optuna.Study:
    """Run (or resume) the search; `n_trials` more trials are added to the study.

    `timeout` is in seconds. The study is kept in `storage`, so calling this
    function again continues the same study.
    """
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    if len(study.trials) == 0:
        study.enqueue_trial(DEFAULT_TRIAL)  # reference configuration first

    study.optimize(make_objective(X, y, seed), n_trials=n_trials, timeout=timeout)
    return study