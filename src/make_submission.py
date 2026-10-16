"""Build the features, train the final model and write the Kaggle submission file.

Usage:
    python make_submission.py [--data-dir data] [--cache-dir cache] [--out submission.csv]

The final model is the average of the five fold models of a stratified CV (each fold
model predicts the test clients). This needs no guess about the number of trees:
early stopping chooses it in every fold, and the out-of-fold AUC printed at the end
is a sanity check of the whole pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cv import run_cv
from features import ID_COL, TARGET, build_features

# Best parameters of the Optuna study (trial 113, see the README)
BEST_PARAMS = {
    "num_leaves": 16,
    "min_child_samples": 251,
    "colsample_bytree": 0.841807636303969,
    "subsample": 0.8673819984099976,
    "reg_alpha": 7.99333163860521,
    "reg_lambda": 13.202174574308918,
    "cat_smooth": 76.03627373365204,
}
FINAL_LR = 0.05
SEED = 42

# Sensitive features that are excluded from the final model (see the README)
EXCLUDE_FEATURES = ["CODE_GENDER"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    X_train, y, X_test, ids_test = build_features(args.data_dir, args.cache_dir)

    drop = [c for c in EXCLUDE_FEATURES if c in X_train.columns]
    X_train = X_train.drop(columns=drop)
    X_test = X_test.drop(columns=drop)
    assert list(X_train.columns) == list(X_test.columns), "train/test columns differ"
    print(f"train: {X_train.shape}, test: {X_test.shape}, excluded: {drop}")

    result = run_cv(
        X_train,
        y,
        params={**BEST_PARAMS, "learning_rate": FINAL_LR},
        seed=SEED,
        X_test=X_test,
    )

    submission = pd.DataFrame({ID_COL: ids_test.to_numpy(), TARGET: result.test_pred})

    # Sanity checks before anything is written
    assert len(submission) == len(X_test), "wrong number of rows"
    assert submission[ID_COL].is_unique, "duplicate client ids"
    assert submission[TARGET].notna().all(), "NaN in predictions"
    assert submission[TARGET].between(0, 1).all(), "predictions outside [0, 1]"
    print(
        f"mean predicted risk: test {submission[TARGET].mean():.4f} "
        f"vs out-of-fold {result.oof.mean():.4f} (should be close)"
    )

    out = Path(args.out)
    submission.to_csv(out, index=False)
    print(f"written: {out} ({len(submission)} rows)")


if __name__ == "__main__":
    main()