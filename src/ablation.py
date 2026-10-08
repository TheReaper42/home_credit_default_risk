"""Feature-group ablation: how much does each group contribute to the final model?

For every group, the CV is re-run without its columns and compared fold by fold
with the full model. Both runs must use the same seed (same folds).

Caveat: a group is measured by its marginal contribution given all the other
groups. If two groups carry overlapping information (for example POS_CASH and
installments), removing either one alone can look harmless even though removing
both would hurt.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from cv import CVResult, run_cv


def columns_with_prefix(
    columns: Iterable[str], prefix: str, exclude: str | None = None
) -> list[str]:
    """Columns that start with `prefix` (optionally skipping those starting with `exclude`)."""
    return [
        c
        for c in columns
        if c.startswith(prefix) and not (exclude and c.startswith(exclude))
    ]


def run_ablation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: dict[str, list[str]],
    reference: CVResult,
    params: dict | None = None,
    seed: int = 42,
    n_folds: int = 5,
    min_gain: float = 0.002,
) -> pd.DataFrame:
    """Drop each feature group in turn and compare with the full-model `reference`.

    Columns of the result:
        n_features   number of dropped columns
        cv_auc       CV AUC without the group
        delta        cv_auc minus the reference CV AUC (negative: the group helps)
        folds_worse  in how many folds the AUC dropped without the group
        verdict      "keep" if dropping costs at least `min_gain` AUC and hurts in
                     all folds but at most one, otherwise "candidate to drop"
    """
    ref_folds = np.array(reference.fold_aucs)
    rows = []

    for name, cols in groups.items():
        present = [c for c in cols if c in X.columns]
        if not present:
            print(f"{name}: no matching columns, skipped")
            continue

        res = run_cv(
            X.drop(columns=present),
            y,
            params=params,
            n_folds=n_folds,
            seed=seed,
            verbose=False,
        )
        diff = np.array(res.fold_aucs) - ref_folds  # negative: dropping hurts
        delta = float(diff.mean())
        folds_worse = int((diff < 0).sum())
        keep = (-delta >= min_gain) and (folds_worse >= n_folds - 1)

        rows.append(
            {
                "group": name,
                "n_features": len(present),
                "cv_auc": res.cv_mean,
                "delta": delta,
                "folds_worse": f"{folds_worse}/{n_folds}",
                "verdict": "keep" if keep else "candidate to drop",
            }
        )
        print(f"{name}: dropped {len(present)} features, CV AUC {res.cv_mean:.5f} "
              f"({delta:+.5f}), worse in {folds_worse}/{n_folds} folds")

    return pd.DataFrame(rows).set_index("group").sort_values("delta")