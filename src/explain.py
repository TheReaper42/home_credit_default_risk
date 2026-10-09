"""Model interpretation with LightGBM's built-in SHAP values (`pred_contrib=True`).

The native implementation is used instead of the `shap` package: it needs no extra
dependency and handles LightGBM's categorical splits natively. Values are in
log-odds of default: a positive value pushes a client towards higher risk.

Notes for reading the results:
- SHAP values describe how the MODEL uses a feature, not a causal effect.
- Correlated features share credit, so the importance of one of them can look
  smaller than its real information content.
- In the beeswarm plot, colors of categorical features follow the (arbitrary)
  order of category codes and carry no meaning.
"""

from __future__ import annotations

import math
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cv import DEFAULT_PARAMS


def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: dict,
    n_estimators: int,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> lgb.LGBMClassifier:
    """Train on all rows with a fixed number of trees (no early stopping)."""
    params = {
        **DEFAULT_PARAMS,
        **best_params,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "random_state": seed,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def shap_values(model: lgb.LGBMClassifier, X_sample: pd.DataFrame):
    """Per-row feature contributions (log-odds) and the base value."""
    contrib = model.booster_.predict(X_sample, pred_contrib=True)
    values = pd.DataFrame(contrib[:, :-1], columns=X_sample.columns, index=X_sample.index)
    base_value = float(contrib[0, -1])  # the same for every row
    return values, base_value


def check_additivity(
    model: lgb.LGBMClassifier,
    X_sample: pd.DataFrame,
    values: pd.DataFrame,
    base_value: float,
) -> float:
    """Max absolute gap between the raw model score and base + sum of contributions.

    Should be close to zero (about 1e-6 or less); otherwise something is wrong.
    """
    raw = model.predict(X_sample, raw_score=True)
    rebuilt = values.sum(axis=1).to_numpy() + base_value
    return float(np.max(np.abs(raw - rebuilt)))


def top_numeric_features(
    values: pd.DataFrame, X_sample: pd.DataFrame, n: int = 4
) -> list[str]:
    """The n most important (by mean |SHAP|) numeric features."""
    ranked = values.abs().mean().sort_values(ascending=False).index
    numeric = [c for c in ranked if str(X_sample[c].dtype) != "category"]
    return numeric[:n]


def _save(fig: plt.Figure, path: str | Path | None) -> None:
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_importance(values: pd.DataFrame, top: int = 20, path=None) -> plt.Figure:
    """Bar chart of the mean absolute SHAP value per feature."""
    imp = values.abs().mean().sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * top + 1))
    ax.barh(imp.index, imp.to_numpy())
    ax.set_xlabel("mean |SHAP value| (log-odds)")
    ax.set_title(f"Top {top} features by mean |SHAP|")
    _save(fig, path)
    return fig


def plot_beeswarm(
    values: pd.DataFrame,
    X_sample: pd.DataFrame,
    top: int = 15,
    path=None,
    seed: int = 0,
) -> plt.Figure:
    """One row per feature: SHAP value of every sampled client, colored by the
    feature value (percentile; gray = missing)."""
    rng = np.random.default_rng(seed)
    order = values.abs().mean().sort_values(ascending=False).head(top).index[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.45 * top + 1.5))
    scatter = None
    for i, feat in enumerate(order):
        x = values[feat].to_numpy()
        col = X_sample[feat]
        if str(col.dtype) == "category":
            raw = col.cat.codes.to_numpy(dtype=float)
            raw[raw < 0] = np.nan  # code -1 means missing
        else:
            raw = col.to_numpy(dtype=float)
        pct = pd.Series(raw).rank(pct=True).to_numpy()  # NaN stays NaN
        y = i + rng.uniform(-0.3, 0.3, size=len(x))
        missing = np.isnan(pct)

        ax.scatter(x[missing], y[missing], s=4, color="lightgray", alpha=0.6,
                   rasterized=True)
        scatter = ax.scatter(x[~missing], y[~missing], c=pct[~missing], cmap="coolwarm",
                             s=4, alpha=0.6, rasterized=True)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("SHAP value (log-odds of default)")
    if scatter is not None:
        fig.colorbar(scatter, ax=ax).set_label("feature value (percentile)")
    _save(fig, path)
    return fig


def plot_dependence(
    values: pd.DataFrame, X_sample: pd.DataFrame, features: list[str], path=None
) -> plt.Figure:
    """Feature value vs its SHAP value, one panel per (numeric) feature."""
    n_cols = 2
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.5 * n_rows), squeeze=False)
    flat = axes.ravel()

    for ax, feat in zip(flat, features):
        ax.scatter(X_sample[feat].to_numpy(dtype=float), values[feat].to_numpy(),
                   s=3, alpha=0.4, rasterized=True)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
    for ax in flat[len(features):]:
        ax.set_visible(False)

    fig.tight_layout()
    _save(fig, path)
    return fig