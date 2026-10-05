# Home Credit Default Risk: credit scoring with LightGBM

A study project based on the Kaggle competition
[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk):
predict the probability that a loan applicant will have repayment difficulties.

**Status:** work in progress. Currently only `application_train.csv` is used;
the other six tables are the next step.

## Goal

Learn the practical tabular-ML workflow on a realistic credit-risk problem:
feature engineering, reliable cross-validation, gradient boosting and model
interpretation. The approach is built step by step, and every change is
measured against the previous version.

## Data

- Source: Kaggle competition data (not included in this repository, see
  "How to run").
- Main table: `application_train.csv`, about 307k applications, binary target
  `TARGET` (1 = payment difficulties). The classes are heavily imbalanced
  (roughly 8% positives).
- Seven related tables in total (bureau history, previous applications,
  instalments, card balances). Only the main table is used so far.

## Method

- **Model:** LightGBM (`LGBMClassifier`), native handling of categorical columns
  and missing values.
- **Validation:** stratified 5-fold CV with a fixed seed (the same folds in every
  experiment), early stopping on AUC with a patience of 100 rounds.
- **Metric:** ROC AUC (the competition metric). The reported numbers are the mean
  and standard deviation over the folds; OOF AUC is computed on the pooled
  out-of-fold predictions.
- **Preprocessing:** the placeholder value `365243` in `DAYS_EMPLOYED` is replaced
  with NaN; string columns are cast to `category`.

Main hyperparameters: `learning_rate=0.05`, `num_leaves=31`, `subsample=0.8`,
`colsample_bytree=0.8`, `reg_lambda=1.0`, up to 5000 trees (early stopping decides
the real number). No tuning has been done yet.

## Results

| # | Experiment | CV AUC (mean ± std) | OOF AUC |
|---|------------|---------------------|---------|
| 1 | Baseline: raw `application_train` columns only | 0.7593 ± 0.0046 | 0.7593 |
| 2 | + 5 ratio features (see below) | 0.7671 ± 0.0047 | 0.7671 |
| 3 | + EXT_SOURCE mean/min/max/std (cumulative) | 0.7672 ± 0.0039 | 0.7672 |
| 4 | + EXT_SOURCE product (cumulative) | 0.7668 ± 0.0042 | 0.7668 |
| 5 | + EXT_SOURCE weighted mean (cumulative) | 0.7673 ± 0.0041 | 0.7673 |
| 6 | + EXT_SOURCE missing count (cumulative) | 0.7669 ± 0.0043 | 0.7669 |

Rows 3-6 add feature groups one after another on top of experiment 2. None of them
gave a stable gain (mean differences between consecutive steps are between -0.0004
and +0.0005, within the noise), so all external-score combinations were discarded
and experiment 2 remains the working feature set.

The ratio features add about **+0.0078 CV AUC**. The fold-to-fold standard deviation
is about 0.005, but all experiments share the same folds, so the comparison is
paired. Differences below roughly 0.002-0.003 are treated as noise.

Ratio features used in experiment 2:

- `CREDIT_INCOME_RATIO`: credit amount / income
- `ANNUITY_INCOME_RATIO`: annuity / income
- `CREDIT_ANNUITY_RATIO`: credit amount / annuity (approximate loan term)
- `GOODS_PRICE_CREDIT_RATIO`: goods price / credit amount
- `EMPLOYED_BIRTH_RATIO`: days employed / age in days

## Observations

- The external scores `EXT_SOURCE_1/2/3` dominate the gain-based importance, with
  `EXT_SOURCE_3` and `EXT_SOURCE_2` clearly in front.
- `CREDIT_ANNUITY_RATIO` is the most important engineered feature and ranks above
  most raw columns.
- Without the ratio features, importance shifts to the raw columns they are built
  from (`AMT_CREDIT`, `AMT_GOODS_PRICE`, `AMT_ANNUITY`, `DAYS_EMPLOYED`). The trees
  partly approximate the missing ratios, which is why importance alone is not a
  reliable measure of a feature's contribution. Ablation (CV with and without a
  feature group) is used instead.
- Gain importance is biased towards high-cardinality categorical features such as
  `ORGANIZATION_TYPE`, so its high rank should be read with caution.
- Combinations of the external scores are a negative result: after adding them,
  `EXT_SOURCE_MEAN` became the most important feature by gain (and the importance of
  `EXT_SOURCE_3` dropped about fivefold), yet CV AUC did not improve. The trees simply
  switched to a new feature that carries the same information as the three originals.
  This is another example of why feature importance is not evidence of a real gain.

## How to run

1. Create an environment and install dependencies:
   ```bash
   pip install lightgbm scikit-learn pandas numpy
   ```
2. Download the competition data (requires a Kaggle account, an API token and
   accepting the competition rules):
   ```bash
   kaggle competitions download -c home-credit-default-risk
   ```
   Unzip it into `data/` (this folder is git-ignored).
3. Run `baseline.py` top to bottom. It is a plain Python script with `# %%` cells,
   so it also works as a notebook in VS Code or Jupyter.

## Roadmap

- [x] Combinations of the external scores (mean, min, max, std, product, weighted
      mean, missing count): no gain, discarded
- [ ] Aggregates from `bureau` and `bureau_balance`
- [ ] Aggregates from `previous_application`, `installments_payments`,
      `POS_CASH_balance` and `credit_card_balance`
- [ ] Hyperparameter tuning (Optuna)
- [ ] Model interpretation with SHAP
- [ ] Feature aggregation in SQL (DuckDB) as an alternative to Pandas
- [ ] Ablation study per feature group

## References

This project is a learning exercise. The following public work is used as study
material and is not my own:

- [LightGBM with Simple Features](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features) (Kaggle notebook)
- [NoxMoon/home-credit-default-risk](https://github.com/NoxMoon/home-credit-default-risk)
  (simplified version of a gold-medal solution)
- [1st place solution write-up](https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution)