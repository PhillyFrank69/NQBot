"""
models/xgb_head.py
------------------
A lightweight wrapper around XGBoost’s scikit‑learn interface that fits an
XGBClassifier on the pre‑engineered feature matrix and returns:
  • the fitted model
  • a cv object with the best score / parameters

Why a separate module?
  • Keeps tree‑based model logic separate from deep‑learning models.
  • Enables fast iterations without touching the high‑level pipeline.

Dependencies: xgboost>=2.0.0, scikit‑learn.  Install via:
    pip install xgboost scikit‑learn

Usage (inside train_model.py):
    from models.xgb_head import fit_xgb, DEFAULT_PARAMS
    model, cv = fit_xgb(X, y, param_grid=DEFAULT_PARAMS)
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBClassifier

# ------------------------------------------------------------------ #
# Default hyper‑parameter search space                               #
# ------------------------------------------------------------------ #
DEFAULT_PARAMS: Dict[str, list] = {
    "n_estimators":    [200, 400, 600],
    "max_depth":       [3, 5, 7],
    "learning_rate":   [0.03, 0.05, 0.1],
    "subsample":       [0.7, 0.9, 1.0],
    "colsample_bytree":[0.7, 0.9, 1.0],
    "gamma":           [0, 1, 5],
}


# ------------------------------------------------------------------ #
# Fit helper                                                         #
# ------------------------------------------------------------------ #

def fit_xgb(
    X,
    y,
    param_grid: Optional[Dict[str, list]] = None,
    n_iter: int = 20,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """Time‑series aware hyper‑parameter search for XGBClassifier."""
    if param_grid is None:
        param_grid = DEFAULT_PARAMS

    base = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        num_class=3,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        verbose=2,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    print("⏳ Fitting XGBoost – this may take a while…")
    search.fit(X, y)
    print("✅ XGBoost done. Best params:", search.best_params_)
    return search.best_estimator_, search
