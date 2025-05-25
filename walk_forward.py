"""
walk_forward.py
===============
Rolling / expanding walk‑forward validation for time‑series ML models.

Why a separate module?
----------------------
* Keeps evaluation logic independent of `train_model.py`.
* Lets you plug in **any** sklearn‑compatible estimator (RF, XGB, etc.) or a
  custom `fit_fn` and quickly measure out‑of‑sample performance over many
  windows.

Typical usage
-------------
```python
from walk_forward import walk_forward_cv, equity_curve
from models.xgb_head import fit_xgb

models, metrics = walk_forward_cv(
    X, y,
    window_train=90_000,   # 90k minutes ≈ 6 weeks
    window_test =15_000,   # 15k minutes ≈ 1 week
    fit_fn      = fit_xgb,
    fit_kwargs  = dict(n_iter=10)
)
print(metrics.tail())
```
"""

from __future__ import annotations
from typing import Callable, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------------------------------------------ #
# Generator for rolling windows                                     #
# ------------------------------------------------------------------ #

def rolling_windows(n: int,
                    window_train: int,
                    window_test: int,
                    step: int | None = None):
    """Yield (train_idx, test_idx) slices for walk‑forward CV."""
    if step is None:
        step = window_test  # non‑overlapping test segments by default
    start = 0
    while start + window_train + window_test <= n:
        train_slice = slice(start, start + window_train)
        test_slice  = slice(start + window_train, start + window_train + window_test)
        yield train_slice, test_slice
        start += step

# ------------------------------------------------------------------ #
# Walk‑forward evaluation                                            #
# ------------------------------------------------------------------ #

def walk_forward_cv(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    window_train: int,
    window_test: int,
    fit_fn: Callable[..., Tuple[Any, Any]],
    fit_kwargs: dict | None = None,
    metrics: List[Callable[[np.ndarray, np.ndarray], float]] | None = None,
) -> Tuple[List[Any], pd.DataFrame]:
    """Perform walk‑forward CV.

    Parameters
    ----------
    X, y          : full feature matrix and labels, chronological order.
    window_train  : length of each expanding train window (bars).
    window_test   : length of each test window.
    fit_fn        : function that takes (X_train, y_train, **fit_kwargs)
                    and returns a fitted model (predict method required).
                    Example: `lambda X,y: RandomForestClassifier().fit(X,y)` or
                    the `fit_xgb` wrapper (discard second return value).
    fit_kwargs    : optional dict forwarded to `fit_fn`.
    metrics       : list of metric callables; default = [accuracy, f1_macro].

    Returns
    -------
    models  : list of fitted models, one per window.
    summary : DataFrame with per‑window metric scores and date range.
    """
    if fit_kwargs is None:
        fit_kwargs = {}
    if metrics is None:
        metrics = [accuracy_score, lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")]

    models   = []
    records  = []
    total_n  = len(X)

    for i, (tr, te) in enumerate(rolling_windows(total_n, window_train, window_test)):
        X_train, y_train = X[tr], y[tr]
        X_test,  y_test  = X[te], y[te]

        model_and_extra = fit_fn(X_train, y_train, **fit_kwargs)
        model = model_and_extra[0] if isinstance(model_and_extra, tuple) else model_and_extra
        models.append(model)

        y_pred = model.predict(X_test)
        row = {f.__name__: f(y_test, y_pred) for f in metrics}
        row.update({"window": i, "train_end": tr.stop, "test_end": te.stop})
        records.append(row)
        print(f"Window {i}:", row)

    summary = pd.DataFrame(records)
    return models, summary

# ------------------------------------------------------------------ #
# Simple P&L aggregation example                                     #
# ------------------------------------------------------------------ #

def equity_curve(pnl_series: List[pd.Series]) -> pd.Series:
    """Concatenate per‑window trade_pnl Series into a cumulative equity curve."""
    equity = pd.concat(pnl_series).cumsum()
    return equity
