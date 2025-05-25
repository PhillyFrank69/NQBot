"""
pipeline.py
===========
End‑to‑end orchestration script for the NQBot ML pipeline.  It wires together
loader, feature builders, model trainers, walk‑forward validation, and optional
vectorised back‑testing – all from simple command‑line flags.

Example
-------
Train XGBoost with walk‑forward CV on 6‑week / 1‑week windows and output
metrics:

```
conda activate nqbot
python pipeline.py --model xgb --wtrain 90000 --wtest 15000 --out metrics.csv
```
"""
from __future__ import annotations

import os
import argparse
from typing import List

import pandas as pd

from data.loader import load_merged_dataset
from features.indicators import add_core_indicators
from features.microstructure import add_core_micro
from walk_forward import walk_forward_cv
from models.xgb_head import fit_xgb

# you may add more models later
MODEL_REGISTRY = {
    "xgb": fit_xgb,               # returns (model, search) tuple
}

# ------------------------------------------------------------------ #
# Configuration paths                                                #
# ------------------------------------------------------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BAR_FILE = os.path.join(BASE_DIR, "..", "data", "bars", "NQ_minute_clean_1min_bars.feather")
TICK_DIR = os.path.join(BASE_DIR, "..", "data", "ticks")

# Features list (update when you add new ones)
FEATURES: List[str] = [
    "vwap", "stoch_k", "stoch_d", "rsi_14", "ema_20", "atr_14",
    "imbalance_10", "tod_sin", "tod_cos", "ret_1", "ret_5",
]

# ------------------------------------------------------------------ #
# Feature engineering                                                #
# ------------------------------------------------------------------ #
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered columns and return df.dropna()."""
    df = add_core_indicators(df)
    # microstructure requires vwap_tick/volume_tick columns
    if {"vwap_tick", "volume_tick"}.issubset(df.columns):
        df = add_core_micro(df)
        df["vwap"] = df["vwap_tick"]
    else:
        df["vwap"] = df["close"]

    # classic returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    return df.dropna()

# ------------------------------------------------------------------ #
# Label creation                                                     #
# ------------------------------------------------------------------ #

def add_labels(df: pd.DataFrame, horizon: int = 5, thresh: float = 0.0005) -> pd.DataFrame:
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = 0
    df.loc[df["future_return"] >  thresh, "label"] =  1
    df.loc[df["future_return"] < -thresh, "label"] = -1
    return df.iloc[:-horizon]

# ------------------------------------------------------------------ #
# Main entry                                                         #
# ------------------------------------------------------------------ #

def main():
    p = argparse.ArgumentParser(description="NQBot walk‑forward pipeline")
    p.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="xgb",
                   help="Which estimator to train")
    p.add_argument("--wtrain", type=int, default=90_000,
                   help="Train window length (bars)")
    p.add_argument("--wtest",  type=int, default=15_000,
                   help="Test window length (bars)")
    p.add_argument("--niter",  type=int, default=20,
                   help="Randomised search iterations (model‑specific)")
    p.add_argument("--out",    type=str,  default="metrics.csv",
                   help="CSV file to save per‑window metrics")
    args = p.parse_args()

    # 1. Load & merge bars/ticks
    df = load_merged_dataset(BAR_FILE, TICK_DIR)

    # 2. Features / labels
    df = build_feature_matrix(df)
    df = add_labels(df)

    X = df[FEATURES].values
    y = df["label"].values

    # 3. Select fit function from registry
    fit_fn = MODEL_REGISTRY[args.model]
    fit_kwargs = dict(n_iter=args.niter) if args.model == "xgb" else {}

    # 4. Walk‑forward CV
    _, metrics = walk_forward_cv(
        X, y,
        window_train=args.wtrain,
        window_test=args.wtest,
        fit_fn=fit_fn,
        fit_kwargs=fit_kwargs,
    )

    # 5. Save metrics
    metrics.to_csv(args.out, index=False)
    print(f"\n💾 Metrics saved to {args.out}\n")


if __name__ == "__main__":
    main()
