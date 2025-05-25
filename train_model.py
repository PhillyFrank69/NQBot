"""
train_model.py
End-to-end pipeline for NQ futures scalping model (LightGBM baseline).

Usage:
    python train_model.py --config configs/config.yaml
"""

import os
import glob
import yaml
import argparse
import joblib
import polars as pl
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta


# ----------------------------------------------------------------------
# 0. Utilities
# ----------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def minute_loader(bar_path: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(bar_path)
        .with_columns(pl.col("timestamp").cast(pl.Datetime).alias("ts"))
        .with_columns(pl.col("ts").dt.set_time_unit("us"))
        .collect()
        .sort("ts")
        .filter(~pl.any_horizontal(pl.col(["open", "high", "low", "close"]).is_null()))
    )


def tick_loader(tick_dir: str) -> pl.DataFrame | None:
    files = glob.glob(os.path.join(tick_dir, "*.parquet"))
    if not files:
        return None
    df = (
        pl.scan_parquet(files)
        .with_columns(pl.col("timestamp").cast(pl.Datetime).alias("ts"))
        .with_columns(pl.col("ts").dt.set_time_unit("us"))
        .collect()
        .sort("ts")
    )
    return df


# ----------------------------------------------------------------------
# 1. Feature Engineering
# ----------------------------------------------------------------------


def engineering_minute(df: pl.DataFrame) -> pl.DataFrame:
    """
    Classic technical & price-action indicators on minute bars
    """
    df = df.sort("ts").with_columns(
        [
            # returns
            (pl.col("close").pct_change(1)).alias("ret1"),
            (pl.col("close").pct_change(5)).alias("ret5"),
            # fast stochastic 9/3
            (
                (pl.col("close") - pl.col("low").rolling_min(9))
                / (pl.col("high").rolling_max(9) - pl.col("low").rolling_min(9))
                * 100
            ).alias("stoch_k"),
        ]
    )
    df = df.with_columns(pl.col("stoch_k").rolling_mean(3).alias("stoch_d"))
    # Z-score normalisation in 30-day rolling window
    for c in ["ret1", "ret5", "stoch_k", "stoch_d"]:
        df = df.with_columns(
            (
                (pl.col(c) - pl.col(c).rolling_mean(7_200))
                / (pl.col(c).rolling_std(7_200) + 1e-9)
            ).alias(f"{c}_z")
        )
    return df


def engineering_tick(df_ticks: pl.DataFrame) -> pl.DataFrame | None:
    """
    Resample ticks to 1-second to derive order-flow features.
    Returns a minute-level Polars frame to join with bar data.
    """
    if df_ticks is None:
        return None
    # Identify price & volume columns heuristically
    price_col = next(
        (c for c in df_ticks.columns if c.lower() in {"price", "last", "close"}), None
    )
    vol_col = next((c for c in df_ticks.columns if c.lower() in {"volume", "size"}), None)

    if price_col is None or vol_col is None:
        return None

    q = (
        df_ticks.lazy()
        .groupby_dynamic(index_column="ts", every="1s", maintain_order=True)
        .agg(
            [
                pl.col(price_col).last().alias("px_last"),
                pl.col(vol_col).sum().alias("vol_sum"),
            ]
        )
    ).collect()

    # VWAP per second
    q = q.with_columns((pl.col("px_last") * pl.col("vol_sum")).alias("dollar"))
    vwap_sec = (
        q.lazy()
        .groupby_dynamic(index_column="ts", every="1m")
        .agg(
            [
                (pl.col("dollar").sum() / pl.col("vol_sum").sum()).alias("vwap_tick"),
                pl.col("vol_sum").sum().alias("vol_tick"),
            ]
        )
        .collect()
    )
    return vwap_sec


def join_features(bars: pl.DataFrame, tick_feats: pl.DataFrame | None) -> pl.DataFrame:
    if tick_feats is not None:
        df = bars.join(tick_feats, on="ts", how="left")
    else:
        df = bars
    df = engineering_minute(df)
    return df.drop_nulls()


# ----------------------------------------------------------------------
# 2. Label Creation
# ----------------------------------------------------------------------


def create_labels(df: pl.DataFrame, horizon: int = 3, thresh: float = 0.00035) -> pl.DataFrame:
    """
    Horizon = 3 min, threshold ≈ 0.035 % (3.5 bp). Adjust for tighter scalps if needed.
    """
    fwd = (df["close"].shift(-horizon) / df["close"] - 1).alias("fwd_ret")
    df = df.with_columns(fwd)
    df = df.with_columns(
        pl.when(pl.col("fwd_ret") > thresh)
        .then(1)
        .when(pl.col("fwd_ret") < -thresh)
        .then(-1)
        .otherwise(0)
        .alias("label")
    )
    return df[:-horizon]  # trim tail with NaNs


# ----------------------------------------------------------------------
# 3. Model Factory & Training
# ----------------------------------------------------------------------


def model_factory(name: str, random_state: int = 42):
    if name.lower() == "lgbm":
        return LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.7,
            colsample_bytree=0.7,
            objective="multiclass",
            random_state=random_state,
        )
    raise ValueError(f"Unknown model {name}")


def walk_forward_cv(df, n_splits: int = 5, embargo: int = 30):
    """
    Generator yielding train/val indices using expanding window +
    purging/embargo as described by López de Prado.
    `embargo` in minutes.
    """
    ts = df["ts"].to_numpy()
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(ts):
        # embargo
        max_train_ts = ts[train_idx].max()
        purged_test = test_idx[ts[test_idx] >= max_train_ts + np.timedelta64(embargo, "m")]
        yield train_idx, purged_test


def train(df: pl.DataFrame, cfg: dict):
    features = [c for c in df.columns if c.endswith("_z") or c in {"vwap_tick", "vol_tick"}]
    X = df[features].to_numpy()
    y = df["label"].to_numpy()

    cv_scores = []
    model = model_factory(cfg["model"])
    for fold, (tr, te) in enumerate(
        walk_forward_cv(df, n_splits=cfg["cv"]["n_splits"], embargo=cfg["cv"]["embargo"])
    ):
        model.fit(X[tr], y[tr])
        y_pred = model.predict(X[te])
        bal_acc = balanced_accuracy_score(y[te], y_pred)
        cv_scores.append(bal_acc)
        print(f"Fold {fold+1}: balanced_acc = {bal_acc:.4f}")
    print("Mean CV balanced_acc:", np.mean(cv_scores))
    model.fit(X, y)  # retrain on full set
    return model, features, cv_scores


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------


def main(cfg_path: str):
    cfg = load_config(cfg_path)

    bars = minute_loader(cfg["paths"]["bars"])
    ticks = tick_loader(cfg["paths"]["ticks"])
    df = join_features(bars, engineering_tick(ticks))
    df = create_labels(df, horizon=cfg["label"]["horizon"], thresh=cfg["label"]["threshold"])

    model, feat_cols, scores = train(df, cfg)

    # Persist artefacts
    os.makedirs(cfg["paths"]["out"], exist_ok=True)
    joblib.dump(
        {"model": model, "features": feat_cols, "cv_scores": scores},
        os.path.join(cfg["paths"]["out"], "lightgbm_scalper.pkl"),
    )
    print(f"Model saved to {cfg['paths']['out']}/lightgbm_scalper.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
