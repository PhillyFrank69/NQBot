"""
features/microstructure.py
Tick-derived microstructure & order-flow features.
Assumes the DataFrame already contains 1-minute columns
  - vwap_tick
  - volume_tick
"""

from __future__ import annotations
import pandas as pd
import numpy as np

VWAP  = "vwap_tick"
VOL   = "volume_tick"


def add_imbalance(df: pd.DataFrame,
                  window: int = 10) -> pd.DataFrame:
    """
    Rolling buy-vs-sell imbalance proxy:
    positive when price closes above rolling VWAP.
    """
    df["imbalance"] = (df["close"] - df[VWAP]) / df[VWAP]
    df[f"imbalance_{window}"] = df["imbalance"].rolling(window).mean()
    return df


def add_time_of_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode time-of-day as sine/cosine features to capture intraday seasonality.
    """
    minutes  = df.index.hour * 60 + df.index.minute
    day_len  = 24 * 60
    df["tod_sin"] = np.sin(2 * np.pi * minutes / day_len)
    df["tod_cos"] = np.cos(2 * np.pi * minutes / day_len)
    return df


# convenience
DEFAULT_MICROSTRUCTURE_FEATURES = ["imbalance_10", "tod_sin", "tod_cos"]


def add_core_micro(df: pd.DataFrame) -> pd.DataFrame:
    return add_time_of_day(add_imbalance(df))
