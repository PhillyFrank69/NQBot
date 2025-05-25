"""
data/loader.py
Utility functions to load minute-bar files, concatenate tick files, resample
ticks to 1-minute, and return a merged DataFrame ready for feature engineering.
"""

import os
import glob
import pandas as pd
from typing import Tuple, Optional


# --------------------------------------------------------------------- #
# 1. Minute bars                                                        #
# --------------------------------------------------------------------- #
def load_minute_bars(path: str) -> pd.DataFrame:
    """
    Load a feather file of cleaned 1-minute bars.
    Expects columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume', ...]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    bars = pd.read_feather(path)
    if 'timestamp' in bars.columns:
        bars.set_index('timestamp', inplace=True)
    bars.index = pd.to_datetime(bars.index)
    return bars


# --------------------------------------------------------------------- #
# 2. Tick data helpers                                                  #
# --------------------------------------------------------------------- #
_PRICE_CANDIDATES  = ("price", "last", "close")
_VOLUME_CANDIDATES = ("volume", "size", "vol")


def _detect_price_volume(df_tick: pd.DataFrame) -> Tuple[str, str]:
    price_col = next((c for c in df_tick.columns
                      if c.lower() in _PRICE_CANDIDATES), None)
    vol_col   = next((c for c in df_tick.columns
                      if c.lower() in _VOLUME_CANDIDATES), None)
    if price_col is None or vol_col is None:
        raise ValueError("Could not detect price/volume columns "
                         f"in tick file. Columns = {df_tick.columns.tolist()}")
    return price_col, vol_col


def concat_tick_feathers(dir_path: str) -> Optional[pd.DataFrame]:
    """
    Concatenate all feather files inside `dir_path`.
    Returns None if directory doesn't exist or is empty.
    """
    if not os.path.isdir(dir_path):
        return None
    files = glob.glob(os.path.join(dir_path, "*.feather"))
    if not files:
        return None

    parts = []
    for f in files:
        tmp = pd.read_feather(f)
        if 'timestamp' in tmp.columns:
            tmp.set_index('timestamp', inplace=True)
        tmp.index = pd.to_datetime(tmp.index)
        parts.append(tmp)

    ticks = pd.concat(parts).sort_index()
    return ticks


def resample_ticks_1min(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with 1-minute VWAP and volume columns from raw ticks.
    """
    price_col, vol_col = _detect_price_volume(ticks)

    pv  = (ticks[price_col] * ticks[vol_col]).resample("1min").sum()
    vol = ticks[vol_col].resample("1min").sum()
    out = pd.DataFrame({
        "vwap_tick": pv / vol,
        "volume_tick": vol
    }).dropna()
    return out


# --------------------------------------------------------------------- #
# 3. High-level convenience                                             #
# --------------------------------------------------------------------- #
def load_merged_dataset(
    bar_file: str,
    tick_dir: str = "",
    join_how: str = "inner"
) -> pd.DataFrame:
    """
    * Load minute bars.
    * Optionally load tick files, resample to 1-minute, merge.
    * Returns a DataFrame indexed by timestamp.
    """
    bars  = load_minute_bars(bar_file)
    ticks = concat_tick_feathers(tick_dir)

    if ticks is not None:
        ticks_1m = resample_ticks_1min(ticks)
        merged   = bars.join(ticks_1m, how=join_how)
    else:
        merged = bars

    return merged
