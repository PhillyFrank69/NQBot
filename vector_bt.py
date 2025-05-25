"""
vector_bt.py
============
A lightning‑fast, **vectorised** back‑tester for minute‑bar futures strategies
based on label or probability outputs from your ML models.

Key design goals
----------------
* **No loops** over bars – pure pandas / NumPy vector ops.
* **Works on signals, not prices** – you feed a column of +1 / -1 / 0 (or
  probabilities) and the engine synthesises entry/exit, P&L, and metrics.
* **Minute‑resolution slippage & commission** built‑in.

Typical usage
-------------
```python
import pandas as pd
from vector_bt import backtest_signals, basic_stats

bars = pd.read_feather("minute_bars.feather").set_index("timestamp")
signals = model.predict(bars[FEATURES])   # +1 / 0 / -1

results = backtest_signals(bars, signals,
                           fee_per_contract=2.04,  # round‑turn
                           slippage_ticks=1,        # 1 tick each side
                           tick_value=5.0,          # NQ tick value
                           tick_size=0.25)          # NQ tick size
print(basic_stats(results))
```
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict

# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #

def _calc_trade_returns(close: pd.Series,
                        position: pd.Series,
                        fee: float,
                        slip_per_side: float) -> pd.Series:
    """Compute trade‑by‑trade returns net of fee & slippage."""
    # position diff identifies entry (+1/-1) and exit
    entry_mask = position.diff().fillna(position).abs() == 1
    exit_mask  = position.diff().abs() == 1

    entry_price = close[entry_mask]
    exit_price  = close[exit_mask]

    # Align by index – each exit matches MOST RECENT entry
    exit_price = exit_price.reindex_like(entry_price, method="bfill")
    direction  = position[entry_mask]    # +1 long, -1 short

    raw_ret = direction * (exit_price.values - entry_price.values)
    gross_ret = raw_ret - 2 * slip_per_side * direction.abs()  # slip both legs
    net_ret   = gross_ret - fee

    pnl = pd.Series(net_ret, index=entry_price.index)
    return pnl


# ------------------------------------------------------------------ #
# Core back‑test                                                     #
# ------------------------------------------------------------------ #

def backtest_signals(
    bars: pd.DataFrame,
    signals: pd.Series,
    fee_per_contract: float = 2.04,
    slippage_ticks: float = 1.0,
    tick_value: float = 5.0,
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """Vectorised long/short back‑test.

    Parameters
    ----------
    bars : DataFrame with at least a 'close' column, indexed by datetime.
    signals : Series of +1 (long), -1 (short), 0 (flat) aligned to bars.
    fee_per_contract : Commission & exchange fees, round‑turn.
    slippage_ticks : Slippage each side (entry + exit) in ticks.
    tick_value : $ value of one tick.
    tick_size : Price move representing one tick.

    Returns
    -------
    DataFrame with columns [position, pnl, equity]. P&L in $ per contract.
    """
    signals = signals.reindex(bars.index).fillna(0).astype(int)

    # Forward‑fill to create holding position until signal changes
    position = signals.replace(0, np.nan).ffill().fillna(0).astype(int)

    slip_dollar = slippage_ticks * tick_value

    trade_pnl = _calc_trade_returns(
        bars['close'], position,
        fee=fee_per_contract,
        slip_per_side=slip_dollar)

    equity = trade_pnl.cumsum()

    out = pd.DataFrame({
        'position': position,
        'trade_pnl': trade_pnl,
        'equity': equity
    })
    return out


# ------------------------------------------------------------------ #
# Performance metrics                                                #
# ------------------------------------------------------------------ #

def basic_stats(bt: pd.DataFrame) -> Dict[str, float]:
    """Return basic equity‑curve stats."""
    trade_pnl = bt['trade_pnl'].dropna()
    equity    = bt['equity'].fillna(method='ffill').fillna(0)

    total   = trade_pnl.sum()
    trades  = len(trade_pnl)
    winrate = (trade_pnl > 0).mean()
    max_dd  = (equity.cummax() - equity).max()
    ret_std = trade_pnl.std() if trades else np.nan
    sharpe  = (trade_pnl.mean() / ret_std * np.sqrt(252*390)) if ret_std else np.nan

    return {
        'total_pnl': total,
        'num_trades': trades,
        'win_rate': winrate,
        'max_drawdown': max_dd,
        'sharpe_like': sharpe,
    }


# ------------------------------------------------------------------ #
# Quick CLI for experimentation                                      #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run a quick vectorised back‑test.")
    p.add_argument("bars",   help="Minute‑bar feather file")
    p.add_argument("signals",help="CSV with timestamp,signal (aligned w/ bars)")
    args = p.parse_args()

    bars    = pd.read_feather(args.bars).set_index('timestamp')
    signals = pd.read_csv(args.signals, index_col='timestamp').squeeze('columns')
    res = backtest_signals(bars, signals)
    print(basic_stats(res))
