# indicators.py

import pandas as pd
import numpy as np

# At top of indicators.py, after imports
COL_OPEN, COL_HIGH, COL_LOW, COL_CLOSE, COL_VOL = \
    "open", "high", "low", "close", "volume"

def add_sma(df, period=20, col=COL_CLOSE):
    ...
def add_ema(df, period=20, col=COL_CLOSE):
    ...
# and so on—replace 'Close','High','Low','Volume' references with constants


def add_sma(df, period=20, col='Close'):
    df[f'SMA_{period}'] = df[col].rolling(window=period, min_periods=1).mean()
    return df

def add_ema(df, period=20, col='Close'):
    df[f'EMA_{period}'] = df[col].ewm(span=period, adjust=False).mean()
    return df

def add_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df[f'ATR_{period}'] = true_range.rolling(window=period, min_periods=1).mean()
    return df

def add_bollinger_bands(df, period=20, num_std=2, col='Close'):
    sma = df[col].rolling(window=period, min_periods=1).mean()
    std = df[col].rolling(window=period, min_periods=1).std()
    df[f'BB_upper_{period}'] = sma + num_std * std
    df[f'BB_lower_{period}'] = sma - num_std * std
    df[f'BB_width_{period}'] = df[f'BB_upper_{period}'] - df[f'BB_lower_{period}']
    df[f'BB_percent_b_{period}'] = (df[col] - df[f'BB_lower_{period}']) / (df[f'BB_upper_{period}'] - df[f'BB_lower_{period}'])
    return df

def add_rsi(df, period=14, col='Close'):
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

def add_stochastic(df, k_period=14, d_period=3):
    low_min = df['Low'].rolling(window=k_period, min_periods=1).min()
    high_max = df['High'].rolling(window=k_period, min_periods=1).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period, min_periods=1).mean()
    return df

def add_macd(df, fast=12, slow=26, signal=9, col='Close'):
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

def add_vwap(df):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP'] = vwap
    df['Dist_Close_VWAP'] = df['Close'] - df['VWAP']
    return df

def add_adx(df, period=14):
    # True Range
    df['TR'] = np.maximum.reduce([
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close'].shift()),
        np.abs(df['Low'] - df['Close'].shift())
    ])
    # Directional Movement
    df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']),
                         np.maximum(df['High'] - df['High'].shift(), 0), 0)
    df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()),
                         np.maximum(df['Low'].shift() - df['Low'], 0), 0)
    # Smooth
    tr_smooth = df['TR'].rolling(window=period, min_periods=1).mean()
    plus_dm_smooth = df['+DM'].rolling(window=period, min_periods=1).mean()
    minus_dm_smooth = df['-DM'].rolling(window=period, min_periods=1).mean()
    plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-9))
    minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-9))
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    df[f'ADX_{period}'] = dx.rolling(window=period, min_periods=1).mean()
    df.drop(['TR', '+DM', '-DM'], axis=1, inplace=True)
    return df

def add_zscore(df, period=20, col='Close'):
    mean = df[col].rolling(window=period, min_periods=1).mean()
    std = df[col].rolling(window=period, min_periods=1).std()
    df[f'ZScore_{col}_{period}'] = (df[col] - mean) / (std + 1e-9)
    return df

# Example usage: chaining indicators
def add_all_indicators(df):
    df = add_sma(df, 20)
    df = add_ema(df, 20)
    df = add_atr(df, 14)
    df = add_bollinger_bands(df, 20)
    df = add_rsi(df, 14)
    df = add_stochastic(df, 14, 3)
    df = add_macd(df)
    df = add_vwap(df)
    df = add_adx(df, 14)
    df = add_zscore(df, 20)
    return df

# If you want to test the script:
if __name__ == '__main__':
    # Load example data
    df = pd.read_csv('your_ohlcv_data.csv', parse_dates=True, index_col=0)
    df = add_all_indicators(df)
    print(df.tail())
