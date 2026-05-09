import pandas as pd
import numpy as np

def bollinger_percent_b(prices, window=20):
    """Calculate Bollinger %B."""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    
    # Avoid division by zero
    diff = upper_band - lower_band
    diff[diff == 0] = np.nan
    
    percent_b = (prices - lower_band) / diff
    return percent_b

def rsi(prices, window=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Avoid division by zero
    avg_loss[avg_loss == 0] = np.nan
    
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    
    # Fill cases where loss is 0
    rsi_val[avg_loss.isna() & avg_gain.notna()] = 100
    
    return rsi_val

def momentum(prices, window=10):
    """Calculate Momentum."""
    return (prices / prices.shift(window)) - 1

def add_indicators(df):
    """Add BB%B, RSI, and Momentum to a dataframe with a 'Close' column."""
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain a 'Close' column")
        
    prices = df['Close']
    
    df['BB_PB'] = bollinger_percent_b(prices, window=20)
    df['RSI'] = rsi(prices, window=14)
    df['Momentum'] = momentum(prices, window=10)
    
    return df.dropna()
