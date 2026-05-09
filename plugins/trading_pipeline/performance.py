import pandas as pd
import numpy as np

def compute_metrics(trades_file_path):
    """
    Calculate performance metrics from the trades log CSV.
    The CSV has columns: Date, Symbol, Order, Price, Target_Position
    """
    df = pd.read_csv(trades_file_path)
    if len(df) < 2:
        return {"Cumulative_Return": 0.0, "Sharpe_Ratio": 0.0, "Mean_Daily_Return": 0.0}
        
    # To compute accurate daily returns, we need the daily portfolio value.
    # Since the trades log only contains days a trade happened, we approximate
    # by just looking at the return between trade dates.
    
    # Sort by date just in case
    df = df.sort_values('Date')
    
    # Simplified calculation based on trade prices
    # (A full implementation would pull daily prices from yfinance and compute
    # daily portfolio value like marketsimcode.py does)
    
    initial_value = 100000.0
    cash = initial_value
    shares = 0
    
    port_vals = []
    
    for _, row in df.iterrows():
        order = row['Order']
        price = row['Price']
        
        # Deduct cash for buying, add for selling
        # No commission or impact in this simplified version
        cash -= (order * price)
        shares += order
        
        # Portfolio value on this day
        val = cash + (shares * price)
        port_vals.append(val)
        
    port_series = pd.Series(port_vals)
    daily_returns = (port_series / port_series.shift(1)) - 1
    
    cum_return = (port_series.iloc[-1] / initial_value) - 1
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    
    # Annualized Sharpe (assuming 252 trading days)
    if std_ret > 0:
        sharpe = np.sqrt(252) * (mean_ret / std_ret)
    else:
        sharpe = 0.0
        
    return {
        "Cumulative_Return": round(float(cum_return), 6),
        "Sharpe_Ratio": round(float(sharpe), 6),
        "Mean_Daily_Return": round(float(mean_ret), 6)
    }
