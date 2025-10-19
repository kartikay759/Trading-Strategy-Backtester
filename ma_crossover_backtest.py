"""
ma_crossover_backtest.py
Simple moving-average crossover backtest (daily OHLC data via yfinance).
Outputs: performance metrics printed, equity curve plot saved as equity_curve.png.

Usage:
    python ma_crossover_backtest.py

Edit parameters in the CONFIG section below (ticker, start/end dates, fees, MA periods).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# -------- CONFIG --------
TICKER = "NSEI"           # example: "AAPL" or "RELIANCE.NS" for Indian Reliance (yfinance supports many symbols)
START = "2018-01-01"
END = datetime.today().strftime("%Y-%m-%d")
SHORT_WINDOW = 20         # short MA (days)
LONG_WINDOW = 50          # long MA (days)
INITIAL_CAPITAL = 100000  # starting capital in currency units
POSITION_SIZE = 1.0       # fraction of portfolio to use when entering trade (1.0 = all-in)
SLIPPAGE = 0.0005         # proportion slippage per trade (0.05%)
COMMISSION = 0.0005       # proportional commission per trade
RISK_FREE_RATE = 0.05     # annual risk free (for Sharpe)
# ------------------------

def fetch_data(ticker, start, end):
    print(f"Downloading {ticker} data from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError("No data downloaded. Check ticker symbol or internet connection.")
    df = df[['Open','High','Low','Close','Adj Close','Volume']].copy()
    return df

def prepare_signals(df):
    df['ma_short'] = df['Adj Close'].rolling(window=SHORT_WINDOW, min_periods=1).mean()
    df['ma_long']  = df['Adj Close'].rolling(window=LONG_WINDOW, min_periods=1).mean()
    # signal: 1 when short crosses above long, 0 otherwise (we will be long only)
    df['signal'] = 0
    df['signal'][SHORT_WINDOW:] = np.where(df['ma_short'][SHORT_WINDOW:] > df['ma_long'][SHORT_WINDOW:], 1, 0)
    df['signal'] = df['signal'].astype(int)
    # generate trade signals (entry=1 when rising edge; exit=-1 when falling edge)
    df['trade'] = df['signal'].diff().fillna(0).astype(int)
    return df

def backtest(df):
    cash = INITIAL_CAPITAL
    position = 0.0       # number of shares
    equity_curve = []
    last_trade_price = 0.0

    # We'll implement a simple "all-in" position size when signal==1, flat when signal==0.
    for idx, row in df.iterrows():
        price = row['Adj Close']
        signal = row['signal']
        # If enter signal and currently flat -> buy
        if signal == 1 and position == 0:
            # buy as many shares as POSITION_SIZE fraction allows
            budget = cash * POSITION_SIZE
            # account for commission & slippage on buy (proportional)
            effective_price = price * (1 + SLIPPAGE)
            qty = np.floor(budget / effective_price)
            if qty > 0:
                cost = qty * effective_price
                commission_cost = cost * COMMISSION
                cash -= (cost + commission_cost)
                position += qty
                last_trade_price = effective_price
        # If exit signal and currently long -> sell all
        elif signal == 0 and position > 0:
            effective_price = price * (1 - SLIPPAGE)
            proceeds = position * effective_price
            commission_cost = proceeds * COMMISSION
            cash += (proceeds - commission_cost)
            position = 0

        # compute total equity
        equity = cash + position * price
        equity_curve.append({'Date': idx, 'Equity': equity, 'Cash': cash, 'Position': position, 'Price': price})
    eq = pd.DataFrame(equity_curve).set_index('Date')
    return eq

def performance_metrics(eq):
    eq = eq.copy()
    eq['returns'] = eq['Equity'].pct_change().fillna(0)
    total_return = (eq['Equity'].iloc[-1] / eq['Equity'].iloc[0]) - 1
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 1/365.25
    cagr = (eq['Equity'].iloc[-1] / eq['Equity'].iloc[0]) ** (1/years) - 1
    # annualized volatility
    ann_vol = eq['returns'].std() * np.sqrt(252)
    # annualized Sharpe (uses returns - rf/252)
    excess_daily = eq['returns'] - (RISK_FREE_RATE / 252)
    sharpe = (excess_daily.mean() / eq['returns'].std()) * np.sqrt(252) if eq['returns'].std() != 0 else np.nan
    # max drawdown
    rolling_max = eq['Equity'].cummax()
    drawdown = (eq['Equity'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }

def plot_equity(eq, filename="equity_curve.png"):
    plt.figure(figsize=(10,6))
    plt.plot(eq.index, eq['Equity'])
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Equity curve saved to {filename}")
    plt.close()

def main():
    df = fetch_data(TICKER, START, END)
    df = prepare_signals(df)
    eq = backtest(df)
    metrics = performance_metrics(eq)
    print("\n=== Performance Metrics ===")
    for k,v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    plot_equity(eq, filename="equity_curve.png")
    # Save equity time series to CSV for analysis / presentation
    eq.to_csv("equity_timeseries.csv")
    print("Equity timeseries saved to equity_timeseries.csv")
    print("\nDone. Files produced: equity_curve.png, equity_timeseries.csv")

if _name_ == "_main_":
    main()
