import numpy as np
import pandas as pd
import yfinance as yf

# List of tickers (you can add more as necessary)
sp500_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Fetch stock data
def download_data(tickers):
    try:
        data = yf.download(tickers, start="2022-01-01", end="2023-01-01", group_by='ticker')
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

stock_data = download_data(sp500_tickers)

# Function to calculate indicators
def calculate_indicators(df):
    print(f"DataFrame columns for {df.name} before calculation: {df.columns.tolist()}")
    
    if 'Close' not in df.columns:
        print(f"'Close' column not found for {df.name}. Available columns: {df.columns.tolist()}")
        return df

    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_100'] = df['Close'].rolling(window=100).mean()

    df['BB_upper'] = df['MA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['MA_20'] - 2 * df['Close'].rolling(window=20).std()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# Backtest function with stop-loss and take-profit
def backtest_mean_reversion(df, stock_ticker):
    initial_balance = 10000
    balance = initial_balance
    position = 0
    stop_loss = 0.98  # 2% loss
    take_profit = 1.05  # 5% gain
    buy_price = 0

    for i in range(1, len(df)):
        if position == 0:
            # Buy signal (mean reversion): Price below 5-day MA and RSI < 30
            if df['Close'].iloc[i] < df['MA_5'].iloc[i] and df['RSI'].iloc[i] < 30:
                position = balance / df['Close'].iloc[i]  # Buy position
                balance = 0
                buy_price = df['Close'].iloc[i]
        
        elif position > 0:
            current_price = df['Close'].iloc[i]
            # Sell signal: if price hits take-profit or stop-loss
            if current_price >= buy_price * take_profit or current_price <= buy_price * stop_loss:
                balance = position * current_price  # Sell position
                position = 0  # Reset position
    
    # If still holding position, sell at the last close price
    if position > 0:
        balance = position * df['Close'].iloc[-1]
    
    return balance

# Running the backtest on each stock
final_results = {}

for ticker in sp500_tickers:
    try:
        stock_df = stock_data[ticker].copy()
        stock_df.name = ticker
        stock_df = calculate_indicators(stock_df)
        final_balance = backtest_mean_reversion(stock_df, ticker)
        final_results[ticker] = final_balance
        print(f"{ticker}: Final balance = {final_balance}")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Display final results
print("Backtest Results for Each Stock:")
for stock, balance in final_results.items():
    print(f"{stock}: {balance}")
