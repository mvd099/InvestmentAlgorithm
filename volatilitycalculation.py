import pandas as pd
import numpy as np
import yfinance as yf

# Download historical data for AAPL
start_date = "2022-01-01"  # replace with your start date
end_date = "2023-01-01"   # replace with your end date
data = yf.download('AAPL', start=start_date, end=end_date)

# Calculate daily returns
data['Daily Return'] = data['Adj Close'].pct_change()

# Calculate daily volatility (standard deviation of daily returns)
daily_volatility = data['Daily Return'].std()

# Annualize the volatility
annualized_volatility = daily_volatility * np.sqrt(252)

print(f'Daily Volatility: {daily_volatility:.2%}')
print(f'Annualized Volatility: {annualized_volatility:.2%}')
