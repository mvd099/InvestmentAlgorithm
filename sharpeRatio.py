import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

starting_amount = 10000  # Starting amount of money

# Step 1: Fetch historical price data for backtesting (5-year period)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Example stocks
data = yf.download(tickers, start='2018-01-01', end='2023-01-01')['Adj Close']

# Step 2: Calculate daily returns
returns = data.pct_change().dropna()

# Step 3: Calculate expected returns and covariance matrix (annualized)
expected_returns = returns.mean() * 252  # Annualized returns
cov_matrix = returns.cov() * 252  # Annualized covariance

# Risk-free rate (e.g., 4.08%)
risk_free_rate = 0.0408 

# Step 4: Define the Sharpe Ratio objective function
def sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -(portfolio_return - risk_free_rate) / portfolio_volatility  # Negative because we minimize

# Step 5: Set up constraints and bounds
num_assets = len(tickers)
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling, weights between 0 and 1

# Step 6: Initial guess (equal weight distribution)
initial_guess = num_assets * [1. / num_assets]

# Step 7: Perform optimization to maximize Sharpe ratio
optimal_weights = minimize(sharpe_ratio, initial_guess, args=(expected_returns, cov_matrix, risk_free_rate),
                           method='SLSQP', bounds=bounds, constraints=constraints)

# Step 8: Backtest portfolio with optimized weights
optimal_weights_values = optimal_weights.x

# Calculate portfolio daily returns using the optimal weights
portfolio_daily_returns = returns.dot(optimal_weights_values)

# Step 9: Calculate cumulative portfolio returns
cumulative_portfolio_returns = (1 + portfolio_daily_returns).cumprod()

# Step 10: Calculate performance metrics
annualized_return = np.mean(portfolio_daily_returns) * 252
portfolio_volatility = np.std(portfolio_daily_returns) * np.sqrt(252)
sharpe_ratio_value = (annualized_return - risk_free_rate) / portfolio_volatility

# Step 11: Adjust for starting investment
portfolio_value = cumulative_portfolio_returns * starting_amount  # Adjusted to the starting amount

# Step 12: Output results
print("Optimal Weights:", optimal_weights_values)
print("Annualized Return:", annualized_return)
print("Cumulative Portfolio Return:", cumulative_portfolio_returns.iloc[-1])
print("Portfolio Volatility:", portfolio_volatility)
print("Sharpe Ratio:", sharpe_ratio_value)
print("Final Portfolio Value:", portfolio_value.iloc[-1])

# Step 13: Plot cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value, label='Optimized Portfolio Value')
plt.title('Portfolio Value Over Time (Starting with €10,000)')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (€)')
plt.legend()
plt.grid(True)
plt.show()
