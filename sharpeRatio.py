import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Step 1: Fetch historical price data for backtesting (5-year period)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Example stocks
data = yf.download(tickers, start='2023-01-01')['Adj Close']  # Fetch data up to now

# Ensure data index is timezone-aware
if data.index.tzinfo is None:
    data.index = data.index.tz_localize('UTC')

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
initial_guess = num_assets * [1. / num_assets]  # 25% for each asset

# Step 7: Initialize variables for portfolio tracking
starting_amount = 10000  # Starting amount of money
portfolio_values = []
weights_over_time = []

# Define rebalancing frequency
rebalance_frequency = 'M'  # Monthly rebalancing
rebalance_dates = pd.date_range(start=data.index[0], end=data.index[-1], freq=rebalance_frequency)

# Step 8: Backtesting with rebalancing
for i, date in enumerate(rebalance_dates):
    # Get data up to the current rebalance date
    data_up_to_date = data.loc[:date]

    # Recalculate returns and metrics
    returns_up_to_date = data_up_to_date.pct_change().dropna()
    expected_returns_up_to_date = returns_up_to_date.mean() * 252
    cov_matrix_up_to_date = returns_up_to_date.cov() * 252
    
    # Use initial guess for the first rebalance date
    if i == 0:
        optimal_weights = initial_guess
    else:
        # Optimize weights for the current date
        optimal_weights = minimize(sharpe_ratio, initial_guess, args=(expected_returns_up_to_date, cov_matrix_up_to_date, risk_free_rate),
                                   method='SLSQP', bounds=bounds, constraints=constraints).x

    # Store the weights
    weights_over_time.append(optimal_weights)
    
    # Calculate portfolio daily returns using the optimal weights
    portfolio_daily_returns = returns_up_to_date.dot(optimal_weights)
    
    # Calculate cumulative portfolio returns
    cumulative_portfolio_returns = (1 + portfolio_daily_returns).cumprod()

    # Adjust for starting investment
    portfolio_value = cumulative_portfolio_returns * starting_amount
    
    # Store the final portfolio value for this rebalance date
    portfolio_values.append(portfolio_value.iloc[-1])

# Convert portfolio values to a DataFrame for plotting
portfolio_values_df = pd.DataFrame(portfolio_values, index=rebalance_dates, columns=['Portfolio Value'])
weights_df = pd.DataFrame(weights_over_time, index=rebalance_dates, columns=tickers)

# Step 12: Output results
final_value = portfolio_values_df.iloc[-1].values[0]
print("Final Portfolio Value:", final_value)
print("Optimal Weights at Final Rebalance Date:")
print(weights_df.iloc[-1])

# Step 13: Plot cumulative returns and weights in the same figure
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Plot for Portfolio Value
axs[0].plot(portfolio_values_df.index, portfolio_values_df, label='Optimized Portfolio Value', color='blue')
axs[0].set_title('Portfolio Value Over Time (Starting with €10,000)')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Portfolio Value (€)')
axs[0].legend()
axs[0].grid(True)

# Plot for Portfolio Weights
weights_plot = weights_df.plot(ax=axs[1], figsize=(10, 6))
axs[1].set_title('Portfolio Weights Over Time')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Weights')

# Show y-ticks explicitly
axs[1].set_yticks(np.arange(0, 1.1, 0.1))  # Adjust this range if needed
axs[1].set_yticklabels([f'{tick:.2f}' for tick in np.arange(0, 1.1, 0.1)])  # Label y-ticks with 2 decimal places

# Create legend with toggle functionality
lines = weights_plot.get_lines()  # Get the lines for each stock
labels = [line.get_label() for line in lines]  # Get labels for the lines

# Create toggle function
def toggle_line(event):
    # Check if the artist clicked is a legend text
    if event.artist.get_gid() in labels:
        index = labels.index(event.artist.get_gid())
        line = lines[index]
        line.set_visible(not line.get_visible())  # Toggle visibility
        plt.draw()  # Redraw the figure

# Create legend
legend = axs[1].legend(tickers, loc='upper left', bbox_to_anchor=(1, 1))

# Set pickable
for text in legend.get_texts():
    text.set_picker(True)  # Make legend text pickable
    text.set_gid(text.get_text())  # Set the text as its gid for identification

# Connect the pick event to the toggle function
fig.canvas.mpl_connect('pick_event', toggle_line)

axs[1].grid(True)

plt.tight_layout()
plt.show()
