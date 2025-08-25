#from moex_utils import *
import moex_utils as moex
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


moex.update_all_stocks()

ticker = 'GAZP'
div_folder = 'G:/YandexDisk/FINANCE/dividends/data/'

# Run the function to update all stocks with the adj_close column
moex.add_adj_close_to_all_stocks(div_folder)

# Get stock price data
df = moex.read_moex_stock(ticker)

# Get dividends data
div_df = pd.read_csv(f"{div_folder}{ticker}.csv", parse_dates=['closing_date'])

# Filter data for the specific ticker
df = df[df['ticker'] == ticker].sort_index()
div_df = div_df[div_df['closing_date'].notnull()].sort_values('closing_date')

# Calculate adjusted close prices using a proportional adjustment factor
adj_close = df['close'].copy()

# Process dividends in reverse chronological order (latest first is crucial)
for _, row in div_df.iloc[::-1].iterrows():
    closing_date = row['closing_date']
    dividend = row['dividend_value']
    
    # Find the last closing price on the day before the dividend record date
    price_lookup_date_series = df.index[df.index < closing_date]
    if price_lookup_date_series.empty:
        continue # Cannot adjust if there's no price data before the dividend

    price_before_div_date = price_lookup_date_series.max()
    price_before_div = df.loc[price_before_div_date, 'close']

    # Skip if price is zero or negative to avoid division errors
    if price_before_div <= 0:
        continue

    # Calculate the adjustment factor
    adj_factor = 1 - (dividend / price_before_div)
    
    # Adjust all prices on and before the ex-dividend date
    mask = df.index <= price_before_div_date
    adj_close.loc[mask] *= adj_factor

# Add adjusted close to dataframe
df['adj_close'] = adj_close

# Plot the results
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(df.index, df['close'], label='Close Price', alpha=0.7)
ax.plot(df.index, df['adj_close'], label='Adjusted Close Price', alpha=0.7)

# Mark dividend dates on the plot
for _, row in div_df.iterrows():
    closing_date = row['closing_date']
    if closing_date in df.index:
        ax.axvline(x=closing_date, color='red', alpha=0.3, linestyle='--')

ax.set_title(f'{ticker} - Close vs Adjusted Close Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Price (RUB)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Calculate CAGR for both close and adj_close prices using log returns
def calculate_cagr_from_log(prices):
    """Calculates CAGR from a price series using log returns."""
    if prices.empty or len(prices) < 2:
        return None, None
        
    # Calculate daily log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    if log_returns.empty:
        return None, None
        
    # Use number of trading days for annualization (252 is standard)
    years = len(log_returns) / 252.0
    
    # The total log return is the sum of daily log returns
    total_log_return = log_returns.sum()
    
    # Annualize the log return and convert to a simple percentage CAGR
    cagr = np.exp(total_log_return / years) - 1
    
    return cagr, log_returns

# Get date range for printing
start_date = df.index.min()
end_date = df.index.max()
years_total = (end_date - start_date).days / 365.25

# Calculate CAGR and get log returns for both series
close_cagr, close_log_returns = calculate_cagr_from_log(df['close'])
adj_close_cagr, adj_close_log_returns = calculate_cagr_from_log(df['adj_close'])

print(f"\nCAGR Analysis (from Log Returns) for {ticker}")
print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({years_total:.2f} years)")
print(f"Close Price CAGR: {close_cagr:.2%}" if close_cagr is not None else "Close Price CAGR: N/A")
print(f"Adjusted Close (Total Return) CAGR: {adj_close_cagr:.2%}" if adj_close_cagr is not None else "Adjusted Close CAGR: N/A")

if close_cagr is not None and adj_close_cagr is not None:
    # The difference in CAGR can be interpreted as the annualized dividend yield
    difference = adj_close_cagr - close_cagr
    print(f"Difference (Annualized Dividend Yield): {difference:.2%}")

if close_log_returns is not None and adj_close_log_returns is not None:
    # Annualized volatility is the standard deviation of daily log returns multiplied by sqrt(252)
    close_volatility = close_log_returns.std() * np.sqrt(252)
    adj_close_volatility = adj_close_log_returns.std() * np.sqrt(252)
    
    print("\nVolatility Analysis (Annualized)")
    print(f"Close Price Volatility: {close_volatility:.2%}")
    print(f"Adjusted Close Volatility: {adj_close_volatility:.2%}")

# Print dividend information
print(f"\nDividend Information for {ticker}")
print(f"Total dividends processed: {len(div_df)}")
for _, row in div_df.iterrows():
    print(f"Date: {row['closing_date'].strftime('%Y-%m-%d')}, Dividend: {row['dividend_value']:.2f} RUB")

