# MOEX Utils

A Python utility library for fetching and managing stock data from the Moscow Exchange (MOEX) API.

## Features

- Fetch historical stock data from MOEX
- Fetch index data from MOEX
- Save data locally in Parquet format
- Update existing data files
- Combine multiple stock datasets
- Calculate and visualize stock performance
- Support for different time frequencies (1min, 10min, 1hour, 1day, 1week, 1month, 1quarter)

## Installation

```bash
pip install requests apimoex pandas pyarrow
```

## Usage

### Basic Import

```python
import moex_utils as moex
```

### Fetching Stock Data

```python
# Fetch stock data for a specific ticker
df = moex.get_moex_stock('SBER', start='2023-01-01', end='2023-12-31')

# Fetch data with different frequency (1 hour candles)
df = moex.get_moex_stock('SBER', start='2023-01-01', frequency=60)
```

### Fetching Index Data

```python
# Fetch index data
df = moex.get_moex_index('IMOEX', start='2023-01-01', end='2023-12-31')
```

### Saving Data Locally

```python
# Save stock data to local Parquet file
moex.save_moex_stock('SBER', start='2023-01-01')

# Save multiple stocks
tickers = ['SBER', 'LKOH', 'GAZP', 'AAPL']
for ticker in tickers:
    moex.save_moex_stock(ticker)
```

### Reading Local Data

```python
# Read local data (creates file if it doesn't exist)
df = moex.read_moex_stock('SBER')
```

### Updating Data

```python
# Update single stock data
moex.update_moex_stock('SBER')

# Update all stocks
moex.update_all_stocks()
```

### Combining Data

```python
# Combine all local stock data into one dataset
combined_df = moex.combine_moex_stocks()
```

### Performance Analysis

```python
# Calculate and visualize last month's performance
performance_data = moex.plot_stocks_performance()
```

## API Reference

### Functions

#### `get_moex_stock(ticker, start='2023-01-01', end=None, session=None, frequency=24)`

Fetches stock data from MOEX API.

**Parameters:**
- `ticker` (str): Stock ticker symbol
- `start` (str): Start date in 'YYYY-MM-DD' format
- `end` (str): End date in 'YYYY-MM-DD' format (default: today)
- `session` (requests.Session): Optional session object
- `frequency` (int): Candle frequency (1=1min, 10=10min, 60=1hour, 24=1day, 7=1week, 31=1month, 4=1quarter)

**Returns:** pandas.DataFrame with columns: date, value_rub, volume, ticker, frequency

#### `get_moex_index(ticker, start='2023-01-01', end=None, session=None)`

Fetches index data from MOEX API.

**Parameters:**
- `ticker` (str): Index ticker symbol
- `start` (str): Start date in 'YYYY-MM-DD' format
- `end` (str): End date in 'YYYY-MM-DD' format (default: today)
- `session` (requests.Session): Optional session object

**Returns:** pandas.DataFrame with columns: date, volume, close

#### `save_moex_stock(ticker, start='2023-01-01', end=None, session=None)`

Downloads and saves stock data to local Parquet file.

#### `read_moex_stock(ticker, start='2023-01-01', end=None, session=None)`

Reads local stock data, creates file if it doesn't exist.

#### `update_moex_stock(ticker, session=None)`

Updates existing local stock data from last date to current date.

#### `update_all_stocks()`

Updates data for all stocks that have existing local files.

#### `combine_moex_stocks()`

Combines all local stock data files into one unified dataset.

#### `plot_stocks_performance()`

Calculates and visualizes last month's performance for all stocks.

## Data Structure

### Stock Data Columns
- `date`: Trading date (index)
- `value_rub`: Stock price in RUB
- `volume`: Trading volume
- `ticker`: Stock ticker symbol
- `frequency`: Data frequency (1, 10, 60, 24, 7, 31, 4)

### Index Data Columns
- `date`: Trading date (index)
- `volume`: Index volume
- `close`: Closing value

## File Organization

Data is stored in the following structure:
```
data/
├── SBER/
│   └── SBER.parquet
├── LKOH/
│   └── LKOH.parquet
└── ...
```

## Error Handling

The library includes comprehensive error handling for:
- Invalid date formats
- Empty API responses
- Missing required columns
- Network connection issues
- File I/O operations

## Dependencies

- `requests`: HTTP requests
- `apimoex`: MOEX API client
- `pandas`: Data manipulation
- `pyarrow`: Parquet file support
- `matplotlib`: Chart visualization (for performance plots)

## License

This project is open source and available under the MIT License. 
