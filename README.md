# MOEX Utils

A Python utility library for fetching and managing stock data from the Moscow Exchange (MOEX) API.

## Features

- Fetch historical stock data from MOEX
- Fetch index data from MOEX
- Save data locally in Parquet format
- Update existing data files
- Combine multiple stock datasets
- Calculate and visualize stock performance
- **Automatic market cap calculation** using shares data from metadata
- **Adjusted close price calculation** based on dividend history
- Support for different time frequencies (1min, 10min, 1hour, 1day, 1week, 1month, 1quarter)

## Installation

```bash
pip install requests apimoex pandas pyarrow openpyxl
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
# Save stock data to local Parquet file (market cap calculated automatically)
moex.save_moex_stock('SBER', start='2023-01-01')

# Save without market cap calculation
moex.save_moex_stock('SBER', start='2023-01-01', calculate_market_cap_flag=False)

# Save multiple stocks
tickers = ['SBER', 'LKOH', 'GAZP']
for ticker in tickers:
    moex.save_moex_stock(ticker, start='2000-01-01')
```

### Reading Local Data

```python
# Read local data (creates file if it doesn't exist)
df = moex.read_moex_stock('SBER')
```

### Updating Data

```python
# Update single stock data (market cap recalculated automatically)
moex.update_moex_stock('SBER')

# Update without market cap recalculation
moex.update_moex_stock('SBER', calculate_market_cap_flag=False)

# Update all stocks
moex.update_all_stocks()
```

### Market Cap Calculation

```python
# Market cap is automatically calculated when saving/updating data
# It uses shares data from metadata/stock-index-base.xlsx

# Calculate market cap for all existing stocks
moex.add_market_cap_to_all_stocks()

# Calculate market cap for a specific DataFrame
df = moex.read_moex_stock('SBER')
df_with_mc = moex.calculate_market_cap(df, 'SBER')
```

### Adjusted Close Price

```python
# Calculate adjusted close for a single stock
df = moex.read_moex_stock('SBER')
df_adj = moex.calculate_adj_close(df, div_folder='path/to/dividends/')

# Add adjusted close to all stocks
moex.add_adj_close_to_all_stocks(div_folder='path/to/dividends/')
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

**Returns:** pandas.DataFrame with columns: date (index), value_rub, close, volume, ticker

#### `get_moex_index(ticker, start='2023-01-01', end=None, session=None)`

Fetches index data from MOEX API.

**Parameters:**
- `ticker` (str): Index ticker symbol
- `start` (str): Start date in 'YYYY-MM-DD' format
- `end` (str): End date in 'YYYY-MM-DD' format (default: today)
- `session` (requests.Session): Optional session object

**Returns:** pandas.DataFrame with columns: date, volume, close

#### `save_moex_stock(ticker, start='2023-01-01', end=None, session=None, frequency=24, out_dir='data', calculate_market_cap_flag=True, metadata_file='metadata/stock-index-base.xlsx')`

Downloads and saves stock data to local Parquet file. Automatically calculates and saves market cap if `calculate_market_cap_flag=True`.

**Parameters:**
- `ticker` (str): Stock ticker symbol
- `start` (str): Start date in 'YYYY-MM-DD' format
- `end` (str): End date in 'YYYY-MM-DD' format (default: today)
- `session` (requests.Session): Optional session object
- `frequency` (int): Candle frequency (1=1min, 10=10min, 60=1hour, 24=1day, 7=1week, 31=1month, 4=1quarter)
- `out_dir` (str): Output directory (default: 'data')
- `calculate_market_cap_flag` (bool): If True, calculates and saves market cap (default: True)
- `metadata_file` (str): Path to Excel file with shares metadata (default: 'metadata/stock-index-base.xlsx')

**Returns:** str | None: Path to saved file or None on error

#### `read_moex_stock(ticker, start='2023-01-01', end=None, session=None)`

Reads local stock data, creates file if it doesn't exist.

**Returns:** pandas.DataFrame with stock data

#### `update_moex_stock(ticker, session=None, calculate_market_cap_flag=True, metadata_file='metadata/stock-index-base.xlsx')`

Updates existing local stock data from last date to current date. Automatically recalculates market cap if `calculate_market_cap_flag=True`.

**Parameters:**
- `ticker` (str): Stock ticker symbol
- `session` (requests.Session): Optional session object
- `calculate_market_cap_flag` (bool): If True, recalculates market cap for all data (default: True)
- `metadata_file` (str): Path to Excel file with shares metadata

#### `update_all_stocks()`

Updates data for all stocks that have existing local files.

#### `combine_moex_stocks()`

Combines all local stock data files into one unified dataset.

#### `load_shares_data(metadata_file='metadata/stock-index-base.xlsx')`

Loads shares data from Excel metadata file.

**Parameters:**
- `metadata_file` (str): Path to Excel file with metadata

**Returns:** pandas.DataFrame with columns: Code, date, Number of issued shares

#### `calculate_market_cap(df, ticker, metadata_file='metadata/stock-index-base.xlsx')`

Calculates market cap for a DataFrame with stock price data.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with stock price data (must contain 'close' or 'value_rub')
- `ticker` (str): Stock ticker symbol
- `metadata_file` (str): Path to Excel file with metadata

**Returns:** pandas.DataFrame with added 'shares' and 'market_cap' columns

#### `add_market_cap_to_all_stocks(metadata_file='metadata/stock-index-base.xlsx')`

Calculates and adds 'shares' and 'market_cap' columns for all stocks in data folder.

**Parameters:**
- `metadata_file` (str): Path to Excel file with shares metadata

#### `calculate_adj_close(df, div_folder)`

Calculates adjusted close price based on dividend history.

**Parameters:**
- `df` (pd.DataFrame): DataFrame with stock data (must contain 'ticker' and 'close')
- `div_folder` (str): Path to folder containing dividend CSV files

**Returns:** pandas.DataFrame with added 'adj_close' column

#### `add_adj_close_to_all_stocks(div_folder)`

Calculates and adds 'adj_close' column for all stocks in data folder.

**Parameters:**
- `div_folder` (str): Path to folder containing dividend CSV files

#### `plot_stocks_performance()`

Calculates and visualizes last month's performance for all stocks.

## Data Structure

### Stock Data Columns
- `date`: Trading date (index)
- `value_rub`: Stock price in RUB
- `close`: Stock closing price (alias for value_rub)
- `volume`: Trading volume
- `ticker`: Stock ticker symbol
- `shares`: Number of issued shares (added when market cap is calculated)
- `market_cap`: Market capitalization = close × shares (added when market cap is calculated)
- `adj_close`: Adjusted close price (added when dividend adjustment is calculated)

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

metadata/
└── stock-index-base.xlsx  # Excel file with shares data (Code, date, Number of issued shares)
```

### Metadata File Format

The `metadata/stock-index-base.xlsx` file should contain:
- Multiple sheets named with dates in format `DD.MM.YYYY`
- Each sheet should have columns: `Code`, `Number of issued shares`
- The library automatically reads all date sheets and combines them

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
- `openpyxl`: Excel file reading (for metadata)
- `matplotlib`: Chart visualization (for performance plots)

## License

This project is open source and available under the MIT License. 
