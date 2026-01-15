# moex_utils.py

import requests
import apimoex
import pandas as pd
from datetime import datetime
import os
import numpy as np

# Constant for the data folder
DATA_FOLDER = "data"
METADATA_FILE = "metadata/stock-index-base.xlsx"

def get_moex_stock(ticker: str, start: str = '2023-01-01', end: str = None, session: requests.Session = None, frequency: int = 24) -> pd.DataFrame:
    """
    Fetches stock data from the Moscow Exchange (MOEX) for a given ticker symbol within a specified date range.
    Parameters:
    ticker (str): The ticker symbol of the stock to fetch data for.
    start (str): The start date for the data in 'YYYY-MM-DD' format. Default is '2023-01-01'.
    end (str): The end date for the data in 'YYYY-MM-DD' format. Default is None, which sets the end date to today.
    session (requests.Session): An optional requests session to use for making the API call. Default is None, which creates a new session.
    frequency (int): The frequency of the candles to fetch. Default is 24 (1 day). Possible values are:
                     1 - 1 minute
                     10 - 10 minutes
                     60 - 1 hour
                     24 - 1 day
                     7 - 1 week
                     31 - 1 month
                     4 - 1 quarter
                     
    Returns:
    pd.DataFrame: A DataFrame containing the stock data with columns 'date', 'value_rub', 'volume', and 'ticker'.
    Raises:
    ValueError: If the date format for 'start' or 'end' is invalid, or if the start date is after the end date.
    KeyError: If the expected columns ('begin' or 'volume') are missing in the API response.
    ConnectionError: If there is an error while making the API request.
    RuntimeError: If there is an error during data processing.
    """
    # Set the default end date to today if not provided
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    # Validate that start and end are correctly formatted dates
    try:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
    except ValueError:
        raise ValueError("Invalid date format for 'start' or 'end'. Please use 'YYYY-MM-DD'.")
    
    # Ensure start date is before end date
    if start_date > end_date:
        raise ValueError("The start date cannot be after the end date.")
    
    # Use the session if provided, otherwise create a new one
    if session is None:
        session = requests.Session()

    try:
        # Fetch data from the MOEX API with the specified frequency
        data = apimoex.get_market_candles(session=session, security=ticker, start=start, end=end, interval=frequency)
        
        # Check if the response data is empty
        if not data:
            raise ValueError("The API response is empty.")
        
        df = pd.DataFrame(data)
        if 'begin' in df.columns:
            df['begin'] = pd.to_datetime(df['begin'])
            df.rename({'begin': 'date', 'value': 'value_rub'}, axis='columns', inplace=True)
        else:
            raise KeyError("The expected 'begin' column is missing in the response.")
        
        df.set_index('date', inplace=True)
        
        # Handle data type conversion
        if 'volume' in df.columns:
            if df['volume'].dtype != 'float64':
                df['volume'] = df['volume'].astype('float64')
        else:
            raise KeyError("The expected 'volume' column is missing in the response.")
        df["ticker"] = ticker
        
        # Добавляем колонку 'close' для совместимости (если её нет)
        if 'close' not in df.columns and 'value_rub' in df.columns:
            df['close'] = df['value_rub']
        
        # df["frequency"] = frequency  # Add frequency column to the DataFrame
        return df
    
    except requests.RequestException as e:
        raise ConnectionError(f"An error occurred while trying to fetch data from the MOEX API: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred during data processing: {e}")

def get_moex_index(ticker: str, start: str = '2023-01-01', end: str = None, session: requests.Session = None) -> pd.DataFrame:
    """
    Fetch historical data for a specified index from the Moscow Exchange (MOEX) API.

    This function retrieves index data for the specified index (defined by ticker) over a defined time range. 
    The data is returned as a pandas DataFrame with relevant columns such as the trade date, 
    index volume, and closing price (index value).

    Parameters:
    ----------
    ticker : str
        The ticker symbol of the index you want to retrieve (e.g., 'IMOEX' for the MOEX Russia Index).
    
    start : str, optional (default: '2023-01-01')
        The start date for the data retrieval period in 'YYYY-MM-DD' format.
    
    end : str, optional (default: today's date)
        The end date for the data retrieval period in 'YYYY-MM-DD' format. 
        If not provided, the current date will be used.

    session (requests.Session): An optional requests session to use for making the API call. Default is None, which creates a new session.
    Returns:
    pd.DataFrame: A DataFrame containing the historical index data with the following columns:
    - 'index_value': The index value for the day (corresponding to the VALUE field in the MOEX API).
    - 'close': The closing price of the index for that date.
    - 'date': The date of the trade, which serves as the DataFrame index.

    Raises:
    -------
    ValueError
        If the provided 'start' or 'end' dates are not valid or if the start date is after the end date.
    
    ConnectionError
        If there is a network error or an issue with the MOEX API request.
    
    KeyError
        If expected fields ('TRADEDATE', 'VALUE', 'CLOSE') are not present in the API response.
    
    RuntimeError
        If there is an error in data processing that does not fall under the above categories.

    Example:
    --------
    >>> df = get_moex_index(ticker='IMOEX', start='2023-01-01', end='2023-12-31')
    >>> print(df.head())
    
    This will return a DataFrame with historical index data for the MOEX Russia Index between January 1, 2023, 
    and December 31, 2023.
    """

    # Set the default end date to today if not provided
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    # Validate that start and end are correctly formatted dates
    try:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
    except ValueError:
        raise ValueError("Invalid date format for 'start' or 'end'. Please use 'YYYY-MM-DD'.")
    
    # Ensure start date is before end date
    if start_date > end_date:
        raise ValueError("The start date cannot be after the end date.")
    
    # Use the session if provided, otherwise create a new one
    if session is None:
        session = requests.Session()
        try:
            # Fetch data from MOEX API
            data = apimoex.get_market_history(
                session=session,
                security=ticker,
                start=start,
                end=end,
                market='index',
                engine='stock'
            )
            
            # Check if the response data is empty
            if not data:
                raise ValueError("The API response is empty.")
                
            df = pd.DataFrame(data)
            required_columns = ['TRADEDATE', 'VALUE', 'CLOSE']
            if not all(col in df.columns for col in required_columns):
                raise KeyError("The expected columns are missing in the API response.")
            
            # Convert date column to datetime and rename columns
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE']).dt.date
            df.rename({'TRADEDATE': 'date', 'VALUE': 'volume', 'CLOSE': 'close'}, axis='columns', inplace=True)
            
            # Set the 'date' column as the index
            df.set_index('date', inplace=True)
        
        except requests.RequestException as e:
            raise ConnectionError(f"An error occurred while fetching data from MOEX API: {e}")
        except Exception as e:
            raise RuntimeError(f"An error occurred during data processing: {e}")
    
    return df

# def save_moex_stock(ticker: str, start: str = '2023-01-01', end: str = None, session: requests.Session = None) -> None:
#     """
#     Downloads stock data for a given ticker symbol and saves it in Parquet format.
    
#     Parameters:
#     ticker (str): The ticker symbol of the stock to fetch data for.
#     start (str): The start date for the data in 'YYYY-MM-DD' format. Default is '2023-01-01'.
#     end (str): The end date for the data in 'YYYY-MM-DD' format. Default is None, which sets the end date to today.
#     session (requests.Session): An optional requests session to use for making the API call. Default is None, which creates a new session.
#     """
#     # Fetch the stock data using the existing function
#     df = get_moex_stock(ticker, start, end, session)
    
#     # Print message with ticker name, start date, and end date
#     print(f"Downloading data for ticker: {ticker}, from {start} to {end if end else 'today'}")
    
#     # Create a directory named after the ticker if it doesn't exist
#     os.makedirs(os.path.join(DATA_FOLDER, ticker), exist_ok=True)

#     # Define the file path for the Parquet file
#     file_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}.parquet")
    
#     # Save the DataFrame to Parquet format
#     df.to_parquet(file_path)


def save_moex_stock(
    ticker: str,
    start: str = "2023-01-01",
    end: str | None = None,
    session: requests.Session | None = None,
    frequency: int = 24,
    out_dir: str = DATA_FOLDER,
    calculate_market_cap_flag: bool = True,
    metadata_file: str = METADATA_FILE,
) -> str | None:
    """
    Скачивает данные по тикеру и сохраняет в Parquet: DATA_FOLDER/<TICKER>/<TICKER>.parquet
    Автоматически рассчитывает и сохраняет market cap, если calculate_market_cap_flag=True.
    
    Parameters:
    ticker (str): Тикер акции.
    start (str): Начальная дата в формате 'YYYY-MM-DD'.
    end (str | None): Конечная дата в формате 'YYYY-MM-DD'. Если None, используется сегодняшняя дата.
    session (requests.Session | None): Опциональная сессия для HTTP запросов.
    frequency (int): Частота свечей (24 = дневные данные).
    out_dir (str): Директория для сохранения данных.
    calculate_market_cap_flag (bool): Если True, рассчитывает и сохраняет market cap.
    metadata_file (str): Путь к Excel файлу с метаданными о количестве акций.
    
    Returns:
    str | None: Путь к сохраненному файлу или None в случае ошибки.
    """
    # нормализуем даты
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # информативный лог
    print(f"[INFO] {ticker}: fetching {start} → {end}, freq={frequency}")

    try:
        df = get_moex_stock(
            ticker=ticker,
            start=start,
            end=end,
            session=session,
            frequency=frequency
        )
        if df is None or df.empty:
            print(f"[SKIP] {ticker}: пустой датафрейм от ISS MOEX (делистинг/нет торгов/не та доска).")
            return None

    except ValueError as e:
        # типично: пустой ответ (например, POLY), неверный период, несуществующий тикер
        print(f"[SKIP] {ticker}: {e}")
        return None
    except requests.RequestException as e:
        # сетевые/HTTP проблемы
        print(f"[ERROR] {ticker}: сетевой сбой — {e}")
        return None
    except Exception as e:
        # любая иная ошибка — не валим цикл
        print(f"[ERROR] {ticker}: неожиданная ошибка — {e}")
        return None

    # Рассчитываем market cap, если требуется
    if calculate_market_cap_flag:
        try:
            df = calculate_market_cap(df, ticker, metadata_file)
            if 'market_cap' in df.columns:
                print(f"[INFO] {ticker}: market cap рассчитан")
        except Exception as e:
            print(f"[WARNING] {ticker}: не удалось рассчитать market cap — {e}")

    # подготовка пути
    tdir = os.path.join(out_dir, ticker.upper())
    os.makedirs(tdir, exist_ok=True)
    file_path = os.path.join(tdir, f"{ticker.upper()}.parquet")

    # атомарная запись
    tmp_path = file_path + ".tmp"
    try:
        df.to_parquet(tmp_path, index=True)
        os.replace(tmp_path, file_path)
        print(f"[OK] {ticker}: {len(df):,} rows → {file_path}")
        return file_path
    except Exception as e:
        # если запись сорвалась — удалим tmp и продолжим
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        print(f"[ERROR] {ticker}: не удалось сохранить Parquet — {e}")
        return None



def update_moex_stock(
    ticker: str, 
    session: requests.Session = None,
    calculate_market_cap_flag: bool = True,
    metadata_file: str = METADATA_FILE,
) -> None:
    """
    Updates the stock data for a given ticker symbol by checking the local Parquet file.
    If the file exists, it fetches new data from the last date in the file to the current date.
    Автоматически пересчитывает market cap, если calculate_market_cap_flag=True.
    
    Parameters:
    ticker (str): The ticker symbol of the stock to update data for.
    session (requests.Session): An optional requests session to use for making the API call. Default is None, which creates a new session.
    calculate_market_cap_flag (bool): Если True, пересчитывает market cap для всех данных.
    metadata_file (str): Путь к Excel файлу с метаданными о количестве акций.
    """
    # Define the file path for the Parquet file
    file_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}.parquet")
    
    # Check if the Parquet file exists
    if os.path.exists(file_path):
        # Load the existing data
        df_existing = pd.read_parquet(file_path)
        
        # Get the last date in the existing data
        last_date = df_existing.index.max()
        
        # Convert last_date to string format for fetching new data
        last_date_str = last_date.strftime('%Y-%m-%d')
        
        # Fetch new data from the last date to today
        new_data = get_moex_stock(ticker, start=last_date_str, session=session)
        
        # Append the new data to the existing DataFrame
        df_updated = pd.concat([df_existing, new_data])
        
        # Удаляем дубликаты по индексу (если есть)
        df_updated = df_updated[~df_updated.index.duplicated(keep='last')]
        df_updated = df_updated.sort_index()
        
        # Пересчитываем market cap для всех данных, если требуется
        if calculate_market_cap_flag:
            try:
                df_updated = calculate_market_cap(df_updated, ticker, metadata_file)
                if 'market_cap' in df_updated.columns:
                    print(f"[INFO] {ticker}: market cap пересчитан")
            except Exception as e:
                print(f"[WARNING] {ticker}: не удалось пересчитать market cap — {e}")
        
        # Save the updated DataFrame back to Parquet format
        df_updated.to_parquet(file_path)
        
        print(f"Updated data for ticker: {ticker} from {last_date_str} to {datetime.today().strftime('%Y-%m-%d')}")
    else:
        print(f"No existing data found for ticker: {ticker}. Please use save_moex_stock to create the initial file.")

def read_moex_stock(ticker: str, start: str = '2023-01-01', end: str = None, session: requests.Session = None) -> pd.DataFrame:
    """
    Reads stock data for a given ticker symbol from a local Parquet file.
    If the file does not exist, it fetches the data using save_moex_stock.
    
    Parameters:
    ticker (str): The ticker symbol of the stock to read data for.
    start (str): The start date for the data in 'YYYY-MM-DD' format. Default is '2023-01-01'.
    end (str): The end date for the data in 'YYYY-MM-DD' format. Default is None, which sets the end date to today.
    session (requests.Session): An optional requests session to use for making the API call. Default is None, which creates a new session.
    
    Returns:
    pd.DataFrame: A DataFrame containing the stock data.
    """
    # Define the file path for the Parquet file
    file_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}.parquet")
    
    # Check if the Parquet file exists
    if os.path.exists(file_path):
        # Load the existing data
        df = pd.read_parquet(file_path)
        print(f"Loaded data for ticker: {ticker} from local file.")
    else:
        # If the file does not exist, create it using save_moex_stock
        print(f"No local data found for ticker: {ticker}. Fetching data...")
        save_moex_stock(ticker, start, end, session)
        df = pd.read_parquet(file_path)  # Load the newly created data
    
    return df

def combine_moex_stocks() -> pd.DataFrame:
    """
    Lists all parquet files in the data folder, reads them, and combines into a unified dataset.
    
    Returns:
    pd.DataFrame: A DataFrame containing combined stock data from all parquet files.
    """
    # Initialize an empty list to store DataFrames
    dfs = []
    
    # Walk through the data directory
    for ticker_dir in os.listdir(DATA_FOLDER):
        dir_path = os.path.join(DATA_FOLDER, ticker_dir)
        
        # Check if it's a directory
        if os.path.isdir(dir_path):
            parquet_file = os.path.join(dir_path, f"{ticker_dir}.parquet")
            
            # Check if parquet file exists
            if os.path.exists(parquet_file):
                try:
                    # Read the parquet file
                    df = pd.read_parquet(parquet_file)
                    dfs.append(df)
                    print(f"Loaded data for {ticker_dir}")
                except Exception as e:
                    print(f"Error loading {ticker_dir}: {e}")
    
    # Combine all DataFrames
    if dfs:
        combined_df = pd.concat(dfs, axis=0)
        print(f"\nCombined {len(dfs)} stocks into unified dataset")
        print(f"Total rows: {len(combined_df)}")
        return combined_df
    else:
        raise ValueError("No parquet files found in data directory")

def load_shares_data(metadata_file: str = METADATA_FILE) -> pd.DataFrame:
    """
    Загружает данные о количестве акций из Excel файла metadata/stock-index-base.xlsx.
    
    Parameters:
    metadata_file (str): Путь к Excel файлу с метаданными. По умолчанию METADATA_FILE.
    
    Returns:
    pd.DataFrame: DataFrame с колонками Code, date, Number of issued shares.
    """
    if not os.path.exists(metadata_file):
        print(f"Warning: Metadata file not found: {metadata_file}")
        return pd.DataFrame()
    
    try:
        xls = pd.ExcelFile(metadata_file)
        sheets = xls.sheet_names
        
        # Фильтруем листы с датами
        date_sheets = []
        for s in sheets:
            try:
                pd.to_datetime(s, format="%d.%m.%Y", errors="raise")
                date_sheets.append(s)
            except Exception:
                pass
        
        # Читаем все листы в один DataFrame
        all_data = []
        for sheet in date_sheets:
            df = pd.read_excel(metadata_file, sheet_name=sheet, skiprows=3)
            df["date"] = pd.to_datetime(sheet, format="%d.%m.%Y")
            all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        shares_df = pd.concat(all_data, ignore_index=True)
        
        # Оставляем только нужные колонки
        if 'Code' in shares_df.columns and 'Number of issued shares' in shares_df.columns:
            shares_df = shares_df[['Code', 'date', 'Number of issued shares']].copy()
            shares_df = shares_df.dropna(subset=['Number of issued shares'])
            shares_df = shares_df.sort_values(['Code', 'date'])
            return shares_df
        else:
            print(f"Warning: Required columns not found in {metadata_file}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading shares data from {metadata_file}: {e}")
        return pd.DataFrame()


def calculate_market_cap(df: pd.DataFrame, ticker: str, metadata_file: str = METADATA_FILE) -> pd.DataFrame:
    """
    Рассчитывает market cap для DataFrame с данными о ценах акций.
    Использует данные о количестве акций из metadata файла.
    
    Parameters:
    df (pd.DataFrame): DataFrame с данными о ценах акций (должен содержать 'close' или 'value_rub').
    ticker (str): Тикер акции.
    metadata_file (str): Путь к Excel файлу с метаданными.
    
    Returns:
    pd.DataFrame: DataFrame с добавленными колонками 'shares' и 'market_cap'.
    """
    if df.empty:
        return df
    
    # Определяем колонку с ценой
    price_col = 'close' if 'close' in df.columns else 'value_rub'
    if price_col not in df.columns:
        print(f"Warning: No price column found for {ticker}. Skipping market cap calculation.")
        return df
    
    # Загружаем данные о количестве акций
    shares_data = load_shares_data(metadata_file)
    if shares_data.empty:
        print(f"Warning: No shares data available. Skipping market cap calculation for {ticker}.")
        return df
    
    # Фильтруем данные для конкретного тикера
    ticker_shares = shares_data[shares_data['Code'] == ticker].copy()
    if ticker_shares.empty:
        print(f"Warning: No shares data found for {ticker}. Skipping market cap calculation.")
        return df
    
    # Создаем временной ряд количества акций
    ticker_shares = ticker_shares.set_index('date')
    ticker_shares = ticker_shares[['Number of issued shares']]
    ticker_shares.columns = ['shares']
    
    # Создаем полный временной ряд от первой до последней даты в df
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            print(f"Warning: Cannot convert index to datetime for {ticker}.")
            return df
    
    date_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='D'
    )
    shares_series = pd.Series(index=date_range, dtype=float)
    
    # Заполняем известные значения
    for date, row in ticker_shares.iterrows():
        shares_series.loc[date] = row['shares']
    
    # Forward fill: используем последнее известное значение
    shares_series = shares_series.ffill()
    
    # Backward fill для начала периода
    shares_series = shares_series.bfill()
    
    # Объединяем с данными о ценах
    df = df.copy()
    shares_aligned = shares_series.reindex(df.index).ffill().bfill()
    df['shares'] = shares_aligned
    
    # Рассчитываем market cap = цена * количество акций
    df['market_cap'] = df[price_col] * df['shares']
    
    # Удаляем строки где нет данных о количестве акций
    df = df.dropna(subset=['market_cap'])
    
    return df


def update_all_stocks():
    """
    Updates data for all stocks that have existing parquet files.
    """
    # Get list of all ticker directories
    ticker_dirs = []
    for item in os.listdir(DATA_FOLDER):
        dir_path = os.path.join(DATA_FOLDER, item)
        if os.path.isdir(dir_path):
            parquet_file = os.path.join(dir_path, f"{item}.parquet")
            if os.path.exists(parquet_file):
                ticker_dirs.append(item)
    
    print(f"Found {len(ticker_dirs)} stocks to update")
    
    # Update each stock
    for ticker in ticker_dirs:
        try:
            print(f"\nUpdating {ticker}...")
            update_moex_stock(ticker)
        except Exception as e:
            print(f"Error updating {ticker}: {e}")
    
    print(f"\nUpdate completed for {len(ticker_dirs)} stocks")

def calculate_adj_close(df: pd.DataFrame, div_folder: str) -> pd.DataFrame:
    """
    Calculates the adjusted close price for a given stock DataFrame based on its dividend history.

    Parameters:
    df (pd.DataFrame): The input DataFrame for a single stock. Must contain 'ticker' and 'close' columns.
    div_folder (str): The path to the folder containing dividend history CSV files (e.g., 'SBER.csv').

    Returns:
    pd.DataFrame: The DataFrame with an added 'adj_close' column. Returns the original DataFrame if no dividend data is found.
    """
    if df.empty or 'ticker' not in df.columns or 'close' not in df.columns:
        print("Warning: DataFrame пуст или нет колонок 'ticker'/'close'. Возвращаю как есть.")
        return df
    # Убедимся, что индекс — даты
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            print("Warning: индекс не преобразуется в даты. Возвращаю без корректировки.")
            df['adj_close'] = df['close']
            return df

    ticker = df['ticker'].iloc[0]
    div_file = os.path.join(div_folder, f"{ticker}.csv")

    if not os.path.exists(div_file):
        print(f"Info: No dividend file found for {ticker} at {div_file}. Setting adj_close equal to close.")
        df['adj_close'] = df['close']
        return df

    try:
        div_df = pd.read_csv(div_file, parse_dates=['closing_date'])
        div_df.dropna(subset=['closing_date'], inplace=True)
        div_df = div_df[div_df['dividend_value'] > 0]
        div_df.sort_values(by='closing_date', inplace=True)
        dividends_df = div_df
    except Exception as e:
        print(f"Error reading dividend file for {ticker}: {e}")
        df['adj_close'] = df['close']
        return df

    # Filter data and sort
    df = df.sort_index()
    div_df = div_df[div_df['closing_date'].notnull()].sort_values('closing_date')

    if div_df.empty:
        df['adj_close'] = df['close']
        return df

    # Calculate adjusted close prices using a proportional adjustment factor
    #adj_close = df['close'].copy()
    adj = df['close'].astype(float).copy()

    # Идём от последних дивидендов к ранним
    for row in div_df.iloc[::-1].itertuples(index=False):
        ex_dividend_date = row.closing_date
        dividend_value = float(row.dividend_value)

        # позиция справа: всё строго ДО экс-даты попадает под корректировку
        pos = adj.index.searchsorted(ex_dividend_date, side='right')

        # если экс-дата раньше всех наших данных — пропускаем
        if pos == 0:
            continue

        close_before = df['close'].iloc[pos - 1]
        if pd.isna(close_before) or close_before <= 0:
            # защита от деления на ноль/NaN
            continue

        adjustment_factor = 1.0 - (dividend_value / float(close_before))
        # опционально: защита от странных значений
        # if adjustment_factor <= 0:
        #     continue

        # применяем ко ВСЕМ прошлым ценам (по индексам до pos)
        adj.iloc[:pos] = adj.iloc[:pos] * adjustment_factor

    df['adj_close'] = adj
    return df

def add_adj_close_to_all_stocks(div_folder: str) -> None:
    """
    Calculates and adds the 'adj_close' column for all stocks in the data folder.

    This function iterates through all stock data files, calculates the adjusted close
    price based on dividend files from the specified folder, and overwrites the
    original Parquet files with the updated data.

    Parameters:
    div_folder (str): The path to the folder containing dividend history CSV files.
    """
    ticker_dirs = [
        item for item in os.listdir(DATA_FOLDER)
        if os.path.isdir(os.path.join(DATA_FOLDER, item)) and
           os.path.exists(os.path.join(DATA_FOLDER, item, f"{item}.parquet"))
    ]

    if not ticker_dirs:
        print("No stock data found in the data folder.")
        return

    print(f"Found {len(ticker_dirs)} stocks to process for adjusted close calculation.")

    for ticker in ticker_dirs:
        try:
            print(f"Processing {ticker}...")
            file_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}.parquet")
            
            df = pd.read_parquet(file_path)
            
            # Calculate adjusted close (will overwrite if column already exists)
            df_adj = calculate_adj_close(df, div_folder)
            
            df_adj.to_parquet(file_path)
            
            print(f"Successfully updated {ticker} with adj_close column.")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    print(f"\nCompleted processing for {len(ticker_dirs)} stocks.")


def add_market_cap_to_all_stocks(metadata_file: str = METADATA_FILE) -> None:
    """
    Рассчитывает и добавляет колонки 'shares' и 'market_cap' для всех акций в папке данных.
    
    Эта функция проходит по всем файлам данных акций, рассчитывает market cap
    на основе данных о количестве акций из metadata файла и перезаписывает
    оригинальные Parquet файлы с обновленными данными.
    
    Parameters:
    metadata_file (str): Путь к Excel файлу с метаданными о количестве акций.
    """
    ticker_dirs = [
        item for item in os.listdir(DATA_FOLDER)
        if os.path.isdir(os.path.join(DATA_FOLDER, item)) and
           os.path.exists(os.path.join(DATA_FOLDER, item, f"{item}.parquet"))
    ]
    
    if not ticker_dirs:
        print("No stock data found in the data folder.")
        return
    
    print(f"Found {len(ticker_dirs)} stocks to process for market cap calculation.")
    
    for ticker in ticker_dirs:
        try:
            print(f"Processing {ticker}...")
            file_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}.parquet")
            
            df = pd.read_parquet(file_path)
            
            # Calculate market cap (will overwrite if columns already exist)
            df_mc = calculate_market_cap(df, ticker, metadata_file)
            
            if 'market_cap' in df_mc.columns:
                df_mc.to_parquet(file_path)
                print(f"Successfully updated {ticker} with market cap data.")
            else:
                print(f"Warning: Could not calculate market cap for {ticker}.")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    print(f"\nCompleted processing for {len(ticker_dirs)} stocks.")
        