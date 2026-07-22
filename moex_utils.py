# moex_utils.py

from __future__ import annotations

import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import Optional

import apimoex
import pandas as pd
import requests

# Пути привязаны к папке модуля, чтобы импорт из nb/ и scripts/ работал при любом cwd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
METADATA_FILE = os.path.join(BASE_DIR, "metadata", "stock-index-base.xlsx")
INDEXES_FOLDER = os.path.join(BASE_DIR, "indexes")
SPLITS_FILE = os.path.join(BASE_DIR, "metadata", "splits.csv")
# Внешний реестр сплитов соседнего проекта dividends (если проекты лежат рядом)
EXTERNAL_SPLITS_FILE = os.path.join(BASE_DIR, "..", "dividends", "metadata", "splits.json")

logger = logging.getLogger("moex_utils")
# Если логирование в приложении не настроено — выводим сообщения в stdout,
# как раньше это делал print (прогресс в ноутбуках и update_data.bat).
# Любая внешняя настройка logging имеет приоритет.
if not logger.handlers and not logging.getLogger().handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

def get_moex_stock(ticker: str, start: str = '2023-01-01', end: Optional[str] = None, session: Optional[requests.Session] = None, frequency: int = 24) -> pd.DataFrame:
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
        if frequency == 24:
            # Дневные данные — из официальной истории торгов (/history):
            # CLOSE здесь — закрытие основной сессии, тот же методический подход,
            # что у индексов. Свечи ISS, в отличие от истории, включают вечернюю
            # сессию, из-за чего закрытия акций и индекса были бы несопоставимы.
            data = apimoex.get_market_history(
                session=session, security=ticker, start=start, end=end,
                market='shares', engine='stock')
            if not data:
                raise ValueError("The API response is empty.")

            df = pd.DataFrame(data)
            required = ['TRADEDATE', 'CLOSE', 'VOLUME', 'VALUE']
            if not all(col in df.columns for col in required):
                raise KeyError("The expected columns are missing in the response.")

            # История отдается по всем режимам торгов — на каждую дату оставляем
            # строку главной доски (с максимальным оборотом)
            df = df.sort_values('VALUE').drop_duplicates(subset='TRADEDATE', keep='last')
            df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
            df = df.rename({'TRADEDATE': 'date', 'CLOSE': 'close',
                            'VOLUME': 'volume', 'VALUE': 'value_rub'}, axis='columns')
            df = df.set_index('date').sort_index()
            df = df.drop(columns=[c for c in ('BOARDID',) if c in df.columns])
            df = df.dropna(subset=['close'])
            df['volume'] = df['volume'].astype('float64')
            df['ticker'] = ticker
            return df

        # Внутридневные и агрегированные частоты — через свечи
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

def get_moex_index(ticker: str, start: str = '2023-01-01', end: Optional[str] = None, session: Optional[requests.Session] = None) -> pd.DataFrame:
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


def save_moex_index(ticker: str = 'IMOEX', start: str = '2010-01-01', end: Optional[str] = None,
                    session: Optional[requests.Session] = None) -> Optional[str]:
    """
    Скачивает историю индекса и сохраняет в INDEXES_FOLDER/<TICKER>.parquet (атомарная запись).

    Returns:
    str | None: путь к файлу или None при ошибке.
    """
    ticker = ticker.upper()
    try:
        df = get_moex_index(ticker, start=start, end=end, session=session)
    except Exception as e:
        logger.error(f"[ERROR] {ticker}: не удалось получить данные индекса — {e}")
        return None

    if df is None or df.empty:
        logger.warning(f"[WARN] {ticker}: пустой ответ по индексу")
        return None

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df['ticker'] = ticker

    os.makedirs(INDEXES_FOLDER, exist_ok=True)
    file_path = os.path.join(INDEXES_FOLDER, f"{ticker}.parquet")
    tmp_path = file_path + ".tmp"
    df.to_parquet(tmp_path)
    os.replace(tmp_path, file_path)
    logger.info(f"[OK] {ticker}: {len(df):,} rows → {file_path}")
    return file_path


def read_moex_index(ticker: str = 'IMOEX') -> pd.DataFrame:
    """
    Читает локальный кэш индекса из INDEXES_FOLDER/<TICKER>.parquet.
    """
    ticker = ticker.upper()
    file_path = os.path.join(INDEXES_FOLDER, f"{ticker}.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Index data file not found: {file_path}. Используйте save_moex_index('{ticker}')."
        )
    return pd.read_parquet(file_path)


def update_moex_index(ticker: str = 'IMOEX', session: Optional[requests.Session] = None) -> None:
    """
    Инкрементально обновляет локальный кэш индекса; при отсутствии файла скачивает историю с 2010 года.
    """
    ticker = ticker.upper()
    file_path = os.path.join(INDEXES_FOLDER, f"{ticker}.parquet")
    if not os.path.exists(file_path):
        save_moex_index(ticker, session=session)
        return

    existing_df = pd.read_parquet(file_path)
    start = pd.to_datetime(existing_df.index.max()).strftime('%Y-%m-%d')
    try:
        new_df = get_moex_index(ticker, start=start, session=session)
    except Exception as e:
        logger.warning(f"[WARN] {ticker}: не удалось обновить индекс — {e}")
        return

    if new_df is None or new_df.empty:
        logger.info(f"[INFO] {ticker}: нет новых данных")
        return

    new_df = new_df.copy()
    new_df.index = pd.to_datetime(new_df.index)
    new_df['ticker'] = ticker

    combined_df = pd.concat([existing_df, new_df])
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()

    tmp_path = file_path + ".tmp"
    combined_df.to_parquet(tmp_path)
    os.replace(tmp_path, file_path)
    logger.info(f"Updated index {ticker}: {start} → {combined_df.index.max().strftime('%Y-%m-%d')}")

def save_moex_stock(
    ticker: str,
    start: str = "2023-01-01",
    end: Optional[str] = None,
    session: Optional[requests.Session] = None,
    frequency: int = 24,
    out_dir: Optional[str] = None,
    calculate_market_cap_flag: bool = True,
    metadata_file: Optional[str] = None,
) -> Optional[str]:
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
    # Тикер нормализуем к верхнему регистру: пути и колонка ticker
    # должны совпадать во всех функциях независимо от регистра на входе
    ticker = ticker.upper()

    # Дефолты путей разрешаем в момент вызова, чтобы работало переопределение
    # moex.DATA_FOLDER / moex.METADATA_FILE (например, из update_data.py)
    if out_dir is None:
        out_dir = DATA_FOLDER
    if metadata_file is None:
        metadata_file = METADATA_FILE

    # нормализуем даты
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # информативный лог
    logger.info(f"[INFO] {ticker}: fetching {start} → {end}, freq={frequency}")

    try:
        df = get_moex_stock(
            ticker=ticker,
            start=start,
            end=end,
            session=session,
            frequency=frequency
        )
        if df is None or df.empty:
            logger.info(f"[SKIP] {ticker}: пустой датафрейм от ISS MOEX (делистинг/нет торгов/не та доска).")
            return None

    except ValueError as e:
        # типично: пустой ответ (например, POLY), неверный период, несуществующий тикер
        logger.info(f"[SKIP] {ticker}: {e}")
        return None
    except requests.RequestException as e:
        # сетевые/HTTP проблемы
        logger.error(f"[ERROR] {ticker}: сетевой сбой — {e}")
        return None
    except Exception as e:
        # любая иная ошибка — не валим цикл
        logger.error(f"[ERROR] {ticker}: неожиданная ошибка — {e}")
        return None

    # Рассчитываем market cap, если требуется
    if calculate_market_cap_flag:
        try:
            df = calculate_market_cap(df, ticker, metadata_file)
            if 'market_cap' in df.columns:
                logger.info(f"[INFO] {ticker}: market cap рассчитан")
        except Exception as e:
            logger.warning(f"[WARNING] {ticker}: не удалось рассчитать market cap — {e}")

    # подготовка пути
    tdir = os.path.join(out_dir, ticker.upper())
    os.makedirs(tdir, exist_ok=True)
    file_path = os.path.join(tdir, f"{ticker.upper()}.parquet")

    # атомарная запись
    tmp_path = file_path + ".tmp"
    try:
        df.to_parquet(tmp_path, index=True)
        os.replace(tmp_path, file_path)
        logger.info(f"[OK] {ticker}: {len(df):,} rows → {file_path}")
        return file_path
    except Exception as e:
        # если запись сорвалась — удалим tmp и продолжим
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        logger.error(f"[ERROR] {ticker}: не удалось сохранить Parquet — {e}")
        return None



def update_moex_stock(
    ticker: str, 
    session: Optional[requests.Session] = None,
    calculate_market_cap_flag: bool = True,
    metadata_file: Optional[str] = None,
    frequency: int = 24,
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
    frequency (int): Частота свечей для дозагрузки (24 = дневные). Должна совпадать
                     с частотой, с которой файл был сохранен изначально.
    """
    ticker = ticker.upper()

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
        new_data = get_moex_stock(ticker, start=last_date_str, session=session, frequency=frequency)
        
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
                    logger.info(f"[INFO] {ticker}: market cap пересчитан")
            except Exception as e:
                logger.warning(f"[WARNING] {ticker}: не удалось пересчитать market cap — {e}")
        
        # Атомарная запись: не оставляем битый файл при прерывании
        tmp_path = file_path + ".tmp"
        df_updated.to_parquet(tmp_path)
        os.replace(tmp_path, file_path)

        logger.info(f"Updated data for ticker: {ticker} from {last_date_str} to {datetime.today().strftime('%Y-%m-%d')}")
    else:
        logger.info(f"No existing data found for ticker: {ticker}. Please use save_moex_stock to create the initial file.")

def read_moex_stock(ticker: str, start: str = '2023-01-01', end: Optional[str] = None, session: Optional[requests.Session] = None) -> pd.DataFrame:
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
    ticker = ticker.upper()

    # Define the file path for the Parquet file
    file_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}.parquet")
    
    # Check if the Parquet file exists
    if os.path.exists(file_path):
        # Load the existing data
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded data for ticker: {ticker} from local file.")
    else:
        # If the file does not exist, create it using save_moex_stock
        logger.info(f"No local data found for ticker: {ticker}. Fetching data...")
        save_moex_stock(ticker, start, end, session)
        df = pd.read_parquet(file_path)  # Load the newly created data
    
    return df

def combine_moex_stocks(data_folder: str | None = None) -> pd.DataFrame:
    """
    Lists all parquet files in the data folder, reads them, and combines into a unified dataset.

    Parameters:
    data_folder (str | None): Папка с подпапками тикеров (<TICKER>/<TICKER>.parquet).
                              По умолчанию — DATA_FOLDER.

    Returns:
    pd.DataFrame: A DataFrame containing combined stock data from all parquet files.
    """
    folder = data_folder if data_folder is not None else DATA_FOLDER
    # Initialize an empty list to store DataFrames
    dfs = []

    # Walk through the data directory
    for ticker_dir in os.listdir(folder):
        dir_path = os.path.join(folder, ticker_dir)
        
        # Check if it's a directory
        if os.path.isdir(dir_path):
            parquet_file = os.path.join(dir_path, f"{ticker_dir}.parquet")
            
            # Check if parquet file exists
            if os.path.exists(parquet_file):
                try:
                    # Read the parquet file
                    df = pd.read_parquet(parquet_file)
                    dfs.append(df)
                    logger.info(f"Loaded data for {ticker_dir}")
                except Exception as e:
                    logger.error(f"Error loading {ticker_dir}: {e}")
    
    # Combine all DataFrames
    if dfs:
        combined_df = pd.concat(dfs, axis=0)
        logger.info(f"\nCombined {len(dfs)} stocks into unified dataset")
        logger.info(f"Total rows: {len(combined_df)}")
        return combined_df
    else:
        raise ValueError("No parquet files found in data directory")

# Кэш разобранных метаданных: ключ (абсолютный путь, mtime файла).
# Полное обновление вызывает load_shares_data на каждый тикер — без кэша
# Excel перечитывался бы ~100 раз за прогон.
_shares_cache: dict = {}


def load_shares_data(metadata_file: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает данные о количестве акций из Excel файла metadata/stock-index-base.xlsx.
    Результат кэшируется в памяти до изменения файла (по mtime).

    Parameters:
    metadata_file (str): Путь к Excel файлу с метаданными. По умолчанию METADATA_FILE.

    Returns:
    pd.DataFrame: DataFrame с колонками Code, date, Number of issued shares.
    """
    if metadata_file is None:
        metadata_file = METADATA_FILE

    if not os.path.exists(metadata_file):
        logger.warning(f"Warning: Metadata file not found: {metadata_file}")
        return pd.DataFrame()

    cache_key = (os.path.abspath(metadata_file), os.path.getmtime(metadata_file))
    cached = _shares_cache.get(cache_key)
    if cached is not None:
        return cached.copy()

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
            _shares_cache.clear()  # храним один актуальный файл
            _shares_cache[cache_key] = shares_df
            return shares_df.copy()
        else:
            logger.warning(f"Warning: Required columns not found in {metadata_file}")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error loading shares data from {metadata_file}: {e}")
        return pd.DataFrame()


def calculate_market_cap(df: pd.DataFrame, ticker: str, metadata_file: Optional[str] = None) -> pd.DataFrame:
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
        logger.warning(f"Warning: No price column found for {ticker}. Skipping market cap calculation.")
        return df
    
    # Загружаем данные о количестве акций
    shares_data = load_shares_data(metadata_file)
    if shares_data.empty:
        logger.warning(f"Warning: No shares data available. Skipping market cap calculation for {ticker}.")
        return df
    
    # Фильтруем данные для конкретного тикера
    ticker_shares = shares_data[shares_data['Code'] == ticker].copy()
    if ticker_shares.empty:
        logger.warning(f"Warning: No shares data found for {ticker}. Skipping market cap calculation.")
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
            logger.warning(f"Warning: Cannot convert index to datetime for {ticker}.")
            return df
    
    # Полный диапазон охватывает и ценовой ряд, и срезы метаданных: срез,
    # датированный раньше начала цен, должен действовать на начало периода (ffill),
    # а не теряться.
    shares_known = ticker_shares['shares']
    shares_known = shares_known[~shares_known.index.duplicated(keep='last')].sort_index()

    # Приводим число акций к пост-событийной базе (kind='shares'/'auto' в реестре):
    # ISS рестейтит историю цен после сплитов/консолидаций, а листы метаданных
    # содержат количество акций на дату листа — без поправки market_cap
    # до события кратно врет (ВТБ ×5000, ГМК и Транснефть ×100, Полюс ×10)
    _shares_adj = load_splits()
    if not _shares_adj.empty:
        _shares_adj = _shares_adj[(_shares_adj['kind'].isin(['shares', 'auto'])) &
                                  (_shares_adj['ticker'] == ticker)]
        for _adj in _shares_adj.itertuples(index=False):
            _adj_mask = shares_known.index < pd.Timestamp(_adj.date)
            if not _adj_mask.any():
                continue
            if _adj.kind == 'shares':
                # ratio в "акционной" семантике: количество акций до даты делится
                shares_known.loc[_adj_mask] = shares_known.loc[_adj_mask] / float(_adj.ratio)
            else:
                # auto: ratio в ценовой семантике; корректируем акции, только если
                # ценовой ряд рестейтнут (разрыва на дату события в ценах нет)
                if _price_jump_matches(df[price_col], _adj.date, float(_adj.ratio)):
                    continue
                shares_known.loc[_adj_mask] = shares_known.loc[_adj_mask] * float(_adj.ratio)

    date_range = pd.date_range(
        start=min(df.index.min(), shares_known.index.min()),
        end=max(df.index.max(), shares_known.index.max()),
        freq='D'
    )
    shares_series = shares_known.reindex(date_range)

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


def load_splits(splits_file: Optional[str] = None,
                external_file: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает объединенный реестр сплитов (колонки: ticker, date, ratio, kind).

    Источники:
    1. metadata/splits.csv этого проекта (явные записи, приоритет);
    2. внешний splits.json проекта dividends (EXTERNAL_SPLITS_FILE), если лежит
       рядом; его записи получают kind='auto'.

    kind:
    - 'price': скачанная история цен содержит разрыв на дату — цены до даты
       делятся на ratio при анализе (adjust_for_splits);
    - 'shares': история цен рестейтнута ISS, но число акций в старых листах
       метаданных в старой базе — количество акций до даты делится на ratio
       (calculate_market_cap);
    - 'auto': тип определяется по данным — если в ценовом ряду есть разрыв,
       соответствующий сплиту, поправка ценовая, иначе корректируются акции.
       ratio для auto — в ценовой семантике (дробление 1:10 → 10,
       консолидация 100:1 → 0.01).
    """
    if splits_file is None:
        splits_file = SPLITS_FILE
    if external_file is None:
        external_file = EXTERNAL_SPLITS_FILE

    if os.path.exists(splits_file):
        df = pd.read_csv(splits_file, parse_dates=['date'])
        if 'kind' not in df.columns:
            df['kind'] = 'price'
        df['kind'] = df['kind'].fillna('price')
    else:
        df = pd.DataFrame(columns=['ticker', 'date', 'ratio', 'kind'])

    # Внешний реестр: {"GMKN": [{"date": "...", "ratio": 100, "kind": "split"|"reverse"}]}
    ext_rows = []
    if os.path.exists(external_file):
        try:
            with open(external_file, encoding='utf-8') as f:
                ext_data = json.load(f)
            for ext_ticker, events in ext_data.items():
                for ev in events:
                    raw = float(ev['ratio'])
                    # приводим к ценовому делителю: дробление → ratio, консолидация → 1/ratio
                    divisor = raw if ev.get('kind') == 'split' else 1.0 / raw
                    ext_rows.append({'ticker': ext_ticker,
                                     'date': pd.Timestamp(ev['date']),
                                     'ratio': divisor,
                                     'kind': 'auto'})
        except Exception as e:
            logger.warning(f"[WARN] Не удалось прочитать внешний реестр сплитов {external_file}: {e}")

    if ext_rows:
        ext_df = pd.DataFrame(ext_rows)
        if not df.empty:
            # Дедупликация: явная запись из csv в пределах 45 дней имеет приоритет
            keep = []
            for row in ext_df.itertuples(index=False):
                dup = df[(df['ticker'] == row.ticker) &
                         ((df['date'] - row.date).abs() <= pd.Timedelta(days=45))]
                if dup.empty:
                    keep.append(row)
            ext_df = pd.DataFrame(keep, columns=ext_df.columns)
        if not ext_df.empty:
            df = ext_df if df.empty else pd.concat([df, ext_df], ignore_index=True)

    return df


def _price_jump_matches(prices: pd.Series, date, divisor: float) -> bool:
    """
    True, если в ценовом ряду на дате события есть разрыв, соответствующий
    сплиту с ценовым делителем divisor (история НЕ рестейтнута источником).
    Допускается расхождение до 2.5x на рыночное движение в день события.
    """
    prices = prices.dropna().sort_index()
    before = prices[prices.index < pd.Timestamp(date)]
    after = prices[prices.index >= pd.Timestamp(date)]
    if before.empty or after.empty:
        return False
    last_before = float(before.iloc[-1])
    first_after = float(after.iloc[0])
    if last_before <= 0 or first_after <= 0:
        return False
    jump = first_after / last_before
    expected = 1.0 / divisor
    return abs(math.log(jump / expected)) < math.log(2.5)


def adjust_for_splits(df: pd.DataFrame, splits_file: Optional[str] = None) -> pd.DataFrame:
    """
    Приводит ценовые колонки к пост-сплитовой базе по реестру splits.csv.

    Свечи MOEX не корректируются на сплиты: например, у T (Т-Технологии)
    20.02.2026 цена «упала» в 10 раз из-за дробления 1:10, и наивный расчет
    доходности через такой разрыв дает бессмыслицу.

    Для каждого сплита: цены (close, adj_close, open, high, low) до даты
    делятся на ratio, volume умножается. value_rub (оборот) и market_cap
    не трогаем — они согласованы во времени и без корректировки.

    Parameters:
    df (pd.DataFrame): данные с колонкой 'ticker' и датами в индексе
                       (одна бумага или combine_moex_stocks()).
    splits_file (str | None): путь к CSV реестра; по умолчанию SPLITS_FILE.

    Returns:
    pd.DataFrame: копия df со скорректированными ценами.
    """
    splits = load_splits(splits_file)
    if not splits.empty:
        splits = splits[splits['kind'].isin(['price', 'auto'])]
    if splits.empty or df.empty or 'ticker' not in df.columns:
        return df

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    price_cols = [c for c in ('close', 'adj_close', 'open', 'high', 'low') if c in df.columns]
    for row in splits.itertuples(index=False):
        ticker_mask = df['ticker'] == row.ticker
        if not ticker_mask.any():
            continue
        # auto: корректируем цены, только если в ряду реально есть разрыв
        # (иначе история уже рестейтнута источником и трогать её нельзя)
        if row.kind == 'auto' and not _price_jump_matches(
                df.loc[ticker_mask, 'close'], row.date, float(row.ratio)):
            continue
        mask = ticker_mask & (df.index < pd.Timestamp(row.date))
        if not mask.any():
            continue
        for col in price_cols:
            df.loc[mask, col] = df.loc[mask, col] / float(row.ratio)
        if 'volume' in df.columns:
            df.loc[mask, 'volume'] = df.loc[mask, 'volume'] * float(row.ratio)

    return df


def update_all_stocks(calculate_market_cap_flag: bool = True, rebuild: bool = False):
    """
    Updates data for all stocks that have existing parquet files.

    Parameters:
    calculate_market_cap_flag (bool): Если False, market cap на этом этапе не пересчитывается
        (полезно, когда пересчет всё равно делается отдельным шагом, как в update_data.py).
    rebuild (bool): Если True, история каждого тикера перескачивается целиком
        (нужно после смены источника данных, чтобы вся история была в единой методике).
    """
    # Get list of all ticker directories
    ticker_dirs = []
    for item in os.listdir(DATA_FOLDER):
        dir_path = os.path.join(DATA_FOLDER, item)
        if os.path.isdir(dir_path):
            parquet_file = os.path.join(dir_path, f"{item}.parquet")
            if os.path.exists(parquet_file):
                ticker_dirs.append(item)
    
    logger.info(f"Found {len(ticker_dirs)} stocks to update")

    # Одна HTTP-сессия на весь прогон: keep-alive вместо нового TLS-соединения на тикер
    session = requests.Session()

    # Update each stock
    for ticker in ticker_dirs:
        try:
            logger.info(f"\nUpdating {ticker}...")
            if rebuild:
                save_moex_stock(ticker, start='2002-01-01', session=session,
                                calculate_market_cap_flag=calculate_market_cap_flag)
            else:
                update_moex_stock(ticker, session=session,
                                  calculate_market_cap_flag=calculate_market_cap_flag)
        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")
    
    logger.info(f"\nUpdate completed for {len(ticker_dirs)} stocks")

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
        logger.warning("Warning: DataFrame пуст или нет колонок 'ticker'/'close'. Возвращаю как есть.")
        return df
    # Убедимся, что индекс — даты
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception:
            logger.warning("Warning: индекс не преобразуется в даты. Возвращаю без корректировки.")
            df['adj_close'] = df['close']
            return df

    ticker = df['ticker'].iloc[0]
    div_file = os.path.join(div_folder, f"{ticker}.csv")

    if not os.path.exists(div_file):
        logger.info(f"Info: No dividend file found for {ticker} at {div_file}. Setting adj_close equal to close.")
        df['adj_close'] = df['close']
        return df

    try:
        div_df = pd.read_csv(div_file, parse_dates=['closing_date'])
        div_df.dropna(subset=['closing_date'], inplace=True)
        div_df = div_df[div_df['dividend_value'] > 0]
        div_df.sort_values(by='closing_date', inplace=True)
    except Exception as e:
        logger.error(f"Error reading dividend file for {ticker}: {e}")
        df['adj_close'] = df['close']
        return df

    # Filter data and sort
    df = df.sort_index()
    div_df = div_df[div_df['closing_date'].notnull()].sort_values('closing_date')

    if div_df.empty:
        df['adj_close'] = df['close']
        return df

    # Calculate adjusted close prices using a proportional adjustment factor
    closes = df['close'].astype(float)
    adj = closes.copy()

    # Идём от последних дивидендов к ранним
    for row in div_df.iloc[::-1].itertuples(index=False):
        ex_dividend_date = row.closing_date
        dividend_value = float(row.dividend_value)

        # позиция справа: всё строго ДО экс-даты попадает под корректировку
        pos = adj.index.searchsorted(ex_dividend_date, side='right')

        # если экс-дата раньше всех наших данных — пропускаем
        if pos == 0:
            continue

        close_before = closes.iloc[pos - 1]
        if pd.isna(close_before) or close_before <= 0:
            # защита от деления на ноль/NaN
            continue

        adjustment_factor = 1.0 - (dividend_value / close_before)
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
        logger.info("No stock data found in the data folder.")
        return

    logger.info(f"Found {len(ticker_dirs)} stocks to process for adjusted close calculation.")

    for ticker in ticker_dirs:
        try:
            logger.info(f"Processing {ticker}...")
            file_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}.parquet")
            
            df = pd.read_parquet(file_path)
            
            # Calculate adjusted close (will overwrite if column already exists)
            df_adj = calculate_adj_close(df, div_folder)
            
            df_adj.to_parquet(file_path)
            
            logger.info(f"Successfully updated {ticker} with adj_close column.")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

    logger.info(f"\nCompleted processing for {len(ticker_dirs)} stocks.")


def add_market_cap_to_all_stocks(metadata_file: Optional[str] = None) -> None:
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
        logger.info("No stock data found in the data folder.")
        return
    
    logger.info(f"Found {len(ticker_dirs)} stocks to process for market cap calculation.")
    
    for ticker in ticker_dirs:
        try:
            logger.info(f"Processing {ticker}...")
            file_path = os.path.join(DATA_FOLDER, ticker, f"{ticker}.parquet")
            
            df = pd.read_parquet(file_path)
            
            # Calculate market cap (will overwrite if columns already exist)
            df_mc = calculate_market_cap(df, ticker, metadata_file)
            
            if 'market_cap' in df_mc.columns:
                df_mc.to_parquet(file_path)
                logger.info(f"Successfully updated {ticker} with market cap data.")
            else:
                logger.warning(f"Warning: Could not calculate market cap for {ticker}.")
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
    
    logger.info(f"\nCompleted processing for {len(ticker_dirs)} stocks.")

# Bonds functions

BONDS_FOLDER = os.path.join(BASE_DIR, "bonds")


def _parse_iss_table(table) -> pd.DataFrame:
    """
    Разбирает таблицу из ответа ISS MOEX в DataFrame.

    Реальный ISS возвращает {'columns': [...], 'data': [...]};
    для совместимости поддерживаются также список словарей
    и список списков с шапкой в первой строке.
    """
    if not table:
        return pd.DataFrame()
    if isinstance(table, dict):
        return pd.DataFrame(table.get('data') or [], columns=table.get('columns'))
    if isinstance(table, list) and isinstance(table[0], dict):
        return pd.DataFrame(table)
    if isinstance(table, list) and isinstance(table[0], list):
        return pd.DataFrame(table[1:], columns=table[0])
    return pd.DataFrame(table)

def get_moex_bonds_list(segment: str = 'TQCB', session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Fetches list of bonds from MOEX by segment.
    
    Parameters:
    segment (str): Bond segment, e.g., 'TQCB' (corporate), 'TQOB' (government).
    
    Returns:
    pd.DataFrame: DataFrame with bond securities data.
    """
    if session is None:
        session = requests.Session()

    # Фильтрация по доске работает только через путь /boards/<board>/:
    # одноимённый query-параметр ISS молча игнорирует
    url = f"https://iss.moex.com/iss/engines/stock/markets/bonds/boards/{segment}/securities.json"
    params = {'iss.only': 'securities'}

    try:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return _parse_iss_table(data.get('securities'))
    except Exception as e:
        raise RuntimeError(f"Error fetching bonds list: {e}")

def get_moex_bond_params(secid: str, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Fetches parameters for a specific bond.
    
    Parameters:
    secid (str): Bond security ID.
    
    Returns:
    pd.DataFrame: DataFrame with bond parameters.
    """
    if session is None:
        session = requests.Session()
    
    url = f"https://iss.moex.com/iss/engines/stock/markets/bonds/securities/{secid}.json"
    params = {'iss.only': 'securities'}

    try:
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return _parse_iss_table(data.get('securities'))
    except Exception as e:
        raise RuntimeError(f"Error fetching bond params for {secid}: {e}")

def get_moex_bond_prices(secid: str, start: str = '2023-01-01', end: Optional[str] = None, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """
    Fetches historical price data for a bond.
    
    Parameters:
    secid (str): Bond security ID.
    start (str): Start date.
    end (str): End date.
    
    Returns:
    pd.DataFrame: DataFrame with historical prices.
    """
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    
    if session is None:
        session = requests.Session()
    
    url = f"https://iss.moex.com/iss/history/engines/stock/markets/bonds/securities/{secid}.json"

    try:
        # ISS отдаёт history страницами (обычно по 100 строк) — листаем через offset
        pages = []
        offset = 0
        for _ in range(1000):  # защита от бесконечного цикла
            params = {'from': start, 'till': end, 'start': offset}
            resp = session.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            page = _parse_iss_table(data.get('history'))
            if page.empty:
                break
            pages.append(page)
            offset += len(page)

            cursor = _parse_iss_table(data.get('history.cursor'))
            if cursor.empty or 'TOTAL' not in cursor.columns:
                break  # курсора нет — считаем ответ одностраничным
            if offset >= int(cursor['TOTAL'].iloc[0]):
                break

        if not pages:
            return pd.DataFrame()

        df = pd.concat(pages, ignore_index=True)
        df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
        df.set_index('TRADEDATE', inplace=True)
        df = df[~df.index.duplicated(keep='last')]
        df['secid'] = secid

        return df
    except Exception as e:
        raise RuntimeError(f"Error fetching bond prices for {secid}: {e}")

def save_moex_bond(secid: str, start: str = '2023-01-01', end: Optional[str] = None, session: Optional[requests.Session] = None) -> None:
    """
    Saves bond data to Parquet file.
    """
    df = get_moex_bond_prices(secid, start, end, session)
    if df.empty:
        logger.warning(f"[WARN] No data for bond {secid}")
        return
    
    os.makedirs(BONDS_FOLDER, exist_ok=True)
    file_path = os.path.join(BONDS_FOLDER, f"{secid}.parquet")
    
    # Atomic write
    temp_path = file_path + ".tmp"
    df.to_parquet(temp_path)
    os.replace(temp_path, file_path)
    
    logger.info(f"[OK] Saved bond {secid} to {file_path}")

def read_moex_bond(secid: str) -> pd.DataFrame:
    """
    Reads bond data from Parquet file.
    """
    file_path = os.path.join(BONDS_FOLDER, f"{secid}.parquet")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Bond data file not found: {file_path}")
    
    df = pd.read_parquet(file_path)
    return df

def update_moex_bond(secid: str, session: Optional[requests.Session] = None) -> None:
    """
    Updates bond data from last saved date to today.
    """
    existing_df = None
    try:
        existing_df = read_moex_bond(secid)
        last_date = existing_df.index.max().strftime('%Y-%m-%d')
        start = (pd.to_datetime(last_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    except FileNotFoundError:
        start = '2023-01-01'

    end = datetime.today().strftime('%Y-%m-%d')
    if start >= end:
        logger.info(f"[INFO] Bond {secid} is up to date")
        return

    new_df = get_moex_bond_prices(secid, start, end, session)
    if new_df.empty:
        logger.info(f"[INFO] No new data for bond {secid}")
        return

    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
    else:
        combined_df = new_df
    
    os.makedirs(BONDS_FOLDER, exist_ok=True)
    file_path = os.path.join(BONDS_FOLDER, f"{secid}.parquet")
    
    temp_path = file_path + ".tmp"
    combined_df.to_parquet(temp_path)
    os.replace(temp_path, file_path)
    
    logger.info(f"[OK] Updated bond {secid}")

def calculate_ytm(price: float, face_value: float, coupon_rate: float, years_to_maturity: float, coupon_freq: int = 2) -> float:
    """
    Calculates Yield to Maturity (YTM) for a bond.
    
    Parameters:
    price (float): Current price (% of face value).
    face_value (float): Face value.
    coupon_rate (float): Annual coupon rate (%).
    years_to_maturity (float): Years to maturity.
    coupon_freq (int): Coupons per year.
    
    Returns:
    float: YTM (%).
    """
    coupon = face_value * (coupon_rate / 100) / coupon_freq
    periods = int(years_to_maturity * coupon_freq)

    if periods == 0:
        return 0

    target = price / 100 * face_value

    def _pv(annual_rate: float) -> float:
        r = annual_rate / coupon_freq
        pv_coupons = sum(coupon / (1 + r) ** i for i in range(1, periods + 1))
        return pv_coupons + face_value / (1 + r) ** periods

    # Бисекция: PV монотонно убывает по ставке, ищем ставку в [-50%, 500%]
    lo, hi = -0.5, 5.0
    if target >= _pv(lo):
        return lo * 100
    if target <= _pv(hi):
        return hi * 100

    for _ in range(200):
        mid = (lo + hi) / 2
        if _pv(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break

    return (lo + hi) / 2 * 100

def calculate_duration(price: float, face_value: float, coupon_rate: float, years_to_maturity: float, ytm: float, coupon_freq: int = 2) -> float:
    """
    Calculates modified duration.
    
    Returns:
    float: Duration in years.
    """
    coupon = face_value * (coupon_rate / 100) / coupon_freq
    periods = int(years_to_maturity * coupon_freq)
    ytm_period = ytm / 100 / coupon_freq
    
    if periods == 0:
        return 0
    
    pv_coupons = sum((i * coupon) / (1 + ytm_period)**i for i in range(1, periods+1))
    pv_face = (periods * face_value) / (1 + ytm_period)**periods
    total_pv_weighted = pv_coupons + pv_face
    
    total_pv = sum(coupon / (1 + ytm_period)**i for i in range(1, periods+1)) + face_value / (1 + ytm_period)**periods
    
    macaulay_duration = total_pv_weighted / total_pv / coupon_freq
    modified_duration = macaulay_duration / (1 + ytm_period)
    
    return modified_duration

def add_bond_metrics(df: pd.DataFrame, params: pd.Series) -> pd.DataFrame:
    """
    Adds YTM and duration to bond price DataFrame.
    
    Parameters:
    df (pd.DataFrame): Price data.
    params (pd.Series): Bond parameters.
    
    Returns:
    pd.DataFrame: DataFrame with added metrics.
    """
    face_value = float(params.get('FACEVALUE', 1000) or 1000)
    coupon_rate = float(params.get('COUPONPERCENT', 0) or 0)
    coupon_freq = 2  # Assume semi-annual

    df = df.copy()

    maturity_raw = params.get('MATDATE')
    maturity_date = pd.to_datetime(maturity_raw) if maturity_raw else pd.NaT
    if pd.isna(maturity_date):
        logger.warning("MATDATE отсутствует или пуст — ytm/duration не рассчитаны")
        df['years_to_maturity'] = float('nan')
        df['ytm'] = float('nan')
        df['duration'] = float('nan')
        return df

    df['years_to_maturity'] = (maturity_date - df.index).days / 365.25

    price_col = 'CLOSE' if 'CLOSE' in df.columns else 'WAPRICE'
    for idx in df.index:
        price_pct = float(df.at[idx, price_col])
        years = float(df.at[idx, 'years_to_maturity'])
        ytm = calculate_ytm(price_pct, face_value, coupon_rate, years, coupon_freq)
        duration = calculate_duration(price_pct, face_value, coupon_rate, years, ytm, coupon_freq)

        df.at[idx, 'ytm'] = ytm
        df.at[idx, 'duration'] = duration

    return df