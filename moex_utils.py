# moex_utils.py

import requests
import apimoex
import pandas as pd
from datetime import datetime

def get_moex_stock(ticker: str, start: str = '2023-01-01', end: str = None, session: requests.Session = None) -> pd.DataFrame:
    """
    Fetches stock data from the Moscow Exchange (MOEX) for a given ticker symbol within a specified date range.
    Parameters:
    ticker (str): The ticker symbol of the stock to fetch data for.
    start (str): The start date for the data in 'YYYY-MM-DD' format. Default is '2023-01-01'.
    end (str): The end date for the data in 'YYYY-MM-DD' format. Default is None, which sets the end date to today.
    session (requests.Session): An optional requests session to use for making the API call. Default is None, which creates a new session.
    Returns:
    pd.DataFrame: A DataFrame containing the stock data with columns 'date', 'value_rub', and 'volume'.
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
        # Fetch data from the MOEX API
        data = apimoex.get_market_candles(session=session, security=ticker, start=start, end=end)
        
        # Check if the response data is empty
        if not data:
            raise ValueError("The API response is empty.")
        
            df.rename(columns={'begin': 'date', 'value': 'value_rub'}, inplace=True)
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
