# holdings_data_module.py

import pandas as pd
import numpy as np
import requests
import logging
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from config import ALPHAVANTAGE_API_KEY
import pandas_ta as ta

logger = logging.getLogger(__name__)

# Directory to cache API responses
CACHE_DIR = 'api_cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def fetch_etf_holdings(etf_symbol: str, api_key: str = ALPHAVANTAGE_API_KEY) -> pd.DataFrame:
    """
    Fetches ETF holdings data from AlphaVantage.
    """
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'ETF_PROFILE',
        'symbol': etf_symbol,
        'apikey': api_key
    }

    cache_file = os.path.join(CACHE_DIR, f"{etf_symbol}_holdings.json")

    # Check if data is cached
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded holdings data for {etf_symbol} from cache.")
    else:
        response = requests.get(base_url, params=params)
        data = response.json()
        if 'Error Message' in data:
            logger.error(f"Error Message from API for {etf_symbol}: {data['Error Message']}")
            return pd.DataFrame()
        if 'Note' in data:
            logger.warning(f"Note from API for {etf_symbol}: {data['Note']}")
            return pd.DataFrame()
        # Save data to cache
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        # Wait to respect rate limits
        time.sleep(12)  # Adjust as per your API limits

    # Parse holdings data from response
    if 'holdings' in data:
        holdings_data = data['holdings']
        holdings_df = pd.DataFrame(holdings_data)
        holdings_df['ETF'] = etf_symbol
        return holdings_df
    else:
        logger.error(f"No holdings data found for {etf_symbol}")
        return pd.DataFrame()

def fetch_etf_price_data(etf_symbols: List[str], start_date: str, end_date: str, api_key: str = ALPHAVANTAGE_API_KEY) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical price data for ETFs from AlphaVantage.
    """
    etf_data = {}
    for symbol in etf_symbols:
        df = fetch_equity_data(symbol, start_date, end_date, api_key)
        if not df.empty:
            etf_data[symbol] = df
        else:
            logger.warning(f"No data for ETF {symbol}")
    return etf_data

def fetch_equity_data(symbol: str, start_date: str, end_date: str, api_key: str = ALPHAVANTAGE_API_KEY) -> pd.DataFrame:
    """
    Fetches historical price data for equities from AlphaVantage.
    """
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'outputsize': 'full',
        'apikey': api_key
    }

    cache_file = os.path.join(CACHE_DIR, f"{symbol}_daily_adjusted.json")

    # Check if data is cached
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded data for {symbol} from cache.")
    else:
        response = requests.get(base_url, params=params)
        data = response.json()
        if 'Error Message' in data:
            logger.error(f"Error Message from API for {symbol}: {data['Error Message']}")
            return pd.DataFrame()
        if 'Note' in data:
            logger.warning(f"Note from API for {symbol}: {data['Note']}")
            return pd.DataFrame()
        # Save data to cache
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        # Wait to respect rate limits
        time.sleep(12)  # Adjust as per your API limits

    ts_key = 'Time Series (Daily)'
    if ts_key in data:
        ts_data = data[ts_key]
        df = pd.DataFrame.from_dict(ts_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df = df.loc[start_date:end_date]
        df.rename(columns={
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. adjusted close': 'adjusted_close',
            '6. volume': 'volume',
            '7. dividend amount': 'dividend_amount',
            '8. split coefficient': 'split_coefficient'
        }, inplace=True)
        numeric_columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        # Ensure date is the index
        df.index.name = 'date'
        return df
    else:
        logger.error(f"Time series key '{ts_key}' not found in API response for {symbol}.")
        return pd.DataFrame()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for the equity data.
    """
    

    df = df.copy()

    # Ensure index is datetime type and sorted
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Ensure adjusted_close is present
    if 'adjusted_close' not in df.columns:
        logger.error("Adjusted close price not found in DataFrame.")
        return pd.DataFrame()

    df['SMA_20'] = df['adjusted_close'].rolling(window=20).mean()
    df['EMA_20'] = df['adjusted_close'].ewm(span=20, adjust=False).mean()
    df['RSI_14'] = ta.rsi(df['adjusted_close'], length=14)
    macd = ta.macd(df['adjusted_close'])
    df['MACD'] = macd['MACD_12_26_9']
    bbands = ta.bbands(df['adjusted_close'], length=20)
    df['BBANDS_upper'] = bbands['BBU_20_2.0']
    df['BBANDS_middle'] = bbands['BBM_20_2.0']
    df['BBANDS_lower'] = bbands['BBL_20_2.0']
    df['ADX_14'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
     # Additional features
    df['ROC_10'] = ta.roc(df['adjusted_close'], length=10)
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['OBV'] = ta.obv(df['close'], df['volume'])

    # Handle any NaNs created by indicator calculations
    df = df.ffill().bfill()  # Forward fill, then backward fill for any remaining NaNs

    
    return df

def fetch_and_prepare_equity_data(equity_symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetches and prepares data for a list of equity symbols.
    """
    equity_data = {}
    for symbol in equity_symbols:
        logger.info(f"Fetching data for equity {symbol}")
        df = fetch_equity_data(symbol, start_date, end_date)
        if not df.empty:
            # Ensure date is set as index and in correct format
            if 'date' in df.columns:
                logger.info(f"Data fetched for {symbol}. Columns: {df.columns}")
                logger.info(f"Index type: {type(df.index)}")
                df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            
            # Verify the date range
            logger.info(f"Technical indicators calculated for {symbol}. Columns: {df.columns}")
            logger.info(f"Date range for {symbol}: {df.index.min()} to {df.index.max()}")
            
            equity_data[symbol] = df
        else:
            logger.warning(f"No data for equity {symbol}")
    return equity_data


def fetch_spy_data(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    symbol = 'SPY'
    spy_data = fetch_equity_data(symbol, start_date, end_date, api_key)
    return spy_data

