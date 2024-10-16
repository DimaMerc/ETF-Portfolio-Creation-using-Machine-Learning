import pandas as pd
import numpy as np
from typing import Dict, Tuple
from holdings_data_module import fetch_etf_holdings, fetch_and_prepare_equity_data, fetch_spy_data
import requests
import logging

logger = logging.getLogger(__name__)

def fetch_and_prepare_data(start_date: str, end_date: str, api_key: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    # Read ETF symbols from CSV file
    etf_csv_file = 'etf_symbols.csv'
    df = pd.read_csv(etf_csv_file)
    etf_symbols = df['Symbol'].tolist() if 'Symbol' in df.columns else df['symbol'].tolist()

    # Fetch holdings for each ETF
    all_holdings = {}
    for etf_symbol in etf_symbols:
        holdings_df = fetch_etf_holdings(etf_symbol, api_key)
        if not holdings_df.empty:
            holdings_df['weight'] = holdings_df['weight'].astype(float)
            holdings_df.sort_values(by='weight', ascending=False, inplace=True)
            all_holdings[etf_symbol] = holdings_df

    # Get unique equity symbols from holdings
    equity_symbols = set()
    for holdings_df in all_holdings.values():
        equity_symbols.update(holdings_df['symbol'].tolist())
    equity_symbols = list(equity_symbols)

    # Fetch and prepare equity data
    equity_data = fetch_and_prepare_equity_data(equity_symbols, start_date, end_date)

    # Fetch SPY data for benchmark comparison
    spy_data = fetch_spy_data(start_date, end_date, api_key)

    return all_holdings, equity_data, spy_data