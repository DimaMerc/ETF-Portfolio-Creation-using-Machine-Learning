import os
import json
import requests
import numpy as np
import pandas as pd
from config import ALPHAVANTAGE_API_KEY
import logging
import time
from datetime import datetime, timedelta
from dateutil.parser import parse  # To parse dates from strings

logger = logging.getLogger(__name__)

def fetch_economic_indicators(api_key: str = ALPHAVANTAGE_API_KEY, years: int = 5, cache_dir='api_cache') -> pd.Series:
    indicators = {
        'REAL_GDP': f"https://www.alphavantage.co/query?function=REAL_GDP&interval=quarterly&apikey={api_key}",
        'TREASURY_YIELD': f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey={api_key}",
        'INFLATION': f"https://www.alphavantage.co/query?function=INFLATION&apikey={api_key}",
        'RETAIL_SALES': f"https://www.alphavantage.co/query?function=RETAIL_SALES&apikey={api_key}"
    }

    data = {}
    cutoff_date = datetime.now() - timedelta(days=years*365)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    for indicator, url in indicators.items():
        cache_file = os.path.join(cache_dir, f"{indicator}.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                response_data = json.load(f)
            logger.info(f"Loaded {indicator} data from cache.")
        else:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    response_data = response.json()
                    # Save response to cache
                    with open(cache_file, 'w') as f:
                        json.dump(response_data, f)
                    # Wait to respect rate limits
                    time.sleep(12)  # Adjust the sleep time based on your API rate limit
                else:
                    logger.warning(f"Error fetching {indicator}: {response.status_code}")
                    response_data = {}
            except Exception as e:
                logger.error(f"Exception occurred while fetching {indicator}: {e}")
                response_data = {}

        # Check for API error messages
        if 'Error Message' in response_data or 'Note' in response_data:
            logger.warning(f"API error for {indicator}: {response_data.get('Error Message') or response_data.get('Note')}")
            data[indicator] = np.nan
            continue

        # Get the indicator data
        indicator_data = response_data.get('data', [])
        if not indicator_data:
            logger.warning(f"No data available for {indicator}")
            data[indicator] = np.nan
            continue

        # Filter data by date
        recent_data = []
        for item in indicator_data:
            try:
                item_date = parse(item['date']).date()
                if item_date >= cutoff_date.date():
                    recent_data.append(item)
            except Exception as e:
                logger.error(f"Error parsing date for {indicator}: {e}")
                continue

        if recent_data:
            # Assuming data is sorted from most recent to oldest
            latest_value = float(recent_data[0]['value'])
            data[indicator] = latest_value
        else:
            logger.warning(f"No recent data available for {indicator}")
            data[indicator] = np.nan

    return pd.Series(data)
