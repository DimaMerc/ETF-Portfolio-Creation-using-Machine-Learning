# config.py

# Data parameters
START_DATE = "2010-01-01"
END_DATE = "2024-10-14"

WINDOW_SIZE = 60
BATCH_SIZE = 32
EPOCHS = 25
RISK_FREE_RATE = 0.03
TOP_N = 10
INITIAL_CAPITAL = 100000
REBALANCE_FREQUENCY = 'M'
TRANSACTION_COST = 0.001
STOP_LOSS = 0.15
TRAILING_STOP = 0.20
TOP_N_HOLDINGS = 10  # Limit to top 5 holdings per ETF
# Path to the output file for ETF weights after each rebalancing
ETF_OUTPUT_FILE = 'etf_weights_rebalancing.csv'

# API keys
ALPHAVANTAGE_API_KEY = "80HLC8UDC38Z6HVE"
