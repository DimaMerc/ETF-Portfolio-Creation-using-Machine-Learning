# main_script.py

import pandas as pd
import numpy as np
from typing import Dict, List
import logging
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from keras import callbacks


import keras
import tensorflow as tf
from keras import backend as K
import torch
from torch_geometric.data import Data
from matplotlib.dates import YearLocator, DateFormatter
from datetime import datetime, timedelta
from config import (
    START_DATE, END_DATE, ALPHAVANTAGE_API_KEY, WINDOW_SIZE, BATCH_SIZE,
    EPOCHS, RISK_FREE_RATE, TOP_N, STOP_LOSS, TRAILING_STOP,
    INITIAL_CAPITAL, TRANSACTION_COST, REBALANCE_FREQUENCY, TOP_N_HOLDINGS
)
from holdings_data_module import (
    fetch_etf_holdings, fetch_equity_data, fetch_etf_price_data, calculate_technical_indicators
)
from feature_engineering_module import (
     generate_gcn_features, create_graph_data
)
from ml_model_module import CustomModel
from backtesting_module import (
    backtest_etf_portfolio, calculate_performance_metrics
)
from sklearn.preprocessing import StandardScaler
from data_preparation import create_sequences, preprocess_data

from risk_management import apply_risk_management, apply_risk_management_to_portfolio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def split_data_by_date(df, train_end_date, validation_end_date):
    # Assume 'date' is already the index
    df.index = pd.to_datetime(df.index)
    train_df = df[df.index <= train_end_date]
    val_df = df[(df.index > train_end_date) & (df.index <= validation_end_date)]
    test_df = df[df.index > validation_end_date]
    return train_df, val_df, test_df


def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Read ETF symbols from CSV file
    etf_csv_file = 'etf_symbols.csv'
    df = pd.read_csv(etf_csv_file)
    etf_symbols = df['Symbol'].tolist() if 'Symbol' in df.columns else df['symbol'].tolist()
    
    # Fetch holdings for each ETF
    all_holdings = {}
    for etf_symbol in etf_symbols:
        logger.info(f"Fetching holdings for ETF {etf_symbol}")
        holdings_df = fetch_etf_holdings(etf_symbol, ALPHAVANTAGE_API_KEY)
        if not holdings_df.empty:
            holdings_df['weight'] = holdings_df['weight'].astype(float)
            holdings_df.sort_values(by='weight', ascending=False, inplace=True)
            # Limit to top N holdings
            top_holdings = holdings_df.head(TOP_N_HOLDINGS)
            all_holdings[etf_symbol] = top_holdings
        else:
            logger.warning(f"No holdings data for {etf_symbol}")
    
    # Get unique equity symbols from holdings
    equity_symbols = set()
    for holdings_df in all_holdings.values():
        equity_symbols.update(holdings_df['symbol'].tolist())
    equity_symbols = list(equity_symbols)
    logger.info(f"Total unique equities to fetch: {len(equity_symbols)}")
    
    # Fetch and prepare equity data
    equity_data = {}
    for symbol in equity_symbols:
        df = fetch_equity_data(symbol, START_DATE, END_DATE, ALPHAVANTAGE_API_KEY)
        if not df.empty:
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            equity_data[symbol] = df
        else:
            logger.warning(f"No data for equity {symbol}")

    # Create graph data with edge weights
    graph_data, node_mapping = create_graph_data(all_holdings, equity_data)

    # Generate GCN embeddings
    embeddings = generate_gcn_features(graph_data)

    # Extract ETF embeddings
    etf_embedding_dict = {}
    for etf_symbol in all_holdings.keys():
        idx = node_mapping[etf_symbol]
        embedding = embeddings[idx].numpy()
        etf_embedding_dict[etf_symbol] = embedding
    
    # Fetch ETF price data
    etf_price_data = {}
    for etf_symbol in etf_symbols:
        df = fetch_equity_data(etf_symbol, START_DATE, END_DATE, ALPHAVANTAGE_API_KEY)
        if not df.empty:
            etf_price_data[etf_symbol] = df
        else:
            logger.warning(f"No price data for ETF {etf_symbol}")
    
    # Generate GCN features
    embeddings = generate_gcn_features(graph_data)
   
    #print(f"gcn_feature.shape for ETF {etf_symbol}: {gcn_feature.shape}")

    # etf_gcn_features is a dictionary mapping ETF symbols to their GCN embeddings
    
    # Prepare LSTM input data
    lstm_sequences = []
    lstm_targets = []
    gcn_inputs = []
    etf_symbols_with_data = []
    for idx, etf_symbol in enumerate(etf_symbols):
        etf_prices = etf_price_data.get(etf_symbol)
        if etf_prices is not None and len(etf_prices) > WINDOW_SIZE:
            # GCN feature for this ETF
            gcn_feature = etf_embedding_dict.get(etf_symbol)
            if gcn_feature is not None:
                # Extract adjusted close prices
                prices = etf_prices['adjusted_close'].values
                # Create sequences
                sequences, targets = create_sequences(prices, WINDOW_SIZE)
                # Append sequences and targets
                lstm_sequences.append(sequences)
                lstm_targets.append(targets)
                # Repeat GCN feature to match the number of sequences
                gcn_input = np.tile(gcn_feature, (len(sequences), 1))
                gcn_inputs.append(gcn_input)
                etf_symbols_with_data.append(etf_symbol)
            else:
                logger.warning(f"No GCN feature for ETF {etf_symbol}")
        else:
            logger.warning(f"Not enough price data for ETF {etf_symbol}")

    print(f"gcn_input.shape before appending: {gcn_input.shape}")

    # Combine data
    if lstm_sequences and gcn_inputs:
        X_lstm = np.concatenate(lstm_sequences, axis=0)
        y = np.concatenate(lstm_targets, axis=0)
        X_gcn = np.concatenate(gcn_inputs, axis=0)
    else:
        logger.error("No data available for training.")
        return
    print(f"X_gcn.shape after concatenation: {X_gcn.shape}")

    # Ensure X_lstm and X_gcn have matching lengths
    assert X_lstm.shape[0] == X_gcn.shape[0] == y.shape[0]
    
    # Reshape X_lstm to have the shape (num_samples, WINDOW_SIZE, 1)
    X_lstm = X_lstm.reshape(-1, WINDOW_SIZE, 1)

    
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_lstm_train, X_lstm_val, X_gcn_train, X_gcn_val, y_train, y_val = train_test_split(
        X_lstm, X_gcn, y, test_size=0.2, random_state=42
    )

    print("Before preprocessing:")
    print(f"X_lstm_train shape: {X_lstm_train.shape}")
    print(f"First element of X_lstm_train:\n{X_lstm_train[0]}")
    print(f"X_gcn_train shape: {X_gcn_train.shape}")
    print(f"First element of X_gcn_train:\n{X_gcn_train[0]}")
    print(f"y_train shape: {y_train.shape}")

    # Preprocess the data
    X_lstm_train, X_gcn_train, y_train, X_lstm_val, X_gcn_val, y_val = preprocess_data(
        X_lstm_train, X_gcn_train, y_train, X_lstm_val, X_gcn_val, y_val
    )

    print("After preprocessing:")
    print(f"X_lstm_train shape: {X_lstm_train.shape}")
    print(f"First element of X_lstm_train:\n{X_lstm_train[0]}")
    print(f"X_gcn_train shape: {X_gcn_train.shape}")
    print(f"First element of X_gcn_train:\n{X_gcn_train[0]}")
    print(f"y_train shape: {y_train.shape}")


    nan_count_gcn = np.isnan(X_gcn).sum()
    inf_count_gcn = np.isinf(X_gcn).sum()
    print(f"X_gcn contains {nan_count_gcn} NaNs and {inf_count_gcn} Infs.")

    if nan_count_gcn > 0 or inf_count_gcn > 0:
        print("Handling NaNs and Infs in X_gcn...")
        X_gcn = np.nan_to_num(X_gcn, nan=0.0, posinf=1e10, neginf=-1e10)


    

    
    print(f"X_lstm_train.shape: {X_lstm_train.shape}")  # Should be (num_samples, 60, 1)
    print(f"X_gcn_train.shape: {X_gcn_train.shape}")    # Should be (num_samples, 32)


    # Build and train the combined model
     # Build and compile the model
    # Build and compile the model
 
    # Assuming X_lstm_train, X_gcn_train, y_train are prepared
  

# Assuming X_lstm_train, X_gcn_train, y_train are prepared
    print("y_train shape:", y_train.shape)
    print("X_lstm_train shape:", X_lstm_train.shape)
    print("X_gcn_train shape:", X_gcn_train.shape)

    # Handle NaN and Inf values in X_gcn_train
    nan_count = np.isnan(X_gcn_train).sum()
    inf_count = np.isinf(X_gcn_train).sum()
    print(f"X_gcn_train contains {nan_count} NaNs and {inf_count} Infs.")
    if nan_count > 0 or inf_count > 0:
        print("Handling NaNs and Infs in X_gcn_train...")
        X_gcn_train = np.nan_to_num(X_gcn_train, nan=0.0, posinf=1e10, neginf=-1e10)

    # Manually split the data into training and validation sets
    val_split = 0.2
    split_idx = int(len(X_lstm_train) * (1 - val_split))

    X_lstm_train, X_lstm_val = X_lstm_train[:split_idx], X_lstm_train[split_idx:]
    X_gcn_train, X_gcn_val = X_gcn_train[:split_idx], X_gcn_train[split_idx:]
    y_train, y_val = y_train[:split_idx], y_train[split_idx:]

    # Create and compile the model
    K.clear_session()
    model = CustomModel(lstm_input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2]), 
                        gcn_input_shape=(X_gcn_train.shape[1],))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(name='mae')]
    )

    # Summary of the model
    model.model.summary()

    # Prepare the datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'lstm_input': X_lstm_train, 'gcn_input': X_gcn_train},
        y_train
    )).batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        {'lstm_input': X_lstm_val, 'gcn_input': X_gcn_val},
        y_val
    )).batch(BATCH_SIZE)



# Define callbacks
    model_file = 'best_model.keras'  # You can change this path as needed
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=False,
        verbose=1
    )
    checkpoint=keras.callbacks.ModelCheckpoint(
        model_file, 
        monitor="val_loss", 
        mode="min", 
        save_best_only=True, 
        verbose=1
    )


    # Train the model
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=val_dataset,
        callbacks=[early_stopping, checkpoint]
    )

     # After training, load the best model
    #best_model = keras.models.load_model(model_file)

    custom_objects = {"CustomModel": CustomModel}
    with keras.utils.custom_object_scope(custom_objects):
        best_model = keras.models.load_model(model_file)

      # Make predictions using the best model
    predictions = best_model.predict(
        {'lstm_input': X_lstm_val, 'gcn_input': X_gcn_val},
        batch_size=BATCH_SIZE
    )
    print("Predictions shape:", predictions.shape)

    # Evaluate the best model
    evaluation = best_model.evaluate(val_dataset)
    print("Validation loss:", evaluation[0])
    print("Validation MAE:", evaluation[1])


    # Plot training history
    plot_training_history(history)
    
    # Make predictions
    print(f"X_lstm_val shape: {X_lstm_val.shape}")  # Should be (num_samples, time_steps, features)
    print(f"X_gcn_val shape: {X_gcn_val.shape}")    # Should be (num_samples, gcn_feature_size)

    

    
    # Map predictions to ETFs
    # Since we have sequences from multiple ETFs, aggregate predictions per ETF
    etf_predictions = {}
    start_idx = 0
    for idx, etf_symbol in enumerate(etf_symbols_with_data):
        num_sequences = lstm_sequences[idx].shape[0]
        etf_pred = predictions[start_idx:start_idx + num_sequences].mean()
        etf_predictions[etf_symbol] = etf_pred
        start_idx += num_sequences
    
    # Select top ETFs based on predictions
    sorted_etfs = sorted(etf_predictions.items(), key=lambda x: x[1], reverse=True)
    selected_etf_symbols = [etf for etf, pred in sorted_etfs[:TOP_N]]
    weights = {symbol: 1.0 / TOP_N for symbol in selected_etf_symbols}
    
    # Prepare price data for backtesting
    price_data = pd.DataFrame()
    for symbol in selected_etf_symbols:
        df = etf_price_data[symbol]
        df = df[~df.index.duplicated(keep='first')]
        price_data[symbol] = df['adjusted_close']
    price_data.dropna(inplace=True)
    
    # Backtest the ETF portfolio
    portfolio_values, transactions = backtest_etf_portfolio(
        weights=weights,
        price_data=price_data,
        initial_capital=INITIAL_CAPITAL,
        rebalance_frequency=REBALANCE_FREQUENCY,
        transaction_cost=TRANSACTION_COST
    )
    
    print("Portfolio values before risk management:")
    print(portfolio_values.head())
    print(f"Type of portfolio_values: {type(portfolio_values)}")

    # Apply risk management
    if isinstance(portfolio_values, pd.Series):
        portfolio_values_with_risk_mgmt = apply_risk_management_to_portfolio(portfolio_values, STOP_LOSS, TRAILING_STOP)
    elif isinstance(portfolio_values, pd.DataFrame):
        portfolio_values_with_risk_mgmt = apply_risk_management_to_portfolio(portfolio_values, STOP_LOSS, TRAILING_STOP)
    else:
        raise ValueError("portfolio_values must be a pandas Series or DataFrame")
    
    print("Portfolio values after risk management:")
    print(portfolio_values_with_risk_mgmt.head())
    
# original sans risk management

    #performance_metrics = calculate_performance_metrics(portfolio_values.pct_change().dropna(), RISK_FREE_RATE)
    
    # After backtesting
    print("Portfolio values (with risk management):")
    print(portfolio_values_with_risk_mgmt)
    print(f"Portfolio values date range: {portfolio_values_with_risk_mgmt.index.min()} to {portfolio_values_with_risk_mgmt.index.max()}")
    print(f"Number of data points in portfolio_values: {len(portfolio_values_with_risk_mgmt)}")
    
    if not transactions.empty:
        total_transactions = len(transactions)
        total_transaction_cost = transactions['cost'].sum()
        logger.info(f"Total Number of Transactions: {total_transactions}")
        logger.info(f"Total Transaction Cost: ${total_transaction_cost:.2f}")
    
        # Save transactions to CSV
        transactions.to_csv('transactions_log.csv', index=False)
        logger.info("Transaction details saved to transactions_log.csv")
    else:
        logger.info("No transactions were made during the backtesting period.")
    
    # Calculate returns (using risk-managed portfolio values)
    returns = portfolio_values_with_risk_mgmt.pct_change().dropna()
    
    print("Portfolio returns (with risk management):")
    print(returns)
    print(f"Number of data points in returns: {len(returns)}")
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(returns, RISK_FREE_RATE)
    
    # Fetch SPY data for benchmark comparison
    spy_df = fetch_equity_data('SPY', START_DATE, END_DATE, ALPHAVANTAGE_API_KEY)
    spy_df = spy_df[~spy_df.index.duplicated(keep='first')]
    spy_prices = spy_df['adjusted_close']
    spy_prices.dropna(inplace=True)
    
    # Calculate SPY portfolio value
    spy_shares = INITIAL_CAPITAL / spy_prices.iloc[0]
    spy_portfolio_values = spy_shares * spy_prices
    
    # Align date ranges
    start_date = max(portfolio_values_with_risk_mgmt.index.min(), spy_portfolio_values.index.min())
    end_date = min(portfolio_values_with_risk_mgmt.index.max(), spy_portfolio_values.index.max())
    portfolio_values_with_risk_mgmt = portfolio_values_with_risk_mgmt.loc[start_date:end_date]
    spy_portfolio_values = spy_portfolio_values.loc[start_date:end_date]
    
    # Calculate SPY returns
    spy_returns = spy_portfolio_values.pct_change().dropna()
    
    # Calculate performance metrics for SPY
    spy_performance_metrics = calculate_performance_metrics(spy_returns, RISK_FREE_RATE)
    
    # Log performance metrics
    logger.info("ETF Portfolio Performance Metrics (with risk management):")
    for metric, value in performance_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info("\nSPY Buy-and-Hold Performance Metrics:")
    for metric, value in spy_performance_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Plot portfolio values comparison
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values_with_risk_mgmt.index, portfolio_values_with_risk_mgmt.values, label='ETF Portfolio Value (with risk management)')
    plt.plot(spy_portfolio_values.index, spy_portfolio_values.values, label='SPY Portfolio Value')
    plt.title('Portfolio Value Over Time (with Risk Management)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()