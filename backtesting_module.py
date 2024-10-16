# backtesting_module.py

import pandas as pd
import numpy as np
from typing import Dict, Tuple

def backtest_etf_portfolio(weights, price_data, initial_capital, rebalance_frequency, transaction_cost):
    if price_data.empty:
        print("Price data is empty.")
        return pd.Series(), pd.DataFrame()
    
    # Ensure the index is datetime and sorted
    if not isinstance(price_data.index, pd.DatetimeIndex):
        price_data.index = pd.to_datetime(price_data.index)
    price_data.sort_index(inplace=True)
    
    # Initialize portfolio value series
    portfolio_value = pd.Series(index=price_data.index, dtype=float)
    portfolio_value.iloc[0] = initial_capital
    
    # Initialize holdings and cash
    holdings = {symbol: 0 for symbol in weights}
    cash = initial_capital
    transactions_list = []
    
    # Set last rebalance date to a date before the first date in price_data
    last_rebalance = price_data.index[0] - pd.DateOffset(days=1)
    
    # Now that holdings is defined, we can print it
    print(f"Initial holdings: {holdings}")
    
    for date in price_data.index:
        # Ensure date is a Timestamp
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)
        
        # Determine if we need to rebalance on this date
        need_rebalance = False
        if rebalance_frequency == 'M':
            if date.month != last_rebalance.month or date == price_data.index[0]:
                need_rebalance = True
        elif rebalance_frequency == 'W':
            if date.isocalendar()[1] != last_rebalance.isocalendar()[1] or date == price_data.index[0]:
                need_rebalance = True
        elif rebalance_frequency == 'D':
            need_rebalance = True
        
        if need_rebalance:
            # Calculate total portfolio value
            total_portfolio_value = cash + sum(
                holdings[symbol] * price_data.loc[date, symbol]
                for symbol in holdings if symbol in price_data.columns
            )
            print(f"\nRebalancing on {date.date()}. Total portfolio value: ${total_portfolio_value:.2f}")
            total_transaction_cost = 0.0
            new_holdings = {}
            
            for symbol, weight in weights.items():
                if symbol not in price_data.columns:
                    print(f"Price data for {symbol} not available on {date.date()}. Skipping.")
                    continue
                price = price_data.loc[date, symbol]
                target_value = total_portfolio_value * weight
                target_shares = int(target_value // price)
                current_shares = holdings.get(symbol, 0)
                trade_shares = target_shares - current_shares
                trade_value = trade_shares * price
                trade_cost = abs(trade_value) * transaction_cost
                total_transaction_cost += trade_cost
                cash -= trade_value + trade_cost  # Update cash
                
                # Update holdings
                new_holdings[symbol] = target_shares
                
                # Record transaction
                if trade_shares != 0:
                    transactions_list.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'buy' if trade_shares > 0 else 'sell',
                        'shares': abs(trade_shares),
                        'price': price,
                        'value': abs(trade_value),
                        'cost': trade_cost
                    })
                    print(f"  {'Bought' if trade_shares > 0 else 'Sold'} {abs(trade_shares)} shares of {symbol} at ${price:.2f} per share.")
            
            # Update holdings with new holdings
            holdings.update(new_holdings)
            last_rebalance = date
            print(f"  Cash after rebalancing: ${cash:.2f}")
        
        # Calculate total holdings value
        total_holdings_value = sum(
            holdings[symbol] * price_data.loc[date, symbol]
            for symbol in holdings if symbol in price_data.columns
        )
        # Update portfolio value
        portfolio_value.loc[date] = cash + total_holdings_value
        # Debug print for daily portfolio value
        print(f"Date: {date.date()}, Portfolio Value: ${portfolio_value.loc[date]:.2f}, Cash: ${cash:.2f}, Holdings Value: ${total_holdings_value:.2f}")
    
    transactions = pd.DataFrame(transactions_list)
    
    # After the loop, check the portfolio_value
    print("\nFinal portfolio values:")
    print(portfolio_value)
    
    return portfolio_value, transactions


def calculate_performance_metrics(returns: pd.Series, risk_free_rate: float) -> Dict[str, float]:
    """
    Calculates performance metrics for the portfolio.
    """
    total_return = (returns + 1).prod() - 1
    annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
    sortino_ratio = (annualized_return - risk_free_rate) / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() != 0 else 0
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    
    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown
    }
