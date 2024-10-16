import pandas as pd
import numpy as np

def apply_risk_management(portfolio_values: pd.Series, stop_loss: float, trailing_stop: float, re_entry_threshold: float = 0.05) -> pd.Series:
    print(f"Applying risk management. Stop loss: {stop_loss}, Trailing stop: {trailing_stop}")
    print(f"Initial portfolio value: {portfolio_values.iloc[0]}")
    
    highest_value = portfolio_values.iloc[0]
    stop_level = highest_value * (1 - stop_loss)
    trailing_stop_level = highest_value * (1 - trailing_stop)
    
    in_market = True
    exit_value = None
    
    for i in range(1, len(portfolio_values)):
        current_value = portfolio_values.iloc[i]
        
        if in_market:
            if current_value > highest_value:
                highest_value = current_value
                trailing_stop_level = highest_value * (1 - trailing_stop)
            
            if current_value < stop_level or current_value < trailing_stop_level:
                print(f"Exit triggered at index {i}. Current value: {current_value}, Stop level: {stop_level}, Trailing stop level: {trailing_stop_level}")
                in_market = False
                exit_value = current_value
        else:
            if current_value > exit_value * (1 + re_entry_threshold):
                print(f"Re-entry triggered at index {i}. Current value: {current_value}")
                in_market = True
                highest_value = current_value
                stop_level = highest_value * (1 - stop_loss)
                trailing_stop_level = highest_value * (1 - trailing_stop)
        
        if not in_market:
            portfolio_values.iloc[i] = exit_value
    
    print(f"Final portfolio value: {portfolio_values.iloc[-1]}")
    return portfolio_values

def apply_risk_management_to_portfolio(portfolio_values: pd.DataFrame, stop_loss: float, trailing_stop: float) -> pd.DataFrame:
    if isinstance(portfolio_values, pd.Series):
        return apply_risk_management(portfolio_values, stop_loss, trailing_stop)
    elif isinstance(portfolio_values, pd.DataFrame):
        return portfolio_values.apply(lambda col: apply_risk_management(col, stop_loss, trailing_stop))
    else:
        raise ValueError("Input must be a pandas Series or DataFrame")