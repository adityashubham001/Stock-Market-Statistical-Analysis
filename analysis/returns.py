import numpy as np
import pandas as pd

#################################### Returns Calculations #############################

def log_returns(df: pd.DataFrame) -> pd.Series:                                  # Function for calculating logarithmic (continuously compounded) returns of a stock
    return np.log(df["Close"] / df["Close"].shift(1)).dropna()                   # Uses today’s Close price vs yesterday’s Close price


def simple_returns(df: pd.DataFrame) -> pd.Series:                               # For further use in reporting, dashboards
    return df["Close"].pct_change().dropna()


################################### PERIOD ########################################

def cumulative_returns_simple(returns: pd.Series) -> pd.Series:                  # Compute cumulative returns from a series of simple (percentage) returns.
    return (1 + returns).cumprod() - 1                                           # Assumes returns are reinvested each period.



def cumulative_returns_log(returns: pd.Series) -> pd.Series:                    # computes cumulative returns from log (continuously compounded) returns.
    return np.exp(returns.cumsum()) - 1


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:    # It computes the annualized return (CAGR) from a series of simple periodic returns
    total_return = (1 + returns).prod() - 1                                         ## Compute total compounded return 
    n_periods = returns.shape[0]                                                    ## Count number of periods
    return (1 + total_return) ** (periods_per_year / n_periods) - 1                 # Count number of periods        


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:        ## computes annualized volatility
    return returns.std() * np.sqrt(periods_per_year)



######################### RISK-ADJUSTED METRICS ###############################

def sharpe_ratio(                                                                     # Computes Sharpe ratio          
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    excess_returns = returns - risk_free_rate / periods_per_year                        # Measures return above risk-free
    ann_return = annualized_return(excess_returns, periods_per_year)                    # Computes CAGR
    ann_vol = annualized_volatility(returns, periods_per_year)                          # Computes annualized volatility
    return np.nan if ann_vol == 0 else ann_return / ann_vol                             # Avoids division by zero


def max_drawdown(returns: pd.Series) -> float:                                                 # Computes maximum drawdown
    cumulative = (1 + returns).cumprod()                                                # Computes cumulative returns
    peak = cumulative.cummax()                                                          # Tracks the highest value reached so far
    drawdown = (cumulative - peak) / peak                                                   # Drawdown(t) = W(t) - max(W)   /  max(W)
    return drawdown.min()                                                               # Returns the worst (most negative) drawdown


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:                            # Computes Calmar ratio
    ann_return = annualized_return(returns, periods_per_year)                                            # Computes CAGR
    max_dd = abs(max_drawdown(returns))                                                                 # Computes maximum drawdown
    return np.nan if max_dd == 0 else ann_return / max_dd                                               # Avoids division by zero


############################### ROLLING METRICS ##############################


def rolling_volatility(                                                                 # Computes rolling volatility            
    returns: pd.Series,                                                                 # Returns series
    window: int = 21,                                                                   # Rolling window
    periods_per_year: int = 252                                                         # Number of periods per year        
) -> pd.Series:                                                                         
    return returns.rolling(window).std() * np.sqrt(periods_per_year)                    # Returns rolling volatility                


def rolling_sharpe_ratio(                                                                # Computes rolling Sharpe ratio   
    returns: pd.Series,                                                                     # Returns series
    window: int = 21,                                                                       # Rolling window
    risk_free_rate: float = 0.0,                                                            # Risk-free rate
    periods_per_year: int = 252                                                                 # Number of periods per year
) -> pd.Series:                                                                             
    excess_returns = returns - risk_free_rate / periods_per_year                                         # Measures return above risk-free
    rolling_return = excess_returns.rolling(window).mean() * periods_per_year                                       # Computes rolling CAGR
    rolling_vol = rolling_volatility(returns, window, periods_per_year)                                           # Computes rolling volatility
    return rolling_return / rolling_vol                                 


def rolling_max_drawdown(returns: pd.Series, window: int = 252) -> pd.Series:    # Computes rolling maximum drawdown
    def _max_dd(x):                                                                # Computes maximum drawdown
        cumulative = (1 + x).cumprod()                                         # Computes cumulative returns
        peak = cumulative.cummax()                                             # Tracks the highest value reached so far
        drawdown = (cumulative - peak) / peak                                          # Drawdown(t) = W(t) - max(W)   /  max(W)
        return drawdown.min()                                                          # Returns the worst (most negative) drawdown

    return returns.rolling(window).apply(_max_dd, raw=False)                        # Returns rolling maximum drawdown


def rolling_calmar_ratio(                                                                 # Computes rolling Calmar ratio
    returns: pd.Series,                                                                  # Returns series
    window: int = 252,                                                                   # Rolling window
    periods_per_year: int = 252                                                          # Number of periods per year
) -> pd.Series:     
    rolling_ann_return = returns.rolling(window).mean() * periods_per_year                # Computes rolling CAGR
    rolling_dd = rolling_max_drawdown(returns, window).abs()                             # Computes rolling maximum drawdown
    return rolling_ann_return / rolling_dd                                                  # Computes rolling Calmar
