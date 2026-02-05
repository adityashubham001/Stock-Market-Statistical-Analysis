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

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = annualized_return(excess_returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    return np.nan if ann_vol == 0 else ann_return / ann_vol


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    ann_return = annualized_return(returns, periods_per_year)
    max_dd = abs(max_drawdown(returns))
    return np.nan if max_dd == 0 else ann_return / max_dd


############################### ROLLING METRICS ##############################


def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    periods_per_year: int = 252
) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 21,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> pd.Series:
    excess_returns = returns - risk_free_rate / periods_per_year
    rolling_return = excess_returns.rolling(window).mean() * periods_per_year
    rolling_vol = rolling_volatility(returns, window, periods_per_year)
    return rolling_return / rolling_vol


def rolling_max_drawdown(returns: pd.Series, window: int = 252) -> pd.Series:
    def _max_dd(x):
        cumulative = (1 + x).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()

    return returns.rolling(window).apply(_max_dd, raw=False)


def rolling_calmar_ratio(
    returns: pd.Series,
    window: int = 252,
    periods_per_year: int = 252
) -> pd.Series:
    rolling_ann_return = returns.rolling(window).mean() * periods_per_year
    rolling_dd = rolling_max_drawdown(returns, window).abs()
    return rolling_ann_return / rolling_dd
