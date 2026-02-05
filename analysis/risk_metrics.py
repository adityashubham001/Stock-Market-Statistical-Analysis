import numpy as np
import pandas as pd
from scipy import stats



######################## DISTRIBUTION METRICS  #############################

def moments(returns: pd.Series) -> dict:         
    r = returns.dropna()

    return {
        "mean": r.mean(),
        "volatility": r.std(),
        "skewness": stats.skew(r),
        "kurtosis": stats.kurtosis(r)
    }



############################ VALUE AT RISK (VaR)   ###################################

def var_historical(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    r = returns.dropna()
    return np.percentile(r, (1 - confidence) * 100)


def cvar_historical(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    r = returns.dropna()
    var = var_historical(r, confidence)
    return r[r <= var].mean()



######################### PROBABILITY METRICS   ##########################

def prob_large_loss(
    returns: pd.Series,
    threshold: float = -0.05
) -> float:
    r = returns.dropna()
    return (r <= threshold).mean()



######################### DOWNSIDE RISK ##########################

def downside_deviation(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = 252
) -> float:
    r = returns.dropna()
    downside = np.minimum(r - target, 0)
    return np.sqrt(np.mean(downside**2)) * np.sqrt(periods_per_year)
