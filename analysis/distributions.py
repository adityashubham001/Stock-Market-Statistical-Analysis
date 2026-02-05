import numpy as np
import pandas as pd
from scipy import stats


# =========================
# DISTRIBUTION FITTING
# =========================

def fit_normal(returns: pd.Series) -> dict:             # This function fits a normal distribution to the returns
   
    r = returns.dropna()
    mu, sigma = stats.norm.fit(r)

    return {
        "distribution": "normal",
        "mu": mu,
        "sigma": sigma
    }


def fit_student_t(returns: pd.Series) -> dict:           # This function fits a Student-t distribution to the returns
    
    r = returns.dropna()
    df, loc, scale = stats.t.fit(r)

    return {
        "distribution": "student_t",
        "df": df,
        "loc": loc,
        "scale": scale
    }


# =========================
# LOG-LIKELIHOOD
# =========================

def log_likelihood_normal(returns: pd.Series, mu: float, sigma: float) -> float:    # This function computes the log-likelihood for normal distribution
    r = returns.dropna()
    return np.sum(stats.norm.logpdf(r, mu, sigma))


def log_likelihood_student_t(                           # This function computes the log-likelihood for Student-t distribution
    returns: pd.Series,
    df: float,
    loc: float,
    scale: float
) -> float:
    r = returns.dropna()
    return np.sum(stats.t.logpdf(r, df, loc, scale))


# =========================
# MODEL COMPARISON
# =========================

def aic(log_likelihood: float, num_params: int) -> float:       # this function computes the AIC for model comparison
    return 2 * num_params - 2 * log_likelihood              # Akaike Information Criterion(AIC) is used to compare the goodness of fit of different statistical models
    


def bic(log_likelihood: float, num_params: int, n_obs: int) -> float:
    """
    Bayesian Information Criterion.
    """
    return np.log(n_obs) * num_params - 2 * log_likelihood


def compare_distributions(returns: pd.Series) -> pd.DataFrame:
    """
    Compare Normal vs Student-t using AIC and BIC.
    """
    r = returns.dropna()
    n = len(r)

    # Normal
    normal_params = fit_normal(r)
    ll_norm = log_likelihood_normal(
        r,
        normal_params["mu"],
        normal_params["sigma"]
    )

    # Student-t
    t_params = fit_student_t(r)
    ll_t = log_likelihood_student_t(
        r,
        t_params["df"],
        t_params["loc"],
        t_params["scale"]
    )

    results = pd.DataFrame({
        "Distribution": ["Normal", "Student-t"],
        "LogLikelihood": [ll_norm, ll_t],
        "AIC": [
            aic(ll_norm, num_params=2),
            aic(ll_t, num_params=3)
        ],
        "BIC": [
            bic(ll_norm, num_params=2, n_obs=n),
            bic(ll_t, num_params=3, n_obs=n)
        ]
    })

    return results.sort_values("AIC")
