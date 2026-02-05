import numpy as np
import pandas as pd
from scipy import stats


# =========================
# PARAMETER ESTIMATION
# =========================

def estimate_normal_params(returns: pd.Series) -> tuple:
    r = returns.dropna()
    return r.mean(), r.std()


def estimate_t_params(returns: pd.Series) -> tuple:
    r = returns.dropna()
    df, loc, scale = stats.t.fit(r)
    return df, loc, scale


# =========================
# MONTE CARLO SIMULATION
# =========================

def simulate_log_returns_normal(
    mu: float,
    sigma: float,
    n_days: int,
    n_simulations: int
) -> np.ndarray:
    return np.random.normal(
        loc=mu,
        scale=sigma,
        size=(n_simulations, n_days)
    )


def simulate_log_returns_t(
    df: float,
    loc: float,
    scale: float,
    n_days: int,
    n_simulations: int
) -> np.ndarray:
    return stats.t.rvs(
        df=df,
        loc=loc,
        scale=scale,
        size=(n_simulations, n_days)
    )


# =========================
# PRICE PATH SIMULATION
# =========================

def simulate_price_paths(
    start_price: float,
    log_returns_sim: np.ndarray
) -> np.ndarray:
    cumulative_log_returns = log_returns_sim.cumsum(axis=1)
    price_paths = start_price * np.exp(cumulative_log_returns)
    return price_paths


# =========================
# FULL PIPELINE HELPERS
# =========================

def monte_carlo_normal(
    returns: pd.Series,
    start_price: float,
    n_days: int = 252,
    n_simulations: int = 10_000
) -> np.ndarray:
    mu, sigma = estimate_normal_params(returns)
    sim_returns = simulate_log_returns_normal(
        mu, sigma, n_days, n_simulations
    )
    return simulate_price_paths(start_price, sim_returns)


def monte_carlo_student_t(
    returns: pd.Series,
    start_price: float,
    n_days: int = 252,
    n_simulations: int = 10_000
) -> np.ndarray:
    df, loc, scale = estimate_t_params(returns)
    sim_returns = simulate_log_returns_t(
        df, loc, scale, n_days, n_simulations
    )
    return simulate_price_paths(start_price, sim_returns)
