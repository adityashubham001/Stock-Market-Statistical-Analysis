import numpy as np
import pandas as pd
from scipy import stats



############################## PARAMETER ESTIMATION ##############################

def estimate_normal_params(returns: pd.Series) -> tuple:     # This function estimates the parameters of a normal distribution
    r = returns.dropna()                                        # Clean the returns by dropping NaN values
    return r.mean(), r.std()                                    # Return the mean and standard deviation


def estimate_t_params(returns: pd.Series) -> tuple:             # This function estimates the parameters of a Student-t distribution
    r = returns.dropna()                                        # Clean the returns by dropping NaN values
    df, loc, scale = stats.t.fit(r)                             # Fit Student-t distribution
    return df, loc, scale                                       # Return the degrees of freedom, location, and scale


# =========================
# MONTE CARLO SIMULATION
# =========================

def simulate_log_returns_normal(                                             # This function simulates log returns for a normal distribution
    mu: float,                                                      # The mean of the normal distribution float because we are working with log returns
    sigma: float,                                                   
    n_days: int,
    n_simulations: int
) -> np.ndarray:
    return np.random.normal(                                        # Return a 2D array of log returns
        loc=mu,
        scale=sigma,
        size=(n_simulations, n_days)
    )


def simulate_log_returns_t(                                             # This function simulates log returns for a Student-t distribution
    df: float,  
    loc: float,
    scale: float,
    n_days: int,
    n_simulations: int
) -> np.ndarray:
    return stats.t.rvs(                                                 # Return a 2D array of log returns
        df=df,                                                          
        loc=loc,
        scale=scale,
        size=(n_simulations, n_days)
    )



########################## PRICE PATH SIMULATION ######################

def simulate_price_paths(                                                                   # This function simulates price paths
    start_price: float,
    log_returns_sim: np.ndarray
) -> np.ndarray:
    cumulative_log_returns = log_returns_sim.cumsum(axis=1)                                  # Return a 2D array of cumulative log returns
    price_paths = start_price * np.exp(cumulative_log_returns)                                  
    return price_paths                                                                      # Return a 2D array of price paths



########################## FULL PIPELINE HELPERS ############################

def monte_carlo_normal(                                                         # This function simulates price paths
    returns: pd.Series,
    start_price: float,
    n_days: int = 252,
    n_simulations: int = 10_000
) -> np.ndarray:
    mu, sigma = estimate_normal_params(returns)                                 # Estimate normal parameters
    sim_returns = simulate_log_returns_normal(
        mu, sigma, n_days, n_simulations
    )
    return simulate_price_paths(start_price, sim_returns)                           # Return a 2D array of price paths


def monte_carlo_student_t(                                              # This function simulates price paths
    returns: pd.Series,
    start_price: float,
    n_days: int = 252,
    n_simulations: int = 10_000
) -> np.ndarray:
    df, loc, scale = estimate_t_params(returns)             # Estimate Student-t parameters
    sim_returns = simulate_log_returns_t(
        df, loc, scale, n_days, n_simulations
    )
    return simulate_price_paths(start_price, sim_returns)
