import numpy as np

from analysis.returns import (
    log_returns,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    calmar_ratio,
    max_drawdown,
    rolling_volatility,
    rolling_sharpe_ratio,
    rolling_max_drawdown,
    rolling_calmar_ratio
)

from analysis.risk_metrics import (
    moments,
    var_historical,
    cvar_historical,
    prob_large_loss,
    downside_deviation
)

from analysis.distributions import compare_distributions
from analysis.monte_carlo import monte_carlo_student_t


def generate_report_data(df):                                                   # Function to generate report data
    returns = log_returns(df)                                                   # Calculate log returns
    last_price = df["Close"].iloc[-1]                                           # Last closing price    

    data = {}                                                                   # Dictionary to store report data

    # Basic 
    data["start_date"] = df["Date"].iloc[0]                                     # Start date of data
    data["end_date"] = df["Date"].iloc[-1]                                      # End date of data
    data["observations"] = len(returns)                                         # Number of observations        
    data["last_price"] = last_price                                             # Last closing price        

    # Moments
    stats = moments(returns)                                                    # Statistical moments
    data.update(stats)                                                                  # Add moments to report data

    # Performance
    data["ann_return"] = annualized_return(returns)                                   
    data["ann_vol"] = annualized_volatility(returns)                        
    data["sharpe"] = sharpe_ratio(returns)
    data["calmar"] = calmar_ratio(returns)
    data["max_dd"] = max_drawdown(returns)
    data["downside_dev"] = downside_deviation(returns)

    # Tail risk
    data["var_95"] = var_historical(returns)
    data["cvar_95"] = cvar_historical(returns)
    data["prob_loss_2"] = prob_large_loss(returns, -0.02)
    data["prob_loss_5"] = prob_large_loss(returns, -0.05)
    data["prob_loss_10"] = prob_large_loss(returns, -0.10)

    # Rolling summaries
    roll_vol_series = rolling_volatility(returns).dropna()
    roll_sharpe_series = rolling_sharpe_ratio(returns).dropna()
    roll_dd_series = rolling_max_drawdown(returns).dropna()
    roll_calmar_series = rolling_calmar_ratio(returns).dropna()

    data["roll_vol"] = (                                                                                # Rolling volatility
        roll_vol_series.iloc[-1] if not roll_vol_series.empty else float("nan")                         # Last rolling volatility
    )
    data["roll_sharpe"] = (                                                                             # Rolling Sharpe ratio
        roll_sharpe_series.iloc[-1] if not roll_sharpe_series.empty else float("nan")                   # Last rolling Sharpe ratio            
    )
    data["worst_roll_dd"] = (                                                                           # Worst rolling drawdown        
        roll_dd_series.min() if not roll_dd_series.empty else float("nan")                               # Worst rolling drawdown  
    )
    data["roll_calmar"] = (                                                                                 # Rolling Calmar ratio
        roll_calmar_series.iloc[-1] if not roll_calmar_series.empty else float("nan")                       # Last rolling Calmar ratio                    
    )

   
    data["distribution_fit"] = compare_distributions(returns)                                       # Distribution comparison

    # Monte Carlo
    paths = monte_carlo_student_t(                                                                  # Monte Carlo simulation
    returns,
    start_price=last_price,
    n_days=252,
    n_simulations=5000
    )

    final_prices = paths[:, -1]

    data["mc_paths"] = paths                                                        # Monte Carlo paths
    data["mc_expected"] = final_prices.mean()
    data["mc_worst_5"] = np.percentile(final_prices, 5)
    data["mc_best_95"] = np.percentile(final_prices, 95)

    return data
