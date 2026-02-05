import numpy as np
import pandas as pd

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


def run_console_dashboard(df: pd.DataFrame, symbol: str):           # Console-based dashboard

    returns = log_returns(df)                        # Calculate log returns

    # =========================
    # BASIC INFO
    # =========================
    start_date = df["Date"].iloc[0].date()      # Start date of data 
    end_date = df["Date"].iloc[-1].date()       # End date of data

    last_price = df["Close"].iloc[-1]      # Last closing price
    n_obs = len(returns)                     # Number of observations


    ################## MOMENTS & RETURN STATS ##################
    stats = moments(returns)                     # Statistical moments
    ann_ret = annualized_return(returns)    # Annualized return
    ann_vol = annualized_volatility(returns)    # Annualized volatility

    
    ################## RISK METRICS ######################

    sharpe = sharpe_ratio(returns)           # Sharpe ratio
    calmar = calmar_ratio(returns)      # Calmar ratio
    max_dd = max_drawdown(returns)           

    var_95 = var_historical(returns)      # 95% VaR
    cvar_95 = cvar_historical(returns)      # 95% CVaR

    downside_dev = downside_deviation(returns)  # Downside deviation

    prob_loss_2 = prob_large_loss(returns, -0.02)   # P(Return ≤ -2%)   
    prob_loss_5 = prob_large_loss(returns, -0.05)   # P(Return ≤ -5%)
    prob_loss_10 = prob_large_loss(returns, -0.10)  # P(Return ≤ -10%)

    # =========================
    # ROLLING METRICS (SUMMARY)
    # =========================
    roll_vol = rolling_volatility(returns)      # Rolling volatility
    roll_sharpe = rolling_sharpe_ratio(returns)  # Rolling Sharpe ratio
    roll_dd = rolling_max_drawdown(returns)    # Rolling max drawdown
    roll_calmar = rolling_calmar_ratio(returns)  # Rolling Calmar ratio

  
    ################# DISTRIBUTION COMPARISON ##########################

    dist_cmp = compare_distributions(returns)   # Distribution comparison (AIC/BIC)

   
    ####################### MONTE CARLO ###########################

    paths = monte_carlo_student_t(
        returns=returns,
        start_price=last_price,
        n_days=252,
        n_simulations=5000
    )


    final_prices = paths[:, -1]                     # Final prices from simulations         

    
    ################## PRINT DASHBOARD #######################

    print("\n" + "=" * 80)
    print(f"STOCK RISK & PERFORMANCE DASHBOARD — {symbol}")
    print("=" * 80)

    print("\nDATA SUMMARY")
    print("-" * 80)
    print(f"Start Date               : {start_date}")
    print(f"End Date                 : {end_date}")
    print(f"Observations             : {n_obs}")
    print(f"Last Close Price         : {last_price:.2f}")

    print("\nDISTRIBUTION MOMENTS")
    print("-" * 80)
    print(f"Mean Return              : {stats['mean']:.2%}")
    print(f"Volatility (Daily)       : {stats['volatility']:.2%}")
    print(f"Skewness                 : {stats['skewness']:.2f}")
    print(f"Kurtosis                 : {stats['kurtosis']:.2f}")

    print("\nANNUALIZED PERFORMANCE")
    print("-" * 80)
    print(f"Annualized Return        : {ann_ret:.2%}")
    print(f"Annualized Volatility    : {ann_vol:.2%}")

    print("\nRISK-ADJUSTED METRICS")
    print("-" * 80)
    print(f"Sharpe Ratio             : {sharpe:.2f}")
    print(f"Calmar Ratio             : {calmar:.2f}")
    print(f"Max Drawdown             : {max_dd:.2%}")
    print(f"Downside Deviation (Ann) : {downside_dev:.2%}")

    print("\nTAIL RISK METRICS")
    print("-" * 80)
    print(f"VaR (95%)                : {var_95:.2%}")
    print(f"CVaR (95%)               : {cvar_95:.2%}")

    print("\nPROBABILITY OF LARGE LOSSES")
    print("-" * 80)
    print(f"P(Return ≤ -2%)          : {prob_loss_2:.2%}")
    print(f"P(Return ≤ -5%)          : {prob_loss_5:.2%}")
    print(f"P(Return ≤ -10%)         : {prob_loss_10:.2%}")

    print("\nROLLING METRICS (LATEST / WORST)")
    print("-" * 80)

    if roll_vol.dropna().empty:
        print("Rolling metrics          : Not enough data")
    else:
        print(f"Latest Rolling Vol       : {roll_vol.dropna().iloc[-1]:.2%}")
        print(f"Latest Rolling Sharpe    : {roll_sharpe.dropna().iloc[-1]:.2f}")
        print(f"Worst Rolling Drawdown   : {roll_dd.min():.2%}")
        print(f"Latest Rolling Calmar    : {roll_calmar.dropna().iloc[-1]:.2f}")


    print("\nDISTRIBUTION FIT (AIC / BIC)")
    print("-" * 80)
    for _, row in dist_cmp.iterrows():
        print(f"{row['Distribution']:10s} | AIC: {row['AIC']:.2f} | BIC: {row['BIC']:.2f}")

    print("\nMONTE CARLO (1Y, 5000 SIMULATIONS)")
    print("-" * 80)
    print(f"Expected Price           : {np.mean(final_prices):.2f}")
    print(f"Worst 5% Outcome         : {np.percentile(final_prices, 5):.2f}")
    print(f"Best 95% Outcome         : {np.percentile(final_prices, 95):.2f}")

    print("\n" + "=" * 80)
