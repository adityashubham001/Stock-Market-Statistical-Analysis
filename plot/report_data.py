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


def generate_report_data(df):
    returns = log_returns(df)
    last_price = df["Close"].iloc[-1]

    data = {}

    # Basic
    data["start_date"] = df["Date"].iloc[0]
    data["end_date"] = df["Date"].iloc[-1]
    data["observations"] = len(returns)
    data["last_price"] = last_price

    # Moments
    stats = moments(returns)
    data.update(stats)

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

    data["roll_vol"] = (
        roll_vol_series.iloc[-1] if not roll_vol_series.empty else float("nan")
    )
    data["roll_sharpe"] = (
        roll_sharpe_series.iloc[-1] if not roll_sharpe_series.empty else float("nan")
    )
    data["worst_roll_dd"] = (
        roll_dd_series.min() if not roll_dd_series.empty else float("nan")
    )
    data["roll_calmar"] = (
        roll_calmar_series.iloc[-1] if not roll_calmar_series.empty else float("nan")
    )

    # Distribution comparison
    data["distribution_fit"] = compare_distributions(returns)

    # Monte Carlo
    paths = monte_carlo_student_t(
    returns,
    start_price=last_price,
    n_days=252,
    n_simulations=5000
    )

    final_prices = paths[:, -1]

    data["mc_paths"] = paths
    data["mc_expected"] = final_prices.mean()
    data["mc_worst_5"] = np.percentile(final_prices, 5)
    data["mc_best_95"] = np.percentile(final_prices, 95)

    return data
