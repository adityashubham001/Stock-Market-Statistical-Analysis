
import streamlit as st
import pandas as pd
from datetime import timedelta
from data.data_loader import fetch_stock
from plot.report_data import generate_report_data
import matplotlib.pyplot as plt
from analysis.returns import (
    log_returns,
    rolling_volatility,
    rolling_sharpe_ratio
)
from plot.distribution_plots import plot_return_distributions
from plot.monte_carlo_plots import plot_monte_carlo_paths


st.set_page_config(page_title="Stock Risk Report", layout="centered")

############################### STREAMLIT DASHBOARD APP ###############################
@st.cache_data
def load_stock(symbol):                                         # Loads stock
    return fetch_stock(symbol)


if "range_days" not in st.session_state:                        # 
    st.session_state.range_days = 365


st.title("📄 Stock Risk & Performance Report")

symbol = st.text_input("Enter Stock Symbol", value="RELIANCE.NS")

if st.button("Generate Report", type="primary"):
    st.session_state.run = True

if "run" in st.session_state and st.session_state.run:
    
    df = load_stock(symbol)
    if df is None or df.empty:
        st.error(f" Invalid symbol or no data found: {symbol}")
        st.stop()       
    
    max_date = df["Date"].max()
    min_date = df["Date"].min()

    st.subheader("Time Range")

    ranges = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "2Y": 730,
        "3Y": 1095,
        "4Y": 1460,
        "5Y": 1825,
        "Max": None,
    }

    cols = st.columns(len(ranges))

    for col, (label, days) in zip(cols, ranges.items()):
        if col.button(label):
            st.session_state.range_days = days


    if st.session_state.range_days is None:
        start_date = min_date
    else:
        start_date = max_date - timedelta(days=st.session_state.range_days)

    end_date = max_date

    st.caption(f"📌 Active range: {label}")


    st.info(
        f"Selected range: {start_date.date()} → {end_date.date()}"
    )

    range_label = (
        "Max"
        if st.session_state.range_days is None
        else f"Last {st.session_state.range_days} days"
    )

    st.caption(f"📌 Active range: **{range_label}**")


    df_filtered = df[
        (df["Date"] >= pd.to_datetime(start_date)) &
        (df["Date"] <= pd.to_datetime(end_date))
    ].copy()

    if len(df_filtered) < 20:
        st.warning("Selected range too short for reliable analysis.")
        st.stop()

    # Recompute analytics on filtered data
    returns = log_returns(df_filtered)
    report = generate_report_data(df_filtered)


    st.divider()
    st.subheader("Data Summary")
    st.write(f"**Start Date:** {report['start_date']}")
    st.write(f"**End Date:** {report['end_date']}")
    st.write(f"**Observations:** {report['observations']}")
    st.write(f"**Last Close Price:** {report['last_price']:.2f}")
    st.subheader(" Price History")
    st.line_chart(df_filtered.set_index("Date")["Close"])



    st.divider()
    st.subheader("Distribution Moments")
    st.write(f"Mean Return: {report['mean']:.2%}")
    st.write(f"Volatility: {report['volatility']:.2%}")
    st.write(f"Skewness: {report['skewness']:.2f}")
    st.write(f"Kurtosis: {report['kurtosis']:.2f}")
    st.subheader("Return Distribution")
    fig_dist = plot_return_distributions(returns)
    st.pyplot(fig_dist)


    st.divider()
    st.subheader("Annualized Performance")
    st.write(f"Annualized Return: {report['ann_return']:.2%}")
    st.write(f"Annualized Volatility: {report['ann_vol']:.2%}")

    st.divider()
    st.subheader("Risk-Adjusted Metrics")
    st.write(f"Sharpe Ratio: {report['sharpe']:.2f}")
    st.write(f"Calmar Ratio: {report['calmar']:.2f}")
    st.write(f"Max Drawdown: {report['max_dd']:.2%}")
    st.write(f"Downside Deviation: {report['downside_dev']:.2%}")

    st.divider()
    st.subheader("Tail Risk Metrics")
    st.write(f"VaR (95%): {report['var_95']:.2%}")
    st.write(f"CVaR (95%): {report['cvar_95']:.2%}")

    st.divider()
    st.subheader("Probability of Large Losses")
    st.write(f"P(Return ≤ -2%): {report['prob_loss_2']:.2%}")
    st.write(f"P(Return ≤ -5%): {report['prob_loss_5']:.2%}")
    st.write(f"P(Return ≤ -10%): {report['prob_loss_10']:.2%}")

    st.divider()
    st.subheader("Rolling Risk Summary")
    st.write(f"Latest Rolling Volatility: {report['roll_vol']:.2%}")
    st.write(f"Latest Rolling Sharpe: {report['roll_sharpe']:.2f}")
    st.write(f"Worst Rolling Drawdown: {report['worst_roll_dd']:.2%}")
    st.write(
        "Latest Rolling Calmar: "
        + ("N/A" if pd.isna(report["roll_calmar"])
        else f"{report['roll_calmar']:.2f}")
    )

    st.divider()
    st.subheader("Rolling Risk Metrics")

    if len(returns) >= 252:
        roll_vol = rolling_volatility(returns)
        roll_sharpe = rolling_sharpe_ratio(returns)

        st.line_chart({
            "Rolling Volatility": roll_vol,
            "Rolling Sharpe": roll_sharpe
        })
    else:
        st.info("Rolling metrics require at least 252 observations.")



    st.divider()
    st.subheader("Distribution Fit (AIC / BIC)")
    st.dataframe(report["distribution_fit"], use_container_width=True)

    st.divider()
    st.subheader("Monte Carlo Summary (1 Year)")
    st.write(f"Expected Price: {report['mc_expected']:.2f}")
    st.write(f"Worst 5% Outcome: {report['mc_worst_5']:.2f}")
    st.write(f"Best 95% Outcome: {report['mc_best_95']:.2f}")
    st.subheader("Monte Carlo Price Paths")
    fig_mc = plot_monte_carlo_paths(
        report["mc_paths"],  # see note below
        n_paths_to_plot=100
    )
    st.pyplot(fig_mc)

    st.caption(
        "Data source: Yahoo Finance • "
        "All metrics computed on selected time range • "
        
    )
