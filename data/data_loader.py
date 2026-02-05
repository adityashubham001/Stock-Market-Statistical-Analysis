import yfinance as yf
import pandas as pd


def fetch_stock(symbol: str, period: str = "5y") -> pd.DataFrame:   # Function to fetch stock data from Yahoo Finance

    df = yf.download(
        symbol,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    # Fix multi-index columns if present
    df.columns = df.columns.get_level_values(0)

    # Ensure Date is a column
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Explicit datetime conversion
    df["Date"] = pd.to_datetime(df["Date"])

    # Numeric date for regression / analytics
    df["date_num"] = df["Date"].map(pd.Timestamp.toordinal)

    return df
