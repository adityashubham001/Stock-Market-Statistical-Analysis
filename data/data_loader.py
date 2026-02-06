import yfinance as yf
import pandas as pd


def fetch_stock(symbol: str, period: str = "5y") -> pd.DataFrame:   # Function to fetch stock data from Yahoo Finance

    df = yf.download(                                                               # Download stock data from Yahoo Finance
        symbol,
        period=period,                                                              # Set period to 5 years
        auto_adjust=False,                                                          # Disable auto-adjustment
        progress=False                                                              # Disable progress bar                
    )

    if df.empty:
        raise ValueError(f"No data found for symbol: {symbol}")

    # Fix multi-index columns if present
    df.columns = df.columns.get_level_values(0)                                     # Get the first level of the multi-index

    # Ensure Date is a column
    df = df.reset_index()                                                           # Reset the index                
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)                        # Rename the first column

    # Explicit datetime conversion
    df["Date"] = pd.to_datetime(df["Date"])                                         # Convert "Date" column to datetime

    # Numeric date for regression / analytics
    df["date_num"] = df["Date"].map(pd.Timestamp.toordinal)                         # Convert "Date" column to numeric

    return df                                                                       # Return the DataFrame
