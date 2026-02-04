User enters stock symbol (e.g. RELIANCE.NS)

App fetches data using yfinance

Data is saved to Excel

App calculates:

Daily returns

Moving averages

Trend direction

GUI displays:

Price chart

Basic insights


The point of this project is to make






Using your saved stock data:

Convert prices → returns

Model returns using probability distributions

Measure risk statistically

Estimate downside probability

Compare empirical vs theoretical distributions


Negative skew → downside risk

High kurtosis → fat tails (market crashes)

returns = df["log_return"].dropna()
So:

returns = daily log returns of a stock

Each value = percentage-like change from one day to the next

This is the core random variable we are statistically analyzing


Row 1:  Histogram + Normal   | Histogram + Student-t
Row 2:  Normal Q-Q Plot      | Student-t Q-Q Plot
Row 3:  Monte Carlo Paths   | (empty / future use)


