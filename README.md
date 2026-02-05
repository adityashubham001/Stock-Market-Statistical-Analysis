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


Stock Market Statistical Analysis/
│
├── main.py                  # Orchestrator
│
├── data/
│   └── data_loader.py       # Data ingestion
│
├── analysis/
│   ├── returns.py           # Returns & performance metrics
│   ├── risk_metrics.py      # VaR, CVaR, tail risk
│   ├── distributions.py     # Normal vs Student-t modeling
│   └── monte_carlo.py       # Forward simulations
│
├── plot/
│   ├── distribution_plots.py
│   ├── monte_carlo_plots.py
│   └── dashboard.py         # Visualization orchestrator
│
├── excel/
│   └── excel_handler.py     # Persistence & Excel dashboard support
│
└── stocks.xlsx



separated ingestion, analytics, simulation, persistence, and visualization so each layer is reusable and testable.


User Input (Symbol)
   ↓
data_loader → price data
   ↓
returns → log returns/SIMPLE RETURNS
   ↓
risk_metrics → VaR / CVaR
   ↓
distributions → fat-tail validation
   ↓
monte_carlo → future scenarios
   ↓
dashboard → insights & visuals



====================================================
ANALYSIS/RETURNS.PY                                =
====================================================

log_returns(df)

Calculates logarithmic (continuously compounded) returns of a stock

Uses today’s Close price vs yesterday’s Close price

df["Close"].shift(1) → moves prices down by one day (yesterday’s price)
df["Close"] / df["Close"].shift(1) → price ratio      ->    P(t)/P(t-1)

np.log() → takes natural log     ->    ln(P(t)/P(t-1))

.dropna() → removes the first NaN since no previous price


-> Time-additive (multi-period returns can be summed)

-> Better for statistical modeling

-> Often closer to normal distribution


simple_returns(df)

Calculates simple percentage returns

Measures actual % gain or loss from one period to the next

pct_change() computes      ->    [P(t) - P(t-1)] / P(t-1)

-> Easy to interpret

-> Common in reporting, dashboards, and investor communication

-> Directly answers: “How much did I gain or lose?”

====================================================

cumulative_returns_simple()

It computes cumulative simple returns over time from a series of simple (percentage) returns.
-> “If I reinvest gains every period, what is my total return up to each point in time?”

-> Takes the returns we get (eg. 0.02, -0.01, 0.03, etc)
-> 1 + returns -> Converts returns into growth factors
-> .cumprod() ->  Computes cumulative product   (eg. 1.02; 1.02 × 0.99 = 1.0098; 1.0098 × 1.03 = 1.040094)
-> Converts back to cumulative return (-1)

Formula ->  R = ∏(1+r)     where:   r = simple return at time i,  R = cumulative return up to time t

It assumes:

Full reinvestment

No transaction costs

No cash inflows/outflows

===================================================

cumulative_returns_log()

It computes cumulative returns from log (continuously compounded) returns.
-> “If returns compound continuously over time, what is the total return so far?”

Since, log returns are additive over time.

Same logic for the formula as cum for simple returns but with change in formula as its for the log returns.

===================================================

annualized_return()

It computes the annualized return (CAGR) from a series of simple periodic returns.
-> “If this return stream continued for a full year, what would the equivalent yearly return be?”


-> Compute total compounded return     ->    total_return = (1 + returns).prod() - 1      ->     R = ∏(1+r) - 1
-> Count number of periods    ->    n_periods = returns.shape[0]
-> Count number of periods   ->    (1 + total_return) ** (periods_per_year / n_periods) - 1


R(annual) = [ (1+R(total)) ^ (periods per year/n) ] - 1     ->    scales the compounded return to a one-year equivalent

=====================================================

annualized_volatility()

It computes annualized volatility, i.e. the standard deviation of returns scaled to one year.
-> “How volatile (risky) is this asset on a yearly basis?”

(1) returns.std()

Calculates standard deviation of periodic returns

Measures dispersion / variability of returns

For daily returns → daily volatility

(2) np.sqrt(periods_per_year)

Scales volatility from one period to one year

Based on the square-root-of-time rule


->    σ(annual) = σ(period) * sq. rt. of N      Where: σ(period) = std of periodic returns,  N = number of periods per year (252 for daily)

this scaling works because:

-> Variance scales linearly with time

-> Standard deviation scales with square root of time

-> Assumes that - 

   Returns are i.i.d.

   No strong autocorrelation

========================================================

sharpe_ratio()

It computes the annualized Sharpe Ratio, which measures risk-adjusted return.
-> “How much extra return am I earning per unit of risk?”

Sharpe ->   E[R - R(f)]  /  σ          Where:   R = asset returns,   R(f) = risk-free rate,  σ = return volatility


excess_returns = returns - risk_free_rate / periods_per_year                                                Excess returns

   -> Converts annual risk-free rate → per-period

   -> Subtracts it from asset returns

   -> This ensures we measure return above “risk-free”

ann_return = annualized_return(excess_returns, periods_per_year)                                            Annualized excess return

   -> Computes CAGR of excess returns
   
   -> This is slightly stricter (and more realistic) than mean-based Sharpe


ann_vol = annualized_volatility(returns, periods_per_year)                                                  Annualized volatility

   -> Uses total volatility, not excess volatility


ann_return / ann_vol                                                                                        Final ratio

   -> Return per unit of risk

   -> Higher = better risk-adjusted performance

==================================================================

max_drawdown()

It computes the Maximum Drawdown (MDD) — the worst peak-to-trough loss experienced by an investment.
-> “What was the maximum percentage loss from the highest point before a new high was reached?”

Focuses on downside risk

Matches investor pain

Critical for

   -> Strategy evaluation

   -> Fund comparison

   -> Risk limits

   -> Stress testing

(1)   Cumulative return/Equity Curve
   -> cumulative = (1 + returns).cumprod()      ->    Transforms periodic returns into a wealth index. (eg. 1.00 → 1.05 → 1.02 → 1.08 → 0.95)

(2)   Running peak
   -> peak = cumulative.cummax()       ->       Tracks the highest value reached so far.  (eg. 1.00 → 1.05 → 1.05 → 1.08 → 1.08)

(3)   Drawdown series
   -> drawdown = (cumulative - peak) / peak        ->       Drawdown(t) = W(t) - max(W)   /  max(W)      -> Values are: 0 at peaks, Negative during losses

(4)   Maximum drawdown
   -> drawdown.min()        ->       Returns the worst (most negative) drawdown. (eg. -0.12  →  -12%)

Interpretation:

   −10% → mild drawdown

   −30% → severe

   −50%+ → catastrophic

Lower (less negative) is better.

=====================================================

calmar_ratio()

It computes the Calmar Ratio, which measures return per unit of maximum drawdown.
-> “How much annual return did I earn for the worst loss I had to sit through?”

Calmar = Annualized Return / | Max Drawdown |

(1)   Annualized return
   -> ann_return = annualized_return(returns, periods_per_year)      ->    Uses CAGR and measures long-term performance

(2)   Maximum drawdown (absolute)
   -> max_dd = abs(max_drawdown(returns))    ->    Converts drawdown to a positive risk measure,   represents worst peak-to-trough loss

(3)   Ratio computation
   -> ann_return / max_dd     ->    Higher = better, penalizes deep crashes, unlike Sharpe

Sharpe ratio:

   -> Penalizes upside volatility

   -> Assumes returns are normal

Calmar ratio:

   -> Focuses on tail risk

   -> Matches investor psychology


Calmar               Interpretation

< 0	               Losing strategy
0 – 0.5	            Weak
0.5 – 1.0	         Acceptable
1.0 – 2.0	         Good
> 3.0	               Excellent


If:

   -> Sharpe is high but Calmar is low → crash risk

   -> Calmar is high but Sharpe is low → choppy but resilient strategy

=======================================================

rolling_volatility()

It computes rolling(moving) annualized volatility over time.
-> “How does the asset’s risk change over time instead of assuming it’s constant?”

Markets are not stationary:

   -> Calm periods → low volatility

   -> Crises → volatility spikes

Rolling volatility captures:

   -> Regime changes

   -> Risk clustering

   -> Stress periods (COVID, 2008, crashes)

(1)   Rolling window
   -> returns.rolling(window)    ->    Creates a moving window of size window

(2)   Standard deviation      
   -> .std()      ->    Computes volatility inside each window, measures short-term risk

(3)   Annualization
   -> * np.sqrt(periods_per_year)      ->    Scales rolling volatility to annual terms, allows direct comparison with annual metrics


σ(t)  =  sq. rt. of N * Std(r(t-w+1),.....,r(t))            Where:   w = rolling window length, N = periods per year

=============================================================
 

rolling_sharpe_ratio()

It computes the rolling (time-varying) Sharpe Ratio.
-> “At each point in time, how good was the return relative to risk over the last N periods?”

Instead of one Sharpe for the entire dataset, you get a Sharpe time series.

A single Sharpe ratio can hide:

   -> Long bad periods

   -> Regime shifts

   -> Strategy decay

Rolling Sharpe shows:

   -> When a strategy worked

   -> When it stopped working

   -> Stability vs fragility

critical for strategy evaluation.

(1)   Excess returns
   -> excess_returns = returns - risk_free_rate / periods_per_year      ->    Converts annual risk-free rate → per-period, measures return above risk-free

(2)   Rolling annualized return
   -> rolling_return = excess_returns.rolling(window).mean() * periods_per_year     ->    Uses rolling average return, annualizes it linearly  ->  E[R−R(f​)] × N

(3)   Rolling annualized volatility
   -> rolling_vol = rolling_volatility(returns, window, periods_per_year)     ->    Uses rolling standard deviation, annualized via √time rule

(4)   Rolling Sharpe
   -> rolling_return / rolling_vol     ->          Positive → good risk-adjusted performance,   Negative → underperforming risk-free


Rolling Sharpe	               Meaning

< 0	                        Losing / bad regime
0 – 0.5	                     Weak
0.5 – 1.0	                  Acceptable
1.0 – 2.0	                  Strong
> 2.0	                        Exceptional

Look for:

   -> Consistency

   -> Duration of positive Sharpe

   -> Sharp collapses (regime breaks)
​
================================================================================

rolling_max_drawdown()

It computes rolling maximum drawdown over a moving window.
-> “Over the last N periods, what was the worst peak-to-trough loss?”

Instead of one max drawdown for the full series, you get a time series of worst-case losses.

A single max drawdown hides:

   -> When the damage happened

   -> Whether risk is increasing or stabilizing

Rolling max drawdown shows:

   -> Stress clusters

   -> Risk regimes

   -> Structural changes in strategy behavior


(1)   Inner function _max_dd
   -> cumulative = (1 + x).cumprod()      -> Operates on one rolling window
   -> peak = cumulative.cummax()    -> Tracks local highs.
   -> drawdown = (cumulative - peak) / peak     -> Computes drawdown series.
   -> return drawdown.min()      -> Returns the worst drawdown in that window.

(2)   Rolling application
   -> returns.rolling(window).apply(_max_dd, raw=False)     ->    Slides a window of length window, applies max drawdown logic to each window returns a Series aligned with time

Interpretation

Values are negative:

   -> −0.10     →   worst loss of 10% in that window

   -> −0.35     →   severe drawdown regime


==============================================================================

rolling_calmar_ratio()

It computes the rolling Calmar Ratio over a moving window.
-> “Over the last N periods, how much annualized return did I earn per unit of worst drawdown?”

Instead of a single Calmar number, you get a Calmar time series.

Sharpe ratio focuses on volatility.
Calmar focuses on crashes.

Rolling Calmar:

   -> Shows drawdown efficiency over time

   -> Highlights fragile vs resilient regimes

Is especially useful for:

   -> Trend-following strategies

   -> Long-only equity systems

   -> Capital allocation decisions

(1)   Rolling annualized return
   -> rolling_ann_return = returns.rolling(window).mean() * periods_per_year     ->    Mean-based annualization (standard for rolling metrics) approximates expected yearly return inside the window

(2)   Rolling maximum drawdown
   -> rolling_dd = rolling_max_drawdown(returns, window).abs()    ->    Measures worst loss in each window, converted to positive risk measure

(3)   Rolling Calmar
   -> rolling_ann_return / rolling_dd     ->    High values → efficient return vs drawdown,  Low / negative → poor risk-adjusted performance


Interpretation guide
Rolling Calmar	                  Meaning
< 0	                           Losing regime
0 – 0.5	                        Weak
0.5 – 1.0	                     Acceptable
1.0 – 2.0	                     Strong
> 3.0	                           Exceptional


====================================================
ANALYSIS/RISK_METRICS.PY                                =
====================================================

moments()

It computes the first four statistical moments of a return distribution.

(1)   Mean (1st moment)
   -> Average return per period

   -> Direction of performance

   -> Positive ≠ good unless risk is controlled

(2)   Volatility / Standard deviation (2nd moment)
   -> Dispersion of returns

   -> Proxy for risk

   -> Penalizes upside & downside equally

(3)   Skewness (3rd moment)
   -> Measures asymmetry of the return distribution:

      -> Positive skew → frequent small losses, rare big gains (ideal)

      -> Negative skew → frequent small gains, rare crashes (dangerous)

   -> Most equity strategies have negative skew.


(4)   Kurtosis (4th moment – excess)
   -> Measures tail heaviness

SciPy returns excess kurtosis:

   0 → normal distribution

   > 0 → fat tails (crashes more likely)

   < 0 → thin tails

Financial returns usually have high positive kurtosis.

Together, they tell us:

   -> Whether returns are normal or fat-tailed

   -> Whether volatility understates risk

   -> Whether Sharpe ratio is reliable

   -> Whether VaR/CVaR should be used

=======================================================================================

var_historical()

It computes Historical Value at Risk (VaR) at a given confidence level using empirical returns.
-> “Based on past data, what is the worst loss I should expect with X% confidence over one period?”

Unlike parametric VaR:

   No distributional assumptions

   Uses actual observed returns

   Fully captures skewness & fat tails (if present in data)


(1)   Clean the data
   -> r = returns.dropna()
      No NaNs
      Correct percentile calculation

(2)   Percentile extraction
   -> np.percentile(r, (1 - confidence) * 100)
      For confidence = 0.95:
      (1 - 0.95) × 100 = 5
      Returns the 5th percentile
   This value is typically negative.

VaR α =inf {x ∣ P (R ≤ x ) ≥ 1 − α}

Where:
   α = confidence level
   R = returns


If:   VaR₉₅% = -0.025

It means:   On 95% of days, losses should not exceed 2.5%.

It says nothing about how bad losses can be beyond this point.

=========================================================================================

cvar_historical()

It computes Historical CVaR (Expected Shortfall) at a given confidence level.
-> “If losses exceed VaR, what is the average loss I should expect?”

CVaR answers the question VaR cannot:

   VaR: How bad is bad?
   CVaR: How bad is really bad?

How CVaR works (conceptually)
   Find the VaR threshold
   Look only at returns worse than VaR
   Compute their average loss

This focuses entirely on the left tail of the distribution.


(1)   Clean the data
   -> r = returns.dropna()

(2)   Compute Historical VaR
   -> var = var_historical(r, confidence)    ->    This gives the loss cutoff at the chosen confidence level.

(3)   Average tail losses
   -> r[r <= var].mean()      
      -> Filters returns below the VaR threshold
      -> Computes mean of extreme losses

The result is more negative than VaR.

CVaR(α) ​= E[ R ∣ R ≤ VaR(α)​]

If:
   VaR₉₅%  = -2.5%
   CVaR₉₅% = -4.1%

It means:

“On the worst 5% of days, the average loss is 4.1%.”

==================================================================================

prob_large_loss()

It computes the empirical probability of a large loss exceeding a specified threshold.
-> “How often do losses worse than X% occur?”

(1)   Clean the data
   -> r = returns.dropna()       ->       Removes missing values so probabilities are correct.

(2)   Threshold comparison
   -> r <= threshold     
      Creates a Boolean array:
         True → loss worse than threshold
         False → otherwise

(3)   Mean of Booleans
   -> .mean()
      the mean is the probability.sINCE, IN PYTHON, True = 1 False = 0


P ( R ≤ T )

Where:
   T = loss threshold (e.g., −5%)

It means:
   “7% of the time, daily losses exceeded 5%.”

=================================================================================

downside_deviation()

It computes annualized downside deviation, which measures volatility of negative returns only, relative to a target return.
-> “How volatile are my losses, ignoring upside moves?”

Standard deviation:

   -> Penalizes gains and losses equally 

Downside deviation:

   -> Penalizes only underperformance 

   -> Aligns with real investor risk perception

   -> Forms the denominator of the Sortino Ratio  


(1)   Clean the data
   -> r = returns.dropna()

(2)   Downside returns
   -> downside = np.minimum(r - target, 0)      
      -> Computes deviation from target
      -> Keeps only negative values
      -> Positive deviations → set to 0

(3)   Root mean square of downside
   -> np.sqrt(np.mean(downside**2))       ->       This is the downside standard deviation.

(4)   Annualization
   -> * np.sqrt(periods_per_year)         ->    Applies square-root-of-time scaling.

Mathematical intuition

If:
   Downside deviation = 12%

It means:
   “My loss volatility is about 12% per year.”

