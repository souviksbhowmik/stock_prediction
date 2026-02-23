# Stock Prediction System — Usage Guide

## Prerequisites

Activate the conda environment before running any commands.

```bash
conda activate stock_prediction
```

## Typical Workflow

```
shortlist → fetch-data → train → predict → analyze → paper trade
```

---

## Step 1: Discover What to Trade

Shortlist buy/short candidates and trending stocks from news.

```bash
stockpredict shortlist -n 5                  # Top 5 buy + short candidates + trending from news
stockpredict shortlist -n 3 --no-news        # Technical-only (faster, no trending section)
stockpredict shortlist -n 5 --no-llm         # News alias matching only, skip LLM discovery
```

Or get a simple ranked watchlist of NIFTY 50 stocks.

```bash
stockpredict suggest -n 10                   # Top 10 stocks ranked by momentum + news
stockpredict suggest -n 5 --no-news          # Technical-only ranking
```

## Step 2: Fetch Data

Fetch historical price data for the stocks you want to trade.

```bash
stockpredict fetch-data                                  # All NIFTY 50
stockpredict fetch-data -s RELIANCE.NS,TCS.NS,SBIN.NS   # Specific stocks
stockpredict fetch-data -s RELIANCE.NS --start-date 2023-01-01
```

## Step 3: Train Models

Train ML models (LSTM + XGBoost ensemble) on fetched data.

```bash
stockpredict train -s RELIANCE.NS,TCS.NS,SBIN.NS        # Specific stocks
stockpredict train                                        # All NIFTY 50
stockpredict train -s RELIANCE.NS --no-news --no-llm     # Technical-only (faster)
stockpredict train -s RELIANCE.NS --start-date 2023-01-01 --end-date 2024-12-31
```

## Step 4: Generate Predictions

Generate buy/hold/sell signals using trained models.

```bash
stockpredict predict -s RELIANCE.NS,TCS.NS               # Specific stocks
stockpredict predict                                      # All NIFTY 50
stockpredict predict -s RELIANCE.NS --export              # Export results to CSV + JSON
stockpredict predict --no-news --no-llm                   # Technical-only predictions
```

## Step 5: Deep Dive (Single Stock Analysis)

Get detailed analysis with LLM broker insights for a specific stock.

```bash
stockpredict analyze -s RELIANCE.NS                      # Full analysis
stockpredict analyze -s RELIANCE.NS --no-llm             # Skip LLM broker analysis
stockpredict analyze -s RELIANCE.NS --no-news            # Skip news features
```

## Step 6: Run the Screener

Screen stocks for top picks, sector momentum, and news alerts.

```bash
stockpredict screen                                      # Full NIFTY 50 screen
stockpredict screen -s RELIANCE.NS,TCS.NS,INFY.NS       # Specific stocks
stockpredict screen --no-news                            # Disable news discovery
stockpredict screen --no-llm                             # Disable LLM discovery
```

## Step 7: Paper Trading

Simulate trades without real money to test your strategy.

```bash
# Open a long position (buy)
stockpredict test-buy -s RELIANCE.NS -a 50000

# Open a short position
stockpredict test-short -s TCS.NS -a 30000

# View open positions and unrealized PnL
stockpredict test-portfolio

# Close a position (sell long / cover short)
stockpredict test-sell -s RELIANCE.NS
stockpredict test-sell -s TCS.NS --trade-id abc123       # Close specific trade

# Generate gain/loss report
stockpredict test-calculate-gain
stockpredict test-calculate-gain --export                # Export to JSON
```

---

## Global Options

```bash
stockpredict --log-level DEBUG <command>                  # Verbose logging
stockpredict --log-level WARNING <command>                # Quieter output
```
