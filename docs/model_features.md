# Model Features Reference

This document lists every feature used by each model and explains how each feature is obtained.
All features flow through `FeaturePipeline.build_features()` in
`src/stock_prediction/features/pipeline.py` unless noted otherwise.

---

## Feature Pipeline Overview

Features are built in five sequential stages:

```
Stage 1  →  OHLCV price data           (yfinance)
Stage 2  →  Technical indicators        (ta library)
Stage 3  →  News sentiment features     (Google RSS + FinBERT, optional)
Stage 4  →  LLM broker scores           (Ollama / OpenAI, optional)
Stage 5  →  Market context              (NIFTY 50, always)
Stage 5b →  Quarterly financial ratios  (yfinance fundamentals, optional)
```

The resulting DataFrame (with labels excluded) forms the **standard feature set**
used by LSTM, XGBoost, Encoder-Decoder, TFT, Q-learning, and DQN.

XGBoost Lag extends this set with 16 additional lag/trend columns.
Prophet uses a separate subset of carry-forward and lagged regressors.

---

## Standard Feature Set

Used by: **LSTM · XGBoost · Encoder-Decoder · TFT · Q-learning · DQN**

### 1. Price / Volume (OHLCV) — 5 features

| Feature | Description | Source |
|---------|-------------|--------|
| Open    | Daily open price | yfinance |
| High    | Daily high price | yfinance |
| Low     | Daily low price | yfinance |
| Close   | Daily close price | yfinance |
| Volume  | Daily traded volume | yfinance |

### 2. Technical Indicators — 30 features

Computed by `add_technical_indicators()` in `src/stock_prediction/features/technical.py`
using the `ta` library. Default periods are set in `config/settings.yaml` under
`features.technical.*`.

#### Momentum (3)

| Feature | Formula / Period | Description |
|---------|-----------------|-------------|
| RSI | 14-day Wilder RSI | Relative Strength Index (0–100); >70 overbought, <30 oversold |
| Stoch_K | 14-day `%K` stochastic | Raw stochastic oscillator (0–100) |
| Stoch_D | 3-day SMA of `%K` | Smoothed signal line for stochastic |

#### Trend (8)

| Feature | Formula / Period | Description |
|---------|-----------------|-------------|
| MACD | EMA(12) − EMA(26) | MACD line |
| MACD_Signal | EMA(9) of MACD | MACD signal line |
| MACD_Histogram | MACD − MACD_Signal | Histogram; positive = bullish momentum |
| SMA_20 | 20-day Simple Moving Average | Medium-term trend baseline |
| SMA_50 | 50-day Simple Moving Average | Long-term trend baseline |
| EMA_12 | 12-day Exponential Moving Average | Fast trend line |
| EMA_26 | 26-day Exponential Moving Average | Slow trend line |
| SMA_20_50_Cross | 1 if SMA_20 > SMA_50, else 0 | Golden/death cross flag |

#### Volatility (5)

| Feature | Formula / Period | Description |
|---------|-----------------|-------------|
| BB_Upper | SMA(20) + 2 × std(20) | Bollinger upper band |
| BB_Middle | SMA(20) | Bollinger middle band |
| BB_Lower | SMA(20) − 2 × std(20) | Bollinger lower band |
| BB_Width | (BB_Upper − BB_Lower) / BB_Middle | Band width normalised by mid; higher = more volatility |
| ATR | 14-day Average True Range | Absolute volatility in price units |

#### Volume (4)

| Feature | Formula / Period | Description |
|---------|-----------------|-------------|
| OBV | Cumulative signed volume | On-Balance Volume; direction of volume flow |
| VWAP | 20-day rolling (TP × Vol) / Vol | Volume-Weighted Average Price; rolling (not intraday) |
| Volume_SMA20 | 20-day SMA of Volume | Rolling average volume for normalisation |
| Volume_Ratio | Volume / Volume_SMA20 | Volume relative to recent average; >1 = high activity |

#### Price-to-indicator Ratios (2)

| Feature | Formula | Description |
|---------|---------|-------------|
| Price_SMA20_Ratio | Close / SMA_20 | How far price is above/below 20-day average |
| Price_SMA50_Ratio | Close / SMA_50 | How far price is above/below 50-day average |

#### Rate-of-Change (Momentum Lags) (8)

| Feature | Formula | Description |
|---------|---------|-------------|
| RSI_Change_1d | RSI[t] − RSI[t−1] | 1-day RSI momentum |
| RSI_Change_3d | RSI[t] − RSI[t−3] | 3-day RSI momentum |
| MACD_Hist_Change_1d | MACD_Histogram[t] − MACD_Histogram[t−1] | 1-day histogram momentum |
| Price_Momentum_3d | Close.pct_change(3) | 3-day price return |
| Price_Momentum_5d | Close.pct_change(5) | 5-day price return |
| Price_Momentum_10d | Close.pct_change(10) | 10-day price return |
| Volume_Change_3d | Volume.pct_change(3) | 3-day volume change |
| Stoch_K_Change_1d | Stoch_K[t] − Stoch_K[t−1] | 1-day stochastic momentum |

### 3. Market Context — 3 features

Computed in `FeaturePipeline._add_market_context()` using NIFTY 50 (`^NSEI`) price data.

| Feature | Formula | Description |
|---------|---------|-------------|
| NIFTY_Return_1d | NIFTY Close.pct_change(1) | Market-wide daily return |
| Relative_Strength_1d | Stock 1d return − NIFTY_Return_1d | Stock outperformance vs market (1 day) |
| Relative_Strength_5d | Stock 5d return − NIFTY 5d return | Stock outperformance vs market (5 days) |

### 4. News Sentiment Features — 35 features *(optional, use_news=True)*

Generated by `NewsFeatureGenerator` in `src/stock_prediction/news/news_features.py`.
Articles are fetched from Google News RSS and scored by a FinBERT-based
`FinancialSentimentAnalyzer`. Features are computed over three time windows
(1d, 7d, 30d) and broadcast as constants onto every row of the training DataFrame.
All column names are prefixed with `news_`.

| Feature pattern | Windows | Description |
|-----------------|---------|-------------|
| `news_sentiment_{w}_mean` | 1d, 7d, 30d | Mean (positive − negative) sentiment score |
| `news_sentiment_{w}_std` | 1d, 7d, 30d | Std dev of sentiment scores |
| `news_volume_{w}` | 1d, 7d, 30d | Article count in window |
| `news_positive_ratio_{w}` | 1d, 7d, 30d | Fraction of articles with score > 0.1 |
| `news_negative_ratio_{w}` | 1d, 7d, 30d | Fraction of articles with score < −0.1 |
| `news_sentiment_{w}_trend` | 7d, 30d only | OLS slope of sentiment scores over window |
| `news_{category}_{w}` | 1d, 7d, 30d | Keyword match counts per category |

Six keyword categories: `earnings`, `merger`, `regulation`, `management`,
`dividend`, `expansion` (defined in `utils/constants.py`).

Total: 5 × 3 + 2 × 2 + 6 × 3 = **35 features**

### 5. LLM Broker Scores — 10 features *(optional, use_llm=True)*

Generated by `BrokerNewsAnalyzer` in `src/stock_prediction/llm/news_analyzer.py`.
Recent headlines are passed to a local LLM (Ollama) or OpenAI with a structured
prompt asking for 0–10 scores on each factor. Results are cached for 6 hours.
All column names are prefixed with `llm_`.

| Feature | Description |
|---------|-------------|
| `llm_earnings_outlook` | Earnings trajectory implied by news |
| `llm_competitive_position` | Market share / competitive moat |
| `llm_management_quality` | Quality of management based on news |
| `llm_sector_momentum` | Sector-level tailwinds / headwinds |
| `llm_risk_level` | Regulatory, operational, and market risk |
| `llm_growth_catalyst` | Near-term growth catalysts |
| `llm_valuation_signal` | Implied valuation attractiveness |
| `llm_institutional_interest` | Institutional activity signals |
| `llm_macro_impact` | Macro / FII / policy impact |
| `llm_overall_broker_score` | Composite broker-like buy/sell score |

### 6. Quarterly Financial Features — 20 features *(optional, use_financials=True)*

Generated by `FinancialFeatureGenerator` in `src/stock_prediction/features/financial.py`.
Quarterly income statement, balance sheet, and cash flow data are fetched via
yfinance. Each row in the daily DataFrame receives the values from the most
recent report whose announcement date is ≤ that trading day
(`pd.merge_asof` backward join).

**Point-in-time correctness:** yfinance returns fiscal quarter-end dates.
An announcement lag of 45 days (configurable) is applied so that results
only enter the DataFrame after they would realistically be public, preventing
look-ahead bias.

#### Fundamental Ratios (16) — prefixed `fin_`

| Feature | Formula | Description |
|---------|---------|-------------|
| `fin_revenue_growth_qoq` | Revenue.pct_change(1) | Quarter-over-quarter revenue growth |
| `fin_revenue_growth_yoy` | Revenue.pct_change(4) | Year-over-year revenue growth |
| `fin_revenue_log` | log1p(Revenue) | Log-scaled absolute revenue size |
| `fin_net_margin` | Net Income / Revenue | Net profit margin |
| `fin_ebitda_margin` | EBITDA / Revenue | EBITDA margin |
| `fin_operating_margin` | EBIT / Revenue | Operating margin |
| `fin_earnings_growth_qoq` | Net Income.pct_change(1) | QoQ earnings growth |
| `fin_earnings_growth_yoy` | Net Income.pct_change(4) | YoY earnings growth |
| `fin_debt_to_equity` | Total Debt / Equity | Financial leverage |
| `fin_debt_to_assets` | Total Debt / Total Assets | Asset leverage ratio |
| `fin_roe_ttm` | TTM Net Income / Equity | Return on equity (trailing 12 months) |
| `fin_cashflow_quality` | Operating CF / Net Income | Cash conversion quality |
| `fin_ocf_margin` | Operating CF / Revenue | Operating cash flow margin |
| `fin_fcf_margin` | (Operating CF + CapEx) / Revenue | Free cash flow margin |
| `fin_interest_coverage` | EBIT / \|Interest Expense\| | Debt service capacity |
| `fin_eps_surprise` | (Actual EPS − Estimated EPS) / \|Estimated\| | EPS beat/miss vs analyst consensus |

#### Report Aging Features (4)

| Feature | Formula | Description |
|---------|---------|-------------|
| `report_age_days` | calendar days since announcement | Age of the current report |
| `report_effect` | exp(−λ × age), λ = ln(2)/30 | Exponential decay; halves every 30 days |
| `report_freshness` | max(0, 91 − age) / 91 | Linear decay to 0 when next report expected |
| `days_to_next_report` | max(0, 91 − age) | Calendar days until next quarterly report |

---

## Per-Model Feature Details

### LSTM

| Property | Value |
|----------|-------|
| Input format | 3-D sequence: `(N, seq_len=60, n_features)` |
| Feature set | Full standard set (OHLCV + technical + market + news + LLM + financial) |
| Scaling | `StandardScaler` fit on training sequences (`seq_scaler` in meta.joblib) |
| How features are used | 60-day rolling window; each timestep is one row of the scaled feature matrix. The LSTM encodes the full temporal sequence into a hidden state for classification. |

---

### XGBoost

| Property | Value |
|----------|-------|
| Input format | 2-D tabular: `(N, n_features)` — one row per day |
| Feature set | Full standard set (same columns as LSTM but no sequence axis) |
| Scaling | `StandardScaler` fit on training rows (`scaler` in meta.joblib) |
| How features are used | Single-row snapshot of the current day fed directly into the gradient-boosted tree classifier. |

---

### XGBoost Lag

| Property | Value |
|----------|-------|
| Input format | 2-D tabular: `(N, n_base + 16)` — one row per day |
| Feature set | Full standard set **plus** 16 lag/trend features from `add_lag_trend_features()` |
| Scaling | Separate `StandardScaler` (`lag_scaler` in meta.joblib) fit on the extended matrix |
| How features are used | Same as XGBoost but with the richer feature set. The separate scaler is needed because the additional columns were not seen during standard scaler fitting. |

#### 16 Additional Lag / Trend Features

Computed by `add_lag_trend_features()` in `src/stock_prediction/features/technical.py`
applied on top of the base feature DataFrame.

**Lagged close ratios (3)**

| Feature | Formula | Description |
|---------|---------|-------------|
| `Close_Lag1_Ratio` | Close[t] / Close[t−1] | 1-day price change ratio |
| `Close_Lag3_Ratio` | Close[t] / Close[t−3] | 3-day price change ratio |
| `Close_Lag5_Ratio` | Close[t] / Close[t−5] | 5-day price change ratio |

Values > 1 indicate an uptrend over the lag period; < 1 indicates a downtrend.

**Lagged indicator snapshots (4)**

| Feature | Formula | Description |
|---------|---------|-------------|
| `RSI_Lag3` | RSI[t−3] | RSI value 3 days ago |
| `RSI_Lag5` | RSI[t−5] | RSI value 5 days ago |
| `MACD_Hist_Lag3` | MACD_Histogram[t−3] | Histogram value 3 days ago |
| `Volume_Ratio_Lag3` | Volume_Ratio[t−3] | Relative volume 3 days ago |

Provides absolute past levels, not just the 1-day diff already in the base set.

**Trend direction — ADX (4) + EMA cross (1)**

| Feature | Formula | Description |
|---------|---------|-------------|
| `ADX` | 14-day Average Directional Index | Trend strength (0–100); >25 = trending |
| `ADX_Pos` | +DI (14-day) | Bullish directional movement |
| `ADX_Neg` | −DI (14-day) | Bearish directional movement |
| `ADX_Cross` | 1 if +DI > −DI | Bullish trend direction flag |
| `EMA_12_26_Cross` | 1 if EMA_12 > EMA_26 | Golden cross (bullish) / death cross (bearish) flag |

**Normalised trend slope (2)**

| Feature | Formula | Description |
|---------|---------|-------------|
| `Trend_Slope_10d` | OLS slope over last 10 Close values, divided by mean price | Normalised directional slope (positive = uptrend) |
| `Trend_Slope_20d` | OLS slope over last 20 Close values, divided by mean price | Same over 20 days |

**Price position within recent range (2)**

| Feature | Formula | Description |
|---------|---------|-------------|
| `Price_Rank_10d` | (Close − 10d_min) / (10d_max − 10d_min) | Position in 10-day range: 0 = at low, 1 = at high |
| `Price_Rank_20d` | (Close − 20d_min) / (20d_max − 20d_min) | Position in 20-day range |

---

### Encoder-Decoder

| Property | Value |
|----------|-------|
| Input format | 3-D sequence: `(N, seq_len=60, n_features)` |
| Feature set | Full standard set (same as LSTM) |
| Scaling | Same `seq_scaler` as LSTM |
| Regression targets | Price ratios `Close[t+k] / Close[t]` for k = 1 … horizon (continuous) |
| How features are used | The encoder reads the 60-day sequence; the decoder outputs a vector of horizon future price ratios. Ratios are converted to BUY/HOLD/SELL probabilities using a Gaussian CDF (µ=1.0, σ estimated from validation residuals). |

---

### TFT (Temporal Fusion Transformer)

| Property | Value |
|----------|-------|
| Input format | 3-D sequence: `(N, seq_len=60, n_features)` |
| Feature set | Full standard set (same as LSTM) |
| Scaling | Same `seq_scaler` as LSTM |
| Regression targets | Same price ratios as Encoder-Decoder |
| How features are used | Same regression → Gaussian CDF probability pipeline as Encoder-Decoder, but uses a richer architecture: Variable Selection Networks select relevant features per timestep, LSTM captures temporal patterns, Multi-head Attention weights important time steps. |

---

### Prophet

| Property | Value |
|----------|-------|
| Input format | Time-series DataFrame with date index |
| Feature set | **Subset** of the standard set — two categories of exogenous regressors |
| Scaling | None — Prophet handles its own internal scaling |
| How features are used | Prophet fits a trend + seasonality model on Close prices and uses exogenous regressors to improve the horizon-ahead forecast. The forecast ratio is converted to BUY/HOLD/SELL the same way as the other regression models. |

Prophet features are prepared in `FeaturePipeline.prepare_prophet_data()`.

#### Category A: Carry-forward Regressors (up to 8)

Future values are approximated by carrying forward the last known value over
the prediction horizon. Only columns that exist in the DataFrame are used.

| Feature | Description |
|---------|-------------|
| `RSI` | 14-day momentum oscillator |
| `MACD_Histogram` | Trend momentum |
| `ATR` | Recent volatility level |
| `NIFTY_Return_1d` | Last known market direction |
| `Relative_Strength_1d` | Last known stock-vs-market outperformance (1d) |
| `Relative_Strength_5d` | Last known stock-vs-market outperformance (5d) |
| `fin_*` columns | All available quarterly financial ratios (carry-forward is valid between report dates) |
| `report_age_days`, `report_effect`, `report_freshness`, `days_to_next_report` | Report aging (carry-forward is a good approximation over 1–10 days) |

*Note: `BB_PB` and `sentiment_compound` are listed as candidates in the code but are not
produced by the pipeline, so they are silently skipped.*

#### Category B: Lagged Regressors (up to 19 × 1 = 19 columns)

For each source column, a lagged version `{col}_lag{horizon}` is created by
shifting back by `horizon` days: `col_lag_h[t+k] = col[t+k−horizon]`.
Because these are exact historical values, no approximation is needed.

| Source column | Lagged name (e.g. horizon=5) |
|---------------|------------------------------|
| `Volume` | `Volume_lag5` |
| `MACD` | `MACD_lag5` |
| `MACD_Signal` | `MACD_Signal_lag5` |
| `SMA_20` | `SMA_20_lag5` |
| `SMA_50` | `SMA_50_lag5` |
| `EMA_12` | `EMA_12_lag5` |
| `EMA_26` | `EMA_26_lag5` |
| `Stoch_K` | `Stoch_K_lag5` |
| `Stoch_D` | `Stoch_D_lag5` |
| `BB_Width` | `BB_Width_lag5` |
| `OBV` | `OBV_lag5` |
| `VWAP` | `VWAP_lag5` |
| `Price_Momentum_5d` | `Price_Momentum_5d_lag5` |
| `Volume_Ratio` | `Volume_Ratio_lag5` |
| `Price_SMA20_Ratio` | `Price_SMA20_Ratio_lag5` |
| `Price_SMA50_Ratio` | `Price_SMA50_Ratio_lag5` |
| `SMA_20_50_Cross` | `SMA_20_50_Cross_lag5` |
| `RSI_Change_1d` | `RSI_Change_1d_lag5` |
| `MACD_Hist_Change_1d` | `MACD_Hist_Change_1d_lag5` |

---

### Q-Learning (Tabular)

| Property | Value |
|----------|-------|
| Input format | Last timestep of scaled sequence: `X_seq[:, −1, :]` |
| Feature set | **5-feature state subset** of the standard set (configurable) |
| Scaling | Same `seq_scaler` as LSTM (applied to full sequence before extracting last row) |
| How features are used | Each of the 5 features is independently discretised into `n_bins` quantile buckets (default 3 bins) fitted on training data. The resulting integer tuple is the state key into a Q-table of shape `(n_bins^5, 3)`. Unknown states at inference return a HOLD-biased prior. |

#### Default State Features (5)

| Feature | Why chosen |
|---------|-----------|
| `RSI` | Momentum / overbought-oversold level |
| `MACD_Histogram` | Trend direction and strength |
| `BB_Width` | Volatility regime |
| `Volume_Ratio` | Volume confirmation |
| `Price_Momentum_5d` | Recent directional momentum |

All remaining standard features are present in the input sequence but are
not used by the Q-table state key.

---

### DQN (Deep Q-Network)

| Property | Value |
|----------|-------|
| Input format | Last timestep of scaled sequence: `X_seq[:, −1, :]` |
| Feature set | **Full standard set** at the current timestep (all n_features columns) |
| Scaling | Same `seq_scaler` as LSTM |
| How features are used | The full feature vector at the last timestep is fed into a fully-connected Q-network: `Input(n_features) → [Linear → ReLU → Dropout] × n_layers → Linear(3)`. No discretisation — the neural network generalises across the continuous feature space. Experience replay and a target network stabilise training. |

---

## Feature Count Summary

| Feature Group | Count | Models |
|---------------|-------|--------|
| OHLCV | 5 | All |
| Technical indicators | 30 | All |
| Market context (NIFTY) | 3 | All |
| News sentiment | 35 | All (if `use_news=True`) |
| LLM broker scores | 10 | All (if `use_llm=True`) |
| Quarterly financials | 20 | All (if `use_financials=True`) |
| **Standard set total** | **~103** | LSTM, XGBoost, ED, TFT, Q-learning, DQN |
| Lag / trend extensions | +16 | XGBoost Lag only |
| **XGBoost Lag total** | **~119** | XGBoost Lag |
| Carry-forward regressors | up to ~28 | Prophet only |
| Lagged regressors | up to 19 | Prophet only |

*Actual counts depend on which optional feature groups are enabled and how many
financial ratio columns are available for a given ticker.*

---

## Label (Target) — Not a Feature

The prediction target `signal` is excluded from all feature matrices.

| Label | Value | Condition |
|-------|-------|-----------|
| SELL | 0 | `return_{horizon}d ≤ sell_threshold` |
| HOLD | 1 | Between thresholds |
| BUY | 2 | `return_{horizon}d ≥ buy_threshold` |

`return_{horizon}d = Close[t+horizon] / Close[t] − 1`

Thresholds widen with horizon (e.g. ±1% at horizon=1, ±2.2% at horizon=5)
to account for increasing expected volatility. Defined in
`config/settings.yaml` under `signals.horizon_thresholds`.
