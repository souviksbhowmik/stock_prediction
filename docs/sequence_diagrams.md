# Sequence Diagrams

## Indian Stock Market Prediction System

Diagrams use [Mermaid](https://mermaid.js.org/) syntax and render natively in GitHub, GitLab, and most Markdown viewers.

---

## 1. `shortlist` — Buy / Short / Trending Discovery

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant StockScreener
    participant YFinance
    participant NewsRSS as Google News RSS
    participant FinBERT
    participant NER as spaCy NER
    participant Ollama as Ollama LLM
    participant ReportFormatter
    participant FileSystem as data/reports/

    User->>CLI: stockpredict shortlist -n 5

    CLI->>StockScreener: shortlist(count=5, use_news, use_llm)

    loop For each NIFTY 50 stock
        StockScreener->>YFinance: fetch_historical(symbol, last 60 days)
        YFinance-->>StockScreener: OHLCV DataFrame
        StockScreener->>StockScreener: compute technical indicators
        StockScreener->>StockScreener: score stock (momentum + RSI + volume + crossover + 52w high)
    end

    opt use_news = true
        StockScreener->>NewsRSS: fetch_articles(market queries)
        NewsRSS-->>StockScreener: news articles (cached 6h)
        StockScreener->>FinBERT: analyze_sentiment(articles)
        FinBERT-->>StockScreener: sentiment scores
        StockScreener->>NER: link_entities(articles)
        NER-->>StockScreener: matched NIFTY tickers + non-NIFTY mentions
        StockScreener->>StockScreener: add news mention count to composite score
    end

    opt use_llm = true
        StockScreener->>Ollama: discover_tickers_from_news(headlines)
        Ollama-->>StockScreener: additional trending tickers
    end

    StockScreener->>StockScreener: rank buy candidates (top N by score)
    StockScreener->>StockScreener: rank short candidates (bottom N by short score)
    StockScreener->>StockScreener: collect trending non-NIFTY stocks
    StockScreener-->>CLI: ShortlistResult(buy_candidates, short_candidates, trending)

    CLI->>ReportFormatter: display_shortlist(result)
    ReportFormatter->>User: Console — Buy Candidates table
    ReportFormatter->>User: Console — Short Candidates table
    ReportFormatter->>User: Console — Trending from News table
    ReportFormatter->>FileSystem: shortlist.csv (Category + all rows)
    ReportFormatter->>User: Console — "Saved: data/reports/shortlist.csv"
```

---

## 2. `suggest` — Ranked Momentum Watchlist

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant StockScreener
    participant YFinance
    participant NewsRSS as Google News RSS
    participant FinBERT
    participant NER as spaCy NER
    participant ReportFormatter
    participant FileSystem as data/reports/

    User->>CLI: stockpredict suggest -n 10

    CLI->>StockScreener: suggest(count=10, use_news)

    loop For each NIFTY 50 stock
        StockScreener->>YFinance: fetch_historical(symbol, last 60 days)
        YFinance-->>StockScreener: OHLCV DataFrame
        StockScreener->>StockScreener: compute technical indicators
        StockScreener->>StockScreener: calculate momentum (1w×0.6 + 1m×0.4)
        StockScreener->>StockScreener: apply bonus scores (volume spike, RSI extremes, crossover, 52w high)
    end

    opt use_news = true
        StockScreener->>NewsRSS: fetch_articles(market queries)
        NewsRSS-->>StockScreener: news articles (cached 6h)
        StockScreener->>FinBERT: analyze_sentiment(articles)
        FinBERT-->>StockScreener: sentiment scores
        StockScreener->>NER: link_entities(articles)
        NER-->>StockScreener: matched NIFTY tickers
        StockScreener->>StockScreener: add news mention bonus (+0.5 each, max 2.0)
    end

    StockScreener->>StockScreener: sort all stocks by composite score descending
    StockScreener-->>CLI: SuggestionResult(top N stocks with reasons)

    CLI->>ReportFormatter: display_suggestions(result)
    ReportFormatter->>User: Console — Suggested Stocks table
    ReportFormatter->>FileSystem: suggestions.csv
    ReportFormatter->>User: Console — "Saved: data/reports/suggestions.csv"
```

---

## 3. `fetch-data` — Data Preview / Connectivity Check

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant YFinanceProvider
    participant YFinance as Yahoo Finance API

    User->>CLI: stockpredict fetch-data -s RELIANCE.NS,TCS.NS

    loop For each symbol
        CLI->>YFinanceProvider: fetch_historical(symbol, start_date)
        YFinanceProvider->>YFinance: yf.Ticker(symbol).history(...)
        YFinance-->>YFinanceProvider: OHLCV DataFrame
        YFinanceProvider->>YFinanceProvider: standardise columns
        YFinanceProvider-->>CLI: StockData(symbol, df, metadata)

        alt data found
            CLI->>User: Console — "RELIANCE.NS: 1200 rows (2020-01-01 to 2024-12-31)"
        else no data
            CLI->>User: Console — "RELIANCE.NS: No data"
        end
    end

    Note over CLI,YFinance: Data is NOT saved to disk. StockData objects are discarded after printing.
```

---

## 4. `train` — Train LSTM + XGBoost Ensemble

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant ModelTrainer
    participant FeaturePipeline
    participant YFinance as Yahoo Finance API
    participant TechIndicators as Technical Indicators (ta)
    participant NewsRSS as Google News RSS
    participant FinBERT
    participant NER as spaCy NER
    participant Ollama as Ollama LLM
    participant LSTMPredictor
    participant XGBoostPredictor
    participant FileSystem as data/models/{SYMBOL}/

    User->>CLI: stockpredict train -s RELIANCE.NS

    CLI->>ModelTrainer: train_batch([RELIANCE.NS])
    ModelTrainer->>FeaturePipeline: prepare_training_data(RELIANCE.NS)

    FeaturePipeline->>YFinance: fetch_historical(RELIANCE.NS, 2020-01-01)
    YFinance-->>FeaturePipeline: OHLCV DataFrame (~1200 rows)

    FeaturePipeline->>TechIndicators: add_technical_indicators(df)
    TechIndicators-->>FeaturePipeline: +22 feature columns (RSI, MACD, BB, ATR, OBV, etc.)

    opt use_news = true
        FeaturePipeline->>NewsRSS: fetch_stock_news(company name)
        NewsRSS-->>FeaturePipeline: articles (cached 6h)
        FeaturePipeline->>FinBERT: analyze_sentiment(articles)
        FinBERT-->>FeaturePipeline: windowed sentiment features (1d, 7d, 30d)
        FeaturePipeline->>NER: link_entities(articles)
        NER-->>FeaturePipeline: +18 news feature columns
    end

    opt use_llm = true
        FeaturePipeline->>Ollama: analyze_stock(headlines)
        Ollama-->>FeaturePipeline: broker scores (cached 24h)
        FeaturePipeline-->>FeaturePipeline: +10 LLM feature columns
    end

    FeaturePipeline->>FeaturePipeline: create labels (return_1d → BUY/HOLD/SELL)
    FeaturePipeline->>FeaturePipeline: dropna (removes indicator warmup rows)
    FeaturePipeline->>FeaturePipeline: build 60-day rolling windows (LSTM sequences)
    FeaturePipeline->>FeaturePipeline: extract tabular snapshots (XGBoost input)

    alt yfinance returned no data OR fewer than 70 rows after dropna
        FeaturePipeline-->>ModelTrainer: raises ValueError(reason)
        ModelTrainer-->>CLI: status=no_data, reason=<specific message>
    else feature build succeeded
        FeaturePipeline-->>ModelTrainer: sequences(N,60,F), tabular(N,F), labels(N,)

        ModelTrainer->>ModelTrainer: chronological split (80% train / 20% val)
        ModelTrainer->>ModelTrainer: StandardScaler.fit_transform(train), transform(val)

        ModelTrainer->>LSTMPredictor: train(X_seq_train, y_train, X_seq_val, y_val)
        Note over LSTMPredictor: Adam + CrossEntropyLoss<br/>up to 50 epochs<br/>early stop patience=10
        LSTMPredictor-->>ModelTrainer: trained LSTM

        ModelTrainer->>XGBoostPredictor: train(X_tab_train, y_train, X_tab_val, y_val)
        Note over XGBoostPredictor: multi:softprob<br/>up to 500 trees<br/>early stop 20 rounds
        XGBoostPredictor-->>ModelTrainer: trained XGBoost

        ModelTrainer->>ModelTrainer: compute validation accuracy
        ModelTrainer->>FileSystem: save lstm.pt
        ModelTrainer->>FileSystem: save xgboost.joblib
        ModelTrainer->>FileSystem: save meta.joblib (scalers + feature_names + trained_at)
        ModelTrainer-->>CLI: status=success, accuracy=0.XXXX
    end

    ModelTrainer-->>CLI: dict[symbol → {status, reason, accuracy, model}]
    CLI->>User: Console — "Training complete: N/M stocks trained successfully"
    CLI->>User: Console — failed symbols with reasons (if any)
    CLI->>FileSystem: train_summary.csv (Symbol, Status, Val Accuracy, Reason)
    CLI->>User: Console — "Saved: data/reports/train_summary.csv"
```

---

## 5. `predict` — Generate Trading Signals

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant ModelTrainer
    participant FeaturePipeline
    participant YFinance as Yahoo Finance API
    participant EnsembleModel
    participant SignalGenerator
    participant StockScreener
    participant ReportFormatter
    participant FileSystem as data/reports/
    participant ProcessedDir as data/processed/

    User->>CLI: stockpredict predict -s RELIANCE.NS

    CLI->>ModelTrainer: load_models(RELIANCE.NS)
    ModelTrainer->>FileSystem: load lstm.pt, xgboost.joblib, meta.joblib
    FileSystem-->>ModelTrainer: models + scalers
    ModelTrainer->>ModelTrainer: check model age (warn if > 30 days)
    ModelTrainer-->>CLI: EnsembleModel, scaler, seq_scaler

    CLI->>FeaturePipeline: build_features(RELIANCE.NS, latest data)
    FeaturePipeline->>YFinance: fetch_historical(RELIANCE.NS)
    YFinance-->>FeaturePipeline: OHLCV DataFrame
    FeaturePipeline->>FeaturePipeline: add technical indicators + news + LLM features
    FeaturePipeline-->>CLI: feature DataFrame

    CLI->>CLI: scale features with saved StandardScaler
    CLI->>EnsembleModel: predict(X_seq, X_tab)
    EnsembleModel->>EnsembleModel: LSTM.predict_proba(X_seq) → (N,3)
    EnsembleModel->>EnsembleModel: XGBoost.predict_proba(X_tab) → (N,3)
    EnsembleModel->>EnsembleModel: weighted sum (0.4 × LSTM + 0.6 × XGBoost)
    EnsembleModel-->>CLI: EnsemblePrediction(signal, confidence, probabilities)

    CLI->>SignalGenerator: generate(prediction, technical_data)
    SignalGenerator->>SignalGenerator: apply confidence thresholds (0.6 / 0.8)
    SignalGenerator->>SignalGenerator: compute short score (sell_prob + RSI + MACD + SMA50)
    SignalGenerator-->>CLI: TradingSignal

    CLI->>StockScreener: screen(symbols)
    StockScreener-->>CLI: ScreenerResult (top_picks, sector_leaders, news_alerts)

    CLI->>ReportFormatter: display_full_report(signals, screener_result)
    ReportFormatter->>User: Console — Top Picks table
    ReportFormatter->>FileSystem: top_picks.csv
    ReportFormatter->>User: Console — Sector Momentum table
    ReportFormatter->>FileSystem: sector_momentum.csv
    ReportFormatter->>User: Console — News Alerts table
    ReportFormatter->>FileSystem: news_alerts.csv
    ReportFormatter->>User: Console — Trading Signals table
    ReportFormatter->>FileSystem: signals.csv
    ReportFormatter->>User: Console — Short Selling Candidates table
    ReportFormatter->>FileSystem: short_candidates.csv

    opt --export flag
        CLI->>ReportFormatter: export_csv(signals)
        ReportFormatter->>ProcessedDir: predictions_YYYY-MM-DD.csv
        CLI->>ReportFormatter: export_json(signals)
        ReportFormatter->>ProcessedDir: predictions_YYYY-MM-DD.json
    end
```

---

## 6. `analyze` — Deep Single-Stock Analysis

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant ModelTrainer
    participant FeaturePipeline
    participant YFinance as Yahoo Finance API
    participant EnsembleModel
    participant SignalGenerator
    participant NewsRSS as Google News RSS
    participant Ollama as Ollama LLM
    participant ReportFormatter
    participant FileSystem as data/reports/

    User->>CLI: stockpredict analyze -s RELIANCE.NS

    CLI->>ModelTrainer: load_models(RELIANCE.NS)
    ModelTrainer->>FileSystem: load lstm.pt, xgboost.joblib, meta.joblib
    FileSystem-->>ModelTrainer: models + scalers
    ModelTrainer-->>CLI: EnsembleModel, scalers

    CLI->>FeaturePipeline: build_features(RELIANCE.NS, latest)
    FeaturePipeline->>YFinance: fetch_historical(RELIANCE.NS)
    YFinance-->>FeaturePipeline: OHLCV DataFrame
    FeaturePipeline->>FeaturePipeline: add technical indicators
    FeaturePipeline-->>CLI: feature DataFrame

    CLI->>EnsembleModel: predict(X_seq, X_tab)
    EnsembleModel-->>CLI: EnsemblePrediction

    CLI->>SignalGenerator: generate(prediction, technical_data)
    SignalGenerator-->>CLI: TradingSignal (signal, confidence, outlooks)

    opt use_news = true
        CLI->>NewsRSS: fetch_stock_news(RELIANCE.NS)
        NewsRSS-->>CLI: top headlines (cached 6h)
    end

    opt use_llm = true
        CLI->>Ollama: analyze_stock(symbol, top 15 headlines)
        Note over Ollama: Scores 10 dimensions on 0-10 scale<br/>+ 1-2 sentence summary<br/>Result cached 24h
        Ollama-->>CLI: llm_scores dict + _summary
    end

    CLI->>ReportFormatter: display_stock_analysis(signal, llm_scores)
    ReportFormatter->>User: Console — Signal, confidence, weekly/monthly outlook
    ReportFormatter->>User: Console — LLM narrative summary panel
    ReportFormatter->>User: Console — Broker Analysis Scores table
    ReportFormatter->>User: Console — Recent headlines list

    opt llm_scores available
        ReportFormatter->>FileSystem: analyze.csv (Factor, Score columns)
        ReportFormatter->>User: Console — "Saved: data/reports/analyze.csv"
    end
```

---

## 7. `screen` — Full Stock Screener

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant StockScreener
    participant YFinance as Yahoo Finance API
    participant TechIndicators as Technical Indicators
    participant NewsRSS as Google News RSS
    participant FinBERT
    participant NER as spaCy NER
    participant Ollama as Ollama LLM
    participant ReportFormatter
    participant FileSystem as data/reports/

    User->>CLI: stockpredict screen

    CLI->>StockScreener: screen(symbols, use_news, use_llm)

    loop For each symbol
        StockScreener->>YFinance: fetch_historical(symbol)
        YFinance-->>StockScreener: OHLCV DataFrame
        StockScreener->>TechIndicators: add_technical_indicators(df)
        TechIndicators-->>StockScreener: enriched DataFrame
        StockScreener->>StockScreener: compute volume spike, RSI extremes, crossover signals
        StockScreener->>StockScreener: compute sector momentum ranking
    end

    opt use_news = true
        StockScreener->>NewsRSS: fetch_articles(market queries)
        NewsRSS-->>StockScreener: articles (cached 6h)
        StockScreener->>FinBERT: analyze_sentiment(articles)
        FinBERT-->>StockScreener: sentiment per article
        StockScreener->>NER: link_entities(articles)
        NER-->>StockScreener: matched tickers (NIFTY + non-NIFTY)
        StockScreener->>StockScreener: build news alerts for non-NIFTY stocks
    end

    opt use_llm = true
        StockScreener->>Ollama: discover additional tickers from news
        Ollama-->>StockScreener: additional tickers
    end

    StockScreener->>StockScreener: rank top picks (volume spikes + technical breakouts)
    StockScreener->>StockScreener: select sector leaders (best per sector)
    StockScreener-->>CLI: ScreenerResult

    CLI->>ReportFormatter: display_full_report(signals=[], screener_result)
    ReportFormatter->>User: Console — Top Picks table
    ReportFormatter->>FileSystem: top_picks.csv
    ReportFormatter->>User: Console — Sector Momentum table
    ReportFormatter->>FileSystem: sector_momentum.csv
    ReportFormatter->>User: Console — News Alerts table
    ReportFormatter->>FileSystem: news_alerts.csv
    ReportFormatter->>User: Console — Full Rankings / Signals table
    ReportFormatter->>FileSystem: signals.csv
    ReportFormatter->>User: Console — Short Candidates table
    ReportFormatter->>FileSystem: short_candidates.csv
```

---

## 8. Paper Trading Flow

### 8a. `test-buy` / `test-short` — Open a Position

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant PaperTradingManager
    participant YFinance as Yahoo Finance API
    participant Ledger as data/trades/ledger.json

    User->>CLI: stockpredict test-buy -s RELIANCE.NS -a 50000

    CLI->>PaperTradingManager: buy(RELIANCE.NS, amount=50000)
    PaperTradingManager->>Ledger: read all trades
    Ledger-->>PaperTradingManager: existing trades

    alt open SHORT exists for symbol
        PaperTradingManager->>YFinance: fetch_latest(RELIANCE.NS)
        YFinance-->>PaperTradingManager: current price
        PaperTradingManager->>PaperTradingManager: close SHORT (cover), compute PnL
        PaperTradingManager->>Ledger: update SHORT trade to CLOSED
        PaperTradingManager->>CLI: "Covered short position"
    else no conflicting position
        PaperTradingManager->>YFinance: fetch_latest(RELIANCE.NS)
        YFinance-->>PaperTradingManager: current price
        PaperTradingManager->>PaperTradingManager: qty = amount / price
        PaperTradingManager->>PaperTradingManager: create PaperTrade(LONG, OPEN, trade_id=UUID)
        PaperTradingManager->>Ledger: append new trade
        PaperTradingManager->>CLI: trade confirmation
    end

    CLI->>User: Console — "Bought X shares of RELIANCE.NS at ₹Y (Trade ID: abc12345)"
```

---

### 8b. `test-sell` — Close a Position

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant PaperTradingManager
    participant YFinance as Yahoo Finance API
    participant Ledger as data/trades/ledger.json

    User->>CLI: stockpredict test-sell -s RELIANCE.NS

    CLI->>PaperTradingManager: sell(RELIANCE.NS, trade_id=None)
    PaperTradingManager->>Ledger: read all trades
    Ledger-->>PaperTradingManager: existing trades

    PaperTradingManager->>PaperTradingManager: find most recent OPEN LONG for RELIANCE.NS
    PaperTradingManager->>YFinance: fetch_latest(RELIANCE.NS)
    YFinance-->>PaperTradingManager: current price (exit price)

    PaperTradingManager->>PaperTradingManager: pnl = (exit_price - entry_price) × qty
    PaperTradingManager->>PaperTradingManager: pnl_pct = (exit_price / entry_price - 1) × 100
    PaperTradingManager->>PaperTradingManager: mark trade CLOSED

    PaperTradingManager->>Ledger: update trade to CLOSED with exit_price and pnl
    PaperTradingManager-->>CLI: closed trade details

    CLI->>User: Console — "Sold RELIANCE.NS at ₹Y | PnL: +₹Z (+2.3%)"
```

---

### 8c. `test-portfolio` — View Open Positions

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant PaperTradingManager
    participant YFinance as Yahoo Finance API
    participant Ledger as data/trades/ledger.json
    participant ReportFormatter
    participant FileSystem as data/reports/

    User->>CLI: stockpredict test-portfolio

    CLI->>PaperTradingManager: get_portfolio()
    PaperTradingManager->>Ledger: read all trades
    Ledger-->>PaperTradingManager: all trades

    PaperTradingManager->>PaperTradingManager: filter OPEN trades

    loop For each open position
        PaperTradingManager->>YFinance: fetch_latest(symbol)
        YFinance-->>PaperTradingManager: current price
        PaperTradingManager->>PaperTradingManager: unrealized_pnl = (current - entry) × qty [LONG]
        Note over PaperTradingManager: or (entry - current) × qty [SHORT]
    end

    PaperTradingManager-->>CLI: list of PaperTrade with current prices + unrealized pnl

    CLI->>ReportFormatter: display_portfolio(trades)
    ReportFormatter->>User: Console — Portfolio table (positions + unrealized PnL)
    ReportFormatter->>User: Console — Total invested and aggregate unrealized PnL
    ReportFormatter->>FileSystem: portfolio.csv
    ReportFormatter->>User: Console — "Saved: data/reports/portfolio.csv"
```

---

### 8d. `test-calculate-gain` — Gain / Loss Report

```mermaid
sequenceDiagram
    actor User
    participant CLI
    participant PaperTradingManager
    participant YFinance as Yahoo Finance API
    participant Ledger as data/trades/ledger.json
    participant ReportFormatter
    participant FileSystem as data/reports/
    participant TradesDir as data/trades/

    User->>CLI: stockpredict test-calculate-gain

    CLI->>PaperTradingManager: calculate_gains()
    PaperTradingManager->>Ledger: read all trades
    Ledger-->>PaperTradingManager: all trades

    PaperTradingManager->>PaperTradingManager: filter CLOSED trades
    PaperTradingManager->>PaperTradingManager: compute totals (trades, wins, losses, total pnl, win rate)
    PaperTradingManager->>PaperTradingManager: identify best and worst trade
    PaperTradingManager->>PaperTradingManager: compute per-symbol breakdown

    loop For each OPEN trade
        PaperTradingManager->>YFinance: fetch_latest(symbol)
        YFinance-->>PaperTradingManager: current price
        PaperTradingManager->>PaperTradingManager: add to unrealized_pnl
    end

    PaperTradingManager-->>CLI: GainReport

    CLI->>ReportFormatter: display_gain_report(report)
    ReportFormatter->>User: Console — Summary metrics table
    ReportFormatter->>FileSystem: gain_summary.csv
    ReportFormatter->>User: Console — Best / worst trade callouts
    ReportFormatter->>User: Console — Per-Stock Breakdown table
    ReportFormatter->>FileSystem: gain_per_stock.csv
    ReportFormatter->>User: Console — "Saved: data/reports/gain_summary.csv"
    ReportFormatter->>User: Console — "Saved: data/reports/gain_per_stock.csv"

    opt --export flag
        CLI->>ReportFormatter: export_gain_report(report)
        ReportFormatter->>TradesDir: report_YYYY-MM-DD.json
    end
```
