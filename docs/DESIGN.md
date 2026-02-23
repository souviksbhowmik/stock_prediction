# Indian Stock Market Prediction System — Design Document

## Overview
Stock prediction system for the Indian market (NIFTY 50) combining technical analysis, NLP-based news sentiment (FinBERT), and LLM-extracted broker-like insights (Ollama). Uses an ensemble of LSTM + XGBoost to generate daily trading signals.

## Architecture
- **Data**: yfinance (modular interface)
- **News**: Google News RSS
- **NLP**: FinBERT for sentiment, spaCy for NER
- **LLM**: Ollama (local) for broker-like analysis
- **ML**: LSTM (40%) + XGBoost (60%) ensemble
- **Signals**: BUY / STRONG BUY / HOLD / SELL / STRONG SELL

## Data Flow
```
yfinance -> OHLCV -> Technical Indicators -> Feature Matrix -> LSTM + XGBoost -> Ensemble -> Signals
Google News RSS -> FinBERT Sentiment + spaCy NER + Ollama LLM -> News Features -> Feature Matrix
```

See config/settings.yaml for all tunable parameters.
