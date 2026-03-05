"""Click-based CLI entry point for stock prediction system."""

import csv
import click
from pathlib import Path
from rich.console import Console

from stock_prediction.config import get_setting
from stock_prediction.utils.logging import setup_logging

console = Console()


def _build_x_tab_lag(df, meta):
    """Build and scale the lag-feature row for xgboost_lag inference.

    Returns a (1, n_lag_features) float32 array, or None if xgboost_lag
    is not part of the loaded ensemble.
    """
    if "xgboost_lag" not in meta.get("selected_models", []):
        return None
    lag_scaler = meta.get("lag_scaler")
    lag_feature_names = meta.get("lag_feature_names", [])
    if lag_scaler is None or not lag_feature_names:
        return None
    try:
        from stock_prediction.features.technical import add_lag_trend_features
        import numpy as np
        lag_df = add_lag_trend_features(df)
        # Select feature columns BEFORE dropna so that rows with NaN labels
        # (last `horizon` rows, whose future returns are unknown) are NOT dropped.
        # Only early rows whose rolling-window features are still NaN get removed.
        available = [c for c in lag_feature_names if c in lag_df.columns]
        if not available:
            return None
        feat_df = lag_df[available].dropna()
        if feat_df.empty:
            return None
        latest_lag = feat_df.values[-1]
        return lag_scaler.transform(latest_lag.reshape(1, -1)).astype(np.float32)
    except Exception:
        return None


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def cli(log_level: str):
    """Indian Stock Market Prediction System."""
    setup_logging(log_level)


@cli.command()
@click.option("--symbols", "-s", default=None, help="Comma-separated stock symbols (e.g. RELIANCE.NS,TCS.NS)")
@click.option("--start-date", default=None, help="Training start date (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="Training end date (YYYY-MM-DD)")
@click.option("--no-news", is_flag=True, help="Disable news features")
@click.option("--no-llm", is_flag=True, help="Disable LLM features")
@click.option(
    "--models", "-m", default=None,
    help=(
        "Comma-separated models to train. "
        "Available: lstm, xgboost, encoder_decoder, prophet. "
        "Multiple entries → weighted ensemble (e.g. --models lstm,encoder_decoder). "
        "Defaults to config setting models.selected_models."
    ),
)
def train(symbols: str | None, start_date: str | None, end_date: str | None,
          no_news: bool, no_llm: bool, models: str | None):
    """Train prediction models for specified stocks."""
    from stock_prediction.models.trainer import ModelTrainer, AVAILABLE_MODELS
    from stock_prediction.utils.constants import NIFTY_50_TICKERS

    # Parse model selection
    selected_models: list[str] | None = None
    if models:
        selected_models = [m.strip().lower() for m in models.split(",")]
        invalid = [m for m in selected_models if m not in AVAILABLE_MODELS]
        if invalid:
            console.print(f"[red]Unknown model(s): {invalid}. Available: {AVAILABLE_MODELS}[/]")
            return
        mode_label = "+".join(selected_models) + (" (ensemble)" if len(selected_models) > 1 else "")
        console.print(f"Model selection: [bold]{mode_label}[/]")

    symbol_list = symbols.split(",") if symbols else NIFTY_50_TICKERS
    console.print(f"Training models for {len(symbol_list)} stocks...")

    trainer = ModelTrainer(use_news=not no_news, use_llm=not no_llm)
    results = trainer.train_batch(symbol_list, start_date, end_date, selected_models)

    success = sum(1 for v in results.values() if v["status"] == "success")
    failed = [sym for sym, v in results.items() if v["status"] != "success"]
    console.print(f"\nTraining complete: {success}/{len(results)} stocks trained successfully")
    for sym in failed:
        r = results[sym]
        console.print(f"[red]  {sym}: {r['status']} — {r['reason']}[/]")

    # Write training summary CSV
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "train_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol", "Status", "Balanced Acc", "Reason"])
        for sym, r in results.items():
            writer.writerow([
                sym,
                r["status"],
                f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "",
                r["reason"],
            ])
    console.print(f"[dim]Saved: {summary_path}[/dim]")


@cli.command()
@click.option("--symbols", "-s", default=None, help="Comma-separated stock symbols")
@click.option("--export", is_flag=True, help="Export results to CSV/JSON")
@click.option("--no-news", is_flag=True, help="Disable news features")
@click.option("--no-llm", is_flag=True, help="Disable LLM features")
def predict(symbols: str | None, export: bool, no_news: bool, no_llm: bool):
    """Generate predictions and trading signals."""
    from stock_prediction.models.trainer import ModelTrainer
    from stock_prediction.features.pipeline import FeaturePipeline
    from stock_prediction.signals.generator import SignalGenerator
    from stock_prediction.signals.screener import StockScreener
    from stock_prediction.signals.report import ReportFormatter
    from stock_prediction.utils.constants import NIFTY_50_TICKERS
    import numpy as np

    use_news = not no_news
    use_llm = not no_llm

    symbol_list = symbols.split(",") if symbols else NIFTY_50_TICKERS
    console.print(f"Generating predictions for {len(symbol_list)} stocks...")

    trainer = ModelTrainer()
    signal_gen = SignalGenerator()
    screener = StockScreener()
    reporter = ReportFormatter()

    # Run screener
    console.print("Running stock screener...")
    screener_result = screener.screen(symbol_list)

    # Generate predictions for each stock
    staleness_threshold = get_setting("models", "staleness_warning_days", default=30)

    signals = []
    for symbol in symbol_list:
        try:
            ensemble, scaler, seq_scaler, model_age_days, meta = trainer.load_models(symbol)
            if model_age_days is not None and model_age_days > staleness_threshold:
                console.print(f"[yellow]Warning: Model for {symbol} is {model_age_days} days old. Consider retraining.[/]")
            pipeline = FeaturePipeline(use_news=use_news, use_llm=use_llm)
            df = pipeline.build_features(symbol)

            if df.empty:
                continue

            label_cols = ["return_1d", "return_5d", "signal"]
            feature_cols = [c for c in df.columns if c not in label_cols]
            features = df[feature_cols].values

            seq_len = pipeline.sequence_length
            if len(features) < seq_len:
                continue

            # Get latest data point
            latest_seq = features[-seq_len:]
            latest_tab = features[-1]

            # Scale
            n_feat = latest_seq.shape[1]
            latest_seq_scaled = seq_scaler.transform(
                latest_seq.reshape(-1, n_feat)
            ).reshape(1, seq_len, n_feat)
            latest_tab_scaled = scaler.transform(latest_tab.reshape(1, -1))

            # Build lag features for xgboost_lag (if trained)
            x_tab_lag = _build_x_tab_lag(df, meta)

            # Predict
            prediction = ensemble.predict_single(
                latest_seq_scaled.astype(np.float32),
                latest_tab_scaled.astype(np.float32),
                X_tab_lag=x_tab_lag,
            )

            # Technical data for signal generator
            tech_cols = ["RSI", "MACD_Histogram", "Price_SMA50_Ratio"]
            tech_data = {}
            for col in tech_cols:
                if col in df.columns:
                    tech_data[col] = float(df[col].iloc[-1])

            signal = signal_gen.generate(symbol, prediction, tech_data)
            signals.append(signal)

        except FileNotFoundError:
            console.print(f"[yellow]No trained model for {symbol} - skipping[/]")
        except Exception as e:
            console.print(f"[red]Error predicting {symbol}: {e}[/]")

    # Display report
    reporter.display_full_report(signals, screener_result)

    if export:
        reporter.export_csv(signals)
        reporter.export_json(signals)
        console.print("[green]Results exported to data/processed/[/]")


@cli.command()
@click.option("--symbol", "-s", required=True, help="Stock symbol (e.g. RELIANCE.NS)")
@click.option("--no-news", is_flag=True, help="Disable news features")
@click.option("--no-llm", is_flag=True, help="Disable LLM features")
def analyze(symbol: str, no_news: bool, no_llm: bool):
    """Deep analysis of a single stock with LLM broker insights."""
    from stock_prediction.models.trainer import ModelTrainer
    from stock_prediction.features.pipeline import FeaturePipeline
    from stock_prediction.signals.generator import SignalGenerator
    from stock_prediction.signals.report import ReportFormatter
    from stock_prediction.llm import get_llm_provider
    from stock_prediction.llm.news_analyzer import BrokerNewsAnalyzer
    from stock_prediction.news.rss_fetcher import GoogleNewsRSSFetcher
    from stock_prediction.utils.constants import TICKER_TO_NAME
    import numpy as np

    use_news = not no_news
    use_llm = not no_llm

    console.print(f"Analyzing {symbol}...")

    # Get LLM analysis
    llm_scores = {}
    headlines = []
    llm_summary = ""
    if use_llm:
        try:
            llm_provider = get_llm_provider()
            analyzer = BrokerNewsAnalyzer(llm_provider)
            fetcher = GoogleNewsRSSFetcher()
            name = TICKER_TO_NAME.get(symbol, symbol.replace(".NS", ""))
            articles = fetcher.fetch_stock_news(name)
            llm_scores = analyzer.analyze_stock(symbol, articles)
            llm_summary = str(llm_scores.pop("_summary", ""))
            headlines = [a.title for a in articles[:5]]
        except Exception as e:
            console.print(f"[yellow]LLM analysis unavailable: {e}[/]")
    elif use_news:
        try:
            fetcher = GoogleNewsRSSFetcher()
            name = TICKER_TO_NAME.get(symbol, symbol.replace(".NS", ""))
            articles = fetcher.fetch_stock_news(name)
            headlines = [a.title for a in articles[:5]]
        except Exception as e:
            console.print(f"[yellow]News fetch failed: {e}[/]")

    # Get model prediction
    signal = None
    try:
        trainer = ModelTrainer()
        ensemble, scaler, seq_scaler, model_age_days, meta = trainer.load_models(symbol)
        staleness_threshold = get_setting("models", "staleness_warning_days", default=30)
        if model_age_days is not None and model_age_days > staleness_threshold:
            console.print(f"[yellow]Warning: Model for {symbol} is {model_age_days} days old. Consider retraining.[/]")
        pipeline = FeaturePipeline(use_news=use_news, use_llm=use_llm)
        df = pipeline.build_features(symbol)

        if not df.empty:
            label_cols = ["return_1d", "return_5d", "signal"]
            feature_cols = [c for c in df.columns if c not in label_cols]
            features = df[feature_cols].values

            seq_len = pipeline.sequence_length
            if len(features) >= seq_len:
                latest_seq = features[-seq_len:]
                latest_tab = features[-1]
                n_feat = latest_seq.shape[1]
                latest_seq_scaled = seq_scaler.transform(
                    latest_seq.reshape(-1, n_feat)
                ).reshape(1, seq_len, n_feat)
                latest_tab_scaled = scaler.transform(latest_tab.reshape(1, -1))

                x_tab_lag = _build_x_tab_lag(df, meta)
                prediction = ensemble.predict_single(
                    latest_seq_scaled.astype(np.float32),
                    latest_tab_scaled.astype(np.float32),
                    X_tab_lag=x_tab_lag,
                )

                tech_cols = ["RSI", "MACD_Histogram", "Price_SMA50_Ratio"]
                tech_data = {}
                for col in tech_cols:
                    if col in df.columns:
                        tech_data[col] = float(df[col].iloc[-1])

                signal_gen = SignalGenerator()
                signal = signal_gen.generate(
                    symbol, prediction, tech_data, llm_summary, headlines
                )
    except FileNotFoundError:
        console.print("[yellow]No trained model found. Run 'stockpredict train' first.[/]")
    except Exception as e:
        console.print(f"[red]Prediction error: {e}[/]")

    reporter = ReportFormatter()
    if signal:
        reporter.display_stock_analysis(signal, llm_scores)
    elif llm_scores:
        # Show LLM analysis even without model
        from stock_prediction.signals.generator import TradingSignal
        fallback = TradingSignal(
            symbol=symbol,
            signal="N/A",
            confidence=0.0,
            strength=0.0,
            llm_summary=llm_summary,
            top_headlines=headlines,
        )
        reporter.display_stock_analysis(fallback, llm_scores)


@cli.command()
@click.option("--count", "-n", default=10, help="Number of stocks to suggest")
@click.option("--no-news", is_flag=True, help="Skip news scoring (technical-only)")
def suggest(count: int, no_news: bool):
    """Suggest top NIFTY 50 stocks ranked by momentum and news."""
    from stock_prediction.signals.screener import StockScreener
    from stock_prediction.signals.report import ReportFormatter

    use_news = not no_news
    mode = "technical-only" if no_news else "technical + news"
    console.print(f"Finding top {count} stocks ({mode})...")

    screener = StockScreener()
    result = screener.suggest(count=count, use_news=use_news)

    reporter = ReportFormatter()
    reporter.display_suggestions(result)


@cli.command()
@click.option("--count", "-n", default=5, help="Number of buy/short candidates")
@click.option("--no-news", is_flag=True, help="Skip news-driven discovery")
@click.option("--no-llm", is_flag=True, help="Skip LLM news discovery")
def shortlist(count: int, no_news: bool, no_llm: bool):
    """Shortlist stocks for buying, shorting, and trending from news."""
    from stock_prediction.signals.screener import StockScreener
    from stock_prediction.signals.report import ReportFormatter

    use_news = not no_news
    use_llm = not no_llm
    mode = "technical-only" if no_news else ("technical + news" if no_llm else "technical + news + LLM")
    console.print(f"Shortlisting top {count} buy/short candidates ({mode})...")

    screener = StockScreener()
    result = screener.shortlist(count=count, use_news=use_news, use_llm=use_llm)

    reporter = ReportFormatter()
    reporter.display_shortlist(result)


@cli.command("fetch-data")
@click.option("--symbols", "-s", default=None, help="Comma-separated stock symbols")
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD)")
def fetch_data(symbols: str | None, start_date: str | None):
    """Fetch and cache stock price data."""
    from stock_prediction.data import get_provider
    from stock_prediction.utils.constants import NIFTY_50_TICKERS

    symbol_list = symbols.split(",") if symbols else NIFTY_50_TICKERS
    provider = get_provider()

    console.print(f"Fetching data for {len(symbol_list)} stocks...")
    results = provider.fetch_batch(symbol_list, start_date=start_date)

    for symbol, data in results.items():
        if data.is_empty:
            console.print(f"[red]{symbol}: No data[/]")
        else:
            console.print(f"[green]{symbol}: {len(data.df)} rows ({data.date_range[0]} to {data.date_range[1]})[/]")


@cli.command()
@click.option("--symbols", "-s", default=None, help="Comma-separated stock symbols")
@click.option("--no-news", is_flag=True, help="Disable news-driven discovery")
@click.option("--no-llm", is_flag=True, help="Disable LLM-based news discovery")
def screen(symbols: str | None, no_news: bool, no_llm: bool):
    """Run stock screener without full model predictions."""
    from stock_prediction.signals.screener import StockScreener
    from stock_prediction.signals.report import ReportFormatter
    from stock_prediction.utils.constants import NIFTY_50_TICKERS

    symbol_list = symbols.split(",") if symbols else NIFTY_50_TICKERS
    console.print(f"Screening {len(symbol_list)} stocks...")

    screener = StockScreener()
    result = screener.screen(symbol_list)

    # Clear LLM/news-based sections if disabled
    if no_llm or no_news:
        result.news_alerts = []

    reporter = ReportFormatter()
    reporter.display_full_report([], result)


@cli.command("test-buy")
@click.option("--symbol", "-s", required=True, help="Stock symbol (e.g. RELIANCE.NS)")
@click.option("--amount", "-a", required=True, type=float, help="Amount in INR to invest")
@click.option("--portfolio", "-p", default="default", show_default=True,
              help="Portfolio name (use 'default' for the main portfolio)")
def test_buy(symbol: str, amount: float, portfolio: str):
    """Paper trade: Buy a stock (open LONG or cover SHORT)."""
    from stock_prediction.signals.paper_trading import PaperTradingManager

    manager = PaperTradingManager(portfolio=portfolio)
    try:
        trade = manager.buy(symbol, amount)
        if trade.status == "CLOSED":
            pnl_color = "green" if (trade.pnl or 0) >= 0 else "red"
            console.print(f"[bold]Covered SHORT {symbol}[/] @ {trade.exit_price:.2f}")
            console.print(f"PnL: [{pnl_color}]INR {trade.pnl:+,.2f} ({trade.pnl_pct:+.1f}%)[/]")
        else:
            console.print(f"[bold green]BUY {symbol}[/] — {trade.quantity:.4f} shares @ INR {trade.entry_price:.2f}")
            console.print(f"Amount: INR {trade.amount:,.2f} | Trade ID: {trade.trade_id}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("test-sell")
@click.option("--symbol", "-s", required=True, help="Stock symbol (e.g. RELIANCE.NS)")
@click.option("--trade-id", default=None, help="Specific trade ID to close")
@click.option("--portfolio", "-p", default="default", show_default=True,
              help="Portfolio name (use 'default' for the main portfolio)")
def test_sell(symbol: str, trade_id: str | None, portfolio: str):
    """Paper trade: Sell a stock (close LONG or cover SHORT)."""
    from stock_prediction.signals.paper_trading import PaperTradingManager

    manager = PaperTradingManager(portfolio=portfolio)
    try:
        trade = manager.sell(symbol, trade_id=trade_id)
        pnl_color = "green" if (trade.pnl or 0) >= 0 else "red"
        action = "SELL" if trade.trade_type == "LONG" else "COVER SHORT"
        console.print(f"[bold]{action} {symbol}[/] — {trade.quantity:.4f} shares @ INR {trade.exit_price:.2f}")
        console.print(f"PnL: [{pnl_color}]INR {trade.pnl:+,.2f} ({trade.pnl_pct:+.1f}%)[/]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("test-short")
@click.option("--symbol", "-s", required=True, help="Stock symbol (e.g. RELIANCE.NS)")
@click.option("--amount", "-a", required=True, type=float, help="Amount in INR")
@click.option("--portfolio", "-p", default="default", show_default=True,
              help="Portfolio name (use 'default' for the main portfolio)")
def test_short(symbol: str, amount: float, portfolio: str):
    """Paper trade: Short sell a stock (open SHORT position)."""
    from stock_prediction.signals.paper_trading import PaperTradingManager

    manager = PaperTradingManager(portfolio=portfolio)
    try:
        trade = manager.short_sell(symbol, amount)
        console.print(f"[bold magenta]SHORT {symbol}[/] — {trade.quantity:.4f} shares @ INR {trade.entry_price:.2f}")
        console.print(f"Amount: INR {trade.amount:,.2f} | Trade ID: {trade.trade_id}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("test-portfolio")
@click.option("--portfolio", "-p", default="default", show_default=True,
              help="Portfolio name (use 'default' for the main portfolio)")
def test_portfolio(portfolio: str):
    """Paper trade: Show open positions and unrealized PnL."""
    from stock_prediction.signals.paper_trading import PaperTradingManager
    from stock_prediction.signals.report import ReportFormatter

    manager = PaperTradingManager(portfolio=portfolio)
    reporter = ReportFormatter()

    trades = manager.get_portfolio()
    reporter.display_portfolio(trades)


@cli.command("test-calculate-gain")
@click.option("--export", is_flag=True, help="Export report to JSON file")
@click.option("--portfolio", "-p", default="default", show_default=True,
              help="Portfolio name (use 'default' for the main portfolio)")
def test_calculate_gain(export: bool, portfolio: str):
    """Paper trade: Calculate gain/loss report for all closed trades."""
    from stock_prediction.signals.paper_trading import PaperTradingManager
    from stock_prediction.signals.report import ReportFormatter

    manager = PaperTradingManager(portfolio=portfolio)
    reporter = ReportFormatter()

    report = manager.calculate_gains()
    reporter.display_gain_report(report)

    if export:
        path = reporter.export_gain_report(report)
        console.print(f"[green]Report exported to {path}[/]")


@cli.command()
@click.argument("query")
def lookup(query: str):
    """Look up trading signal by company name or partial name.

    QUERY can be any part of a company name, ticker, or alias
    e.g. "tata", "bank", "infosys", "reliance"
    """
    from stock_prediction.models.trainer import ModelTrainer
    from stock_prediction.features.pipeline import FeaturePipeline
    from stock_prediction.signals.generator import SignalGenerator
    from stock_prediction.utils.constants import COMPANY_ALIASES, TICKER_TO_NAME
    from rich.table import Table
    from rich.text import Text
    import numpy as np

    q = query.strip().lower()

    # --- Match tickers ---
    matched: dict[str, str] = {}  # ticker -> company name

    # 1. Substring match in alias keys
    for alias, ticker in COMPANY_ALIASES.items():
        if q in alias:
            matched[ticker] = TICKER_TO_NAME.get(ticker, ticker)

    # 2. Substring match in canonical company names
    for ticker, name in TICKER_TO_NAME.items():
        if q in name.lower():
            matched[ticker] = name

    # 3. Substring match in ticker symbol itself (without .NS)
    for ticker in TICKER_TO_NAME:
        if q in ticker.lower().replace(".ns", ""):
            matched[ticker] = TICKER_TO_NAME[ticker]

    if not matched:
        console.print(f"[yellow]No companies found matching '{query}'.[/]")
        console.print("Try a broader term, e.g. 'tata', 'bank', 'pharma'")
        return

    console.print(f"\nFound [bold]{len(matched)}[/] match(es) for '[cyan]{query}[/]':\n")

    # --- Fetch signals ---
    SIGNAL_COLORS = {
        "STRONG BUY": "bold green",
        "BUY": "green",
        "HOLD": "yellow",
        "SELL": "red",
        "STRONG SELL": "bold red",
    }

    trainer = ModelTrainer()
    signal_gen = SignalGenerator()

    table = Table(title=f"Signal Lookup — '{query}'", show_lines=True)
    table.add_column("Symbol", style="cyan", width=16)
    table.add_column("Company", width=28)
    table.add_column("Signal", width=14)
    table.add_column("Confidence", justify="right", width=11)
    table.add_column("BUY %", justify="right", width=7)
    table.add_column("HOLD %", justify="right", width=7)
    table.add_column("SELL %", justify="right", width=7)
    table.add_column("Note", width=22)

    csv_rows = []
    for symbol, name in sorted(matched.items()):
        try:
            ensemble, scaler, seq_scaler, model_age_days, meta = trainer.load_models(symbol)
            pipeline = FeaturePipeline(use_news=False, use_llm=False)
            df = pipeline.build_features(symbol)

            if df.empty or len(df) < pipeline.sequence_length:
                raise ValueError("Insufficient feature data")

            label_cols = ["return_1d", "return_5d", "signal"]
            feature_cols = [c for c in df.columns if c not in label_cols]
            features = df[feature_cols].values
            seq_len = pipeline.sequence_length

            latest_seq = features[-seq_len:]
            latest_tab = features[-1]
            n_feat = latest_seq.shape[1]

            latest_seq_scaled = seq_scaler.transform(
                latest_seq.reshape(-1, n_feat)
            ).reshape(1, seq_len, n_feat)
            latest_tab_scaled = scaler.transform(latest_tab.reshape(1, -1))

            x_tab_lag = _build_x_tab_lag(df, meta)
            prediction = ensemble.predict_single(
                latest_seq_scaled.astype(np.float32),
                latest_tab_scaled.astype(np.float32),
                X_tab_lag=x_tab_lag,
            )

            tech_cols = ["RSI", "MACD_Histogram", "Price_SMA50_Ratio"]
            tech_data = {c: float(df[c].iloc[-1]) for c in tech_cols if c in df.columns}

            sig = signal_gen.generate(symbol, prediction, tech_data)
            color = SIGNAL_COLORS.get(sig.signal, "white")
            note = f"Model age: {model_age_days}d" if model_age_days else ""

            table.add_row(
                symbol, name[:28],
                Text(sig.signal, style=color),
                f"{sig.confidence:.1%}",
                f"{sig.probabilities.get('BUY', 0):.0%}",
                f"{sig.probabilities.get('HOLD', 0):.0%}",
                f"{sig.probabilities.get('SELL', 0):.0%}",
                note,
            )
            csv_rows.append([
                symbol, name, sig.signal, f"{sig.confidence:.1%}",
                f"{sig.probabilities.get('BUY', 0):.0%}",
                f"{sig.probabilities.get('HOLD', 0):.0%}",
                f"{sig.probabilities.get('SELL', 0):.0%}",
                note,
            ])

        except FileNotFoundError:
            table.add_row(symbol, name[:28], Text("N/A", style="dim"), "", "", "", "", "No trained model")
            csv_rows.append([symbol, name, "N/A", "", "", "", "", "No trained model"])
        except Exception as e:
            table.add_row(symbol, name[:28], Text("ERROR", style="red"), "", "", "", "", str(e)[:22])
            csv_rows.append([symbol, name, "ERROR", "", "", "", "", str(e)])

    console.print(table)

    # Save CSV
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    lookup_path = reports_dir / "lookup.csv"
    with open(lookup_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Symbol", "Company", "Signal", "Confidence", "BUY %", "HOLD %", "SELL %", "Note"])
        writer.writerows(csv_rows)
    console.print(f"[dim]Saved: {lookup_path}[/dim]")


@cli.command("experiment")
@click.option("--symbol", "-s", required=True, help="Stock symbol (e.g. INFY.NS)")
@click.option("--models", "-m", required=True,
              help="Comma-separated algorithms to test (e.g. lstm,encoder_decoder)")
@click.option("--start-date", default=None, help="Training start date (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="Training end date (YYYY-MM-DD)")
@click.option("--horizon", default=None, type=int,
              help="Prediction horizon in days (overrides config)")
@click.option("--no-news", is_flag=True, help="Disable news features")
@click.option("--no-llm", is_flag=True, help="Disable LLM features")
@click.option("--no-financials", is_flag=True, help="Disable financial features")
def experiment(symbol: str, models: str, start_date: str | None, end_date: str | None,
               horizon: int | None, no_news: bool, no_llm: bool, no_financials: bool):
    """Train individual algorithms in an isolated sandbox for a symbol.

    Each algorithm is saved to its own timestamped directory under
    data/models/<SYMBOL>/experimental/ and never touches the production model.
    Use list-experiments to compare results and promote to promote the best one.
    """
    from datetime import datetime as _dt
    from stock_prediction.models.trainer import ModelTrainer, AVAILABLE_MODELS
    from stock_prediction.config import load_settings
    from rich.table import Table

    selected = [m.strip().lower() for m in models.split(",")]
    invalid = [m for m in selected if m not in AVAILABLE_MODELS]
    if invalid:
        console.print(f"[red]Unknown algorithm(s): {invalid}. Available: {AVAILABLE_MODELS}[/]")
        return

    if horizon is not None:
        load_settings()["features"]["prediction_horizon"] = horizon

    save_dir = Path(get_setting("models", "save_dir", default="data/models"))
    sym_key  = symbol.replace(".", "_")

    trainer = ModelTrainer(
        use_news=not no_news, use_llm=not no_llm, use_financials=not no_financials
    )

    results = []
    for alg in selected:
        run_id  = f"{_dt.now():%Y%m%d_%H%M%S}_{alg}"
        run_dir = save_dir / sym_key / "experimental" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Training [bold]{alg}[/] → {run_dir.name} …")
        try:
            model, accuracy, _ = trainer.train_stock(
                symbol, start_date, end_date, [alg], experiment_dir=run_dir
            )
            if model is None:
                results.append((run_id, alg, "no_data", None))
                console.print(f"  [yellow]⚠ No training data[/]")
            else:
                results.append((run_id, alg, "success", accuracy))
                acc_str = f"{accuracy:.1%}" if accuracy is not None else "—"
                console.print(f"  [green]✓ val accuracy: {acc_str}[/]")
        except Exception as e:
            results.append((run_id, alg, "failed", None))
            console.print(f"  [red]✗ {e}[/]")

    # Summary table
    table = Table(title=f"Experimental Results — {symbol}", show_header=True)
    table.add_column("Run ID",       style="dim")
    table.add_column("Algorithm",    style="bold")
    table.add_column("Val Accuracy", justify="right")
    table.add_column("Status")
    for run_id, alg, status, acc in results:
        acc_str    = f"{acc:.1%}" if acc is not None else "—"
        status_str = {"success": "[green]success[/]", "no_data": "[yellow]no data[/]",
                      "failed": "[red]failed[/]"}.get(status, status)
        table.add_row(run_id, alg, acc_str, status_str)
    console.print(table)
    console.print("[dim]Use 'stockpredict list-experiments -s SYMBOL' to compare all runs.[/dim]")
    console.print("[dim]Use 'stockpredict promote -s SYMBOL --run-id RUN_ID' to go live.[/dim]")


@cli.command("list-experiments")
@click.option("--symbol", "-s", required=True, help="Stock symbol (e.g. INFY.NS)")
def list_experiments_cmd(symbol: str):
    """List all experimental runs for a symbol with their val accuracies."""
    from stock_prediction.models.trainer import list_experiments
    from rich.table import Table
    from rich.text import Text

    save_dir    = Path(get_setting("models", "save_dir", default="data/models"))
    experiments = list_experiments(symbol, save_dir)

    if not experiments:
        console.print(f"[yellow]No experimental runs found for {symbol}.[/yellow]")
        return

    table = Table(title=f"Experimental Runs — {symbol}", show_header=True)
    table.add_column("Run ID",       style="dim")
    table.add_column("Algorithm",    style="bold")
    table.add_column("Val Accuracy", justify="right")
    table.add_column("Trained At")
    table.add_column("Horizon", justify="right")

    for exp in experiments:
        sel        = exp.get("selected_models") or []
        alg_label  = ", ".join(sel) if sel else exp["run_id"].split("_", 2)[-1]
        acc        = exp.get("val_accuracy")
        acc_str    = f"{acc:.1%}" if acc is not None else "—"
        trained_at = (exp.get("trained_at") or "")[:16].replace("T", " ") or "—"
        horizon    = str(exp.get("horizon", "—"))
        table.add_row(exp["run_id"], alg_label, acc_str, trained_at, horizon)

    console.print(table)
    console.print("[dim]Use 'stockpredict promote -s SYMBOL --run-id RUN_ID' to promote a run.[/dim]")


@cli.command("promote")
@click.option("--symbol", "-s", required=True, help="Stock symbol (e.g. INFY.NS)")
@click.option("--run-id", "-r", required=True, help="Experimental run ID to promote")
@click.confirmation_option(prompt="This will overwrite the production model. Continue?")
def promote_cmd(symbol: str, run_id: str):
    """Promote an experimental run to production."""
    from stock_prediction.models.trainer import promote_experiment

    save_dir = Path(get_setting("models", "save_dir", default="data/models"))
    run_dir  = save_dir / symbol.replace(".", "_") / "experimental" / run_id

    if not run_dir.exists():
        console.print(f"[red]Run not found: {run_dir}[/]")
        console.print("[dim]Use 'stockpredict list-experiments -s SYMBOL' to see available runs.[/dim]")
        return

    promote_experiment(symbol, run_dir, save_dir)
    console.print(f"[green]✓ Promoted {run_id} → production for {symbol}[/]")
    console.print(f"[dim]All prediction commands will now use the promoted model.[/dim]")


@cli.command("delete-experiment")
@click.option("--symbol", "-s", required=True, help="Stock symbol (e.g. INFY.NS)")
@click.option("--run-id", "-r", required=True, help="Experimental run ID to delete")
@click.confirmation_option(prompt="Delete this experimental run?")
def delete_experiment(symbol: str, run_id: str):
    """Delete an experimental run directory."""
    import shutil

    save_dir = Path(get_setting("models", "save_dir", default="data/models"))
    run_dir  = save_dir / symbol.replace(".", "_") / "experimental" / run_id

    if not run_dir.exists():
        console.print(f"[red]Run not found: {run_dir}[/]")
        return

    shutil.rmtree(run_dir)
    console.print(f"[green]✓ Deleted {run_id}[/]")


@cli.command("purge-experiments")
@click.option("--symbol", "-s", default=None, help="Limit purge to one symbol (e.g. INFY.NS). Omit for all symbols.")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without removing anything.")
def purge_experiments_cmd(symbol: str | None, dry_run: bool):
    """Delete all experimental training runs (across all symbols, or one symbol).

    Use --symbol to restrict the purge to a single stock.
    Use --dry-run to preview what would be removed without deleting anything.
    """
    import shutil
    from stock_prediction.models.trainer import purge_experiments

    save_dir = Path(get_setting("models", "save_dir", default="data/models"))

    if dry_run:
        # Preview mode — count without deleting
        sym_dirs = (
            [save_dir / symbol.replace(".", "_")]
            if symbol
            else ([p for p in save_dir.iterdir() if p.is_dir()] if save_dir.exists() else [])
        )
        total_runs = 0
        total_bytes = 0
        for sym_dir in sym_dirs:
            exp_root = sym_dir / "experimental"
            if not exp_root.exists():
                continue
            for run_dir in exp_root.iterdir():
                if run_dir.is_dir():
                    size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
                    console.print(f"  [dim]{run_dir}[/dim]  ({size / 1024:.1f} KB)")
                    total_runs += 1
                    total_bytes += size
        if total_runs == 0:
            console.print("[yellow]No experimental runs found.[/yellow]")
        else:
            console.print(
                f"\n[cyan]Would delete {total_runs} run(s) "
                f"({total_bytes / 1024 / 1024:.2f} MB)[/cyan]"
            )
        return

    scope = f"[bold]{symbol}[/]" if symbol else "[bold]ALL symbols[/]"
    click.confirm(f"Delete all experimental runs for {scope}?", abort=True)

    runs, freed = purge_experiments(save_dir, symbol=symbol)
    if runs == 0:
        console.print("[yellow]No experimental runs found — nothing deleted.[/yellow]")
    else:
        console.print(
            f"[green]✓ Deleted {runs} experimental run(s) "
            f"({freed / 1024 / 1024:.2f} MB freed)[/green]"
        )


@cli.command("model-info")
@click.option("--symbol", "-s", required=True, help="Stock symbol (e.g. INFY.NS)")
def model_info(symbol: str):
    """Show production model details for a symbol."""
    import joblib
    from rich.table import Table
    from rich.text import Text

    save_dir = Path(get_setting("models", "save_dir", default="data/models"))
    meta_path = save_dir / symbol.replace(".", "_") / "meta.joblib"

    if not meta_path.exists():
        console.print(f"[yellow]No trained model found for {symbol}.[/yellow]")
        return

    meta = joblib.load(meta_path)

    sel       = meta.get("selected_models") or []
    acc       = meta.get("val_accuracy")
    trained   = (meta.get("trained_at") or "")[:16].replace("T", " ")
    horizon   = meta.get("horizon", "—")
    use_news  = meta.get("use_news", False)
    use_llm   = meta.get("use_llm", False)
    use_fin   = meta.get("use_financials", False)

    table = Table(title=f"Production Model — {symbol}", show_header=False, box=None, padding=(0, 2))
    table.add_column("Field", style="bold cyan", no_wrap=True)
    table.add_column("Value")

    table.add_row("Algorithms",    ", ".join(sel) if sel else "—")
    table.add_row("Val Accuracy",  f"{acc:.1%}" if acc is not None else "—")
    table.add_row("Trained At",    trained or "—")
    table.add_row("Horizon",       f"{horizon} day{'s' if horizon != 1 else ''}")
    table.add_row("News features", Text("✓", style="green") if use_news else Text("✗", style="red"))
    table.add_row("LLM features",  Text("✓", style="green") if use_llm  else Text("✗", style="red"))
    table.add_row("Financials",    Text("✓", style="green") if use_fin  else Text("✗", style="red"))
    table.add_row("Model path",    str(meta_path.parent))

    console.print(table)


@cli.command("catalogue")
@click.option("--symbol", "-s", default=None,
              help="Symbol to inspect in detail. Omit to list all trained models.")
def catalogue(symbol: str | None):
    """Browse all trained models or inspect one in full detail.

    \b
    Examples:
      stockpredict catalogue              # summary table of all trained models
      stockpredict catalogue -s INFY.NS   # full detail for INFY.NS
    """
    from rich.table import Table
    from rich.text import Text
    from stock_prediction.models.trainer import list_trained_models

    save_dir = Path(get_setting("models", "save_dir", default="data/models"))

    WEIGHT_KEYS = {
        "lstm":            "lstm_weight",
        "xgboost":         "xgb_weight",
        "xgboost_lag":     "xgb_lag_weight",
        "encoder_decoder": "ed_weight",
        "prophet":         "prophet_weight",
        "tft":             "tft_weight",
        "qlearning":       "ql_weight",
        "dqn":             "dqn_weight",
        "dqn_lag":         "dqn_lag_weight",
    }

    all_models = list_trained_models(save_dir)

    if not all_models:
        console.print("[yellow]No trained models found in[/yellow]", str(save_dir))
        return

    if symbol:
        # ── Detail view for one symbol ───────────────────────────────────────
        sym_key = symbol.replace(".", "_")
        entry = next((m for m in all_models if m["model_dir"].name == sym_key), None)
        if entry is None:
            console.print(f"[yellow]No trained model found for {symbol}.[/yellow]")
            return

        meta = entry
        sel      = meta.get("selected_models") or []
        acc      = meta.get("val_accuracy")
        trained  = (meta.get("trained_at") or "")[:16].replace("T", " ")
        horizon  = meta.get("horizon", "—")
        use_news = meta.get("use_news", False)
        use_llm  = meta.get("use_llm", False)
        use_fin  = meta.get("use_financials", False)
        age      = meta.get("model_age_days")
        in_sz    = meta.get("input_size", 0)
        lag_sz   = meta.get("lag_input_size", 0)
        feat_n   = len(meta.get("feature_names") or [])
        lag_feat = len(meta.get("lag_feature_names") or [])

        # ── Overview ─────────────────────────────────────────────────────────
        info = Table(title=f"📚 Model — {symbol}", show_header=False, box=None, padding=(0, 2))
        info.add_column("Field", style="bold cyan", no_wrap=True)
        info.add_column("Value")

        ck = lambda v: Text("✓", style="green") if v else Text("✗", style="red")
        info.add_row("Algorithms",    ", ".join(sel) if sel else "—")
        info.add_row("Val Accuracy",  f"{acc:.1%}" if acc is not None else "—")
        info.add_row("Horizon",       f"{horizon} day{'s' if horizon != 1 else ''}")
        info.add_row("Trained At",    trained or "—")
        info.add_row("Model Age",     f"{age} days" if age is not None else "—")
        info.add_row("News features", ck(use_news))
        info.add_row("LLM features",  ck(use_llm))
        info.add_row("Financials",    ck(use_fin))
        info.add_row("Input size",    f"{in_sz} features" if in_sz else "—")
        info.add_row("Lag input size",f"{lag_sz} features" if lag_sz else "—")
        info.add_row("Feature names", f"{feat_n} columns" if feat_n else "—")
        info.add_row("Lag features",  f"{lag_feat} columns" if lag_feat else "—")
        info.add_row("Model path",    str(meta["model_dir"]))
        console.print(info)

        # ── Ensemble weights ──────────────────────────────────────────────────
        if sel:
            wt = Table(title="Ensemble Weights", box=None, padding=(0, 2))
            wt.add_column("Model",  style="bold")
            wt.add_column("Weight", justify="right")
            for m in sel:
                w = meta.get(WEIGHT_KEYS.get(m, ""), None)
                wt.add_row(m, f"{w:.4f}" if w is not None else "—")
            console.print(wt)

        # ── Model files ───────────────────────────────────────────────────────
        files = meta.get("model_files", {})
        if files:
            ft = Table(title="Model Files on Disk", box=None, padding=(0, 2))
            ft.add_column("File", style="bold")
            ft.add_column("Size", justify="right")
            for fname, sz in sorted(files.items()):
                ft.add_row(fname, f"{sz/1024:.1f} KB")
            console.print(ft)

    else:
        # ── Summary table — all symbols ───────────────────────────────────────
        tbl = Table(title="📚 Trained Model Catalogue", box=None, padding=(0, 2))
        tbl.add_column("Symbol",     style="bold cyan")
        tbl.add_column("Algorithms")
        tbl.add_column("Val Acc",    justify="right")
        tbl.add_column("Horizon",    justify="center")
        tbl.add_column("Trained At")
        tbl.add_column("Age",        justify="right")
        tbl.add_column("News",       justify="center")
        tbl.add_column("LLM",        justify="center")
        tbl.add_column("Fin",        justify="center")

        ck = lambda v: Text("✓", style="green") if v else Text("✗", style="red")
        for entry in all_models:
            sel     = entry.get("selected_models") or []
            acc     = entry.get("val_accuracy")
            trained = (entry.get("trained_at") or "")[:10]
            horizon = entry.get("horizon", "—")
            age     = entry.get("model_age_days")
            tbl.add_row(
                entry["symbol"],
                ", ".join(sel) if sel else "—",
                f"{acc:.1%}" if acc is not None else "—",
                str(horizon),
                trained or "—",
                f"{age}d" if age is not None else "—",
                ck(entry.get("use_news", False)),
                ck(entry.get("use_llm", False)),
                ck(entry.get("use_financials", False)),
            )

        console.print(tbl)
        console.print(
            f"\n[dim]{len(all_models)} model(s) found. "
            "Run [bold]stockpredict catalogue -s SYMBOL[/bold] for full detail.[/dim]"
        )


if __name__ == "__main__":
    cli()
