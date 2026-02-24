"""Click-based CLI entry point for stock prediction system."""

import csv
import click
from pathlib import Path
from rich.console import Console

from stock_prediction.config import get_setting
from stock_prediction.utils.logging import setup_logging

console = Console()


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
def train(symbols: str | None, start_date: str | None, end_date: str | None,
          no_news: bool, no_llm: bool):
    """Train prediction models for specified stocks."""
    from stock_prediction.models.trainer import ModelTrainer
    from stock_prediction.utils.constants import NIFTY_50_TICKERS

    symbol_list = symbols.split(",") if symbols else NIFTY_50_TICKERS
    console.print(f"Training models for {len(symbol_list)} stocks...")

    trainer = ModelTrainer(use_news=not no_news, use_llm=not no_llm)
    results = trainer.train_batch(symbol_list, start_date, end_date)

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
        writer.writerow(["Symbol", "Status", "Val Accuracy", "Reason"])
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
            ensemble, scaler, seq_scaler, model_age_days = trainer.load_models(symbol)
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

            # Predict
            prediction = ensemble.predict_single(
                latest_seq_scaled.astype(np.float32),
                latest_tab_scaled.astype(np.float32),
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
        ensemble, scaler, seq_scaler, model_age_days = trainer.load_models(symbol)
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

                prediction = ensemble.predict_single(
                    latest_seq_scaled.astype(np.float32),
                    latest_tab_scaled.astype(np.float32),
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
def test_buy(symbol: str, amount: float):
    """Paper trade: Buy a stock (open LONG or cover SHORT)."""
    from stock_prediction.signals.paper_trading import PaperTradingManager

    manager = PaperTradingManager()
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
def test_sell(symbol: str, trade_id: str | None):
    """Paper trade: Sell a stock (close LONG or cover SHORT)."""
    from stock_prediction.signals.paper_trading import PaperTradingManager

    manager = PaperTradingManager()
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
def test_short(symbol: str, amount: float):
    """Paper trade: Short sell a stock (open SHORT position)."""
    from stock_prediction.signals.paper_trading import PaperTradingManager

    manager = PaperTradingManager()
    try:
        trade = manager.short_sell(symbol, amount)
        console.print(f"[bold magenta]SHORT {symbol}[/] — {trade.quantity:.4f} shares @ INR {trade.entry_price:.2f}")
        console.print(f"Amount: INR {trade.amount:,.2f} | Trade ID: {trade.trade_id}")
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")


@cli.command("test-portfolio")
def test_portfolio():
    """Paper trade: Show open positions and unrealized PnL."""
    from stock_prediction.signals.paper_trading import PaperTradingManager
    from stock_prediction.signals.report import ReportFormatter

    manager = PaperTradingManager()
    reporter = ReportFormatter()

    trades = manager.get_portfolio()
    reporter.display_portfolio(trades)


@cli.command("test-calculate-gain")
@click.option("--export", is_flag=True, help="Export report to JSON file")
def test_calculate_gain(export: bool):
    """Paper trade: Calculate gain/loss report for all closed trades."""
    from stock_prediction.signals.paper_trading import PaperTradingManager
    from stock_prediction.signals.report import ReportFormatter

    manager = PaperTradingManager()
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
            ensemble, scaler, seq_scaler, model_age_days = trainer.load_models(symbol)
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

            prediction = ensemble.predict_single(
                latest_seq_scaled.astype(np.float32),
                latest_tab_scaled.astype(np.float32),
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


if __name__ == "__main__":
    cli()
