"""Rich console report formatter and CSV/JSON export."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from stock_prediction.config import get_setting
from stock_prediction.signals.generator import TradingSignal
from stock_prediction.signals.screener import ScreenerResult, SuggestionResult
from stock_prediction.utils.constants import TICKER_TO_NAME
from stock_prediction.utils.logging import get_logger

if TYPE_CHECKING:
    from stock_prediction.signals.paper_trading import GainReport, PaperTrade

logger = get_logger("signals.report")

SIGNAL_COLORS = {
    "STRONG BUY": "bold green",
    "BUY": "green",
    "HOLD": "yellow",
    "SELL": "red",
    "STRONG SELL": "bold red",
}


class ReportFormatter:
    """Format and display trading signal reports."""

    def __init__(self):
        self.console = Console()
        self.export_dir = Path(get_setting("report", "export_dir", default="data/processed"))
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def display_full_report(
        self,
        signals: list[TradingSignal],
        screener_result: ScreenerResult | None = None,
    ) -> None:
        """Display full report with all sections."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.console.print(
            Panel(f"Stock Prediction Report - {now}", style="bold blue")
        )

        if screener_result:
            self._display_top_picks(screener_result.top_picks)
            self._display_sector_overview(screener_result.sector_leaders)
            self._display_news_alerts(screener_result.news_alerts)

        self._display_signals(signals)
        self._display_short_candidates(signals)

    def _display_signals(self, signals: list[TradingSignal]) -> None:
        """Display main signal table."""
        table = Table(title="Trading Signals", show_lines=True)
        table.add_column("Symbol", style="cyan", width=14)
        table.add_column("Name", width=22)
        table.add_column("Signal", width=12)
        table.add_column("Confidence", justify="right", width=10)
        table.add_column("BUY %", justify="right", width=8)
        table.add_column("HOLD %", justify="right", width=8)
        table.add_column("SELL %", justify="right", width=8)
        table.add_column("Outlook", width=30)

        for sig in signals:
            color = SIGNAL_COLORS.get(sig.signal, "white")
            name = TICKER_TO_NAME.get(sig.symbol, sig.symbol)[:22]
            table.add_row(
                sig.symbol,
                name,
                Text(sig.signal, style=color),
                f"{sig.confidence:.1%}",
                f"{sig.probabilities.get('BUY', 0):.0%}",
                f"{sig.probabilities.get('HOLD', 0):.0%}",
                f"{sig.probabilities.get('SELL', 0):.0%}",
                sig.weekly_outlook,
            )

        self.console.print(table)

    def _display_short_candidates(self, signals: list[TradingSignal]) -> None:
        """Display short selling candidates."""
        shorts = [s for s in signals if s.is_short_candidate]
        if not shorts:
            return

        table = Table(title="Short Selling Candidates", show_lines=True)
        table.add_column("Symbol", style="cyan", width=14)
        table.add_column("Name", width=22)
        table.add_column("Signal", width=12)
        table.add_column("Short Score", justify="right", width=12)
        table.add_column("Confidence", justify="right", width=10)
        table.add_column("Reasons", width=30)

        for sig in shorts:
            reasons = []
            tech = sig.technical_summary
            if tech.get("RSI", 50) > 70:
                reasons.append(f"Overbought RSI={tech['RSI']:.0f}")
            if tech.get("MACD_Histogram", 0) < 0:
                reasons.append("Bearish MACD")
            if tech.get("Price_SMA50_Ratio", 1) < 1:
                reasons.append("Below SMA50")

            name = TICKER_TO_NAME.get(sig.symbol, sig.symbol)[:22]
            table.add_row(
                sig.symbol,
                name,
                Text(sig.signal, style="bold red"),
                f"{sig.short_score:.2f}",
                f"{sig.confidence:.1%}",
                ", ".join(reasons) if reasons else "Model signal",
            )

        self.console.print(table)

    def _display_top_picks(self, top_picks: list[dict]) -> None:
        """Display top picks section."""
        if not top_picks:
            return

        table = Table(title="Top Picks (Pre-Screened)", show_lines=True)
        table.add_column("Symbol", style="cyan", width=14)
        table.add_column("Name", width=22)
        table.add_column("Score", justify="right", width=8)
        table.add_column("Price", justify="right", width=10)
        table.add_column("RSI", justify="right", width=6)
        table.add_column("Reasons", width=40)

        for pick in top_picks:
            table.add_row(
                pick["symbol"],
                pick["name"][:22],
                f"{pick['score']:.1f}",
                f"{pick['price']:.2f}",
                f"{pick['rsi']:.0f}",
                "; ".join(pick["reasons"]),
            )

        self.console.print(table)

    def _display_sector_overview(self, sector_data: dict[str, list[dict]]) -> None:
        """Display sector momentum overview."""
        if not sector_data:
            return

        table = Table(title="Sector Momentum", show_lines=True)
        table.add_column("Sector", style="cyan", width=20)
        table.add_column("Leader", width=22)
        table.add_column("1W Return", justify="right", width=10)
        table.add_column("1M Return", justify="right", width=10)
        table.add_column("Momentum", justify="right", width=10)

        for sector, stocks in sector_data.items():
            if stocks:
                leader = stocks[0]
                ret_1w = leader["return_1w"]
                ret_1m = leader["return_1m"]
                color = "green" if leader["momentum"] > 0 else "red"
                table.add_row(
                    sector.replace("_", " "),
                    leader["name"][:22],
                    Text(f"{ret_1w:+.1f}%", style=color),
                    Text(f"{ret_1m:+.1f}%", style=color),
                    Text(f"{leader['momentum']:+.1f}", style=color),
                )

        self.console.print(table)

    def _display_news_alerts(self, alerts: list[dict]) -> None:
        """Display news-discovered stocks."""
        if not alerts:
            return

        table = Table(title="News Alerts (Non-NIFTY 50)", show_lines=True)
        table.add_column("Company", style="cyan", width=25)
        table.add_column("Ticker", width=14)
        table.add_column("Reason", width=45)

        for alert in alerts:
            table.add_row(
                alert["company"],
                alert.get("ticker", "N/A"),
                alert.get("reason", ""),
            )

        self.console.print(table)

    def display_suggestions(self, result: SuggestionResult) -> None:
        """Display stock suggestions as a Rich table."""
        if not result.suggestions:
            self.console.print("[yellow]No suggestions found.[/]")
            return

        table = Table(title="Suggested Stocks", show_lines=True)
        table.add_column("Rank", justify="right", width=5)
        table.add_column("Symbol", style="cyan", width=14)
        table.add_column("Name", width=22)
        table.add_column("Price", justify="right", width=10)
        table.add_column("1W Ret", justify="right", width=9)
        table.add_column("1M Ret", justify="right", width=9)
        table.add_column("RSI", justify="right", width=6)
        table.add_column("News", justify="right", width=5)
        table.add_column("Score", justify="right", width=7)
        table.add_column("Reasons", width=38)

        for s in result.suggestions:
            w_color = "green" if s.return_1w >= 0 else "red"
            m_color = "green" if s.return_1m >= 0 else "red"
            table.add_row(
                str(s.rank),
                s.symbol,
                s.name[:22],
                f"{s.price:.2f}",
                Text(f"{s.return_1w:+.1f}%", style=w_color),
                Text(f"{s.return_1m:+.1f}%", style=m_color),
                f"{s.rsi:.0f}",
                str(s.news_mentions),
                f"{s.score:.1f}",
                "; ".join(s.reasons),
            )

        self.console.print(table)
        self.console.print(
            f"Screened {result.total_screened} stocks"
            + (f" | {result.news_articles_scanned} news articles scanned"
               if result.news_articles_scanned else "")
        )

    def display_stock_analysis(
        self,
        signal: TradingSignal,
        llm_scores: dict[str, float] | None = None,
    ) -> None:
        """Display detailed analysis for a single stock."""
        name = TICKER_TO_NAME.get(signal.symbol, signal.symbol)
        color = SIGNAL_COLORS.get(signal.signal, "white")

        self.console.print(Panel(f"Analysis: {name} ({signal.symbol})", style="bold blue"))
        self.console.print(f"Signal: [{color}]{signal.signal}[/] (Confidence: {signal.confidence:.1%})")
        self.console.print(f"Weekly: {signal.weekly_outlook}")
        self.console.print(f"Monthly: {signal.monthly_outlook}")

        if signal.is_short_candidate:
            self.console.print(f"[bold red]Short Selling Candidate[/] (Score: {signal.short_score:.2f})")

        if signal.llm_summary:
            self.console.print(Panel(signal.llm_summary, title="LLM Broker Analysis"))

        if llm_scores:
            table = Table(title="Broker Analysis Scores", show_lines=True)
            table.add_column("Factor", width=25)
            table.add_column("Score (0-10)", justify="right", width=12)

            for key, value in llm_scores.items():
                if key.startswith("_"):
                    continue
                score_color = "green" if value >= 6 else "red" if value <= 4 else "yellow"
                table.add_row(
                    key.replace("_", " ").title(),
                    Text(f"{value:.1f}", style=score_color),
                )

            self.console.print(table)

        if signal.top_headlines:
            self.console.print("\n[bold]Recent Headlines:[/]")
            for headline in signal.top_headlines[:5]:
                self.console.print(f"  - {headline}")

    def export_csv(self, signals: list[TradingSignal], filename: str = "signals.csv") -> Path:
        """Export signals to CSV."""
        path = self.export_dir / filename
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Symbol", "Signal", "Confidence", "Strength",
                "Buy%", "Hold%", "Sell%", "Short_Score",
                "Weekly_Outlook", "Monthly_Outlook",
            ])
            for sig in signals:
                writer.writerow([
                    sig.symbol, sig.signal, f"{sig.confidence:.4f}", f"{sig.strength:.4f}",
                    f"{sig.probabilities.get('BUY', 0):.4f}",
                    f"{sig.probabilities.get('HOLD', 0):.4f}",
                    f"{sig.probabilities.get('SELL', 0):.4f}",
                    f"{sig.short_score:.4f}",
                    sig.weekly_outlook, sig.monthly_outlook,
                ])
        logger.info(f"Exported CSV to {path}")
        return path

    def export_json(self, signals: list[TradingSignal], filename: str = "signals.json") -> Path:
        """Export signals to JSON."""
        path = self.export_dir / filename
        data = []
        for sig in signals:
            data.append({
                "symbol": sig.symbol,
                "signal": sig.signal,
                "confidence": sig.confidence,
                "strength": sig.strength,
                "short_score": sig.short_score,
                "is_short_candidate": sig.is_short_candidate,
                "probabilities": sig.probabilities,
                "weekly_outlook": sig.weekly_outlook,
                "monthly_outlook": sig.monthly_outlook,
                "llm_summary": sig.llm_summary,
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported JSON to {path}")
        return path

    def display_portfolio(self, trades: list[PaperTrade]) -> None:
        """Display open positions with unrealized PnL."""
        if not trades:
            self.console.print("[yellow]No open positions.[/]")
            return

        table = Table(title="Paper Trading Portfolio", show_lines=True)
        table.add_column("ID", width=8)
        table.add_column("Symbol", style="cyan", width=14)
        table.add_column("Type", width=6)
        table.add_column("Entry Date", width=12)
        table.add_column("Entry Price", justify="right", width=12)
        table.add_column("Qty", justify="right", width=10)
        table.add_column("Amount", justify="right", width=12)
        table.add_column("Current", justify="right", width=12)
        table.add_column("PnL", justify="right", width=12)
        table.add_column("PnL %", justify="right", width=8)

        total_invested = 0.0
        total_pnl = 0.0

        for t in trades:
            pnl = t.pnl or 0
            pnl_pct = t.pnl_pct or 0
            color = "green" if pnl >= 0 else "red"
            type_color = "green" if t.trade_type == "LONG" else "magenta"
            current = t.exit_price or t.entry_price  # exit_price used for current in portfolio

            table.add_row(
                t.trade_id,
                t.symbol,
                Text(t.trade_type, style=type_color),
                t.entry_date[:10],
                f"{t.entry_price:.2f}",
                f"{t.quantity:.4f}",
                f"{t.amount:.2f}",
                f"{current:.2f}",
                Text(f"{pnl:+.2f}", style=color),
                Text(f"{pnl_pct:+.1f}%", style=color),
            )
            total_invested += t.amount
            total_pnl += pnl

        self.console.print(table)

        pnl_color = "green" if total_pnl >= 0 else "red"
        self.console.print(f"Total Invested: INR {total_invested:,.2f}")
        self.console.print(f"Unrealized PnL: [{pnl_color}]INR {total_pnl:+,.2f}[/]")

    def display_gain_report(self, report: GainReport) -> None:
        """Display gain/loss analysis."""
        if report.total_trades == 0:
            self.console.print("[yellow]No closed trades to analyze.[/]")
            return

        self.console.print(Panel("Paper Trading Gain Report", style="bold blue"))

        # Summary
        pnl_color = "green" if report.total_pnl >= 0 else "red"
        win_rate = (report.winning_trades / report.total_trades * 100) if report.total_trades else 0

        table = Table(title="Summary", show_lines=True)
        table.add_column("Metric", width=25)
        table.add_column("Value", justify="right", width=20)

        table.add_row("Total Closed Trades", str(report.total_trades))
        table.add_row("Winning Trades", f"[green]{report.winning_trades}[/]")
        table.add_row("Losing Trades", f"[red]{report.losing_trades}[/]")
        table.add_row("Win Rate", f"{win_rate:.1f}%")
        table.add_row("Total PnL", Text(f"INR {report.total_pnl:+,.2f}", style=pnl_color))
        table.add_row("Total PnL %", Text(f"{report.total_pnl_pct:+.2f}%", style=pnl_color))
        table.add_row("Open Positions", str(report.open_positions))

        ur_color = "green" if report.unrealized_pnl >= 0 else "red"
        table.add_row("Unrealized PnL", Text(f"INR {report.unrealized_pnl:+,.2f}", style=ur_color))

        self.console.print(table)

        # Best/Worst trades
        if report.best_trade:
            b = report.best_trade
            self.console.print(
                f"[green]Best Trade:[/] {b['symbol']} ({b['trade_type']}) — "
                f"PnL: INR {b.get('pnl', 0):+,.2f} ({b.get('pnl_pct', 0):+.1f}%)"
            )
        if report.worst_trade:
            w = report.worst_trade
            self.console.print(
                f"[red]Worst Trade:[/] {w['symbol']} ({w['trade_type']}) — "
                f"PnL: INR {w.get('pnl', 0):+,.2f} ({w.get('pnl_pct', 0):+.1f}%)"
            )

        # Per-stock breakdown
        if report.per_stock:
            stock_table = Table(title="Per-Stock Breakdown", show_lines=True)
            stock_table.add_column("Symbol", style="cyan", width=14)
            stock_table.add_column("Trades", justify="right", width=8)
            stock_table.add_column("PnL", justify="right", width=14)
            stock_table.add_column("PnL %", justify="right", width=10)

            for sym, data in report.per_stock.items():
                color = "green" if data["pnl"] >= 0 else "red"
                stock_table.add_row(
                    sym,
                    str(data["trades"]),
                    Text(f"INR {data['pnl']:+,.2f}", style=color),
                    Text(f"{data['pnl_pct']:+.2f}%", style=color),
                )

            self.console.print(stock_table)

    def export_gain_report(self, report: GainReport) -> Path:
        """Write gain report to a dated JSON file."""
        from dataclasses import asdict

        report_dir = Path(
            get_setting("paper_trading", "report_dir", default="data/trades")
        )
        report_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime("%Y-%m-%d")
        path = report_dir / f"report_{date_str}.json"

        with open(path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        logger.info(f"Exported gain report to {path}")
        return path
