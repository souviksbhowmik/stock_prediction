"""Paper trading system for simulated buy/sell/short trades."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from stock_prediction.config import get_setting
from stock_prediction.data import get_provider
from stock_prediction.utils.logging import get_logger

logger = get_logger("signals.paper_trading")


@dataclass
class PaperTrade:
    trade_id: str
    symbol: str
    trade_type: str  # "LONG" or "SHORT"
    entry_date: str  # ISO date
    entry_price: float
    quantity: float
    amount: float  # INR invested
    exit_date: str | None = None
    exit_price: float | None = None
    status: str = "OPEN"  # "OPEN" or "CLOSED"
    pnl: float | None = None
    pnl_pct: float | None = None


@dataclass
class GainReport:
    report_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_pnl_pct: float
    best_trade: dict | None
    worst_trade: dict | None
    per_stock: dict[str, dict]
    open_positions: int
    unrealized_pnl: float


class PaperTradingManager:
    """Manages the paper trade ledger file."""

    def __init__(self, ledger_path: str | Path | None = None):
        if ledger_path is not None:
            self.ledger_path = Path(ledger_path)
        else:
            self.ledger_path = Path(
                get_setting("paper_trading", "ledger_file", default="data/trades/ledger.json")
            )
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def buy(self, symbol: str, amount: float) -> PaperTrade:
        """Open a LONG position or cover an existing SHORT."""
        trades = self._load_ledger()

        # Check if there's an open SHORT to cover
        open_short = next(
            (t for t in trades if t.symbol == symbol and t.trade_type == "SHORT" and t.status == "OPEN"),
            None,
        )
        if open_short:
            return self.cover_short(symbol)

        price = self._get_current_price(symbol)
        quantity = amount / price

        trade = PaperTrade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            trade_type="LONG",
            entry_date=datetime.now().isoformat(),
            entry_price=price,
            quantity=quantity,
            amount=amount,
        )

        trades.append(trade)
        self._save_ledger(trades)
        logger.info(f"BUY {symbol}: {quantity:.4f} shares @ {price:.2f} (INR {amount:.2f})")
        return trade

    def sell(self, symbol: str, trade_id: str | None = None) -> PaperTrade:
        """Close an existing LONG position or cover a SHORT."""
        trades = self._load_ledger()

        # Find open position for this symbol
        if trade_id:
            position = next(
                (t for t in trades if t.trade_id == trade_id and t.status == "OPEN"),
                None,
            )
        else:
            # Try LONG first, then SHORT
            position = next(
                (t for t in trades if t.symbol == symbol and t.trade_type == "LONG" and t.status == "OPEN"),
                None,
            )
            if position is None:
                position = next(
                    (t for t in trades if t.symbol == symbol and t.trade_type == "SHORT" and t.status == "OPEN"),
                    None,
                )

        if position is None:
            raise ValueError(f"No open position found for {symbol}")

        if position.trade_type == "SHORT":
            return self.cover_short(symbol, trade_id=position.trade_id)

        price = self._get_current_price(symbol)
        position.exit_date = datetime.now().isoformat()
        position.exit_price = price
        position.status = "CLOSED"
        position.pnl = (price - position.entry_price) * position.quantity
        position.pnl_pct = ((price - position.entry_price) / position.entry_price) * 100

        self._save_ledger(trades)
        logger.info(f"SELL {symbol}: {position.quantity:.4f} shares @ {price:.2f} (PnL: {position.pnl:+.2f})")
        return position

    def short_sell(self, symbol: str, amount: float) -> PaperTrade:
        """Open a SHORT position."""
        price = self._get_current_price(symbol)
        quantity = amount / price

        trade = PaperTrade(
            trade_id=str(uuid.uuid4())[:8],
            symbol=symbol,
            trade_type="SHORT",
            entry_date=datetime.now().isoformat(),
            entry_price=price,
            quantity=quantity,
            amount=amount,
        )

        trades = self._load_ledger()
        trades.append(trade)
        self._save_ledger(trades)
        logger.info(f"SHORT {symbol}: {quantity:.4f} shares @ {price:.2f} (INR {amount:.2f})")
        return trade

    def cover_short(self, symbol: str, trade_id: str | None = None) -> PaperTrade:
        """Close an existing SHORT position (buy to cover)."""
        trades = self._load_ledger()

        if trade_id:
            position = next(
                (t for t in trades if t.trade_id == trade_id and t.status == "OPEN"),
                None,
            )
        else:
            position = next(
                (t for t in trades if t.symbol == symbol and t.trade_type == "SHORT" and t.status == "OPEN"),
                None,
            )

        if position is None:
            raise ValueError(f"No open SHORT position found for {symbol}")

        price = self._get_current_price(symbol)
        position.exit_date = datetime.now().isoformat()
        position.exit_price = price
        position.status = "CLOSED"
        # SHORT PnL: profit when price drops
        position.pnl = (position.entry_price - price) * position.quantity
        position.pnl_pct = ((position.entry_price - price) / position.entry_price) * 100

        self._save_ledger(trades)
        logger.info(f"COVER SHORT {symbol}: {position.quantity:.4f} shares @ {price:.2f} (PnL: {position.pnl:+.2f})")
        return position

    def get_portfolio(self) -> list[PaperTrade]:
        """Return all OPEN trades with unrealized PnL."""
        trades = self._load_ledger()
        open_trades = [t for t in trades if t.status == "OPEN"]

        for trade in open_trades:
            try:
                price = self._get_current_price(trade.symbol)
                if trade.trade_type == "LONG":
                    trade.pnl = (price - trade.entry_price) * trade.quantity
                    trade.pnl_pct = ((price - trade.entry_price) / trade.entry_price) * 100
                else:  # SHORT
                    trade.pnl = (trade.entry_price - price) * trade.quantity
                    trade.pnl_pct = ((trade.entry_price - price) / trade.entry_price) * 100
                trade.exit_price = price  # current price for display
            except Exception as e:
                logger.warning(f"Could not fetch price for {trade.symbol}: {e}")

        return open_trades

    def calculate_gains(self) -> GainReport:
        """Analyze all CLOSED trades and generate a gain report."""
        trades = self._load_ledger()
        closed = [t for t in trades if t.status == "CLOSED"]
        open_trades = [t for t in trades if t.status == "OPEN"]

        if not closed:
            return GainReport(
                report_date=datetime.now().isoformat(),
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
                best_trade=None,
                worst_trade=None,
                per_stock={},
                open_positions=len(open_trades),
                unrealized_pnl=0.0,
            )

        winners = [t for t in closed if (t.pnl or 0) > 0]
        losers = [t for t in closed if (t.pnl or 0) < 0]

        total_pnl = sum(t.pnl or 0 for t in closed)
        total_invested = sum(t.amount for t in closed)
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0.0

        best = max(closed, key=lambda t: t.pnl or 0)
        worst = min(closed, key=lambda t: t.pnl or 0)

        # Per-stock breakdown
        per_stock: dict[str, dict] = {}
        for trade in closed:
            sym = trade.symbol
            if sym not in per_stock:
                per_stock[sym] = {"trades": 0, "pnl": 0.0, "total_invested": 0.0}
            per_stock[sym]["trades"] += 1
            per_stock[sym]["pnl"] += trade.pnl or 0
            per_stock[sym]["total_invested"] += trade.amount
        for sym in per_stock:
            invested = per_stock[sym]["total_invested"]
            per_stock[sym]["pnl_pct"] = (per_stock[sym]["pnl"] / invested * 100) if invested else 0.0

        # Unrealized PnL for open positions
        unrealized = 0.0
        for trade in open_trades:
            try:
                price = self._get_current_price(trade.symbol)
                if trade.trade_type == "LONG":
                    unrealized += (price - trade.entry_price) * trade.quantity
                else:
                    unrealized += (trade.entry_price - price) * trade.quantity
            except Exception:
                pass

        return GainReport(
            report_date=datetime.now().isoformat(),
            total_trades=len(closed),
            winning_trades=len(winners),
            losing_trades=len(losers),
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            best_trade=asdict(best),
            worst_trade=asdict(worst),
            per_stock=per_stock,
            open_positions=len(open_trades),
            unrealized_pnl=unrealized,
        )

    def _load_ledger(self) -> list[PaperTrade]:
        """Load trades from the JSON ledger file."""
        if not self.ledger_path.exists():
            return []
        with open(self.ledger_path) as f:
            data = json.load(f)
        return [PaperTrade(**item) for item in data]

    def _save_ledger(self, trades: list[PaperTrade]) -> None:
        """Save trades to the JSON ledger file."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "w") as f:
            json.dump([asdict(t) for t in trades], f, indent=2)

    def _get_current_price(self, symbol: str) -> float:
        """Fetch current market price for a symbol."""
        provider = get_provider()
        data = provider.fetch_latest(symbol)
        if data.is_empty:
            raise ValueError(f"Could not fetch price for {symbol}")
        return float(data.df["Close"].iloc[-1])
