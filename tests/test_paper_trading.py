"""Tests for paper trading system."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from stock_prediction.signals.paper_trading import GainReport, PaperTrade, PaperTradingManager


@pytest.fixture
def ledger_path(tmp_path):
    """Provide a temporary ledger file path."""
    return tmp_path / "ledger.json"


@pytest.fixture
def manager(ledger_path):
    """Create a PaperTradingManager with a temp ledger."""
    return PaperTradingManager(ledger_path=ledger_path)


@pytest.fixture
def mock_price():
    """Mock _get_current_price to return controllable prices."""
    with patch.object(PaperTradingManager, "_get_current_price") as mock:
        yield mock


class TestBuy:
    def test_buy_creates_long_position(self, manager, mock_price):
        mock_price.return_value = 2500.0
        trade = manager.buy("RELIANCE.NS", 50000)

        assert trade.trade_type == "LONG"
        assert trade.status == "OPEN"
        assert trade.symbol == "RELIANCE.NS"
        assert trade.entry_price == 2500.0
        assert trade.quantity == pytest.approx(20.0)
        assert trade.amount == 50000

    def test_buy_persists_to_ledger(self, manager, mock_price, ledger_path):
        mock_price.return_value = 1000.0
        manager.buy("TCS.NS", 10000)

        data = json.loads(ledger_path.read_text())
        assert len(data) == 1
        assert data[0]["symbol"] == "TCS.NS"

    def test_buy_covers_existing_short(self, manager, mock_price):
        mock_price.return_value = 2500.0
        manager.short_sell("RELIANCE.NS", 50000)

        mock_price.return_value = 2400.0  # price dropped, short profits
        trade = manager.buy("RELIANCE.NS", 50000)

        assert trade.status == "CLOSED"
        assert trade.trade_type == "SHORT"
        assert trade.pnl == pytest.approx(100.0 * 20.0)  # (2500-2400)*20


class TestSell:
    def test_sell_closes_long_position(self, manager, mock_price):
        mock_price.return_value = 1000.0
        manager.buy("TCS.NS", 10000)

        mock_price.return_value = 1100.0
        trade = manager.sell("TCS.NS")

        assert trade.status == "CLOSED"
        assert trade.exit_price == 1100.0
        assert trade.pnl == pytest.approx(100.0 * 10.0)  # (1100-1000)*10
        assert trade.pnl_pct == pytest.approx(10.0)

    def test_sell_no_position_raises(self, manager, mock_price):
        mock_price.return_value = 1000.0
        with pytest.raises(ValueError, match="No open position"):
            manager.sell("NONEXISTENT.NS")

    def test_sell_covers_short_if_no_long(self, manager, mock_price):
        mock_price.return_value = 2000.0
        manager.short_sell("INFY.NS", 20000)

        mock_price.return_value = 1800.0
        trade = manager.sell("INFY.NS")

        assert trade.status == "CLOSED"
        assert trade.trade_type == "SHORT"
        assert trade.pnl == pytest.approx(200.0 * 10.0)

    def test_sell_by_trade_id(self, manager, mock_price):
        mock_price.return_value = 500.0
        t1 = manager.buy("SYM.NS", 5000)
        t2 = manager.buy("SYM.NS", 10000)

        mock_price.return_value = 600.0
        trade = manager.sell("SYM.NS", trade_id=t2.trade_id)

        assert trade.trade_id == t2.trade_id
        assert trade.status == "CLOSED"

        # First trade should still be open
        trades = manager._load_ledger()
        open_trades = [t for t in trades if t.status == "OPEN"]
        assert len(open_trades) == 1
        assert open_trades[0].trade_id == t1.trade_id


class TestShortSell:
    def test_short_creates_short_position(self, manager, mock_price):
        mock_price.return_value = 3000.0
        trade = manager.short_sell("HDFCBANK.NS", 30000)

        assert trade.trade_type == "SHORT"
        assert trade.status == "OPEN"
        assert trade.entry_price == 3000.0
        assert trade.quantity == pytest.approx(10.0)

    def test_short_profit_when_price_drops(self, manager, mock_price):
        mock_price.return_value = 3000.0
        manager.short_sell("HDFCBANK.NS", 30000)

        mock_price.return_value = 2800.0
        trade = manager.cover_short("HDFCBANK.NS")

        assert trade.pnl == pytest.approx(200.0 * 10.0)
        assert trade.pnl_pct == pytest.approx(200.0 / 3000.0 * 100)

    def test_short_loss_when_price_rises(self, manager, mock_price):
        mock_price.return_value = 3000.0
        manager.short_sell("HDFCBANK.NS", 30000)

        mock_price.return_value = 3300.0
        trade = manager.cover_short("HDFCBANK.NS")

        assert trade.pnl == pytest.approx(-300.0 * 10.0)


class TestCoverShort:
    def test_cover_no_short_raises(self, manager, mock_price):
        mock_price.return_value = 1000.0
        with pytest.raises(ValueError, match="No open SHORT"):
            manager.cover_short("NONEXISTENT.NS")


class TestPortfolio:
    def test_empty_portfolio(self, manager):
        trades = manager.get_portfolio()
        assert trades == []

    def test_portfolio_shows_open_with_unrealized_pnl(self, manager, mock_price):
        mock_price.return_value = 1000.0
        manager.buy("A.NS", 10000)
        manager.short_sell("B.NS", 20000)

        mock_price.return_value = 1100.0
        trades = manager.get_portfolio()

        assert len(trades) == 2
        long_trade = next(t for t in trades if t.trade_type == "LONG")
        short_trade = next(t for t in trades if t.trade_type == "SHORT")

        # LONG: price went up = profit
        assert long_trade.pnl == pytest.approx(100.0 * 10.0)
        # SHORT: price went up = loss
        assert short_trade.pnl == pytest.approx(-100.0 * 20.0)

    def test_portfolio_excludes_closed(self, manager, mock_price):
        mock_price.return_value = 1000.0
        manager.buy("A.NS", 10000)
        manager.buy("B.NS", 10000)

        mock_price.return_value = 1100.0
        manager.sell("A.NS")

        trades = manager.get_portfolio()
        assert len(trades) == 1
        assert trades[0].symbol == "B.NS"


class TestCalculateGains:
    def test_no_closed_trades(self, manager, mock_price):
        report = manager.calculate_gains()
        assert report.total_trades == 0
        assert report.total_pnl == 0.0

    def test_gain_report_with_closed_trades(self, manager, mock_price):
        # Buy and sell two stocks
        mock_price.return_value = 1000.0
        manager.buy("A.NS", 10000)
        manager.buy("B.NS", 20000)

        mock_price.return_value = 1200.0
        manager.sell("A.NS")  # +20% win

        mock_price.return_value = 900.0
        manager.sell("B.NS")  # -10% loss

        report = manager.calculate_gains()

        assert report.total_trades == 2
        assert report.winning_trades == 1
        assert report.losing_trades == 1
        assert report.total_pnl == pytest.approx(2000.0 + (-2000.0))  # (200*10) + (-100*20)
        assert report.best_trade["symbol"] == "A.NS"
        assert report.worst_trade["symbol"] == "B.NS"
        assert "A.NS" in report.per_stock
        assert "B.NS" in report.per_stock

    def test_gain_report_per_stock_breakdown(self, manager, mock_price):
        mock_price.return_value = 100.0
        manager.buy("X.NS", 1000)
        manager.buy("X.NS", 2000)

        mock_price.return_value = 110.0
        manager.sell("X.NS")  # closes first

        mock_price.return_value = 120.0
        manager.sell("X.NS")  # closes second

        report = manager.calculate_gains()
        assert report.per_stock["X.NS"]["trades"] == 2
        assert report.per_stock["X.NS"]["pnl"] > 0


class TestLedgerPersistence:
    def test_ledger_roundtrip(self, manager, mock_price, ledger_path):
        mock_price.return_value = 500.0
        manager.buy("A.NS", 5000)
        manager.short_sell("B.NS", 10000)

        # Create new manager pointing to same file
        manager2 = PaperTradingManager(ledger_path=ledger_path)
        trades = manager2._load_ledger()
        assert len(trades) == 2
        assert trades[0].symbol == "A.NS"
        assert trades[1].symbol == "B.NS"

    def test_empty_ledger_returns_empty_list(self, manager):
        trades = manager._load_ledger()
        assert trades == []
