"""Streamlit web app for the Stock Prediction System.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the installed package importable when running from the project root.
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockPredict India",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Reduce top padding */
.main .block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

/* Section headers */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e3a5f;
    border-left: 4px solid #2563eb;
    padding-left: 8px;
    margin: 1.2rem 0 0.4rem;
}

/* Watchlist chips in sidebar */
.wl-chip {
    display: inline-block;
    background: #dbeafe;
    color: #1e40af;
    padding: 2px 9px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
}

/* Larger metric values */
[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session state initialisation ────────────────────────────────────────────
for _key in (
    "watchlist",
    "suggest_result",
    "shortlist_result",
    "screen_result",
    "train_results",
    "predict_signals",
    "predict_warnings",
    "portfolio_trades",
    "gain_report",
):
    if _key not in st.session_state:
        st.session_state[_key] = set() if _key == "watchlist" else None

# ─── Constants ───────────────────────────────────────────────────────────────
SIGNAL_BG: dict[str, str] = {
    "STRONG BUY":  "#166534",
    "BUY":         "#16a34a",
    "HOLD":        "#d97706",
    "SELL":        "#dc2626",
    "STRONG SELL": "#7f1d1d",
    "N/A":         "#6b7280",
    "ERROR":       "#374151",
}

PAGES = [
    "📊 Suggest",
    "📋 Shortlist",
    "🔍 Lookup",
    "📥 Fetch Data",
    "🧠 Train",
    "🔮 Predict",
    "🔬 Analyze",
    "📡 Screen",
    "💰 Trade",
    "💼 Portfolio",
    "📈 Gain Report",
]

# ─── Shared helpers ───────────────────────────────────────────────────────────

def _watchlist_csv() -> str:
    return ",".join(sorted(st.session_state.watchlist))


def _parse_symbols(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


# Pandas Styler colour functions
def _color_signal(val: str) -> str:
    bg = SIGNAL_BG.get(val, "#6b7280")
    return f"background-color:{bg}; color:white; font-weight:700"


def _color_return(val: str) -> str:
    try:
        v = float(str(val).replace("%", "").replace("+", ""))
        if "+" in str(val) or v > 0:
            return "color:#16a34a; font-weight:600"
        if v < 0:
            return "color:#dc2626; font-weight:600"
    except Exception:
        pass
    return ""


def _color_pnl(val: str) -> str:
    try:
        s = str(val).replace("₹", "").replace(",", "").replace("%", "").strip()
        v = float(s)
        if "+" in str(val) or v > 0:
            return "color:#16a34a; font-weight:600"
        if v < 0:
            return "color:#dc2626; font-weight:600"
    except Exception:
        pass
    return ""


def _color_status(val: str) -> str:
    return {
        "success": "color:#16a34a; font-weight:600",
        "no_data": "color:#d97706; font-weight:600",
        "failed":  "color:#dc2626; font-weight:600",
    }.get(val, "")


def _suggestion_df(suggestions) -> pd.DataFrame:
    rows = []
    for s in suggestions:
        rows.append({
            "Select":    False,
            "Rank":      s.rank,
            "Symbol":    s.symbol,
            "Name":      s.name,
            "Price (₹)": f"{s.price:,.2f}",
            "1W Ret":    f"{s.return_1w:+.1f}%",
            "1M Ret":    f"{s.return_1m:+.1f}%",
            "RSI":       f"{s.rsi:.0f}",
            "News":      s.news_mentions,
            "Score":     f"{s.score:.2f}",
            "Reasons":   " | ".join(s.reasons),
        })
    return pd.DataFrame(rows)


def _section(title: str, color: str = "#2563eb") -> None:
    st.markdown(
        f'<div class="section-header" style="border-color:{color}">{title}</div>',
        unsafe_allow_html=True,
    )


def _signal_badge(signal: str) -> None:
    bg = SIGNAL_BG.get(signal, "#6b7280")
    st.markdown(
        f'<div style="background:{bg};color:white;padding:14px 20px;'
        f'border-radius:10px;text-align:center;margin-bottom:0.5rem">'
        f'<div style="font-size:1.5rem;font-weight:800">{signal}</div>'
        f'<div style="font-size:0.82rem;opacity:0.9">Signal</div></div>',
        unsafe_allow_html=True,
    )


def _predict_for_symbol(symbol, trainer, signal_gen, use_news, use_llm):
    """Shared prediction logic — returns TradingSignal or raises."""
    from stock_prediction.features.pipeline import FeaturePipeline
    import numpy as np

    ensemble, scaler, seq_scaler, model_age = trainer.load_models(symbol)
    pipeline = FeaturePipeline(use_news=use_news, use_llm=use_llm)
    df = pipeline.build_features(symbol)

    if df.empty:
        raise ValueError("No feature data returned")

    label_cols = ["return_1d", "return_5d", "signal"]
    feature_cols = [c for c in df.columns if c not in label_cols]
    features = df[feature_cols].values
    seq_len = pipeline.sequence_length

    if len(features) < seq_len:
        raise ValueError(f"Insufficient data ({len(features)} rows < {seq_len} needed)")

    lseq = features[-seq_len:]
    ltab = features[-1]
    n_f = lseq.shape[1]
    lss = seq_scaler.transform(lseq.reshape(-1, n_f)).reshape(1, seq_len, n_f)
    lts = scaler.transform(ltab.reshape(1, -1))
    pred = ensemble.predict_single(lss.astype(np.float32), lts.astype(np.float32))

    tech = {
        c: float(df[c].iloc[-1])
        for c in ["RSI", "MACD_Histogram", "Price_SMA50_Ratio"]
        if c in df.columns
    }
    return signal_gen.generate(symbol, pred, tech), model_age, df


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockPredict")
    st.markdown("*Indian Market · NIFTY 50*")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Watchlist**")
    wl_sorted = sorted(st.session_state.watchlist)
    if wl_sorted:
        chips = " ".join(
            f'<span class="wl-chip">{s.replace(".NS", "")}</span>' for s in wl_sorted
        )
        st.markdown(chips, unsafe_allow_html=True)
        if st.button("🗑 Clear Watchlist", key="clear_wl"):
            st.session_state.watchlist.clear()
            st.rerun()
    else:
        st.caption("Empty — add symbols from Suggest / Shortlist")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Suggest ──────────────────────────────────────────────────────────────────
def page_suggest() -> None:
    st.title("📊 Suggest — Top NIFTY 50 Stocks")
    st.caption(
        "Ranks all NIFTY 50 stocks by technical momentum and news mentions. "
        "Select stocks to add them to your Watchlist."
    )

    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        count = st.number_input("Number of stocks", 1, 50, 10, key="sug_count")
    with col2:
        use_news = st.checkbox("Include news", value=True, key="sug_news")

    if st.button("▶ Run Suggest", type="primary", key="sug_run"):
        with st.spinner("Scoring NIFTY 50 stocks …"):
            try:
                from stock_prediction.signals.screener import StockScreener
                st.session_state.suggest_result = StockScreener().suggest(
                    count=int(count), use_news=use_news
                )
            except Exception as e:
                st.error(f"Error: {e}")
                return

    result = st.session_state.suggest_result
    if result is None:
        return

    st.success(
        f"Screened **{result.total_screened}** stocks · "
        f"**{result.news_articles_scanned}** news articles scanned"
    )

    _section("Top Suggestions")
    df = _suggestion_df(result.suggestions)
    edited = st.data_editor(
        df,
        column_config={"Select": st.column_config.CheckboxColumn("✓", width="small")},
        hide_index=True,
        use_container_width=True,
        key="sug_editor",
    )
    selected = edited[edited["Select"]]["Symbol"].tolist()
    if selected and st.button(f"➕ Add {len(selected)} symbol(s) to Watchlist", key="sug_add"):
        st.session_state.watchlist.update(selected)
        st.success(f"Added: {', '.join(selected)}")
        st.rerun()


# ─── Shortlist ────────────────────────────────────────────────────────────────
def page_shortlist() -> None:
    st.title("📋 Shortlist — Buy / Short / Trending")
    st.caption(
        "Identifies top buy candidates, short candidates, and news-trending stocks. "
        "Select any to add them to your Watchlist."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        count = st.number_input("Candidates per category", 1, 20, 5, key="sl_count")
    with col2:
        use_news = st.checkbox("Include news", value=True, key="sl_news")
    with col3:
        use_llm = st.checkbox("Include LLM", value=True, key="sl_llm")

    if st.button("▶ Run Shortlist", type="primary", key="sl_run"):
        with st.spinner("Shortlisting stocks …"):
            try:
                from stock_prediction.signals.screener import StockScreener
                st.session_state.shortlist_result = StockScreener().shortlist(
                    count=int(count), use_news=use_news, use_llm=use_llm
                )
            except Exception as e:
                st.error(f"Error: {e}")
                return

    result = st.session_state.shortlist_result
    if result is None:
        return

    st.success(
        f"Screened **{result.total_screened}** stocks · "
        f"**{result.news_articles_scanned}** news articles scanned"
    )

    all_selected: list[str] = []
    for label, candidates, color in (
        ("🟢 Buy Candidates",   result.buy_candidates,   "#16a34a"),
        ("🔴 Short Candidates", result.short_candidates, "#dc2626"),
        ("📰 Trending (News)",  result.trending,         "#7c3aed"),
    ):
        _section(label, color)
        if candidates:
            df = _suggestion_df(candidates)
            edited = st.data_editor(
                df,
                column_config={"Select": st.column_config.CheckboxColumn("✓", width="small")},
                hide_index=True,
                use_container_width=True,
                key=f"sl_{label[:2]}",
            )
            all_selected.extend(edited[edited["Select"]]["Symbol"].tolist())
        else:
            st.caption("No candidates found.")

    if all_selected and st.button(
        f"➕ Add {len(all_selected)} symbol(s) to Watchlist", key="sl_add"
    ):
        st.session_state.watchlist.update(all_selected)
        st.success(f"Added: {', '.join(all_selected)}")
        st.rerun()


# ─── Lookup ───────────────────────────────────────────────────────────────────
def page_lookup() -> None:
    st.title("🔍 Lookup — Signal by Company Name")
    st.caption(
        "Search by any part of a company name, ticker, or alias "
        "(e.g. 'tata', 'bank', 'infosys'). Requires trained models."
    )

    query = st.text_input(
        "Search query", placeholder="e.g. tata, reliance, hdfc bank", key="lk_q"
    )
    if not st.button("🔍 Search", type="primary", key="lk_run") or not query.strip():
        return

    q = query.strip().lower()
    with st.spinner(f"Searching for '{query}' …"):
        try:
            from stock_prediction.utils.constants import COMPANY_ALIASES, TICKER_TO_NAME

            matched: dict[str, str] = {}
            for alias, ticker in COMPANY_ALIASES.items():
                if q in alias:
                    matched[ticker] = TICKER_TO_NAME.get(ticker, ticker)
            for ticker, name in TICKER_TO_NAME.items():
                if q in name.lower():
                    matched[ticker] = name
            for ticker in TICKER_TO_NAME:
                if q in ticker.lower().replace(".ns", ""):
                    matched[ticker] = TICKER_TO_NAME[ticker]

            if not matched:
                st.warning(f"No companies found matching '{query}'. Try a broader term.")
                return

            from stock_prediction.models.trainer import ModelTrainer
            from stock_prediction.signals.generator import SignalGenerator

            trainer = ModelTrainer()
            signal_gen = SignalGenerator()
            rows = []

            for symbol, name in sorted(matched.items()):
                try:
                    sig, model_age, _ = _predict_for_symbol(
                        symbol, trainer, signal_gen,
                        use_news=False, use_llm=False,
                    )
                    rows.append({
                        "Symbol":     symbol,
                        "Company":    name,
                        "Signal":     sig.signal,
                        "Confidence": f"{sig.confidence:.1%}",
                        "BUY %":      f"{sig.probabilities.get('BUY', 0):.0%}",
                        "HOLD %":     f"{sig.probabilities.get('HOLD', 0):.0%}",
                        "SELL %":     f"{sig.probabilities.get('SELL', 0):.0%}",
                        "Model Age":  f"{model_age}d" if model_age else "—",
                    })
                except FileNotFoundError:
                    rows.append({
                        "Symbol": symbol, "Company": name, "Signal": "N/A",
                        "Confidence": "—", "BUY %": "—", "HOLD %": "—",
                        "SELL %": "—", "Model Age": "No model",
                    })
                except Exception as ex:
                    rows.append({
                        "Symbol": symbol, "Company": name, "Signal": "ERROR",
                        "Confidence": "—", "BUY %": "—", "HOLD %": "—",
                        "SELL %": "—", "Model Age": str(ex)[:40],
                    })

        except Exception as e:
            st.error(f"Lookup error: {e}")
            return

    st.success(f"Found **{len(rows)}** result(s) for '{query}'")
    styled = pd.DataFrame(rows).style.map(_color_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─── Fetch Data ───────────────────────────────────────────────────────────────
def page_fetch_data() -> None:
    st.title("📥 Fetch Data — Preview Stock Prices")
    st.caption(
        "Fetches price data from yfinance for a preview. "
        "**Note:** data is not saved to disk — `train` fetches independently."
    )

    if "fd_syms" not in st.session_state:
        st.session_state["fd_syms"] = _watchlist_csv()

    symbols_input = st.text_input(
        "Stock Symbols",
        placeholder="e.g. RELIANCE.NS,TCS.NS",
        key="fd_syms",
    )
    start_date = st.date_input("Start Date (optional)", value=None, key="fd_start")

    if st.button("▶ Fetch", type="primary", key="fd_run"):
        symbol_list = _parse_symbols(symbols_input)
        if not symbol_list:
            st.warning("Please enter at least one symbol.")
            return

        with st.spinner(f"Fetching data for {len(symbol_list)} symbol(s) …"):
            try:
                from stock_prediction.data import get_provider
                provider = get_provider()
                sd = str(start_date) if start_date else None
                results = provider.fetch_batch(symbol_list, start_date=sd)

                summary = []
                preview_sym, preview_data = None, None
                for sym, data in results.items():
                    if data.is_empty:
                        summary.append({"Symbol": sym, "Rows": 0, "From": "—", "To": "—", "Status": "No data"})
                    else:
                        summary.append({
                            "Symbol": sym,
                            "Rows":   len(data.df),
                            "From":   str(data.date_range[0]),
                            "To":     str(data.date_range[1]),
                            "Status": "OK",
                        })
                        if preview_sym is None:
                            preview_sym, preview_data = sym, data

                st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

                if preview_data is not None:
                    _section(f"Preview: {preview_sym} — last 10 rows")
                    st.dataframe(preview_data.df.tail(10).round(2), use_container_width=True)

            except Exception as e:
                st.error(f"Fetch error: {e}")


# ─── Train ────────────────────────────────────────────────────────────────────
def page_train() -> None:
    st.title("🧠 Train — Model Training")
    st.caption(
        "Trains an LSTM + XGBoost ensemble for each symbol. "
        "Trained models are saved to `data/models/`. May take several minutes per stock."
    )

    # NIFTY 50 button must come BEFORE the text_input so session state is set first
    if st.button("Use All NIFTY 50 Stocks", key="tr_nifty"):
        from stock_prediction.utils.constants import NIFTY_50_TICKERS
        st.session_state["tr_syms"] = ",".join(NIFTY_50_TICKERS)
        st.rerun()

    if "tr_syms" not in st.session_state:
        st.session_state["tr_syms"] = _watchlist_csv()

    symbols_input = st.text_input(
        "Stock Symbols",
        placeholder="e.g. RELIANCE.NS,TCS.NS,INFY.NS",
        key="tr_syms",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input("Start Date", value=None, key="tr_start")
    with col2:
        end_date = st.date_input("End Date", value=None, key="tr_end")
    with col3:
        use_news = st.checkbox("News features", value=True, key="tr_news")
    with col4:
        use_llm = st.checkbox("LLM features", value=True, key="tr_llm")

    if st.button("▶ Start Training", type="primary", key="tr_run"):
        symbol_list = _parse_symbols(symbols_input)
        if not symbol_list:
            st.warning("Please enter at least one symbol.")
            return

        st.info(f"Training {len(symbol_list)} stock(s) — this may take a while …")
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            from stock_prediction.models.trainer import ModelTrainer
            trainer = ModelTrainer(use_news=use_news, use_llm=use_llm)
            sd = str(start_date) if start_date else None
            ed = str(end_date) if end_date else None

            results: dict[str, dict] = {}
            for i, sym in enumerate(symbol_list):
                status_text.markdown(f"Training **{sym}** ({i + 1} / {len(symbol_list)}) …")
                try:
                    model, accuracy = trainer.train_stock(sym, sd, ed)
                    if model is None:
                        results[sym] = {"status": "no_data", "accuracy": None, "reason": "No training data"}
                    else:
                        results[sym] = {"status": "success", "accuracy": accuracy, "reason": ""}
                except ValueError as e:
                    results[sym] = {"status": "no_data", "accuracy": None, "reason": str(e)}
                except Exception as e:
                    results[sym] = {"status": "failed", "accuracy": None, "reason": str(e)}
                progress_bar.progress((i + 1) / len(symbol_list))

            status_text.empty()
            st.session_state.train_results = results

        except Exception as e:
            st.error(f"Training error: {e}")
            return

    results = st.session_state.train_results
    if results is None:
        return

    ok = sum(1 for v in results.values() if v["status"] == "success")
    st.success(f"Training complete: **{ok} / {len(results)}** stocks trained successfully")

    rows = [
        {
            "Symbol":       sym,
            "Status":       r["status"],
            "Val Accuracy": f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "—",
            "Reason":       r["reason"],
        }
        for sym, r in results.items()
    ]
    styled = pd.DataFrame(rows).style.map(_color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─── Predict ──────────────────────────────────────────────────────────────────
def page_predict() -> None:
    st.title("🔮 Predict — Trading Signals")
    st.caption(
        "Generates BUY / SELL / HOLD signals using trained models. "
        "Symbols pre-filled from your Watchlist."
    )

    if st.button("Use All NIFTY 50 Stocks", key="pr_nifty"):
        from stock_prediction.utils.constants import NIFTY_50_TICKERS
        st.session_state["pr_syms"] = ",".join(NIFTY_50_TICKERS)
        st.rerun()

    if "pr_syms" not in st.session_state:
        st.session_state["pr_syms"] = _watchlist_csv()

    symbols_input = st.text_input(
        "Stock Symbols",
        placeholder="e.g. RELIANCE.NS,TCS.NS",
        key="pr_syms",
    )
    col1, col2 = st.columns(2)
    with col1:
        use_news = st.checkbox("News features", value=True, key="pr_news")
    with col2:
        use_llm = st.checkbox("LLM features", value=True, key="pr_llm")

    if st.button("▶ Run Predict", type="primary", key="pr_run"):
        symbol_list = _parse_symbols(symbols_input)
        if not symbol_list:
            st.warning("Please enter at least one symbol.")
            return

        with st.spinner(f"Generating predictions for {len(symbol_list)} symbol(s) …"):
            try:
                from stock_prediction.models.trainer import ModelTrainer
                from stock_prediction.signals.generator import SignalGenerator
                from stock_prediction.config import get_setting

                trainer = ModelTrainer()
                signal_gen = SignalGenerator()
                staleness = get_setting("models", "staleness_warning_days", default=30)

                signals = []
                warn_msgs: list[str] = []

                for symbol in symbol_list:
                    try:
                        sig, model_age, _ = _predict_for_symbol(
                            symbol, trainer, signal_gen, use_news, use_llm
                        )
                        if model_age and model_age > staleness:
                            warn_msgs.append(
                                f"{symbol}: model is {model_age} days old — consider retraining"
                            )
                        signals.append(sig)
                    except FileNotFoundError:
                        warn_msgs.append(f"{symbol}: no trained model — run Train first")
                    except Exception as ex:
                        warn_msgs.append(f"{symbol}: {ex}")

                st.session_state.predict_signals = signals
                st.session_state.predict_warnings = warn_msgs

            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

    signals = st.session_state.predict_signals
    if signals is None:
        return

    for w in (st.session_state.predict_warnings or []):
        st.warning(w)

    if not signals:
        st.info("No signals generated. Ensure models are trained first.")
        return

    # Summary counts
    counts = Counter(s.signal for s in signals)
    cols = st.columns(5)
    for col, lbl in zip(cols, ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]):
        with col:
            bg = SIGNAL_BG[lbl]
            st.markdown(
                f'<div style="background:{bg};color:white;border-radius:8px;'
                f'padding:10px;text-align:center;font-weight:700">'
                f'{lbl}<br><span style="font-size:1.8rem">{counts.get(lbl,0)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _section("Signal Table")

    def _short_color(val: str) -> str:
        return "color:#dc2626; font-weight:600" if val == "Yes" else ""

    rows = [
        {
            "Symbol":          sig.symbol,
            "Signal":          sig.signal,
            "Confidence":      f"{sig.confidence:.1%}",
            "BUY %":           f"{sig.probabilities.get('BUY', 0):.0%}",
            "HOLD %":          f"{sig.probabilities.get('HOLD', 0):.0%}",
            "SELL %":          f"{sig.probabilities.get('SELL', 0):.0%}",
            "RSI":             f"{sig.technical_summary.get('RSI', 0):.0f}" if sig.technical_summary.get("RSI") else "—",
            "Short?":          "Yes" if sig.is_short_candidate else "No",
            "Weekly Outlook":  sig.weekly_outlook,
        }
        for sig in signals
    ]
    styled = (
        pd.DataFrame(rows)
        .style
        .map(_color_signal, subset=["Signal"])
        .map(_short_color, subset=["Short?"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─── Analyze ──────────────────────────────────────────────────────────────────
def page_analyze() -> None:
    st.title("🔬 Analyze — Deep Stock Analysis")
    st.caption(
        "Single-stock deep dive: model signal, LLM broker scores, and recent headlines."
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.text_input("Symbol", placeholder="e.g. RELIANCE.NS", key="an_sym")
    with col2:
        use_news = st.checkbox("News", value=True, key="an_news")
    with col3:
        use_llm = st.checkbox("LLM", value=True, key="an_llm")

    if not st.button("▶ Analyze", type="primary", key="an_run") or not symbol.strip():
        return

    sym = symbol.strip().upper()
    with st.spinner(f"Analyzing {sym} …"):
        try:
            from stock_prediction.utils.constants import TICKER_TO_NAME
            from stock_prediction.models.trainer import ModelTrainer
            from stock_prediction.signals.generator import SignalGenerator
            from stock_prediction.config import get_setting

            name = TICKER_TO_NAME.get(sym, sym.replace(".NS", ""))
            llm_scores: dict = {}
            headlines: list[str] = []
            llm_summary = ""

            if use_llm:
                try:
                    from stock_prediction.llm import get_llm_provider
                    from stock_prediction.llm.news_analyzer import BrokerNewsAnalyzer
                    from stock_prediction.news.rss_fetcher import GoogleNewsRSSFetcher
                    articles = GoogleNewsRSSFetcher().fetch_stock_news(name)
                    llm_scores = BrokerNewsAnalyzer(get_llm_provider()).analyze_stock(sym, articles)
                    llm_summary = str(llm_scores.pop("_summary", ""))
                    headlines = [a.title for a in articles[:5]]
                except Exception as ex:
                    st.warning(f"LLM unavailable: {ex}")
            elif use_news:
                try:
                    from stock_prediction.news.rss_fetcher import GoogleNewsRSSFetcher
                    articles = GoogleNewsRSSFetcher().fetch_stock_news(name)
                    headlines = [a.title for a in articles[:5]]
                except Exception as ex:
                    st.warning(f"News fetch failed: {ex}")

            signal = None
            try:
                trainer = ModelTrainer()
                signal_gen = SignalGenerator()
                staleness = get_setting("models", "staleness_warning_days", default=30)
                # _predict_for_symbol returns the base signal; re-generate with LLM context
                _, model_age, df = _predict_for_symbol(
                    sym, trainer, signal_gen, use_news, use_llm
                )
                if model_age and model_age > staleness:
                    st.warning(f"Model for {sym} is {model_age} days old — consider retraining.")
                label_cols = ["return_1d", "return_5d", "signal"]
                feature_cols = [c for c in df.columns if c not in label_cols]
                tech = {
                    c: float(df[c].iloc[-1])
                    for c in ["RSI", "MACD_Histogram", "Price_SMA50_Ratio"]
                    if c in df.columns
                }
                import numpy as np
                ensemble, scaler, seq_scaler, _ = trainer.load_models(sym)
                seq_len = 60
                features = df[feature_cols].values
                n_f = features.shape[1]
                lss = seq_scaler.transform(features[-seq_len:].reshape(-1, n_f)).reshape(1, seq_len, n_f)
                lts = scaler.transform(features[-1].reshape(1, -1))
                pred = ensemble.predict_single(lss.astype(np.float32), lts.astype(np.float32))
                signal = signal_gen.generate(sym, pred, tech, llm_summary, headlines)
            except FileNotFoundError:
                st.warning(f"No trained model for {sym}. Run Train first.")
            except Exception as ex:
                st.error(f"Prediction error: {ex}")

        except Exception as e:
            st.error(f"Analysis error: {e}")
            return

    st.markdown(f"## {name} ({sym})")

    if signal:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            _signal_badge(signal.signal)
        with c2:
            st.metric("Confidence", f"{signal.confidence:.1%}")
        with c3:
            st.metric("BUY Prob", f"{signal.probabilities.get('BUY', 0):.0%}")
        with c4:
            st.metric("SELL Prob", f"{signal.probabilities.get('SELL', 0):.0%}")

        if signal.weekly_outlook:
            st.info(f"**Short-term:** {signal.weekly_outlook}")
        if signal.monthly_outlook:
            st.info(f"**Medium-term:** {signal.monthly_outlook}")

        if signal.technical_summary:
            _section("Technical Indicators")
            items = list(signal.technical_summary.items())
            tech_cols = st.columns(len(items))
            for col, (k, v) in zip(tech_cols, items):
                with col:
                    st.metric(k.replace("_", " "), f"{v:.3f}")

    if llm_scores:
        broker_rows = [
            {"Broker": k, "Score": v}
            for k, v in llm_scores.items()
            if not k.startswith("_")
        ]
        if broker_rows:
            _section("Broker Analysis Scores")
            st.dataframe(pd.DataFrame(broker_rows), use_container_width=True, hide_index=True)

    if llm_summary:
        _section("LLM Summary")
        st.markdown(llm_summary)

    if headlines:
        _section("Recent Headlines")
        for h in headlines:
            st.markdown(f"• {h}")


# ─── Screen ───────────────────────────────────────────────────────────────────
def page_screen() -> None:
    st.title("📡 Screen — Full Stock Screener")
    st.caption(
        "Multi-layer screening: technical signals, sector momentum, and LLM-based news discovery."
    )

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        symbols_input = st.text_input(
            "Symbols (leave blank for all NIFTY 50)",
            placeholder="e.g. RELIANCE.NS,TCS.NS",
            key="sc_syms",
        )
    with col2:
        use_news = st.checkbox("News", value=True, key="sc_news")
    with col3:
        use_llm = st.checkbox("LLM", value=True, key="sc_llm")

    if st.button("▶ Run Screener", type="primary", key="sc_run"):
        with st.spinner("Screening stocks …"):
            try:
                from stock_prediction.signals.screener import StockScreener
                from stock_prediction.utils.constants import NIFTY_50_TICKERS
                symbol_list = _parse_symbols(symbols_input) or NIFTY_50_TICKERS
                result = StockScreener().screen(symbol_list)
                if not use_news or not use_llm:
                    result.news_alerts = []
                st.session_state.screen_result = result
            except Exception as e:
                st.error(f"Screen error: {e}")
                return

    result = st.session_state.screen_result
    if result is None:
        return

    # Top Picks
    _section("🏆 Top Picks")
    if result.top_picks:
        st.dataframe(pd.DataFrame(result.top_picks), use_container_width=True, hide_index=True)
    else:
        st.caption("No top picks found.")

    # Sector Leaders
    _section("🏭 Sector Leaders")
    if result.sector_leaders:
        for sector, stocks in result.sector_leaders.items():
            with st.expander(f"{sector}  ({len(stocks)} stocks)"):
                st.dataframe(pd.DataFrame(stocks), use_container_width=True, hide_index=True)
    else:
        st.caption("No sector data.")

    # News Alerts
    if result.news_alerts:
        _section("📰 News Alerts (LLM Discovery)")
        st.dataframe(pd.DataFrame(result.news_alerts), use_container_width=True, hide_index=True)

    # Full Rankings
    _section("📊 Full Rankings")
    if result.full_rankings:
        st.dataframe(pd.DataFrame(result.full_rankings), use_container_width=True, hide_index=True)


# ─── Trade ────────────────────────────────────────────────────────────────────
def page_trade() -> None:
    st.title("💰 Trade — Paper Trading")
    st.caption("Simulate buy, sell, and short-sell trades with virtual money. No real capital at risk.")

    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Symbol", placeholder="e.g. RELIANCE.NS", key="td_sym")
        action = st.selectbox(
            "Action",
            ["Buy (Long)", "Sell / Cover", "Short Sell"],
            key="td_action",
        )
    with col2:
        amount = st.number_input(
            "Amount (₹ INR)", min_value=0.0, value=10_000.0, step=1_000.0, key="td_amount"
        )
        trade_id_input = ""
        if action == "Sell / Cover":
            trade_id_input = st.text_input(
                "Trade ID (optional — leave blank to auto-select)",
                key="td_tid",
            )

    if st.button("▶ Execute Trade", type="primary", key="td_run"):
        if not symbol.strip():
            st.warning("Please enter a symbol.")
            return

        sym = symbol.strip().upper()
        with st.spinner(f"Executing {action} for {sym} …"):
            try:
                from stock_prediction.signals.paper_trading import PaperTradingManager
                manager = PaperTradingManager()

                if action == "Buy (Long)":
                    trade = manager.buy(sym, float(amount))
                    if trade.status == "CLOSED":
                        st.success(f"Covered SHORT {sym} @ ₹{trade.exit_price:,.2f}")
                        st.metric("Realized PnL", f"₹{trade.pnl:+,.2f} ({trade.pnl_pct:+.1f}%)")
                    else:
                        st.success(
                            f"BUY {sym} — {trade.quantity:.4f} shares @ ₹{trade.entry_price:,.2f}"
                        )
                        c1, c2 = st.columns(2)
                        c1.metric("Amount Invested", f"₹{trade.amount:,.0f}")
                        c2.info(f"Trade ID: **{trade.trade_id}**")

                elif action == "Sell / Cover":
                    tid = trade_id_input.strip() or None
                    trade = manager.sell(sym, trade_id=tid)
                    action_lbl = "SELL" if trade.trade_type == "LONG" else "COVER SHORT"
                    st.success(
                        f"{action_lbl} {sym} — {trade.quantity:.4f} shares @ ₹{trade.exit_price:,.2f}"
                    )
                    st.metric("Realized PnL", f"₹{trade.pnl:+,.2f} ({trade.pnl_pct:+.1f}%)")

                elif action == "Short Sell":
                    trade = manager.short_sell(sym, float(amount))
                    st.success(
                        f"SHORT {sym} — {trade.quantity:.4f} shares @ ₹{trade.entry_price:,.2f}"
                    )
                    c1, c2 = st.columns(2)
                    c1.metric("Amount at Risk", f"₹{trade.amount:,.0f}")
                    c2.info(f"Trade ID: **{trade.trade_id}**")

            except Exception as e:
                st.error(f"Trade error: {e}")


# ─── Portfolio ────────────────────────────────────────────────────────────────
def page_portfolio() -> None:
    st.title("💼 Portfolio — Open Positions")
    st.caption("Shows all open paper trades with live unrealized P&L (fetches current prices).")

    if st.button("🔄 Refresh Portfolio", type="primary", key="pf_refresh"):
        with st.spinner("Fetching current prices …"):
            try:
                from stock_prediction.signals.paper_trading import PaperTradingManager
                st.session_state.portfolio_trades = PaperTradingManager().get_portfolio()
            except Exception as e:
                st.error(f"Portfolio error: {e}")
                return

    trades = st.session_state.portfolio_trades
    if trades is None:
        return

    if not trades:
        st.info("No open positions. Open trades on the Trade page.")
        return

    rows = [
        {
            "Trade ID":         t.trade_id,
            "Symbol":           t.symbol,
            "Type":             t.trade_type,
            "Entry Date":       t.entry_date[:10],
            "Entry ₹":          f"{t.entry_price:,.2f}",
            "Current ₹":        f"{t.exit_price:,.2f}" if t.exit_price else "—",
            "Qty":              f"{t.quantity:.4f}",
            "Invested ₹":       f"{t.amount:,.0f}",
            "Unrealized PnL ₹": f"{t.pnl:+,.2f}" if t.pnl is not None else "—",
            "PnL %":            f"{t.pnl_pct:+.1f}%" if t.pnl_pct is not None else "—",
        }
        for t in trades
    ]
    styled = pd.DataFrame(rows).style.map(_color_pnl, subset=["Unrealized PnL ₹", "PnL %"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    total_pnl = sum(t.pnl or 0 for t in trades)
    total_inv = sum(t.amount for t in trades)
    c1, c2, c3 = st.columns(3)
    c1.metric("Open Positions", len(trades))
    c2.metric("Total Invested", f"₹{total_inv:,.0f}")
    c3.metric("Total Unrealized PnL", f"₹{total_pnl:+,.2f}")


# ─── Gain Report ──────────────────────────────────────────────────────────────
def page_gain_report() -> None:
    st.title("📈 Gain Report — Closed Trade Analysis")
    st.caption("Calculates realized gains and losses across all closed paper trades.")

    if st.button("📊 Calculate Gains", type="primary", key="gr_run"):
        with st.spinner("Calculating …"):
            try:
                from stock_prediction.signals.paper_trading import PaperTradingManager
                st.session_state.gain_report = PaperTradingManager().calculate_gains()
            except Exception as e:
                st.error(f"Error: {e}")
                return

    report = st.session_state.gain_report
    if report is None:
        return

    if report.total_trades == 0:
        st.info("No closed trades yet. Open and close positions on the Trade page.")
        return

    # Summary metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Closed Trades", report.total_trades)
    c2.metric("Winners ✅", report.winning_trades)
    c3.metric("Losers ❌", report.losing_trades)
    c4.metric("Total Realized PnL", f"₹{report.total_pnl:+,.2f}")
    c5.metric("Overall PnL %", f"{report.total_pnl_pct:+.2f}%")

    c1, c2 = st.columns(2)
    c1.metric("Open Positions", report.open_positions)
    c2.metric("Unrealized PnL", f"₹{report.unrealized_pnl:+,.2f}")

    # Best / Worst
    if report.best_trade or report.worst_trade:
        _section("Notable Trades")
        c1, c2 = st.columns(2)
        with c1:
            if report.best_trade:
                st.success(
                    f"**Best:** {report.best_trade['symbol']} — "
                    f"₹{report.best_trade.get('pnl', 0):+,.2f} "
                    f"({report.best_trade.get('pnl_pct', 0):+.1f}%)"
                )
        with c2:
            if report.worst_trade:
                st.error(
                    f"**Worst:** {report.worst_trade['symbol']} — "
                    f"₹{report.worst_trade.get('pnl', 0):+,.2f} "
                    f"({report.worst_trade.get('pnl_pct', 0):+.1f}%)"
                )

    # Per-stock breakdown
    if report.per_stock:
        _section("Per-Stock Breakdown")
        rows = [
            {
                "Symbol":    sym,
                "Trades":    d["trades"],
                "Total PnL ₹": f"{d['pnl']:+,.2f}",
                "Invested ₹":  f"{d['total_invested']:,.0f}",
                "PnL %":       f"{d.get('pnl_pct', 0):+.2f}%",
            }
            for sym, d in report.per_stock.items()
        ]
        styled = pd.DataFrame(rows).style.map(_color_pnl, subset=["Total PnL ₹", "PnL %"])
        st.dataframe(styled, use_container_width=True, hide_index=True)


# ─── Router ───────────────────────────────────────────────────────────────────
PAGE_MAP = {
    "📊 Suggest":    page_suggest,
    "📋 Shortlist":  page_shortlist,
    "🔍 Lookup":     page_lookup,
    "📥 Fetch Data": page_fetch_data,
    "🧠 Train":      page_train,
    "🔮 Predict":    page_predict,
    "🔬 Analyze":    page_analyze,
    "📡 Screen":     page_screen,
    "💰 Trade":      page_trade,
    "💼 Portfolio":  page_portfolio,
    "📈 Gain Report": page_gain_report,
}

PAGE_MAP[page]()
