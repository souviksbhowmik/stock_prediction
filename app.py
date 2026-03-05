"""Streamlit web app for the Stock Prediction System.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import threading
import time
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

/* Watchlist action panel */
.wl-action-panel {
    background: #f0f7ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 4px;
    font-size: 0.78rem;
}

/* Larger metric values */
[data-testid="stMetricValue"] { font-size: 1.5rem !important; }

/* Copyable HTML tables */
.copy-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.875rem;
    margin-bottom: 1rem;
}
.copy-table th {
    background: #1e3a5f;
    color: white;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    white-space: nowrap;
}
.copy-table td {
    padding: 7px 12px;
    border-bottom: 1px solid #e5e7eb;
    white-space: nowrap;
    user-select: text;
    -webkit-user-select: text;
    cursor: text;
}
.copy-table tr:nth-child(even) td { background: #f8fafc; }
.copy-table tr:hover td { background: #eff6ff !important; }
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
    "predict_signal_metas",
    "portfolio_trades",
    "gain_report",
):
    if _key not in st.session_state:
        st.session_state[_key] = set() if _key == "watchlist" else None

if "settings_cache" not in st.session_state:
    st.session_state["settings_cache"] = {}

# Apply session overrides so all downstream get_setting() calls pick them up.
from stock_prediction.config import set_ui_overrides as _set_ui_overrides
_set_ui_overrides(st.session_state.get("settings_cache", {}))

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
    "📊 Suggest (deprecated)",
    "📋 Shortlist",
    "🔍 Lookup",
    "📥 Fetch Data",
    "🧠 Train",
    "🧪 Experimental",
    "🔮 Predict",
    "🔬 Analyze",
    "📡 Screen",
    "💰 Trade",
    "💼 Portfolio",
    "📈 Gain Report",
    "⚙️ Settings",
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


def _color_short(val: str) -> str:
    return "color:#dc2626; font-weight:600" if val == "Yes" else ""


def _show_table(
    df: pd.DataFrame,
    style_map: dict | None = None,
    tooltip_cols: set[str] | None = None,
) -> None:
    """Render a DataFrame as a copyable HTML table with optional per-column cell styling.

    style_map:   {column_name: callable(value) -> css_string}
    tooltip_cols: set of column names whose content is truncated in the cell and shown
                  in full as a native browser tooltip on hover.
    """
    cols = list(df.columns)
    headers = "".join(f"<th>{c}</th>" for c in cols)

    body_rows: list[str] = []
    for _, row in df.iterrows():
        cells: list[str] = []
        for col in cols:
            val = "" if row[col] is None else str(row[col])
            css = style_map[col](val) if (style_map and col in style_map) else ""
            if tooltip_cols and col in tooltip_cols and val not in ("", "—"):
                css = (css + ";" if css else "") + (
                    "max-width:160px;overflow:hidden;text-overflow:ellipsis;"
                    "white-space:nowrap;cursor:default"
                )
                cells.append(f'<td style="{css}" title="{val}">{val}</td>')
            else:
                style_attr = f' style="{css}"' if css else ""
                cells.append(f"<td{style_attr}>{val}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    html = (
        f'<div style="overflow-x:auto">'
        f'<table class="copy-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


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


# ─── Background training worker ──────────────────────────────────────────────

def _run_training_bg(
    symbol_list: list[str],
    sd: str | None,
    ed: str | None,
    use_news: bool,
    use_llm: bool,
    use_financials: bool,
    selected_models: list[str],
    horizon: int,
    progress: dict,
) -> None:
    """Runs in a daemon thread — keeps going even if the user navigates away."""
    try:
        from stock_prediction.models.trainer import ModelTrainer
        from stock_prediction.config import load_settings
        # Override horizon in the cached config so all pipeline calls use it
        load_settings()["features"]["prediction_horizon"] = horizon
        trainer = ModelTrainer(use_news=use_news, use_llm=use_llm, use_financials=use_financials)
        for i, sym in enumerate(symbol_list):
            if progress.get("cancelled"):
                break
            progress["current_sym"] = sym
            progress["done"] = i
            try:
                model, accuracy, plot_paths = trainer.train_stock(sym, sd, ed, selected_models)
                if model is None:
                    progress["results"][sym] = {
                        "status": "no_data", "accuracy": None, "reason": "No training data",
                        "plot_paths": {},
                    }
                else:
                    progress["results"][sym] = {
                        "status": "success", "accuracy": accuracy, "reason": "",
                        "plot_paths": plot_paths,
                    }
            except ValueError as e:
                progress["results"][sym] = {
                    "status": "no_data", "accuracy": None, "reason": str(e),
                    "plot_paths": {},
                }
            except Exception as e:
                progress["results"][sym] = {
                    "status": "failed", "accuracy": None, "reason": str(e),
                    "plot_paths": {},
                }
            progress["done"] = i + 1
    except Exception as e:
        progress["error"] = str(e)
    finally:
        import gc
        gc.collect()
        progress["complete"] = True
        progress["current_sym"] = None


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


def _add_sym_to_input(session_key: str, symbol: str) -> None:
    """Append symbol to a comma-separated session state input field (no duplicates)."""
    existing = _parse_symbols(st.session_state.get(session_key, ""))
    if symbol not in existing:
        existing.append(symbol)
    st.session_state[session_key] = ",".join(existing)


def _predict_for_symbol(symbol, trainer, signal_gen):
    """Shared prediction logic — returns (TradingSignal, model_age, meta, df) or raises.

    Feature flags (use_news / use_llm / use_financials) and horizon are loaded
    from the model's saved meta so prediction always matches training conditions.
    """
    from stock_prediction.features.pipeline import FeaturePipeline
    import numpy as np

    ensemble, scaler, seq_scaler, model_age, meta = trainer.load_models(symbol)

    use_news       = meta.get("use_news", True)
    use_llm        = meta.get("use_llm", True)
    use_financials = meta.get("use_financials", True)
    horizon        = meta.get("horizon", 5)
    feature_names  = meta.get("feature_names", [])

    pipeline = FeaturePipeline(use_news=use_news, use_llm=use_llm, use_financials=use_financials)
    df = pipeline.build_features(symbol)

    if df.empty:
        raise ValueError("No feature data returned")

    label_cols = {"return_1d", "return_5d", f"return_{horizon}d", "signal"}

    # Use training feature columns in training order for scaler compatibility
    if feature_names:
        available = [c for c in feature_names if c in df.columns]
        features = df[available].values if available else df[
            [c for c in df.columns if c not in label_cols]
        ].values
    else:
        features = df[[c for c in df.columns if c not in label_cols]].values

    seq_len = pipeline.sequence_length

    if len(features) < seq_len:
        raise ValueError(f"Insufficient data ({len(features)} rows < {seq_len} needed)")

    lseq = features[-seq_len:]
    ltab = features[-1]
    n_f = lseq.shape[1]
    lss = seq_scaler.transform(lseq.reshape(-1, n_f)).reshape(1, seq_len, n_f)
    lts = scaler.transform(ltab.reshape(1, -1))

    # Build lag-feature row for xgboost_lag (None if model not in ensemble)
    x_tab_lag = None
    if "xgboost_lag" in meta.get("selected_models", []):
        _lag_scaler = meta.get("lag_scaler")
        _lag_feat   = meta.get("lag_feature_names", [])
        if _lag_scaler is not None and _lag_feat:
            from stock_prediction.features.technical import add_lag_trend_features
            _lag_df = add_lag_trend_features(df)
            # Select feature columns first so label-NaN rows (last `horizon` rows)
            # are not dropped — only early rolling-window NaN rows are removed.
            _avail = [c for c in _lag_feat if c in _lag_df.columns]
            if _avail:
                _feat_df = _lag_df[_avail].dropna()
                if not _feat_df.empty:
                    _row = _feat_df.values[-1]
                    x_tab_lag = _lag_scaler.transform(_row.reshape(1, -1)).astype(np.float32)

    pred = ensemble.predict_single(
        lss.astype(np.float32), lts.astype(np.float32), X_tab_lag=x_tab_lag
    )

    tech = {
        c: float(df[c].iloc[-1])
        for c in ["RSI", "MACD_Histogram", "Price_SMA50_Ratio"]
        if c in df.columns
    }
    return signal_gen.generate(symbol, pred, tech), model_age, meta, df


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockPredict")
    st.markdown("*Indian Market · NIFTY 50*")
    st.markdown("---")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    # Training status badge — visible from every page
    _tp = st.session_state.get("train_progress")
    if _tp and not _tp.get("complete"):
        _done  = _tp.get("done", 0)
        _total = _tp.get("total", 1)
        st.markdown("---")
        st.markdown(
            f'<div style="background:#1e40af;color:white;border-radius:8px;'
            f'padding:8px 12px;font-size:0.8rem;font-weight:600">'
            f'⏳ Training {_done}/{_total}</div>',
            unsafe_allow_html=True,
        )
    # Experimental training status badge
    _ep = st.session_state.get("exp_progress")
    if _ep and not _ep.get("complete"):
        st.markdown("---")
        st.caption(f"🧪 Exp {_ep.get('done', 0)}/{_ep.get('total', 1)} ({_ep.get('symbol', '')})")
    st.markdown("---")
    st.markdown("**Watchlist** — click a symbol to act on it")
    wl_sorted = sorted(st.session_state.watchlist)
    if wl_sorted:
        selected_sym = st.session_state.get("wl_selected")

        # One button per symbol — highlighted when selected
        for sym in wl_sorted:
            label = sym.replace(".NS", "")
            is_sel = sym == selected_sym
            if st.button(
                f"{'▶ ' if is_sel else ''}{label}",
                key=f"wl_chip_{sym}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                st.session_state["wl_selected"] = None if is_sel else sym
                st.rerun()

        # Action panel shown below when a symbol is selected
        if selected_sym and selected_sym in st.session_state.watchlist:
            sname = selected_sym.replace(".NS", "")
            st.markdown(
                f'<div class="wl-action-panel">Add <b>{sname}</b> to …</div>',
                unsafe_allow_html=True,
            )

            # Pages that accept multiple symbols (appended)
            multi_actions = [
                ("🧠 Train",      "tr_syms"),
                ("🔮 Predict",    "pr_syms"),
                ("📥 Fetch Data", "fd_syms"),
                ("📡 Screen",     "sc_syms"),
            ]
            # Pages that accept a single symbol (set directly)
            single_actions = [
                ("🔬 Analyze", "an_sym"),
                ("💰 Trade",   "td_sym"),
            ]

            col1, col2 = st.columns(2)
            for i, (lbl, key) in enumerate(multi_actions):
                with (col1 if i % 2 == 0 else col2):
                    if st.button(lbl, key=f"wl_act_{key}", use_container_width=True):
                        _add_sym_to_input(key, selected_sym)
                        st.toast(f"{sname} → {lbl.split()[-1]}", icon="✅")

            for i, (lbl, key) in enumerate(single_actions):
                with (col1 if i % 2 == 0 else col2):
                    if st.button(lbl, key=f"wl_act_{key}", use_container_width=True):
                        st.session_state[key] = selected_sym
                        st.toast(f"{sname} → {lbl.split()[-1]}", icon="✅")

            if st.button("⭐ All Pages", key="wl_act_all", use_container_width=True):
                for _, key in multi_actions:
                    _add_sym_to_input(key, selected_sym)
                for _, key in single_actions:
                    st.session_state[key] = selected_sym
                st.toast(f"{sname} added to all pages", icon="⭐")

            if st.button("🗑 Remove", key="wl_remove", use_container_width=True):
                st.session_state.watchlist.discard(selected_sym)
                st.session_state["wl_selected"] = None
                st.rerun()

        st.markdown("")
        if st.button("🗑 Clear All", key="clear_wl", use_container_width=True):
            st.session_state.watchlist.clear()
            st.session_state["wl_selected"] = None
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
                    sig, model_age, _, __ = _predict_for_symbol(
                        symbol, trainer, signal_gen
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
    _show_table(pd.DataFrame(rows), style_map={"Signal": _color_signal})


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

                _show_table(pd.DataFrame(summary))

                if preview_data is not None:
                    _section(f"Preview: {preview_sym} — last 10 rows")
                    _show_table(preview_data.df.tail(10).round(2).reset_index())

            except Exception as e:
                st.error(f"Fetch error: {e}")


# ─── Plot popup dialog ────────────────────────────────────────────────────────
@st.dialog("Plot Viewer", width="large")
def _show_plots_popup(symbol: str, plot_paths: dict) -> None:
    """Modal dialog with interactive Plotly plots for train/val/pred."""
    st.markdown(f"### {symbol} — Training Plots")
    tab_tr, tab_val, tab_pred = st.tabs(["📊 Training Period", "📈 Train+Val Fit", "🔮 Prediction"])
    for tab, key, label in [
        (tab_tr,   "train_plot", "Training Period"),
        (tab_val,  "val_plot",   "Train+Val Fit"),
        (tab_pred, "pred_plot",  "Prediction"),
    ]:
        with tab:
            path = plot_paths.get(key)
            if path:
                from pathlib import Path as _Path
                p = _Path(path)
                if p.exists():
                    html = p.read_text(encoding="utf-8")
                    st.components.v1.html(html, height=520, scrolling=False)
                    st.download_button(
                        label=f"⬇ Download {label} Plot",
                        data=html.encode("utf-8"),
                        file_name=f"{symbol}_{key}.html",
                        mime="text/html",
                        key=f"dl_{symbol}_{key}",
                    )
                else:
                    st.info(f"Plot file not found: {path}")
            else:
                st.info(f"No {label} plot was generated for {symbol}.")


# ─── Model info dialog ────────────────────────────────────────────────────────
@st.dialog("Model Info", width="small")
def _show_model_info_popup(symbol: str, meta: dict) -> None:
    """Modal dialog showing saved training metadata for a symbol."""
    from datetime import datetime as _dt
    st.markdown(f"### {symbol}")

    trained_at = meta.get("trained_at", "")
    if trained_at:
        date_str = trained_at[:10]
        try:
            age_days = (_dt.now() - _dt.fromisoformat(trained_at)).days
            age_str = f"{age_days} day{'s' if age_days != 1 else ''} ago"
        except ValueError:
            age_str = ""
        st.write(f"**Trained:** {date_str}" + (f"  ·  {age_str}" if age_str else ""))
    else:
        st.write("**Trained:** unknown")

    st.write(f"**Horizon:** {meta.get('horizon', '—')} trading days")

    val_acc = meta.get("val_accuracy")
    acc_str = f"{val_acc:.4f}" if val_acc is not None else "— (retrain to record)"
    st.write(f"**Val accuracy:** {acc_str}")

    st.markdown("**Models & ensemble weights:**")
    selected = meta.get("selected_models", [])
    weight_keys = {
        "lstm": "lstm_weight", "xgboost": "xgb_weight",
        "encoder_decoder": "ed_weight", "prophet": "prophet_weight",
    }
    for m in selected:
        w = meta.get(weight_keys.get(m, ""), 0.0)
        bar = "█" * int(round(w * 10))
        st.write(f"  • **{m}** — {w:.1%}  `{bar}`")

    st.markdown("**Features used during training:**")
    feats = [
        ("News",       meta.get("use_news", True)),
        ("LLM",        meta.get("use_llm", True)),
        ("Financials", meta.get("use_financials", True)),
    ]
    for name, enabled in feats:
        icon = "✅" if enabled else "❌"
        st.write(f"  {icon} {name}")


# ─── Train ────────────────────────────────────────────────────────────────────
def _render_train_results(results: dict) -> None:
    ok = sum(1 for v in results.values() if v["status"] == "success")
    st.success(f"Training complete: **{ok} / {len(results)}** stocks trained successfully")

    # Header row
    hcols = st.columns([1.2, 1, 1.2, 3, 0.8])
    for col, hdr in zip(hcols, ["Symbol", "Status", "Balanced Acc", "Reason", "Plots"]):
        col.markdown(f"**{hdr}**")

    # One row per symbol with a "View Plots" button
    csv_rows = []
    for sym, r in results.items():
        rcols = st.columns([1.2, 1, 1.2, 3, 0.8])
        rcols[0].write(sym)
        status = r["status"]
        status_css = {"success": "#16a34a", "no_data": "#d97706", "failed": "#dc2626"}.get(status, "#6b7280")
        rcols[1].markdown(f'<span style="color:{status_css};font-weight:600">{status}</span>', unsafe_allow_html=True)
        rcols[2].write(f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "—")
        rcols[3].write(r.get("reason") or "—")
        plot_paths = r.get("plot_paths") or {}
        if plot_paths and status == "success":
            if rcols[4].button("📊", key=f"plt_{sym}", help="View plots"):
                _show_plots_popup(sym, plot_paths)
        else:
            rcols[4].write("—")
        csv_rows.append({
            "Symbol":       sym,
            "Status":       status,
            "Balanced Acc": f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "—",
            "Reason":       r.get("reason") or "",
        })

    df_csv = pd.DataFrame(csv_rows)
    st.download_button(
        label="⬇ Download CSV",
        data=df_csv.to_csv(index=False),
        file_name="train_results.csv",
        mime="text/csv",
        key="tr_download",
    )


def page_train() -> None:
    st.title("🧠 Train — Model Training")
    st.caption(
        "Trains models for each symbol and saves them to `data/models/`. "
        "Select one or more models below — multiple selections use ensemble (weighted average). "
        "**Training runs in the background — you can freely navigate to other pages.**"
    )

    tp = st.session_state.get("train_progress")
    training_active = tp is not None and not tp.get("complete", False)

    # ── In-progress view ──────────────────────────────────────────────────────
    if training_active:
        done  = tp.get("done", 0)
        total = tp.get("total", 1)
        current = tp.get("current_sym") or ""
        frac  = done / total if total else 0

        st.info("⏳ Training is running in the background — safe to navigate away and return.")
        st.progress(frac)
        st.markdown(
            f"**{done} / {total}** complete"
            + (f" — currently training **{current}**" if current else "")
        )

        if tp.get("error"):
            st.error(f"Training error: {tp['error']}")

        # Show partial results so far
        if tp["results"]:
            _section("Results so far")
            _render_train_results(tp["results"])

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("⏹ Cancel", key="tr_cancel"):
                tp["cancelled"] = True
        # Auto-refresh every 2 s to show live progress
        time.sleep(2)
        st.rerun()
        return

    # ── Completed view ────────────────────────────────────────────────────────
    if tp is not None and tp.get("complete"):
        if tp.get("error"):
            st.error(f"Training stopped with error: {tp['error']}")
        _render_train_results(tp["results"])
        if st.button("🔄 Train Again", key="tr_again"):
            st.session_state["train_progress"] = None
            st.rerun()
        return

    # ── Setup form (only shown when not training) ─────────────────────────────
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

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None, key="tr_start")
    with col2:
        end_date = st.date_input("End Date", value=None, key="tr_end")

    col3, col4, col5, col6, col7 = st.columns(5)
    with col3:
        from stock_prediction.models.trainer import AVAILABLE_MODELS
        selected_models = st.multiselect(
            "Models to train",
            options=AVAILABLE_MODELS,
            default=["lstm"],
            key="tr_models",
            help="Select one or more. Multiple selections → ensemble (weighted average).",
        )
        if len(selected_models) > 1:
            st.caption("Ensemble mode: models will be weighted by validation accuracy.")
    with col4:
        _HORIZON_OPTIONS = [1, 3, 5, 7, 10]
        horizon = st.selectbox(
            "Prediction horizon",
            options=_HORIZON_OPTIONS,
            index=_HORIZON_OPTIONS.index(5),
            key="tr_horizon",
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
            help="Trading days ahead the model predicts. Saved with the model.",
        )
    with col5:
        use_news = st.checkbox("News features", value=True, key="tr_news")
    with col6:
        use_llm = st.checkbox("LLM features", value=True, key="tr_llm")
    with col7:
        use_financials = st.checkbox(
            "Financial features", value=True, key="tr_fin",
            help="Quarterly P&L / balance sheet ratios + report aging features"
        )

    if st.button("▶ Start Training", type="primary", key="tr_run"):
        symbol_list = _parse_symbols(symbols_input)
        if not symbol_list:
            st.warning("Please enter at least one symbol.")
            return
        if not selected_models:
            st.warning("Please select at least one model.")
            return

        progress: dict = {
            "symbols":     symbol_list,
            "total":       len(symbol_list),
            "done":        0,
            "current_sym": None,
            "results":     {},
            "complete":    False,
            "cancelled":   False,
            "error":       None,
        }
        st.session_state["train_progress"] = progress

        thread = threading.Thread(
            target=_run_training_bg,
            args=(
                symbol_list,
                str(start_date) if start_date else None,
                str(end_date) if end_date else None,
                use_news,
                use_llm,
                use_financials,
                selected_models,
                horizon,
                progress,
            ),
            daemon=True,
        )
        thread.start()
        st.rerun()


# ─── Experimental Training ────────────────────────────────────────────────────

def _run_experimental_bg(
    symbol: str,
    algorithms: list[str],
    sd: str | None,
    ed: str | None,
    use_news: bool,
    use_llm: bool,
    use_financials: bool,
    horizon: int,
    save_dir: Path,
    progress: dict,
) -> None:
    """Background thread: trains each algorithm in an isolated subprocess.

    Each algorithm runs inside a ``ProcessPoolExecutor`` worker so that
    native-library crashes or resource leaks (Prophet/Stan semaphores,
    PyTorch MPS, XGBoost) are contained within the worker and cannot
    bring down the Streamlit server.
    """
    import concurrent.futures
    from datetime import datetime as _dt
    from stock_prediction.models.trainer import train_single_algorithm

    sym_key = symbol.replace(".", "_")
    try:
        for alg in algorithms:
            if progress.get("cancelled"):
                break
            progress["current_alg"] = alg
            run_id = f"{_dt.now():%Y%m%d_%H%M%S}_{alg}"
            run_dir = save_dir / sym_key / "experimental" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        train_single_algorithm,
                        symbol, alg, sd, ed,
                        use_news, use_llm, use_financials,
                        horizon, str(run_dir),
                    )
                    accuracy = future.result(timeout=7200)  # 2-hour hard cap
                if accuracy is None:
                    progress["results"][alg] = {
                        "status": "no_data", "val_accuracy": None, "run_dir": str(run_dir),
                    }
                else:
                    progress["results"][alg] = {
                        "status": "success", "val_accuracy": accuracy, "run_dir": str(run_dir),
                    }
            except concurrent.futures.TimeoutError:
                progress["results"][alg] = {
                    "status": "failed", "val_accuracy": None, "run_dir": str(run_dir),
                    "reason": "Training timed out (>2 hours)",
                }
            except Exception as e:
                progress["results"][alg] = {
                    "status": "failed", "val_accuracy": None, "run_dir": str(run_dir),
                    "reason": str(e),
                }
            progress["done"] = progress["done"] + 1
    except Exception as e:
        progress["error"] = str(e)
    finally:
        import gc
        gc.collect()
        progress["complete"] = True
        progress["current_alg"] = None


def page_experimental_train() -> None:
    st.title("🧪 Experimental — Algorithm Sandbox")
    st.caption(
        "Train individual algorithms for a single symbol in an isolated sandbox. "
        "Compare validation accuracies, then promote the best candidate to production. "
        "Production models are never touched during experimental runs."
    )

    from stock_prediction.config import get_setting
    from stock_prediction.models.trainer import AVAILABLE_MODELS, list_experiments, promote_experiment
    import shutil

    save_dir = Path(get_setting("models", "save_dir", default="data/models"))

    ep = st.session_state.get("exp_progress")
    exp_active = ep is not None and not ep.get("complete", False)

    # ── Section 1: Setup form ─────────────────────────────────────────────────
    _section("Setup")
    symbol_input = st.text_input(
        "Stock Symbol",
        placeholder="e.g. INFY.NS",
        key="exp_sym",
        disabled=exp_active,
    )

    col1, col2 = st.columns(2)
    with col1:
        algorithms = st.multiselect(
            "Algorithms to test",
            options=AVAILABLE_MODELS,
            default=["lstm"],
            key="exp_algs",
            disabled=exp_active,
        )
    with col2:
        _HORIZON_OPTIONS = [1, 3, 5, 7, 10]
        horizon = st.selectbox(
            "Prediction horizon",
            options=_HORIZON_OPTIONS,
            index=_HORIZON_OPTIONS.index(5),
            key="exp_horizon",
            format_func=lambda x: f"{x} day{'s' if x > 1 else ''}",
            disabled=exp_active,
        )

    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("Start Date", value=None, key="exp_start", disabled=exp_active)
    with col4:
        end_date = st.date_input("End Date", value=None, key="exp_end", disabled=exp_active)

    col5, col6, col7 = st.columns(3)
    with col5:
        use_news = st.checkbox("News features", value=True, key="exp_news", disabled=exp_active)
    with col6:
        use_llm = st.checkbox("LLM features", value=True, key="exp_llm", disabled=exp_active)
    with col7:
        use_financials = st.checkbox("Financial features", value=True, key="exp_fin", disabled=exp_active)

    if not exp_active:
        if st.button("▶ Start Experimental Training", type="primary", key="exp_run"):
            sym = symbol_input.strip().upper()
            if not sym:
                st.warning("Please enter a stock symbol.")
                st.stop()
            if not algorithms:
                st.warning("Please select at least one algorithm.")
                st.stop()

            progress: dict = {
                "symbol":      sym,
                "algorithms":  algorithms,
                "total":       len(algorithms),
                "done":        0,
                "current_alg": None,
                "results":     {},
                "complete":    False,
                "cancelled":   False,
                "error":       None,
            }
            st.session_state["exp_progress"] = progress

            thread = threading.Thread(
                target=_run_experimental_bg,
                args=(
                    sym,
                    algorithms,
                    str(start_date) if start_date else None,
                    str(end_date) if end_date else None,
                    use_news,
                    use_llm,
                    use_financials,
                    horizon,
                    save_dir,
                    progress,
                ),
                daemon=True,
            )
            thread.start()
            st.rerun()

    # ── Section 2: Training progress ──────────────────────────────────────────
    if ep is not None:
        _section("Training Progress")
        done  = ep.get("done", 0)
        total = ep.get("total", 1)
        current = ep.get("current_alg")
        frac  = done / total if total else 0

        if exp_active:
            st.info("⏳ Experimental training running — safe to navigate away and return.")
            st.progress(frac)
            st.markdown(
                f"**{done} / {total}** complete"
                + (f" — training **{current}**" if current else "")
            )
            if ep.get("error"):
                st.error(f"Error: {ep['error']}")

        # Per-algorithm status rows
        algorithms_in_progress = ep.get("algorithms", [])
        results_so_far = ep.get("results", {})
        for alg in algorithms_in_progress:
            if alg in results_so_far:
                r = results_so_far[alg]
                if r["status"] == "success":
                    acc_str = f"{r['val_accuracy']:.1%}" if r.get("val_accuracy") is not None else "—"
                    st.markdown(f"- **{alg}** ✓  {acc_str}")
                elif r["status"] == "no_data":
                    st.markdown(f"- **{alg}** ⚠️  no data")
                else:
                    reason = r.get("reason", "")
                    st.markdown(f"- **{alg}** ✗  failed — {reason}")
            elif alg == current:
                st.markdown(f"- **{alg}** ⏳ training…")
            else:
                st.markdown(f"- **{alg}** —")

        col_cancel, _ = st.columns([1, 5])
        with col_cancel:
            if exp_active and st.button("⏹ Cancel", key="exp_cancel"):
                ep["cancelled"] = True

        if exp_active:
            time.sleep(2)
            st.rerun()

    # ── Section 3: Experimental results table ─────────────────────────────────
    sym_for_table = (ep.get("symbol") if ep else None) or symbol_input.strip().upper()
    if sym_for_table:
        experiments = list_experiments(sym_for_table, save_dir)
        if experiments:
            _section("Experimental Results")
            for exp in experiments:
                run_id   = exp["run_id"]
                run_dir  = exp["run_dir"]
                sel_models = exp.get("selected_models", [])
                alg_label  = ", ".join(sel_models) if sel_models else run_id.split("_", 2)[-1]
                acc        = exp.get("val_accuracy")
                acc_str    = f"{acc:.1%}" if acc is not None else "—"
                trained_at = exp.get("trained_at", "")[:16].replace("T", " ") if exp.get("trained_at") else "—"

                col_id, col_alg, col_acc, col_at, col_act = st.columns([3, 2, 1.5, 2, 2])
                col_id.code(run_id, language=None)
                col_alg.write(alg_label)
                col_acc.write(acc_str)
                col_at.write(trained_at)

                with col_act:
                    promote_key = f"exp_promote_{run_id}"
                    delete_key  = f"exp_delete_{run_id}"
                    confirm_key = f"exp_confirm_{run_id}"

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Promote", key=promote_key, use_container_width=True):
                            st.session_state[confirm_key] = True
                    with c2:
                        if st.button("🗑", key=delete_key, use_container_width=True):
                            import shutil as _shutil
                            _shutil.rmtree(run_dir, ignore_errors=True)
                            st.toast(f"Deleted {run_id}", icon="🗑")
                            st.rerun()

                if st.session_state.get(confirm_key):
                    st.warning(
                        f"Promote **{run_id}** to production? This overwrites existing production files for {sym_for_table}."
                    )
                    c_yes, c_no, _ = st.columns([1, 1, 5])
                    with c_yes:
                        if st.button("✅ Confirm", key=f"exp_confirm_yes_{run_id}"):
                            promote_experiment(sym_for_table, run_dir, save_dir)
                            st.session_state.pop(confirm_key, None)
                            st.toast(f"Promoted {run_id} to production!", icon="✅")
                            st.rerun()
                    with c_no:
                        if st.button("✗ Cancel", key=f"exp_confirm_no_{run_id}"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()

    # ── Section 4: Current production model ───────────────────────────────────
    sym_for_prod = (ep.get("symbol") if ep else None) or symbol_input.strip().upper()
    if sym_for_prod:
        with st.expander(f"Current Production Model for {sym_for_prod}", expanded=False):
            prod_meta_path = save_dir / sym_for_prod.replace(".", "_") / "meta.joblib"
            if prod_meta_path.exists():
                import joblib as _joblib
                prod_meta = _joblib.load(prod_meta_path)
                sel = prod_meta.get("selected_models") or []
                acc = prod_meta.get("val_accuracy")
                trained_at = (prod_meta.get("trained_at") or "")[:16].replace("T", " ")
                hor = prod_meta.get("horizon", "—")
                news_flag = "✓" if prod_meta.get("use_news") else "✗"
                llm_flag  = "✓" if prod_meta.get("use_llm")  else "✗"
                fin_flag  = "✓" if prod_meta.get("use_financials") else "✗"
                st.markdown(f"**Algorithms** : {', '.join(sel) if sel else '—'}")
                st.markdown(f"**Val Accuracy**: {f'{acc:.1%}' if acc is not None else '—'}")
                st.markdown(f"**Trained At** : {trained_at or '—'}")
                st.markdown(f"**Horizon**    : {hor} day{'s' if hor != 1 else ''}")
                st.markdown(f"**Features**   : News {news_flag}   LLM {llm_flag}   Financials {fin_flag}")
            else:
                st.info(f"No production model trained yet for {sym_for_prod}.")


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
    st.caption("Feature flags (News / LLM / Financials) and prediction horizon are loaded automatically from each symbol's trained model.")

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
                signal_metas: dict[str, dict] = {}
                warn_msgs: list[str] = []

                for symbol in symbol_list:
                    try:
                        sig, model_age, meta, _ = _predict_for_symbol(
                            symbol, trainer, signal_gen
                        )
                        if model_age and model_age > staleness:
                            warn_msgs.append(
                                f"{symbol}: model is {model_age} days old — consider retraining"
                            )
                        signals.append(sig)
                        signal_metas[symbol] = meta
                    except FileNotFoundError:
                        warn_msgs.append(f"{symbol}: no trained model — run Train first")
                    except Exception as ex:
                        warn_msgs.append(f"{symbol}: {ex}")

                st.session_state.predict_signals = signals
                st.session_state.predict_signal_metas = signal_metas
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
    signal_metas = st.session_state.get("predict_signal_metas") or {}
    # Header — 11 columns: …existing 10… + ℹ️
    _PR_WIDTHS = [1, 1, 1, 0.7, 0.7, 0.7, 0.6, 0.6, 2, 0.5, 0.5]
    _PR_HEADERS = ["Symbol", "Signal", "Confidence", "BUY %", "HOLD %", "SELL %",
                   "RSI", "Short?", "Weekly Outlook", "📊", "ℹ️"]
    pr_cols = st.columns(_PR_WIDTHS)
    for col, hdr in zip(pr_cols, _PR_HEADERS):
        col.markdown(f"**{hdr}**")
    from pathlib import Path as _PPath
    _plot_base = _PPath("data/plots")
    for sig in signals:
        sym_plots = _plot_base / sig.symbol.replace(".", "_")
        pred_path = sym_plots / f"{sig.symbol}_pred_plot.html"
        val_path  = sym_plots / f"{sig.symbol}_val_plot.html"
        tr_path   = sym_plots / f"{sig.symbol}_train_plot.html"
        has_plots = pred_path.exists()

        rcols = st.columns(_PR_WIDTHS)
        rcols[0].write(sig.symbol)
        sig_css = {"STRONG BUY": "#166534", "BUY": "#16a34a", "HOLD": "#d97706",
                   "SELL": "#dc2626", "STRONG SELL": "#7f1d1d"}.get(sig.signal, "#6b7280")
        rcols[1].markdown(f'<span style="color:{sig_css};font-weight:700">{sig.signal}</span>', unsafe_allow_html=True)
        rcols[2].write(f"{sig.confidence:.1%}")
        rcols[3].write(f"{sig.probabilities.get('BUY', 0):.0%}")
        rcols[4].write(f"{sig.probabilities.get('HOLD', 0):.0%}")
        rcols[5].write(f"{sig.probabilities.get('SELL', 0):.0%}")
        rcols[6].write(f"{sig.technical_summary.get('RSI', 0):.0f}" if sig.technical_summary.get("RSI") else "—")
        short_css = "color:#dc2626;font-weight:600" if sig.is_short_candidate else ""
        rcols[7].markdown(f'<span style="{short_css}">{"Yes" if sig.is_short_candidate else "No"}</span>', unsafe_allow_html=True)
        rcols[8].write(sig.weekly_outlook or "—")
        if has_plots:
            if rcols[9].button("📊", key=f"pr_plt_{sig.symbol}", help="View plots"):
                _show_plots_popup(sig.symbol, {
                    "train_plot": str(tr_path),
                    "val_plot":   str(val_path),
                    "pred_plot":  str(pred_path),
                })
        else:
            rcols[9].write("—")
        meta = signal_metas.get(sig.symbol, {})
        if rcols[10].button("ℹ️", key=f"pr_info_{sig.symbol}", help="Model info"):
            _show_model_info_popup(sig.symbol, meta)


# ─── Analyze ──────────────────────────────────────────────────────────────────
def page_analyze() -> None:
    st.title("🔬 Analyze — Deep Stock Analysis")
    st.caption(
        "Single-stock deep dive: model signal, LLM broker scores, and recent headlines."
    )

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        symbol = st.text_input("Symbol", placeholder="e.g. RELIANCE.NS", key="an_sym")
    with col2:
        use_news = st.checkbox("News", value=True, key="an_news")
    with col3:
        use_llm = st.checkbox("LLM", value=True, key="an_llm")
    with col4:
        use_fin_an = st.checkbox("Financials", value=True, key="an_fin")

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
                # _predict_for_symbol handles feature flags/horizon from meta
                _, model_age, meta_an, df = _predict_for_symbol(
                    sym, trainer, signal_gen
                )
                if model_age and model_age > staleness:
                    st.warning(f"Model for {sym} is {model_age} days old — consider retraining.")
                horizon_an = meta_an.get("horizon", 5)
                label_cols = {"return_1d", "return_5d", f"return_{horizon_an}d", "signal"}
                feature_names_an = meta_an.get("feature_names", [])
                tech = {
                    c: float(df[c].iloc[-1])
                    for c in ["RSI", "MACD_Histogram", "Price_SMA50_Ratio"]
                    if c in df.columns
                }
                import numpy as np
                ensemble, scaler, seq_scaler, _, __ = trainer.load_models(sym)
                seq_len = 60
                if feature_names_an:
                    avail = [c for c in feature_names_an if c in df.columns]
                    features = df[avail].values if avail else df[
                        [c for c in df.columns if c not in label_cols]
                    ].values
                else:
                    features = df[[c for c in df.columns if c not in label_cols]].values
                n_f = features.shape[1]
                lss = seq_scaler.transform(features[-seq_len:].reshape(-1, n_f)).reshape(1, seq_len, n_f)
                lts = scaler.transform(features[-1].reshape(1, -1))

                x_tab_lag_an = None
                if "xgboost_lag" in meta_an.get("selected_models", []):
                    _ls = meta_an.get("lag_scaler")
                    _lf = meta_an.get("lag_feature_names", [])
                    if _ls is not None and _lf:
                        from stock_prediction.features.technical import add_lag_trend_features
                        _ld = add_lag_trend_features(df)
                        _av = [c for c in _lf if c in _ld.columns]
                        if _av:
                            _feat_ld = _ld[_av].dropna()
                            if not _feat_ld.empty:
                                x_tab_lag_an = _ls.transform(_feat_ld.values[-1].reshape(1, -1)).astype(np.float32)

                pred = ensemble.predict_single(
                    lss.astype(np.float32), lts.astype(np.float32), X_tab_lag=x_tab_lag_an
                )
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
            _show_table(pd.DataFrame(broker_rows))

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
        _show_table(pd.DataFrame(result.top_picks))
    else:
        st.caption("No top picks found.")

    # Sector Leaders
    _section("🏭 Sector Leaders")
    if result.sector_leaders:
        for sector, stocks in result.sector_leaders.items():
            with st.expander(f"{sector}  ({len(stocks)} stocks)"):
                _show_table(pd.DataFrame(stocks))
    else:
        st.caption("No sector data.")

    # News Alerts
    if result.news_alerts:
        _section("📰 News Alerts (LLM Discovery)")
        _show_table(pd.DataFrame(result.news_alerts))

    # Full Rankings
    _section("📊 Full Rankings")
    if result.full_rankings:
        _show_table(pd.DataFrame(result.full_rankings))


# ─── Trade ────────────────────────────────────────────────────────────────────
def page_trade() -> None:
    st.title("💰 Trade — Paper Trading")
    st.caption("Simulate buy, sell, and short-sell trades with virtual money. No real capital at risk.")

    from stock_prediction.signals.paper_trading import list_portfolios as _list_portfolios
    portfolios = _list_portfolios()
    p_col1, p_col2 = st.columns([2, 2])
    with p_col1:
        portfolio_choice = st.selectbox(
            "Portfolio", options=portfolios + ["➕ Create new…"], key="td_portfolio"
        )
    with p_col2:
        new_port = ""
        if portfolio_choice == "➕ Create new…":
            new_port = st.text_input("New portfolio name", placeholder="e.g. swing_trades", key="td_new_port")
    portfolio_name = (new_port.strip() or "default") if portfolio_choice == "➕ Create new…" else portfolio_choice

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

    comment_input = st.text_input(
        "Comment (optional)",
        placeholder="e.g. based on RSI signal, earnings play …",
        key="td_comment",
    )

    if st.button("▶ Execute Trade", type="primary", key="td_run"):
        if not symbol.strip():
            st.warning("Please enter a symbol.")
            return

        sym = symbol.strip().upper()
        with st.spinner(f"Executing {action} for {sym} …"):
            try:
                from stock_prediction.signals.paper_trading import PaperTradingManager
                manager = PaperTradingManager(portfolio=portfolio_name)

                comment = comment_input.strip()

                if action == "Buy (Long)":
                    trade = manager.buy(sym, float(amount), comment=comment)
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
                    trade = manager.short_sell(sym, float(amount), comment=comment)
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

    from stock_prediction.signals.paper_trading import list_portfolios as _list_portfolios
    portfolios = _list_portfolios()
    portfolio_name = st.selectbox("Portfolio", options=portfolios, key="pf_portfolio")

    # Clear stale cache when portfolio changes
    if st.session_state.get("_pf_last_portfolio") != portfolio_name:
        st.session_state["portfolio_trades"] = None
        st.session_state["_pf_last_portfolio"] = portfolio_name

    if st.button("🔄 Refresh Portfolio", type="primary", key="pf_refresh"):
        with st.spinner("Fetching current prices …"):
            try:
                from stock_prediction.signals.paper_trading import PaperTradingManager
                st.session_state.portfolio_trades = PaperTradingManager(portfolio=portfolio_name).get_portfolio()
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
            "Comment":          getattr(t, "comment", "") or "—",
        }
        for t in trades
    ]
    _show_table(
        pd.DataFrame(rows),
        style_map={"Unrealized PnL ₹": _color_pnl, "PnL %": _color_pnl},
        tooltip_cols={"Comment"},
    )

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

    from stock_prediction.signals.paper_trading import list_portfolios as _list_portfolios
    portfolios = _list_portfolios()
    portfolio_name = st.selectbox("Portfolio", options=portfolios, key="gr_portfolio")

    # Clear stale cache when portfolio changes
    if st.session_state.get("_gr_last_portfolio") != portfolio_name:
        st.session_state["gain_report"] = None
        st.session_state["_gr_last_portfolio"] = portfolio_name

    if st.button("📊 Calculate Gains", type="primary", key="gr_run"):
        with st.spinner("Calculating …"):
            try:
                from stock_prediction.signals.paper_trading import PaperTradingManager
                st.session_state.gain_report = PaperTradingManager(portfolio=portfolio_name).calculate_gains()
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
        _show_table(
            pd.DataFrame(rows),
            style_map={"Total PnL ₹": _color_pnl, "PnL %": _color_pnl},
        )


# ─── Settings ─────────────────────────────────────────────────────────────────
def page_settings() -> None:
    from stock_prediction.config import get_setting, set_ui_overrides as _apply

    st.title("⚙️ Session Settings")
    st.info(
        "Settings here apply to this browser session only. "
        "Restart the app to reset. "
        "Edit `config/settings.yaml` for permanent changes."
    )

    # ── Signal Thresholds ────────────────────────────────────────────────────
    _section("Signal Thresholds")
    st.markdown(
        "These thresholds determine the minimum predicted return required to "
        "generate a **BUY** or **SELL** signal for each forecast horizon. "
        "Values are stored as decimals (e.g. 2.2% → 0.022)."
    )

    # Load current effective thresholds (session override or YAML default)
    current_thresholds: dict = get_setting("signals", "horizon_thresholds") or {}

    # Build editable inputs for horizons 1–10
    horizons = list(range(1, 11))
    buy_vals: dict[int, float] = {}
    sell_vals: dict[int, float] = {}

    header_cols = st.columns([1.5, 2, 2])
    header_cols[0].markdown("**Horizon (days)**")
    header_cols[1].markdown("**Buy threshold (%)**")
    header_cols[2].markdown("**Sell threshold (%, shown as positive)**")

    for h in horizons:
        defaults = current_thresholds.get(h, [0.0, 0.0])
        buy_default = float(defaults[0]) * 100 if defaults else 0.0
        sell_default = abs(float(defaults[1])) * 100 if defaults else 0.0

        row = st.columns([1.5, 2, 2])
        row[0].markdown(f"**{h}d**")
        buy_vals[h] = row[1].number_input(
            label=f"Buy {h}d",
            value=round(buy_default, 2),
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            format="%.2f",
            label_visibility="collapsed",
            key=f"settings_buy_{h}",
        )
        sell_vals[h] = row[2].number_input(
            label=f"Sell {h}d",
            value=round(sell_default, 2),
            min_value=0.0,
            max_value=10.0,
            step=0.1,
            format="%.2f",
            label_visibility="collapsed",
            key=f"settings_sell_{h}",
        )

    col_apply, col_reset = st.columns([1, 1])

    with col_apply:
        if st.button("Apply to session", type="primary", key="settings_apply"):
            new_thresholds = {
                h: [round(buy_vals[h] / 100, 6), -round(sell_vals[h] / 100, 6)]
                for h in horizons
            }
            cache = dict(st.session_state.get("settings_cache", {}))
            cache.setdefault("signals", {})["horizon_thresholds"] = new_thresholds
            st.session_state["settings_cache"] = cache
            _apply(cache)
            st.success("Thresholds applied for this session.")
            st.rerun()

    with col_reset:
        if st.button("Reset to config defaults", key="settings_reset"):
            cache = dict(st.session_state.get("settings_cache", {}))
            cache.get("signals", {}).pop("horizon_thresholds", None)
            if not cache.get("signals"):
                cache.pop("signals", None)
            st.session_state["settings_cache"] = cache
            _apply(cache)
            st.success("Reset to settings.yaml defaults.")
            st.rerun()


# ─── Model Catalogue ──────────────────────────────────────────────────────────
def page_catalogue() -> None:
    from pathlib import Path
    from stock_prediction.config import get_setting
    from stock_prediction.models.trainer import list_trained_models

    st.title("📚 Model Catalogue")
    st.caption("Browse all trained models. Select a symbol to see full details.")

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
        st.warning(f"No trained models found in `{save_dir}`. Run **Train** first.")
        return

    # ── Symbol search / filter ───────────────────────────────────────────────
    search = st.text_input("🔍 Filter by symbol", placeholder="e.g. INFY or leave blank for all")
    filtered = [m for m in all_models
                if not search or search.upper() in m["symbol"].upper()]

    if not filtered:
        st.info(f"No models match '{search}'.")
        return

    # ── Summary table ────────────────────────────────────────────────────────
    st.subheader(f"All Trained Models ({len(filtered)} of {len(all_models)})")

    def _ck(v: bool) -> str:
        return "✓" if v else "✗"

    rows = [
        {
            "Symbol":     m["symbol"],
            "Algorithms": ", ".join(m.get("selected_models") or []) or "—",
            "Val Acc":    f"{m['val_accuracy']:.1%}" if m.get("val_accuracy") is not None else "—",
            "Horizon":    f"{m.get('horizon', '—')}d",
            "Trained At": (m.get("trained_at") or "")[:10] or "—",
            "Age":        f"{m['model_age_days']}d" if m.get("model_age_days") is not None else "—",
            "News":       _ck(m.get("use_news", False)),
            "LLM":        _ck(m.get("use_llm", False)),
            "Fin":        _ck(m.get("use_financials", False)),
            "Files":      str(len(m.get("model_files", {}))),
        }
        for m in filtered
    ]

    def _color_ck(val: str) -> str:
        if val == "✓": return "color:#16a34a; font-weight:700"
        if val == "✗": return "color:#dc2626"
        return ""

    def _color_age(val: str) -> str:
        try:
            d = int(val.replace("d", ""))
            if d > 30: return "color:#d97706; font-weight:600"
            if d > 7:  return "color:#ca8a04"
        except Exception:
            pass
        return ""

    _show_table(
        pd.DataFrame(rows),
        style_map={
            "News": _color_ck,
            "LLM":  _color_ck,
            "Fin":  _color_ck,
            "Age":  _color_age,
        },
    )

    st.divider()

    # ── Detail panel ─────────────────────────────────────────────────────────
    st.subheader("Symbol Detail")
    symbol_options = [m["symbol"] for m in filtered]
    selected = st.selectbox("Select a symbol to inspect", options=symbol_options, key="cat_symbol")

    entry = next((m for m in filtered if m["symbol"] == selected), None)
    if entry is None:
        return

    sel      = entry.get("selected_models") or []
    acc      = entry.get("val_accuracy")
    trained  = (entry.get("trained_at") or "")[:19].replace("T", " ")
    horizon  = entry.get("horizon", "—")
    age      = entry.get("model_age_days")
    use_news = entry.get("use_news", False)
    use_llm  = entry.get("use_llm", False)
    use_fin  = entry.get("use_financials", False)
    in_sz    = entry.get("input_size", 0)
    lag_sz   = entry.get("lag_input_size", 0)
    feat_n   = len(entry.get("feature_names") or [])
    lag_feat = len(entry.get("lag_feature_names") or [])
    files    = entry.get("model_files", {})

    # ── Key metrics row ───────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Val Accuracy", f"{acc:.1%}" if acc is not None else "—")
    c2.metric("Horizon", f"{horizon}d")
    c3.metric("Trained At", trained[:10] if trained else "—")
    c4.metric("Model Age", f"{age} days" if age is not None else "—",
              delta="stale" if (age or 0) > 30 else None,
              delta_color="inverse")
    c5.metric("Model Files", len(files))

    # ── Feature flags ─────────────────────────────────────────────────────────
    st.markdown("**Feature Flags**")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("News Sentiment", "✓ Enabled" if use_news else "✗ Disabled")
    fc2.metric("LLM Broker Scores", "✓ Enabled" if use_llm else "✗ Disabled")
    fc3.metric("Quarterly Financials", "✓ Enabled" if use_fin else "✗ Disabled")

    # ── Input dimensions ──────────────────────────────────────────────────────
    st.markdown("**Input Dimensions**")
    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Standard Input Size", in_sz or "—")
    dc2.metric("Lag Input Size", lag_sz or "—")
    dc3.metric("Standard Features", feat_n or "—")
    dc4.metric("Lag Features", lag_feat or "—")

    # ── Ensemble weights ──────────────────────────────────────────────────────
    if sel:
        st.markdown("**Ensemble Weights**")
        wt_rows = []
        for m in sel:
            w = entry.get(WEIGHT_KEYS.get(m, ""), None)
            wt_rows.append({
                "Model":  m,
                "Weight": f"{w:.4f}" if w is not None else "—",
                "Bar":    f"{(w or 0)*100:.1f}%",
            })
        _show_table(pd.DataFrame(wt_rows))

    # ── Model files ───────────────────────────────────────────────────────────
    if files:
        st.markdown("**Model Files on Disk**")
        file_rows = [
            {"File": fname, "Size": f"{sz/1024:.1f} KB"}
            for fname, sz in sorted(files.items())
        ]
        _show_table(pd.DataFrame(file_rows))

    # ── Model directory path ──────────────────────────────────────────────────
    st.caption(f"📁 Model directory: `{entry['model_dir']}`")


# ─── Router ───────────────────────────────────────────────────────────────────
PAGE_MAP = {
    "📊 Suggest (deprecated)": page_suggest,
    "📋 Shortlist":  page_shortlist,
    "🔍 Lookup":     page_lookup,
    "📥 Fetch Data": page_fetch_data,
    "🧠 Train":      page_train,
    "🧪 Experimental": page_experimental_train,
    "🔮 Predict":    page_predict,
    "🔬 Analyze":    page_analyze,
    "📡 Screen":     page_screen,
    "📚 Model Catalogue": page_catalogue,
    "💰 Trade":      page_trade,
    "💼 Portfolio":  page_portfolio,
    "📈 Gain Report": page_gain_report,
    "⚙️ Settings":   page_settings,
}

PAGE_MAP[page]()
