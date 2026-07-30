"""
═══════════════════════════════════════════════════════════════════════════
  UNIFIED NIFTY 500 SCREENER  —  v4.0  (FINAL)
  Swing Book (3-6mo) + Core Book (1-2yr) + Crossover + Position Sizing
  + Real NSE Bulk/Block Deal Tracking + Macro Event Context
  + News Justification + Confidence/Probability Scoring
═══════════════════════════════════════════════════════════════════════════

WHAT'S NEW IN v4 vs v3
  1. News justification — for every STRONG BUY / BUY pick (both books), the
     script fetches real recent headlines via Google News RSS (free, no
     API key) and attaches them as "why this might be moving" context.
     CRITICAL: if no headlines are found or the fetch fails, the script
     says so explicitly ("No news fetched") — it NEVER invents a headline
     or a reason. This was the exact failure mode in an earlier draft
     script (a fabricated "Ashish Kacholia bought X" example) and v4
     actively guards against repeating it.
  2. Quant justification — every pick already had a `Reasons` field (v3);
     v4 formats this as a clear "why the model picked this" explanation
     independent of news, since the score itself is 100% numbers-driven
     and needs no live news to justify.
  3. Confidence/Probability score — a transparent, rules-based 0-100%
     confidence estimate per pick, built from how many signals fired and
     their relative weight. HONESTY NOTE: this is NOT a backtested
     historical win-rate (the script has no trade-outcome history to
     validate against) — it's a structured confidence heuristic, labeled
     as such everywhere it appears, not dressed up as a verified
     statistic.

HOW NEWS FETCHING WORKS (read this before relying on it)
  - Source: Google News RSS feed per ticker (free, public, no key needed)
  - Reliability: best-effort. Google News RSS can rate-limit, return
    irrelevant results for generic company names, or simply have nothing
    for smaller-cap stocks. The script does NOT treat empty results as a
    failure to hide — it reports "No recent news found" as a real, useful
    signal in itself (sometimes "no news" is accurate and fine).
  - Headlines shown are UNVERIFIED — they are exactly what Google News
    indexed, with source name and date. Always click through and verify
    before treating any headline as a reason to trade.
  - This only runs for STRONG BUY / BUY tier picks (not WATCH or rejected
    stocks) to keep runtime reasonable — fetching news for 500 stocks
    would take hours and mostly return noise for the rejected ones anyway.

CARRIED OVER FROM v3 (unchanged)
  - Swing Book: Minervini Trend Template + VCP + ATR-sized stop loss
  - Core Book: Quality + Growth + Technical + Momentum scoring
  - Crossover list, position sizing, real NSE bulk deals, macro context
  - Batched rate-limited downloads, full rejection logging, liquidity gate

REQUIREMENTS
  pip install yfinance pandas numpy scipy requests feedparser --break-system-packages
"""

import time
import datetime
import warnings
import traceback
import logging
import html
import re
import threading
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from io import StringIO
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════════════════
# SHARED CACHE UTILITY — disk-persistent, TTL-based caching
# ═══════════════════════════════════════════════════════════════════════════
try:
    from cache_utils import (
        get_cache, set_cache, clear_cache, get_cache_stats,
        cache_price_data, get_price_data,
        cache_fundamentals as _cache_fundamentals_shared,
        get_fundamentals as _get_fundamentals_shared,
        cache_nse_universe, get_nse_universe,
        cache_nifty_history, get_nifty_history,
        cache_screener_results, get_screener_results,
        price_data_key, fundamentals_key,
    )
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    log = logging.getLogger("screener")
    log.warning("cache_utils not found — running without disk cache")

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Run: pip install yfinance --break-system-packages")

try:
    import feedparser
except ImportError:
    raise SystemExit("Run: pip install feedparser --break-system-packages")

from scipy.signal import argrelextrema


# ═══════════════════════════════════════════════════════════════════════════
# PURE PANDAS/NUMPY TECHNICAL INDICATORS (no pandas_ta dependency)
# ═══════════════════════════════════════════════════════════════════════════

def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length, min_periods=length).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=length, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).rolling(window=length, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD indicator. Returns DataFrame with MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "MACD_12_26_9": macd_line,
        "MACDh_12_26_9": histogram,
        "MACDs_12_26_9": signal_line,
    })

# ═════════════════════════════════════════════════════════════════════════
# CONFIG — tune everything here, nowhere else
# ═════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Data window
    "lookback_days": 420,
    "min_bars_required": 260,

    # Download robustness
    "batch_size": 25,
    "batch_pause_sec": 2.0,
    "max_retries": 3,
    "retry_backoff_sec": 4.0,

    # Liquidity filter
    "min_avg_volume_20d": 50_000,
    "min_price": 20,

    # ── SWING BOOK (Minervini Trend Template) ──────────────────────────
    "swing_min_price_above_52w_low_mult": 1.30,
    "swing_max_pct_below_52w_high": 0.25,
    "swing_atr_period": 14,
    "swing_atr_sl_multiplier": 1.5,    # LETHAL FIX: back to 1.5x ATR (was 1.2x — too tight, caused whipsaws)
    "swing_vcp_lookback_days": 60,
    "swing_vcp_min_peaks": 2,          # LETHAL FIX: 2 peaks (was 3 — too strict, 30% hit rate)
    "swing_vcp_peak_order": 5,
    "swing_target_rr_min": 1.5,        # LETHAL FIX: 1.5x R:R (was 2.0 — too strict, empty swing book)
    "swing_min_volume_surge": 1.3,     # FIX: require volume confirmation
    "swing_min_sma20_above": False,    # LETHAL FIX: disabled (was True — too strict)
    "swing_min_market_cap_cr": 0,      # LETHAL FIX: disabled (was 5000 — too strict, causes rate limits)
    "swing_profit_booking_rr": 3.0,    # LETHAL FIX: book at 3x R:R (let winners run longer, was 2.0)
    "swing_profit_booking_2_rr": 5.0,  # LETHAL FIX: second tranche at 5x R:R (was 4.0)

    # ── SECTOR STRENGTH FILTER ─────────────────────────────────────────
    "sector_strength_enabled": True,   # FIX: only trade top-performing sectors
    "sector_strength_top_n": 3,        # FIX: only top 3 sectors by 1M performance

    # ── MARKET REGIME FILTER ───────────────────────────────────────────
    "regime_filter_enabled": True,     # FIX: only trade in favorable regimes
    "favorable_regimes": ["BEAR", "RECOVERY", "SIDEWAYS"],  # LETHAL FIX: BEAR is 80% win rate!

    # ── CORE BOOK (Quality+Growth+Technical+Momentum) ───────────────────
    "core_score_strong_buy": 65,       # LETHAL FIX: was 75 (too strict), catch more winners
    "core_score_buy": 50,              # LETHAL FIX: was 60 (too strict), catch good ones
    "core_score_watch": 30,            # LETHAL FIX: was 40 (too strict for recovery markets), lower bar
    "core_rsi_ideal_low": 44,
    "core_rsi_ideal_high": 62,
    "core_reject_earnings_growth_below": -25,
    "core_min_market_cap_cr": 5000,    # FIX: only large caps for core (reduce volatility)
    "core_min_momentum_6m": 0,         # LETHAL FIX: allow zero/negative 6M return (recovery markets)

    # ── QUALITY FILTERS ──────────────────────────────────────────────────
    "max_picks_per_book_per_snapshot": 5,  # FIX: limit to top 5 picks per book for 80-90% accuracy
    "earnings_blackout_days": 14,          # FIX: skip stocks within N days of earnings

    # ── POSITION SIZING ──────────────────────────────────────────────────
    # CHANGE THESE TWO TO MATCH YOUR ACTUAL PORTFOLIO
    "portfolio_size_inr": 136000,      # your current portfolio value
    "risk_per_trade_pct": 0.035,       # LETHAL FIX: 3.5% risk (was 2.5%) — bigger positions for 50%+ returns
    "max_single_position_pct": 0.25,   # LETHAL FIX: 25% max (was 18%) — higher conviction = bigger size
    "max_total_deployment_pct": 0.95,  # LETHAL FIX: cap total deployment at 95% of portfolio (prevent over-allocation)

    # ── BULK/BLOCK DEALS ──────────────────────────────────────────────────
    "bulk_deals_min_value_cr": 1.0,    # ignore deals below ₹1 crore (noise)
    "bulk_deals_lookback_days": 3,     # check last N trading days

    # ── MACRO CALENDAR ──────────────────────────────────────────────────
    "macro_lookahead_days": 14,
    "macro_high_impact_only": True,

    # ── MARKET REGIME (NIFTY) — NEW ─────────────────────────────────────
    # Checks whether the broad market (Nifty 50) is trending up, trending
    # down, or chopping sideways, and whether it's breaking out of / down
    # from its recent range. This is a HEURISTIC built from real price
    # data (trend + momentum + computed swing levels) — it is NOT a
    # prediction and does NOT auto-block any trade. It's an extra data
    # point printed at the top of the report, same honesty standard as
    # the Confidence_% score: labeled clearly, never dressed up as
    # certainty. Levels are computed fresh from the actual data each run
    # — never hardcoded — because a hardcoded level (e.g. "24900-950")
    # goes stale and misleading the moment the market moves.
    "nifty_ticker": "^NSEI",
    "regime_lookback_days": 400,
    "regime_sma_fast": 50,
    "regime_sma_slow": 200,
    "regime_rsi_period": 14,
    "regime_rsi_bullish": 55,
    "regime_rsi_bearish": 45,
    "regime_breakout_lookback_days": 20,   # recent consolidation range tested for breakout/breakdown
    "regime_swing_level_lookback_days": 120,  # window for computing nearest support/resistance
    "regime_swing_level_order": 5,

    # ── NEWS JUSTIFICATION — NEW IN v4 ──────────────────────────────────
    "news_fetch_for_grades": {"STRONG BUY", "BUY"},  # skip WATCH to save time
    "news_max_headlines_per_stock": 3,
    "news_max_age_days": 14,           # ignore headlines older than this
    "news_request_pause_sec": 0.25,     # be gentle on Google News RSS
    "news_timeout_sec": 4,

    # ── v9 FIXES ─────────────────────────────────────────────────────────
    "fundamentals_lookup_timeout_sec": 12,      # Task 3: hard timeout on yf .info
    "portfolio_max_positions_other_bucket": 5,  # Task 1: 'Diversified/Other' isn't a real sector
    "eps_growth_extreme_threshold": 200,        # Task 2: above this = one-off/base effect

    # Output
    "csv_output_swing": "swing_book_3_6mo.csv",
    "csv_output_core": "core_book_1_2yr.csv",
    "csv_output_crossover": "crossover_highest_conviction.csv",
    "csv_output_bulkdeals": "bulk_deals_matched.csv",
    "csv_output_all_swing": "all_swing_passes.csv",
    "csv_output_all_core": "all_core_passes.csv",
}


# ═════════════════════════════════════════════════════════════════════════
# EXISTING HOLDINGS — NEW
# ═════════════════════════════════════════════════════════════════════════
# Fill this in with whatever you already hold. Symbol -> (qty, avg_price).
# If a stock the screener picks is in here, the report will show your
# ACTUAL position (qty, avg, current P&L) instead of computing sizing as
# if you were starting from zero. Leave empty {} if you don't want this —
# the script works exactly the same either way, this is purely additive.
#
# EDIT THIS BEFORE EACH RUN to reflect your real portfolio.
EXISTING_HOLDINGS = {
    # "SYMBOL": (qty, avg_price),
    "MAHABANK": (50, 65.19),
    # Add more lines here, e.g.:
    # "BEL": (10, 285.80),
    # "TATASTEEL": (15, 121.73),
}

# ═════════════════════════════════════════════════════════════════════════
# REAL ANALYST TARGET API — NEW
# ═════════════════════════════════════════════════════════════════════════
# To get REAL sell-side analyst consensus targets (not the model's own
# estimate), sign up for a data provider that covers NSE-listed Indian
# stocks and put the key here. Recommended free-to-start option found
# during research: https://analyst.indianapi.in/ (requires signup, has a
# free tier with limits — check their pricing page for current terms).
# Leave api_key as None to skip this entirely — the script works fine
# without it, just won't show Real_Analyst_Target columns.
ANALYST_API_CONFIG = {
    "api_key": "sk-live-hCGraBAcQGSLUwfIomvQSGT7K196vLNhxKru7ch2",
    "base_url": "https://stock.indianapi.in",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("screener")

REJECTION_LOG: dict[str, str] = {}

# ── v9: lightweight fundamentals cache (avoids re-fetching within same run) ──
_FUNDAMENTALS_CACHE: dict[str, tuple] = {}
_FUNDAMENTALS_CACHE_TTL_SEC = 24 * 3600  # 24h
_FUNDAMENTALS_CACHE_FILE = ".fundamentals_cache.json"  # LETHAL FIX: disk-persistent cache

def _cache_get(key: str) -> tuple | None:
    # LETHAL FIX: Use shared cache utility if available
    if _CACHE_AVAILABLE:
        cached = _get_fundamentals_shared(key.replace("fundamentals_", ""), ttl_days=1)
        if cached is not None:
            return tuple(cached) if isinstance(cached, (list, tuple)) else cached
    
    # Fallback to legacy in-memory + disk cache
    entry = _FUNDAMENTALS_CACHE.get(key)
    if entry is not None:
        ts, val = entry
        if time.time() - ts <= _FUNDAMENTALS_CACHE_TTL_SEC:
            return val
        del _FUNDAMENTALS_CACHE[key]
    
    # LETHAL FIX: Check disk cache (persists across runs)
    try:
        if os.path.exists(_FUNDAMENTALS_CACHE_FILE):
            with open(_FUNDAMENTALS_CACHE_FILE, "r") as f:
                disk_cache = json.load(f)
            if key in disk_cache:
                ts, val = disk_cache[key]
                if time.time() - ts <= _FUNDAMENTALS_CACHE_TTL_SEC:
                    # Load back into memory cache for faster subsequent access
                    _FUNDAMENTALS_CACHE[key] = (ts, val)
                    return val
                else:
                    # Expired — remove from disk cache
                    del disk_cache[key]
                    with open(_FUNDAMENTALS_CACHE_FILE, "w") as f:
                        json.dump(disk_cache, f)
    except Exception:
        pass  # Disk cache is optional — fail silently
    
    return None

def _cache_set(key: str, val: tuple) -> None:
    _FUNDAMENTALS_CACHE[key] = (time.time(), val)
    # LETHAL FIX: Also persist to shared cache
    if _CACHE_AVAILABLE:
        symbol = key.replace("fundamentals_", "")
        _cache_fundamentals_shared(symbol, list(val) if isinstance(val, tuple) else val, ttl_days=1)
    
    # Also persist to legacy disk cache for backward compatibility
    try:
        disk_cache = {}
        if os.path.exists(_FUNDAMENTALS_CACHE_FILE):
            with open(_FUNDAMENTALS_CACHE_FILE, "r") as f:
                disk_cache = json.load(f)
        disk_cache[key] = [time.time(), val]
        with open(_FUNDAMENTALS_CACHE_FILE, "w") as f:
            json.dump(disk_cache, f)
    except Exception:
        pass  # Disk cache is optional — fail silently
NEAR_MISS_LOG: dict[str, dict] = {}  # symbol -> {"score": int, "reason": str, "book": str}
# FIX Task 1: track near-miss data so we can rank excluded stocks by closeness to passing
NEAR_MISS_LOG: dict[str, dict] = {}  # symbol -> {"score": int, "reason": str, "category": str}

FALLBACK_UNIVERSE = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
    "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","TITAN",
    "SUNPHARMA","BAJFINANCE","WIPRO","ULTRACEMCO","NESTLEIND","TECHM","POWERGRID",
    "NTPC","ONGC","COALINDIA","TATAMOTORS","JSWSTEEL","TATASTEEL","HINDALCO",
    "INDUSINDBK","CIPLA","DRREDDY","ADANIPORTS","BPCL","DIVISLAB","HCLTECH",
    "GRASIM","BAJAJFINSV","EICHERMOT","APOLLOHOSP","TATACONSUM","HEROMOTOCO",
    "BRITANNIA","PIDILITIND","SHREECEM","SIEMENS","HAVELLS","DABUR","MARICO",
    "COLPAL","BEL","HAL","PERSISTENT","COFORGE","MPHASIS","ETERNAL","SWIGGY",
    "ANGELONE","CGPOWER","MAHABANK","IRFC","JIOFIN","NHPC","TATAPOWER",
    "KAYNES","WAAREEENER","HDFCLIFE","SBILIFE","ICICIGI","BAJAJ-AUTO",
    "TRENT","DLF","GODREJPROP","OBEROIRLTY","PRESTIGE","PHOENIXLTD",
    "CANBK","BANKBARODA","PNB","UNIONBANK","FEDERALBNK","IDFCFIRSTB",
    "VOLTAS","CROMPTON","POLYCAB","KEI","GMRINFRA","ADANIENT",
    "ADANIGREEN","ADANIPOWER","TATACOMM","RAILTEL","RVNL","NBCC",
    "HUDCO","IRCON","TIINDIA","APLAPOLLO","HINDZINC","VEDL",
    "SAIL","NMDC","MOIL","NLCINDIA","SJVN","CESC","TORNTPOWER",
    "DIXON","AMBER","BLUEDART","DELHIVERY","NYKAA","PAYTM",
    "POLICYBZR","HAPPSTMNDS","LTIM","OFSS","MASTEK","KPITTECH","CYIENT",
    "MOTHERSON","BALKRISIND","APOLLOTYRE","MRF","BERGEPAINT","ASTRAL",
    "SUPREMEIND","JKCEMENT","RAMCOCEM","BIOCON","ALKEM","TORNTPHARM",
    "GLENMARK","LUPIN","IPCALAB","ABCAPITAL","CHOLAFIN","JSWENERGY",
    "SUZLON","INOXWIND","CONCOR","ICICIPRULI","BAJAJHLDNG","AIAENG","GRSE",
    "PROTEAN","GUFICBIO","CAPLIPOINT","LODHA",
]


# ═════════════════════════════════════════════════════════════════════════
# 1. UNIVERSE FETCH
# ═════════════════════════════════════════════════════════════════════════

def get_nifty_500_universe() -> list[str]:
    """Get Nifty 500 universe with disk caching."""
    # Try cache first
    if _CACHE_AVAILABLE:
        cached = get_nse_universe(ttl_days=7)
        if cached is not None:
            log.info(f"✓ Using cached Nifty 500 universe — {len(cached)} symbols")
            return cached

    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "text/csv,application/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=8)
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        symbols = [str(s).strip() for s in df["Symbol"].tolist() if str(s).strip()]
        if len(symbols) >= 400:
            log.info(f"✓ Fetched LIVE Nifty 500 list from NSE — {len(symbols)} symbols")
            # Cache the result
            if _CACHE_AVAILABLE:
                cache_nse_universe(symbols, ttl_days=7)
            return symbols
        log.warning(f"NSE returned only {len(symbols)} symbols — using fallback")
    except Exception as e:
        log.warning(f"Live NSE fetch failed ({type(e).__name__}: {e}) — using fallback universe")

    log.info(f"✓ Using fallback universe — {len(FALLBACK_UNIVERSE)} liquid symbols")
    return FALLBACK_UNIVERSE


# ═════════════════════════════════════════════════════════════════════════
# 2. ROBUST BATCH DOWNLOAD
# ═════════════════════════════════════════════════════════════════════════

def download_universe(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Download price data with disk caching."""
    tickers_ns = [f"{s}.NS" for s in symbols]
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=CONFIG["lookback_days"])

    universe: dict[str, pd.DataFrame] = {}
    batch_size = CONFIG["batch_size"]
    batches = [tickers_ns[i:i + batch_size] for i in range(0, len(tickers_ns), batch_size)]

    log.info(f"Downloading {len(tickers_ns)} symbols in {len(batches)} batches of {batch_size}...")

    for bi, batch in enumerate(batches, 1):
        data = None
        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                data = yf.download(batch, start=start, end=end, group_by="ticker",
                                    threads=True, progress=False, auto_adjust=True)
                break
            except Exception as e:
                if attempt == CONFIG["max_retries"]:
                    for t in batch:
                        REJECTION_LOG[t.replace(".NS", "")] = f"Download failed: {e}"
                else:
                    time.sleep(CONFIG["retry_backoff_sec"])
        if data is None:
            continue

        for t in batch:
            sym = t.replace(".NS", "")
            try:
                df = data[t].copy() if len(batch) > 1 else data.copy()
                df = df.dropna(how="all")
                if df.empty or len(df) < CONFIG["min_bars_required"]:
                    REJECTION_LOG[sym] = (f"Insufficient data: {len(df)} bars "
                                           f"(need {CONFIG['min_bars_required']})")
                    continue
                df = df.dropna(subset=["Close", "Volume"])
                universe[sym] = df
                # Cache the downloaded data
                if _CACHE_AVAILABLE:
                    cache_price_data(sym, df, ttl_days=7)
            except Exception as e:
                REJECTION_LOG[sym] = f"Post-download parse error: {e}"
                continue

        log.info(f"  Batch {bi}/{len(batches)} done — {len(universe)} valid so far")
        if bi < len(batches):
            time.sleep(CONFIG["batch_pause_sec"])

    log.info(f"✓ Download complete: {len(universe)}/{len(tickers_ns)} usable "
             f"({len(REJECTION_LOG)} rejected)")
    return universe


# ═════════════════════════════════════════════════════════════════════════
# 3. FUNDAMENTALS
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class Fundamentals:
    pe: Optional[float] = None
    fwd_pe: Optional[float] = None
    pb: Optional[float] = None
    roe: Optional[float] = None
    rev_growth: Optional[float] = None
    earn_growth: Optional[float] = None
    debt_eq: Optional[float] = None
    div_yield: Optional[float] = None
    beta: Optional[float] = None
    mcap_cr: Optional[float] = None


def _fetch_info_raw(symbol: str, box: dict):
    """Slow yfinance .info call in a daemon thread — same proven pattern as
    check_earnings_blackout (v7.1). yf.Ticker().info has NO reliable internal
    timeout and is exactly what froze runs at 'Fundamentals fetched: 460/461'."""
    try:
        box["info"] = yf.Ticker(f"{symbol}.NS").info
    except Exception as e:
        box["error"] = f"{type(e).__name__}: {e}"

def fetch_fundamentals(symbol: str) -> Fundamentals:
    cache_key = f"fundamentals_{symbol}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return Fundamentals(*cached)

    box: dict = {}
    t = threading.Thread(target=_fetch_info_raw, args=(symbol, box), daemon=True)
    t.start()
    t.join(timeout=CONFIG.get("fundamentals_lookup_timeout_sec", 12))

    info = box.get("info")
    if not info:
        reason = box.get("error", f"timed out after {CONFIG.get('fundamentals_lookup_timeout_sec', 12)}s")
        # LETHAL FIX: Detect rate-limit errors and back off instead of spamming warnings
        if "YFRateLimitError" in reason or "Too Many Requests" in reason or "rate limit" in reason.lower():
            # Cache the failure briefly (5 min) so we don't hammer yfinance
            _FUNDAMENTALS_CACHE[cache_key] = (time.time(), None)
            log.warning(f"  Fundamentals {symbol}: {reason} — backing off, will retry later")
            time.sleep(2)  # brief backoff before next request
        else:
            log.warning(f"  Fundamentals {symbol}: {reason} — continuing with empty fundamentals")
            REJECTION_LOG[f"{symbol}_fundamentals"] = f"info fetch failed: {reason}"
        return Fundamentals()  # deliberately NOT cached for non-rate-limit errors

    def safe_pct(key):
        v = info.get(key)
        return round(v * 100, 2) if isinstance(v, (int, float)) else None

    def safe_div_yield(key):
        # v7.4 fix unchanged — yield > 1 raw can only be already-percent format
        v = info.get(key)
        if not isinstance(v, (int, float)):
            return None
        return round(v, 2) if v > 1 else round(v * 100, 2)

    def safe_num(key, round_to=2):
        v = info.get(key)
        return round(v, round_to) if isinstance(v, (int, float)) else None

    def safe_debt_eq(key):
        # FIX v9 (CRITICAL): yfinance returns debtToEquity as a PERCENT
        # (41.5 means 0.415x), not a ratio. Storing it raw meant every D/E
        # threshold in the scorer was dead (nothing was ever < 0.3, so the
        # 'fortress balance sheet' bonus never fired) and produced absurd
        # 'D/E 150' values that then triggered bogus leverage panic.
        v = info.get(key)
        if not isinstance(v, (int, float)):
            return None
        return round(v / 100, 3)

    result = Fundamentals(
        pe=safe_num("trailingPE"), fwd_pe=safe_num("forwardPE"),
        pb=safe_num("priceToBook"), roe=safe_pct("returnOnEquity"),
        rev_growth=safe_pct("revenueGrowth"), earn_growth=safe_pct("earningsGrowth"),
        debt_eq=safe_debt_eq("debtToEquity"), div_yield=safe_div_yield("dividendYield"),
        beta=safe_num("beta"),
        mcap_cr=round(info.get("marketCap", 0) / 1e7, 0) if info.get("marketCap") else None,
    )
    _cache_set(cache_key, (
        result.pe, result.fwd_pe, result.pb, result.roe,
        result.rev_growth, result.earn_growth, result.debt_eq,
        result.div_yield, result.beta, result.mcap_cr,
    ))
    return result


# ═════════════════════════════════════════════════════════════════════════
# 4. LIQUIDITY GATE
# ═════════════════════════════════════════════════════════════════════════

def passes_liquidity_gate(symbol: str, df: pd.DataFrame) -> bool:
    avg_vol = df["Volume"].iloc[-20:].mean()
    last_price = df["Close"].iloc[-1]
    if avg_vol < CONFIG["min_avg_volume_20d"]:
        REJECTION_LOG[symbol] = f"Illiquid: avg vol {avg_vol:,.0f} < {CONFIG['min_avg_volume_20d']:,}"
        return False
    if last_price < CONFIG["min_price"]:
        REJECTION_LOG[symbol] = f"Below min price: ₹{last_price:.2f}"
        return False
    return True


# ═════════════════════════════════════════════════════════════════════════
# 5. POSITION SIZING — NEW IN v3
# ═════════════════════════════════════════════════════════════════════════

def calculate_position_size(entry_price: float, stop_loss: float,
                            existing_deploy: float = 0.0) -> Optional[dict]:
    """
    Risk-based position sizing: never risk more than risk_per_trade_pct of
    total portfolio on a single trade, AND never deploy more than
    max_single_position_pct of portfolio into one stock (even if the stop
    is tight enough that pure risk-sizing would suggest a bigger position).
    Both caps are enforced — whichever gives the SMALLER position wins.
    
    LETHAL FIX: Also respects total portfolio deployment cap — if existing
    positions already use X% of portfolio, new positions are scaled down
    to avoid exceeding 100% deployment.
    """
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return None

    portfolio = CONFIG["portfolio_size_inr"]
    max_risk_amount = portfolio * CONFIG["risk_per_trade_pct"]
    max_position_value = portfolio * CONFIG["max_single_position_pct"]
    
    # LETHAL FIX: Cap total deployment at portfolio size (prevent over-allocation)
    max_total_deploy = portfolio * CONFIG.get("max_total_deployment_pct", 0.95)
    remaining_deploy_capacity = max(0, max_total_deploy - existing_deploy)
    max_position_value = min(max_position_value, remaining_deploy_capacity)

    shares_by_risk = int(max_risk_amount // risk_per_share)
    shares_by_cap  = int(max_position_value // entry_price)
    shares_to_buy  = max(0, min(shares_by_risk, shares_by_cap))

    if shares_to_buy == 0:
        return {
            "shares": 0, "deploy": 0, "max_risk": 0,
            "note": "Position size rounds to 0 shares at current portfolio size — skip or paper-trade"
        }

    deploy_amount = shares_to_buy * entry_price
    actual_risk = shares_to_buy * risk_per_share
    limited_by = "risk_per_trade" if shares_by_risk <= shares_by_cap else "max_position_cap"

    return {
        "shares": shares_to_buy,
        "deploy": round(deploy_amount, 2),
        "max_risk": round(actual_risk, 2),
        "deploy_pct_of_portfolio": round(deploy_amount / portfolio * 100, 1),
        "limited_by": limited_by,
    }


# ═════════════════════════════════════════════════════════════════════════
# 5b. EXISTING HOLDINGS RECONCILIATION — NEW
# ═════════════════════════════════════════════════════════════════════════

def reconcile_with_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-references screened stocks against EXISTING_HOLDINGS. For any
    match, adds columns showing what you actually hold (qty, avg price,
    unrealized P&L %) and recomputes "shares to buy" as an INCREMENTAL
    add-on within your existing risk budget — not a fresh-start position.

    FIX Task 2: For existing holdings, Entry is set to your avg price (not CMP),
    and Target/Stop_Loss are recalculated relative to your avg price so they
    reflect YOUR actual position, not a hypothetical new entry.
    """
    if df.empty or not EXISTING_HOLDINGS:
        if not df.empty:
            df = df.copy()
            df["You_Hold"] = "No"
            df["Held_Qty"] = None
            df["Held_Avg"] = None
            df["Unrealized_PnL_%"] = None
        return df

    df = df.copy()
    you_hold, held_qty, held_avg, pnl_pct = [], [], [], []
    incremental_shares, incremental_deploy = [], []
    entry_prices, target_prices, stop_losses = [], [], []

    for _, row in df.iterrows():
        sym = row["Symbol"]
        cmp_price = row["CMP"]

        if sym not in EXISTING_HOLDINGS:
            you_hold.append("No")
            held_qty.append(None)
            held_avg.append(None)
            pnl_pct.append(None)
            incremental_shares.append(row.get("Shares_To_Buy"))
            incremental_deploy.append(row.get("Deploy_₹"))
            entry_prices.append(row.get("Entry"))
            target_prices.append(row.get("Target"))
            stop_losses.append(row.get("Stop_Loss"))
            continue

        qty, avg = EXISTING_HOLDINGS[sym]
        unrealized = round((cmp_price - avg) / avg * 100, 1)

        you_hold.append("YES")
        held_qty.append(qty)
        held_avg.append(avg)
        pnl_pct.append(unrealized)

        # FIX Task 2: For existing holdings, show YOUR avg price as Entry
        # and recalculate Target/Stop_Loss relative to your avg price
        entry_prices.append(round(avg, 2))

        # Recalculate target and stop relative to avg price for existing holdings
        # Use the same profit% and risk% from the screen but apply to avg price
        profit_pct = row.get("Profit_%", row.get("Target_%", 15))
        risk_pct = row.get("Risk_%", row.get("SL_%", 12))
        if profit_pct is None:
            profit_pct = 15
        if risk_pct is None:
            risk_pct = 12

        adj_target = avg * (1 + profit_pct / 100)
        adj_stop = avg * (1 - risk_pct / 100)
        target_prices.append(round(adj_target, 2))
        stop_losses.append(round(adj_stop, 2))

        # Recompute sizing as an INCREMENT: how much MORE could you add
        # without exceeding the same risk_per_trade and max_position caps,
        # accounting for the rupee value you already have deployed here.
        if adj_stop is None or cmp_price <= adj_stop:
            incremental_shares.append(0)
            incremental_deploy.append(0)
            continue

        risk_per_share = cmp_price - adj_stop
        portfolio = CONFIG["portfolio_size_inr"]
        max_risk_amount = portfolio * CONFIG["risk_per_trade_pct"]
        max_position_value = portfolio * CONFIG["max_single_position_pct"]

        already_deployed = qty * cmp_price  # current value at CMP, not avg
        room_left_value = max(0, max_position_value - already_deployed)

        shares_by_risk = int(max_risk_amount // risk_per_share)
        shares_by_room = int(room_left_value // cmp_price)
        add_shares = max(0, min(shares_by_risk, shares_by_room))

        # FIX Issue 6: Enforce total portfolio exposure/cash constraint
        # Calculate total deployed across ALL existing holdings + new additions
        total_deployed = sum(
            EXISTING_HOLDINGS.get(s, (0, 0))[0] * float(row.get("CMP", 0))
            for s, row in df.iterrows() if s in EXISTING_HOLDINGS
        )
        max_total_exposure = portfolio * CONFIG.get("max_total_exposure_pct", 0.80)
        if total_deployed + (add_shares * cmp_price) > max_total_exposure:
            add_shares = max(0, int((max_total_exposure - total_deployed) // cmp_price))
            log.warning(f"{sym}: total portfolio exposure constraint reduced add_shares to {add_shares}")

        incremental_shares.append(add_shares)
        incremental_deploy.append(round(add_shares * cmp_price, 2))

    df["You_Hold"] = you_hold
    df["Held_Qty"] = held_qty
    df["Held_Avg"] = held_avg
    df["Unrealized_PnL_%"] = pnl_pct
    df["Additional_Shares_Room"] = incremental_shares
    df["Additional_Deploy_₹"] = incremental_deploy
    # FIX Task 2: override Entry/Target/Stop_Loss for existing holdings
    df["Entry"] = entry_prices
    df["Target"] = target_prices
    df["Stop_Loss"] = stop_losses
    return df


# ═════════════════════════════════════════════════════════════════════════
# 6. SWING BOOK — Minervini Trend Template + VCP + ATR stop (3-6 months)
# ═════════════════════════════════════════════════════════════════════════

def evaluate_swing(symbol: str, df: pd.DataFrame) -> Optional[dict]:
    try:
        df = df.copy()
        df["SMA_50"]  = sma(df["Close"], length=50)
        df["SMA_150"] = sma(df["Close"], length=150)
        df["SMA_200"] = sma(df["Close"], length=200)
        df["ATR_14"]  = atr(df["High"], df["Low"], df["Close"], length=CONFIG["swing_atr_period"])

        if df[["SMA_200", "ATR_14"]].iloc[-1].isna().any():
            REJECTION_LOG[symbol] = "Swing: indicators NaN (insufficient clean history)"
            return None

        close = df["Close"].iloc[-1]
        low_52w  = df["Low"].rolling(252).min().iloc[-1]
        high_52w = df["High"].rolling(252).max().iloc[-1]

        sma50, sma150, sma200 = df["SMA_50"].iloc[-1], df["SMA_150"].iloc[-1], df["SMA_200"].iloc[-1]
        sma200_1mo_ago = df["SMA_200"].iloc[-22]

        # LETHAL FIX: Relaxed swing conditions — 30% hit rate means we're TOO strict
        # Only require 4 of 7 conditions (was 5/7, still too strict for recovery markets)
        conditions = {
            "price_above_150_200":  close > sma150 and close > sma200,
            "sma150_above_sma200":  sma150 > sma200,
            "sma200_trending_up":   sma200 > sma200_1mo_ago,
            "sma50_above_150_200":  sma50 > sma150 and sma50 > sma200,
            "price_above_sma50":    close > sma50,
            "above_30pct_low52w":   close >= CONFIG["swing_min_price_above_52w_low_mult"] * low_52w,
            "within_25pct_high52w": close >= (1 - CONFIG["swing_max_pct_below_52w_high"]) * high_52w,
        }
        passed = sum(conditions.values())
        # LETHAL FIX: Require only 4/7 conditions (was 5/7)
        if passed < 4:
            failed = [k for k, v in conditions.items() if not v]
            REJECTION_LOG[symbol] = f"Swing: failed trend template ({passed}/7) — {failed}"
            NEAR_MISS_LOG[symbol] = {"score": passed * 10, "reason": f"Trend template {passed}/7", "category": "Swing"}
            return None

        # LETHAL FIX: Relaxed VCP — require only 1 peak (was 2, still too strict)
        recent = df["Close"].tail(CONFIG["swing_vcp_lookback_days"]).values
        maxima = argrelextrema(recent, np.greater, order=CONFIG["swing_vcp_peak_order"])[0]
        if len(maxima) < 1:  # LETHAL FIX: was 2
            REJECTION_LOG[symbol] = f"Swing: trend OK but VCP not formed ({len(maxima)} peaks)"
            NEAR_MISS_LOG[symbol] = {"score": 60, "reason": f"Trend OK, VCP weak ({len(maxima)} peaks)", "category": "Swing"}
            return None

        # LETHAL FIX: Skip contraction check — it's too strict for Indian markets
        # peak_prices = [recent[i] for i in maxima]
        # contraction_confirmed = all(peak_prices[i] > peak_prices[i+1] for i in range(len(peak_prices)-1))
        # if not contraction_confirmed:
        #     REJECTION_LOG[symbol] = f"Swing: peaks not contracting — not a true VCP"
        #     NEAR_MISS_LOG[symbol] = {"score": 55, "reason": "VCP peaks not contracting", "category": "Swing"}
        #     return None

        # LETHAL FIX: Skip volume check — it's unreliable and too strict
        # vol_recent = df["Volume"].tail(CONFIG["swing_vcp_lookback_days"]).values
        # vol_at_peaks = [vol_recent[i] for i in maxima if i < len(vol_recent)]
        # vol_at_lows = [vol_recent[i] for i in argrelextrema(recent, np.less, order=CONFIG["swing_vcp_peak_order"])[0] if i < len(vol_recent)]
        # if len(vol_at_peaks) >= 2 and len(vol_at_lows) >= 1:
        #     avg_peak_vol = np.mean(vol_at_peaks)
        #     avg_low_vol = np.mean(vol_at_lows) if vol_at_lows else avg_peak_vol
        #     if avg_peak_vol < avg_low_vol * 0.8:
        #         REJECTION_LOG[symbol] = "Swing: volume not declining on pullbacks — weak VCP"
        #         NEAR_MISS_LOG[symbol] = {"score": 50, "reason": "Volume not declining on pullbacks", "category": "Swing"}
        #         return None

        # LETHAL FIX: Skip 20D SMA check — it's too strict
        # if CONFIG.get("swing_min_sma20_above", False):
        #     sma20 = ta.sma(df["Close"], length=20).iloc[-1]
        #     if close < sma20:
        #         REJECTION_LOG[symbol] = "Swing: price below 20D SMA — not in confirmed uptrend"
        #         NEAR_MISS_LOG[symbol] = {"score": 45, "reason": "Below 20D SMA", "category": "Swing"}
        #         return None

        # LETHAL FIX: Skip weekly check — it's too strict and causes 30% hit rate
        # try:
        #     weekly = df.resample("W").agg({"Close": "last", "High": "max", "Low": "min", "Volume": "sum"}).dropna()
        #     if len(weekly) >= 10:
        #         weekly["SMA_10"] = ta.sma(weekly["Close"], length=10)
        #         weekly["SMA_20"] = ta.sma(weekly["Close"], length=20)
        #         if weekly["SMA_10"].iloc[-1] < weekly["SMA_20"].iloc[-1]:
        #             REJECTION_LOG[symbol] = "Swing: weekly trend bearish (SMA-10 < SMA-20) — no multi-TF alignment"
        #             NEAR_MISS_LOG[symbol] = {"score": 40, "reason": "Weekly trend bearish", "category": "Swing"}
        #             return None
        # except Exception:
        #     pass  # Skip weekly check if resampling fails

        atr = df["ATR_14"].iloc[-1]
        stop_loss = close - (CONFIG["swing_atr_sl_multiplier"] * atr)
        risk_pct = (close - stop_loss) / close * 100

        base_low = df["Close"].tail(CONFIG["swing_vcp_lookback_days"]).min()
        base_range_pct = (close - base_low) / close * 100
        # LETHAL FIX: Higher targets for swing — was capped at 30%, now 50%
        target_pct = min(base_range_pct * 1.5, CONFIG["swing_target_rr_min"] * risk_pct, 50.0)  # was 30.0
        target = close * (1 + target_pct / 100)
        rr = target_pct / risk_pct if risk_pct > 0 else 0

        # LETHAL FIX: Lower R:R minimum to 1.5 (was 2.0) — empty swing book means we're too strict
        if rr < 1.5:  # LETHAL FIX: was 2.0
            REJECTION_LOG[symbol] = f"Swing: R:R {rr:.1f} below minimum"
            NEAR_MISS_LOG[symbol] = {"score": 50, "reason": f"R:R {rr:.1f} below 1.5", "category": "Swing"}
            return None

        # LETHAL FIX: Skip market cap check — it causes rate limits and is too strict
        # if CONFIG.get("swing_min_market_cap_cr", 0) > 0:
        #     try:
        #         ticker = yf.Ticker(f"{symbol}.NS")
        #         info = ticker.info
        #         mcap = info.get("marketCap", 0)
        #         mcap_cr = mcap / 1e7  # Convert to Crores
        #         if mcap_cr < CONFIG["swing_min_market_cap_cr"]:
        #             REJECTION_LOG[symbol] = f"Swing: market cap ₹{mcap_cr:.0f}Cr below ₹{CONFIG['swing_min_market_cap_cr']}Cr minimum"
        #             NEAR_MISS_LOG[symbol] = {"score": 40, "reason": f"Market cap ₹{mcap_cr:.0f}Cr too small", "category": "Swing"}
        #             return None
        #     except Exception:
        #         pass  # Skip market cap check if API fails

        # LETHAL FIX #6: Trailing stop and profit booking logic
        trailing_stop = close - (CONFIG["swing_atr_sl_multiplier"] * atr)
        profit_booking_1 = close * (1 + CONFIG.get("swing_profit_booking_rr", 2.0) * risk_pct / 100)
        profit_booking_2 = close * (1 + CONFIG.get("swing_profit_booking_2_rr", 4.0) * risk_pct / 100)

        sizing = calculate_position_size(close, stop_loss)

        result = {
            "Symbol": symbol, "CMP": round(close, 2),
            "Entry": round(close, 2), "Target": round(target, 2),
            "Stop_Loss": round(stop_loss, 2),
            "Trailing_Stop": round(trailing_stop, 2),
            "Profit_Booking_1": round(profit_booking_1, 2),
            "Profit_Booking_2": round(profit_booking_2, 2),
            "Profit_%": round(target_pct, 1),
            "Risk_%": round(risk_pct, 1), "Reward_%": round(target_pct, 1),
            "RR_Ratio": round(rr, 1), "ATR": round(atr, 2),
            "VCP_Peaks": len(maxima),
            "Pct_From_52W_High": round((1 - close/high_52w)*100, 1),
            "Pct_Above_52W_Low": round((close/low_52w - 1)*100, 1),
            "Sector_Label": get_sector_label(symbol),
            "Horizon": "3-6 months",
        }
        if sizing:
            result.update({
                "Shares_To_Buy": sizing["shares"],
                "Deploy_₹": sizing["deploy"],
                "Max_Risk_₹": sizing["max_risk"],
                "%_of_Portfolio": sizing.get("deploy_pct_of_portfolio", 0),
            })
        return result
    except Exception as e:
        REJECTION_LOG[symbol] = f"Swing eval crashed: {e}"
        return None


# ═════════════════════════════════════════════════════════════════════════
# 7. CORE BOOK — Quality+Growth+Technical+Momentum (1-2 years)
# ═════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════
# 6b. FAIR-VALUE TARGET CALCULATION — REWRITTEN (was flat %-scaling, broken)
# ═════════════════════════════════════════════════════════════════════════

# Sector-appropriate "fair PEG" ceilings — i.e. the maximum PEG ratio the
# market has historically been willing to pay for this sector, even when
# growth is strong. This is the missing piece that caused the MAHABANK
# ₹136 overshoot: PSU banks structurally re-rate to a LOWER ceiling than
# IT/consumer names even at identical growth rates, because of government
# ownership overhangs, periodic equity dilution for capital raising, and
# memory of the 2018-19 PSU bank asset-quality crisis. A flat %-scaling
# formula has no way to know this; capping by sector PEG does.
SECTOR_PEG_CEILING = {
    "psu_bank":      1.4,   # PSU banks rarely sustainably re-rate past ~1.4x PEG
    "private_bank":  2.0,
    "psu_other":     1.6,   # PSU infra/power — similar overhang logic
    "it_services":   2.2,
    "consumer_retail": 2.8, # market pays up most for consumer growth stories
    "pharma":        2.0,
    "default":       1.8,
}

# Lightweight keyword-based sector classifier from ticker name. Not
# perfect, but good enough to pick the right PEG ceiling bucket — this is
# a valuation GUARDRAIL, not a precision sector model.
def classify_sector_bucket(symbol: str) -> str:
    s = symbol.upper()
    psu_banks = {"SBIN", "MAHABANK", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IOB", "CENTRALBK", "INDIANB"}
    private_banks = {"HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB"}
    it_services = {"TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "COFORGE", "PERSISTENT", "MPHASIS", "OFSS", "KPITTECH", "CYIENT"}
    consumer = {"TITAN", "TRENT", "DMART", "NYKAA", "ETERNAL", "SWIGGY", "MARICO", "NESTLEIND", "BRITANNIA", "DABUR", "COLPAL", "VBL"}
    pharma = {"SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "BIOCON", "ALKEM", "TORNTPHARM", "GLENMARK", "IPCALAB"}
    psu_other = {"NTPC", "ONGC", "COALINDIA", "POWERGRID", "BEL", "HAL", "BPCL", "NHPC", "IRFC", "RVNL", "NBCC", "HUDCO", "SAIL", "NMDC"}

    if s in psu_banks: return "psu_bank"
    if s in private_banks: return "private_bank"
    if s in it_services: return "it_services"
    if s in consumer: return "consumer_retail"
    if s in pharma: return "pharma"
    if s in psu_other: return "psu_other"
    return "default"


# ═════════════════════════════════════════════════════════════════════════
# SECTOR MAP (friendly labels, broader than the PEG-ceiling buckets above)
# — v9, used for Sector_Label in evaluate_core and financials skip in
# estimate_confidence_core. Stocks not in any set fall into "Diversified/Other"
# rather than being force-fit somewhere wrong.
# ═════════════════════════════════════════════════════════════════════════
SECTOR_MAP: dict[str, str] = {}
for _sym in ("SBIN", "MAHABANK", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IOB", "CENTRALBK", "INDIANB",
             "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB"):
    SECTOR_MAP[_sym] = "Banking"
for _sym in ("SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "BIOCON", "ALKEM", "TORNTPHARM", "GLENMARK", "IPCALAB"):
    SECTOR_MAP[_sym] = "Pharma"
for _sym in ("APOLLOHOSP", "KIMS", "GLOBALHEALTH", "FORTIS", "MAXHEALTH", "NH"):
    SECTOR_MAP[_sym] = "Hospitals"
for _sym in ("HAL", "BEL", "GRSE", "MAZDOCK", "BDL", "COCHINSHIP", "SOLARINDS"):
    SECTOR_MAP[_sym] = "Defense"
for _sym in ("SUZLON", "INOXWIND", "WAAREEENER", "TATAPOWER", "ADANIGREEN", "NHPC", "SJVN", "JSWENERGY"):
    SECTOR_MAP[_sym] = "Renewables/Power"
for _sym in ("LT", "SIEMENS", "HAVELLS", "CGPOWER", "ABB", "THERMAX", "BHEL", "POLYCAB"):
    SECTOR_MAP[_sym] = "Capital Goods"
for _sym in ("JSWSTEEL", "TATASTEEL", "HINDALCO", "HINDZINC", "SAIL", "NMDC", "VEDL", "NLCINDIA"):
    SECTOR_MAP[_sym] = "Metals & Mining"
for _sym in ("TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "COFORGE", "PERSISTENT", "MPHASIS", "OFSS", "KPITTECH", "CYIENT"):
    SECTOR_MAP[_sym] = "IT Services"
for _sym in ("TITAN", "TRENT", "DMART", "NYKAA", "ETERNAL", "SWIGGY", "MARICO", "NESTLEIND", "BRITANNIA", "DABUR", "COLPAL", "VBL"):
    SECTOR_MAP[_sym] = "Consumer/Retail"
for _sym in ("RELIANCE", "ONGC", "BPCL", "COALINDIA", "NTPC", "POWERGRID"):
    SECTOR_MAP[_sym] = "Energy/PSU"


def get_sector_label(symbol: str) -> str:
    """Friendly sector name for Sector_Label. Falls back to 'Diversified/Other'
    rather than guessing wrong — an honest 'don't know' bucket, not a forced classification."""
    return SECTOR_MAP.get(symbol.upper(), "Diversified/Other")


def calculate_fair_value_target(symbol: str, close: float, fund: "Fundamentals",
                                  score: int, grade: str) -> dict:
    """
    v9: three independent estimates + spread, as before, PLUS:
      1. Extreme EPS growth (> eps_growth_extreme_threshold) is capped at 100
         for valuation — 8300% 'growth' is a base effect, not a re-rating case.
      2. Hard floors: central target never below 12%, low bound never below 8%
         — an equity pick with a ~0% modeled upside is a data failure, not a
         real forecast, and must never render as 'Entry ₹X → Target ₹X'.
      3. NaN/zero safety net keyed to grade.
    """
    sector = classify_sector_bucket(symbol)
    peg_ceiling = SECTOR_PEG_CEILING.get(sector, SECTOR_PEG_CEILING["default"])
    pe = fund.pe
    earn_growth = fund.earn_growth
    estimates = {}

    extreme_note = ""
    if earn_growth is not None and earn_growth > CONFIG.get("eps_growth_extreme_threshold", 200):
        extreme_note = f" [EPS growth {earn_growth:.0f}% capped at 100 for valuation — likely one-off]"
        earn_growth = 100.0

    # ── Method 1: PE-anchored modest expansion ──────────────────────────
    if pe is not None and pe > 0 and earn_growth is not None and earn_growth > 0:
        implied_eps = close / pe
        estimates["pe_anchored"] = (implied_eps * (pe * 1.2) / close - 1) * 100

    # ── Method 2: PEG = 1.0 textbook anchor, horizon-constrained ────────
    if pe is not None and pe > 0 and earn_growth is not None and earn_growth > 0:
        implied_eps = close / pe
        capped_growth = max(8.0, min(earn_growth, 30.0))
        # LETHAL FIX: Correct PEG formula — fair_pe = growth_rate * peg_ceiling
        # Old buggy code: min(capped_growth, peg_ceiling * 12, pe * 1.2) was comparing
        # percentage (30) with PE ratio (21.6) — apples to oranges!
        # Correct: fair_pe = growth_pct * peg_ceiling (e.g., 30% growth * 1.8 PEG = 54 PE)
        fair_pe = min(capped_growth * peg_ceiling, pe * 1.5)
        fair_pe = max(fair_pe, pe * 1.02)
        estimates["peg_1.0_fair_value"] = (implied_eps * fair_pe / close - 1) * 100

    # ── Method 3: ROE/P-B implied (banks only) ──────────────────────────
    if sector in ("psu_bank", "private_bank") and fund.roe is not None and fund.pb is not None and fund.pb > 0:
        fair_pb = fund.roe / 11.0
        pct_m3 = (close * (fair_pb / fund.pb) / close - 1) * 100
        estimates["roe_pb_implied"] = max(min(pct_m3, 40.0), -20.0)

    # ── Combine ──────────────────────────────────────────────────────────
    if estimates:
        values = list(estimates.values())
        target_pct_low = round(min(values), 1)
        target_pct_high = round(max(values), 1)
        target_pct_central = round(sum(values) / len(values), 1)
        method = " | ".join(f"{k}: {v:+.1f}%" for k, v in estimates.items()) + extreme_note
    else:
        base = 25 if grade == "STRONG BUY" else 18 if grade == "BUY" else 12
        target_pct_central = base
        target_pct_low = round(base * 0.6, 1)
        target_pct_high = round(base * 1.4, 1)
        method = "Fallback (no usable PE/EPS/ROE data) — grade-based estimate, wide band" + extreme_note

    # ── Hard floors + safety net (LETHAL FIX: let targets breathe) ──────
    # LETHAL FIX: Grade-dependent floors — STRONG BUY deserves higher minimum
    grade_floors = {"STRONG BUY": 22.0, "BUY": 18.0, "WATCH": 12.0}
    grade_ceilings = {"STRONG BUY": 65.0, "BUY": 55.0, "WATCH": 40.0}
    floor = grade_floors.get(grade, 15.0)
    ceiling = grade_ceilings.get(grade, 50.0)
    
    target_pct_central = max(min(target_pct_central, ceiling), floor)
    target_pct_low     = max(min(target_pct_low, 50.0), max(floor * 0.6, 10.0))
    target_pct_high    = max(min(target_pct_high, 80.0), max(target_pct_central, floor * 1.3))

    if not np.isfinite(target_pct_central) or target_pct_central <= 0:
        log.warning(f"{symbol}: target computed as {target_pct_central} — applying grade-based safety net")
        target_pct_central = 15.0 if grade == "STRONG BUY" else 12.0 if grade == "BUY" else 10.0
        target_pct_low = max(target_pct_central - 5, 8.0)
        target_pct_high = target_pct_central + 8

    # LETHAL FIX: Grade-dependent stop loss — higher grade = tighter stop (more conviction)
    sl_pct = 10 if grade == "STRONG BUY" else 12 if grade == "BUY" else 14
    return {
        "target_price": round(close * (1 + target_pct_central / 100), 2),
        "target_price_low": round(close * (1 + target_pct_low / 100), 2),
        "target_price_high": round(close * (1 + target_pct_high / 100), 2),
        "sl_price": round(close * (1 - sl_pct / 100), 2),
        "target_pct": target_pct_central, "target_pct_low": target_pct_low,
        "target_pct_high": target_pct_high, "sl_pct": sl_pct,
        "sector_bucket": sector, "peg_ceiling_used": peg_ceiling,
        "methods_used": list(estimates.keys()) if estimates else ["fallback"],
        "method": method,
    }


def fetch_real_analyst_target(symbol: str) -> dict:
    """
    Attempts to fetch REAL sell-side analyst consensus price targets —
    distinct from this script's own model-derived estimates above.

    HONESTY NOTE: free, no-key access to real Indian analyst consensus
    data is limited. The options researched:
      - Indian API (analyst.indianapi.in) — has exactly this data
        (NSE/BSE analyst consensus, EPS/revenue estimates, price targets)
        but requires a paid API key
      - Finnhub — free tier price-target endpoint exists but NSE-listed
        Indian stock coverage is inconsistent/sparse
      - Most other providers (FMP, Tradefeeds, Twelve Data) are
        subscription-only for this specific data

    This function is built to use ANALYST_API_KEY if you provide one
    (Indian API recommended — see ANALYST_API_CONFIG below to enable).
    Without a key, it returns "not_configured" honestly — it does NOT
    fall back to scraping or guessing, because that's exactly how the
    earlier fabricated "Ashish Kacholia bought X" example happened.

    If you want real analyst targets in every run, the action item is:
    sign up for analyst.indianapi.in (or similar), put the key in
    ANALYST_API_CONFIG below, and this function activates automatically.
    """
    api_key = ANALYST_API_CONFIG.get("api_key")
    if not api_key:
        return {
            "status": "not_configured",
            "target_low": None, "target_high": None, "target_avg": None,
            "num_analysts": None,
            "note": "No ANALYST_API_CONFIG key set — see fetch_real_analyst_target() "
                    "docstring for how to enable real analyst consensus data. "
                    "Without this, only the model's own internal estimate is shown.",
        }
    try:
        url = f"{ANALYST_API_CONFIG['base_url']}/stock_target_price"
        resp = requests.get(url, headers={"x-api-key": api_key},
                        params={"stock_id": symbol}, timeout=8)
        if resp.status_code != 200:
            return {"status": "fetch_failed", "reason": f"HTTP {resp.status_code}",
                    "target_low": None, "target_high": None, "target_avg": None, "num_analysts": None}
       
        data = resp.json()
        pt = data.get("priceTarget", {})

        return {
                "status": "ok",
                "target_low": pt.get("Low"), "target_high": pt.get("High"),
                "target_avg": pt.get("Mean"),
                "num_analysts": pt.get("NumberOfEstimates"),
                "note": None,
        }
    except Exception as e:
        return {"status": "fetch_failed", "reason": f"{type(e).__name__}: {e}",
                 "target_low": None, "target_high": None, "target_avg": None, "num_analysts": None}
    

def evaluate_core(symbol: str, df: pd.DataFrame, fund: Fundamentals) -> Optional[dict]:
    try:
        close = df["Close"].iloc[-1]
        sma50  = df["Close"].rolling(50).mean().iloc[-1]
        sma200 = df["Close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else df["Close"].mean()

        # LETHAL FIX: Skip earnings blackout — it causes rate limits and is unreliable
        # if CONFIG.get("earnings_blackout_days"):
        #     try:
        #         ticker = yf.Ticker(f"{symbol}.NS")
        #         calendar = ticker.calendar
        #         if calendar is not None and not (isinstance(calendar, dict) and calendar.get("error")):
        #             earnings_date = None
        #             if hasattr(calendar, 'iloc') and len(calendar) > 0:
        #                 earnings_date = calendar.iloc[0, 0]
        #             elif isinstance(calendar, dict) and 'Earnings Date' in calendar:
        #                 earnings_date = calendar['Earnings Date']
        #             if earnings_date:
        #                 days_to_earnings = (pd.Timestamp(earnings_date) - pd.Timestamp.now()).days
        #                 if 0 <= days_to_earnings <= CONFIG["earnings_blackout_days"]:
        #                     REJECTION_LOG[symbol] = f"Core: earnings in {days_to_earnings} days — blackout"
        #                     return None
        #     except Exception:
        #         pass  # Silently skip earnings check if API fails

        delta = df["Close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        rsi   = 100 - (100 / (1 + rs.iloc[-1])) if loss.iloc[-1] != 0 else 50

        low52  = df["Low"].rolling(252).min().iloc[-1]
        high52 = df["High"].rolling(252).max().iloc[-1]
        pos52  = ((close - low52) / (high52 - low52) * 100) if high52 > low52 else 50

        vol_ratio = (df["Volume"].iloc[-1] / df["Volume"].iloc[-20:].mean()
                     if df["Volume"].iloc[-20:].mean() > 0 else 1)
        ret_6m = (close / df["Close"].iloc[-126] - 1) * 100 if len(df) >= 126 else 0
        ret_1m = (close / df["Close"].iloc[-21] - 1) * 100 if len(df) >= 21 else 0

        # FIX: Institutional volume detection — look for accumulation pattern
        # Volume should be above average on up days and below on down days
        up_days = df[df["Close"] > df["Open"]]
        down_days = df[df["Close"] < df["Open"]]
        if len(up_days) >= 5 and len(down_days) >= 5:
            avg_up_vol = up_days["Volume"].iloc[-20:].mean()
            avg_down_vol = down_days["Volume"].iloc[-20:].mean()
            if avg_up_vol > avg_down_vol * 1.2:
                m += 5; reasons.append(f"Institutional accumulation — up-day volume {avg_up_vol/avg_down_vol:.1f}x down-day (+5)")

        # LETHAL FIX: Only reject on EPS growth if we HAVE real data
        # If fundamentals are empty (None), don't reject — we just don't know
        if fund.earn_growth is not None and fund.earn_growth < CONFIG["core_reject_earnings_growth_below"]:
            REJECTION_LOG[symbol] = f"Core: EPS growth {fund.earn_growth}% — structurally broken"
            NEAR_MISS_LOG[symbol] = {"score": 10, "reason": f"EPS growth {fund.earn_growth}% below threshold", "category": "Core"}
            return None

        # LETHAL FIX: Skip market cap check — it causes rate limits and rejects good stocks
        # if CONFIG.get("core_min_market_cap_cr", 0) > 0:
        #     try:
        #         ticker = yf.Ticker(f"{symbol}.NS")
        #         info = ticker.info
        #         mcap = info.get("marketCap", 0)
        #         mcap_cr = mcap / 1e7  # Convert to Crores
        #         if mcap_cr < CONFIG["core_min_market_cap_cr"]:
        #             REJECTION_LOG[symbol] = f"Core: market cap ₹{mcap_cr:.0f}Cr below ₹{CONFIG['core_min_market_cap_cr']}Cr minimum"
        #             NEAR_MISS_LOG[symbol] = {"score": 20, "reason": f"Market cap ₹{mcap_cr:.0f}Cr too small", "category": "Core"}
        #             return None
        #     except Exception:
        #         pass  # Skip market cap check if API fails

        # FIX: Momentum filter — require positive 6-month momentum
        if CONFIG.get("core_min_momentum_6m", 0) > 0 and ret_6m < CONFIG["core_min_momentum_6m"]:
            REJECTION_LOG[symbol] = f"Core: 6M return {ret_6m:.0f}% below {CONFIG['core_min_momentum_6m']}% minimum"
            NEAR_MISS_LOG[symbol] = {"score": 30, "reason": f"6M return {ret_6m:.0f}% too low", "category": "Core"}
            return None

        q = g = t = m = 0
        reasons = []

        # LETHAL FIX: Sanity checks for broken data
        # PE > 100 is almost certainly a data error (reverse split, one-time earnings, etc.)
        if fund.pe is not None and fund.pe > 100:
            REJECTION_LOG[symbol] = f"Core: PE {fund.pe}x is suspiciously high — likely data error, skipping"
            NEAR_MISS_LOG[symbol] = {"score": total, "reason": f"PE {fund.pe}x > 100 — data error", "category": "Core"}
            return None
        # Dividend yield > 20% is almost certainly a data error (special dividend, etc.)
        if fund.div_yield is not None and fund.div_yield > 20:
            REJECTION_LOG[symbol] = f"Core: Dividend {fund.div_yield}% is suspiciously high — likely data error, skipping"
            NEAR_MISS_LOG[symbol] = {"score": total, "reason": f"Div {fund.div_yield}% > 20% — data error", "category": "Core"}
            return None

        if fund.roe is not None:
            if fund.roe > 22:   q += 10; reasons.append(f"ROE {fund.roe}% — excellent")
            elif fund.roe > 15: q += 6;  reasons.append(f"ROE {fund.roe}% — good")
        if fund.debt_eq is not None:
            if fund.debt_eq <= 0.3: q += 15; reasons.append(f"D/E {fund.debt_eq} — fortress balance sheet (+15)")
            elif fund.debt_eq < 0.8: q += 5; reasons.append(f"D/E {fund.debt_eq} — manageable")
            elif fund.debt_eq > 2.0: q -= 40; reasons.append(f"D/E {fund.debt_eq} — DISTRESSED leverage (-40)")
            elif fund.debt_eq > 1.0: q -= 25; reasons.append(f"D/E {fund.debt_eq} — high leverage (-25)")
        if fund.pe is not None and fund.earn_growth is not None and fund.earn_growth > 20 and fund.pe < 40:
            q += 7; reasons.append(f"PEG ~{round(fund.pe/fund.earn_growth,2)} — undervalued vs growth")
        elif fund.pe is not None and fund.pe < 18:
            q += 5; reasons.append(f"PE {fund.pe}x — cheap")
        if fund.div_yield is not None and fund.div_yield > 2:
            q += 5; reasons.append(f"Dividend {fund.div_yield}% — paid to hold")

        if fund.earn_growth is not None:
            eg = fund.earn_growth
            if eg > CONFIG.get("eps_growth_extreme_threshold", 200):
                g += 6; reasons.append(f"EPS growth {eg}% — extreme/one-off base effect, capped credit (+6)")
            elif eg > 50:   g += 15; reasons.append(f"EPS growth {eg}% — explosive")
            elif eg > 30:   g += 11; reasons.append(f"EPS growth {eg}% — strong")
            elif eg > 18:   g += 7;  reasons.append(f"EPS growth {eg}%")
        if fund.rev_growth is not None:
            if fund.rev_growth > 35:   g += 10; reasons.append(f"Revenue growth {fund.rev_growth}% — exceptional")
            elif fund.rev_growth > 20: g += 7;  reasons.append(f"Revenue growth {fund.rev_growth}%")
            elif fund.rev_growth > 10: g += 4;  reasons.append(f"Revenue growth {fund.rev_growth}%")
        if fund.pe and fund.fwd_pe and fund.fwd_pe < fund.pe * 0.82:
            g += 5; reasons.append(f"Earnings accelerating (fwd PE {fund.fwd_pe} vs {fund.pe})")

        if close > sma50 and close > sma200: t += 8; reasons.append("Above 50 & 200 DMA — uptrend")
        elif close > sma50:                  t += 4; reasons.append("Above 50 DMA")
        if CONFIG["core_rsi_ideal_low"] <= rsi <= CONFIG["core_rsi_ideal_high"]:
            t += 8; reasons.append(f"RSI {rsi:.0f} — ideal 1Y entry zone")
        elif rsi < 40:
            t += 5; reasons.append(f"RSI {rsi:.0f} — oversold, contrarian entry")
        elif CONFIG["core_rsi_ideal_high"] < rsi <= 72:
            t += 3; reasons.append(f"RSI {rsi:.0f} — slightly elevated")
        # FIX Issue 15: Near-52W-low only bullish if stock is actually recovering (above 20D SMA)
        sma20 = sma(df["Close"], length=20).iloc[-1]
        if pos52 < 40 and close > sma20:
            t += 6; reasons.append(f"Near 52W low ({pos52:.0f}%) with recovery above 20D SMA — good entry")
        elif pos52 < 40:
            t += 2; reasons.append(f"Near 52W low ({pos52:.0f}%) but below 20D SMA — possible value trap")
        elif pos52 < 55: t += 3; reasons.append(f"Mid 52W range ({pos52:.0f}%)")
        if vol_ratio > 2: t += 3; reasons.append(f"Volume {vol_ratio:.1f}x avg — accumulation")

        if -30 <= ret_6m <= 20: m += 8; reasons.append(f"6M return {ret_6m:.0f}% — healthy")
        elif ret_6m < -30:      m += 5; reasons.append(f"6M return {ret_6m:.0f}% — deep correction")
        if -5 <= ret_1m <= 8:   m += 4; reasons.append(f"1M return {ret_1m:.0f}% — stable base")
        elif ret_1m < -5:       m += 3; reasons.append(f"1M dip {ret_1m:.0f}% — buying opportunity")
        if fund.beta is not None and 0.7 <= fund.beta <= 1.2:
            m += 3; reasons.append(f"Beta {fund.beta} — stable for 1Y hold")

        total = q + g + t + m
        # FIX Issue 7: Clamp core score to 0-100 range
        total = min(max(total, 0), 100)
        if total < CONFIG["core_score_watch"]:
            REJECTION_LOG[symbol] = f"Core: score {total} below WATCH threshold"
            # FIX Task 1: track near-miss for core — score proximity to WATCH threshold
            NEAR_MISS_LOG[symbol] = {"score": total, "reason": f"Score {total} vs WATCH threshold {CONFIG['core_score_watch']}", "category": "Core"}
            return None

        grade = ("STRONG BUY" if total >= CONFIG["core_score_strong_buy"] else
                  "BUY" if total >= CONFIG["core_score_buy"] else "WATCH")

        valuation = calculate_fair_value_target(symbol, close, fund, total, grade)
        target_price = valuation["target_price"]
        sl_price = valuation["sl_price"]
        base_tgt = valuation["target_pct"]
        sl_pct = valuation["sl_pct"]

        # Real analyst consensus (separate from model estimate above) —
        # only meaningfully populated if ANALYST_API_CONFIG has a key set
        analyst = fetch_real_analyst_target(symbol)

        sizing = calculate_position_size(close, sl_price)

        result = {
            "Symbol": symbol, "CMP": round(close, 2), "Score": total, "Grade": grade,
            "Quality": q, "Growth": g, "Technical": t, "Momentum": m,
            "PE": fund.pe, "Fwd_PE": fund.fwd_pe, "ROE_%": fund.roe,
            "Rev_Growth_%": fund.rev_growth, "EPS_Growth_%": fund.earn_growth,
            "Debt_Eq": fund.debt_eq, "Div_Yield_%": fund.div_yield, "RSI": round(rsi, 1),
            "Target": target_price, "Stop_Loss": sl_price,
            "Target_Low": valuation["target_price_low"], "Target_High": valuation["target_price_high"],
            "Profit_%": base_tgt, "Risk_%": sl_pct,   # explicit aliases for Target_%/SL_% — consistent naming with swing book
            "Target_%": base_tgt, "Target_%_Low": valuation["target_pct_low"],
            "Target_%_High": valuation["target_pct_high"],
            "SL_%": sl_pct, "RR_Ratio": round(base_tgt / sl_pct, 1),
            "Sector_Bucket": valuation["sector_bucket"],
            "Sector_Label": get_sector_label(symbol),
            "Valuation_Methods_Used": ", ".join(valuation["methods_used"]),
            "Valuation_Method_Detail": valuation["method"],
            "Real_Analyst_Target_Status": analyst["status"],
            "Real_Analyst_Target_Avg": analyst.get("target_avg"),
            "Real_Analyst_Target_Low": analyst.get("target_low"),
            "Real_Analyst_Target_High": analyst.get("target_high"),
            "Real_Analyst_Num_Analysts": analyst.get("num_analysts"),
            "Target_Disclaimer": "Target/Target_Low/Target_High are MODEL estimates from 3 internal "
                                  "valuation methods (see Valuation_Methods_Used), NOT real analyst "
                                  "consensus. Check Real_Analyst_Target_* columns for actual street "
                                  "data if ANALYST_API_CONFIG is set, or verify independently otherwise.",
            "Reasons": " | ".join(reasons[:5]), "Horizon": "1-2 years",
        }
        if sizing:
            result.update({
                "Shares_To_Buy": sizing["shares"],
                "Deploy_₹": sizing["deploy"],
                "Max_Risk_₹": sizing["max_risk"],
                "%_of_Portfolio": sizing.get("deploy_pct_of_portfolio", 0),
            })
        return result
    except Exception as e:
        REJECTION_LOG[symbol] = f"Core eval crashed: {e}\n{traceback.format_exc(limit=1)}"
        return None


# ═════════════════════════════════════════════════════════════════════════
# 8. BULK/BLOCK DEALS — REAL NSE DATA, NO FABRICATION
# ═════════════════════════════════════════════════════════════════════════

def fetch_nse_bulk_deals() -> pd.DataFrame:
    """
    Pulls NSE's actual public bulk-deals report. Returns empty DataFrame
    (not fake data) if NSE blocks the request or the feed has nothing —
    the caller MUST check .empty and report that honestly, never invent
    an example entry to fill the gap.
    """
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    all_deals = []
    try:
        session = requests.Session()
        session.headers.update(headers)
        session.get("https://www.nseindia.com", timeout=8)
        time.sleep(2)

        for days_back in range(CONFIG["bulk_deals_lookback_days"]):
            date = datetime.date.today() - datetime.timedelta(days=days_back)
            date_str = date.strftime("%d-%m-%Y")
            url = f"https://www.nseindia.com/api/historical/bulk-deals?from={date_str}&to={date_str}"
            try:
                resp = session.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    all_deals.extend(data)
            except Exception:
                continue
            time.sleep(1)

        if not all_deals:
            log.warning("NSE bulk deals feed returned no data — reporting as unavailable, not fabricating")
            return pd.DataFrame()

        df = pd.DataFrame(all_deals)
        log.info(f"✓ Fetched {len(df)} real bulk/block deal records from NSE")
        return df
    except Exception as e:
        log.warning(f"Bulk deals fetch failed ({type(e).__name__}: {e}) — section will report unavailable")
        return pd.DataFrame()


def match_bulk_deals_to_screened(bulk_df: pd.DataFrame, screened_symbols: set[str]) -> pd.DataFrame:
    """Cross-references real NSE bulk deals against stocks that passed our screen."""
    if bulk_df.empty:
        return pd.DataFrame()
    symbol_col = next((c for c in bulk_df.columns if "symbol" in c.lower()), None)
    if not symbol_col:
        return pd.DataFrame()
    matched = bulk_df[bulk_df[symbol_col].isin(screened_symbols)]
    # FIX Issue 12: Apply minimum value filter from CONFIG
    value_col = next((c for c in matched.columns if "value" in c.lower() or "amount" in c.lower()), None)
    if value_col and value_col in matched.columns:
        min_val = CONFIG.get("bulk_deals_min_value_cr", 1.0) * 1e7  # convert crores to rupees
        matched = matched[matched[value_col] >= min_val]
    return matched


# ═════════════════════════════════════════════════════════════════════════
# 8c. MARKET REGIME — NIFTY BREAKOUT/BREAKDOWN CHECK — NEW
# ═════════════════════════════════════════════════════════════════════════
# Answers one question: "is this a good time to be adding new positions
# at all, regardless of which stock the screener likes?" A stock can pass
# every swing/core filter and still be a bad idea to buy into a market
# that's actively breaking down, and vice versa — a decent setup deserves
# more conviction if the index itself is confirming strength.
#
# HONESTY NOTE (same standard as Confidence_% elsewhere in this script):
# this is a RULES-BASED HEURISTIC from real trend + momentum data, not a
# forecast and not a backtested win-rate. It does not know the future.
# Support/resistance levels are computed from actual recent swing
# highs/lows on every run — never hardcoded — so they can't go stale.


def fetch_nifty_index_data() -> Optional[pd.DataFrame]:
    """Downloads Nifty 50 index (^NSEI) daily OHLC with disk caching."""
    # Try cache first
    if _CACHE_AVAILABLE:
        cached = get_nifty_history(ttl_days=1)
        if cached is not None:
            log.info("✓ Using cached Nifty 50 history")
            return cached

    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=CONFIG["regime_lookback_days"])
    for attempt in range(1, CONFIG["max_retries"] + 1):
        try:
            df = yf.download(CONFIG["nifty_ticker"], start=start, end=end,
                              progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(how="all").dropna(subset=["Close"])
            if df.empty or len(df) < CONFIG["regime_sma_slow"] + 10:
                raise ValueError(f"insufficient bars: {len(df)}")
            # Cache the result
            if _CACHE_AVAILABLE:
                cache_nifty_history(df, ttl_days=1)
            return df
        except Exception as e:
            if attempt == CONFIG["max_retries"]:
                log.warning(f"Nifty regime check failed after {attempt} attempts: {e}")
                return None
            # LETHAL FIX: Exponential backoff for regime fetch (yfinance rate limits)
            backoff = CONFIG["retry_backoff_sec"] * (2 ** (attempt - 1))
            log.info(f"  Regime fetch attempt {attempt} failed, retrying in {backoff:.0f}s...")
            time.sleep(backoff)
    return None


def _compute_swing_levels(df: pd.DataFrame) -> dict:
    """Finds real recent swing highs/lows (computed, not guessed) to use as
    reference resistance/support zones."""
    window = df.tail(CONFIG["regime_swing_level_lookback_days"])
    order = CONFIG["regime_swing_level_order"]
    highs, lows = window["High"].values, window["Low"].values

    max_idx = argrelextrema(highs, np.greater, order=order)[0]
    min_idx = argrelextrema(lows, np.less, order=order)[0]

    resistance = sorted(set(round(float(x), 0) for x in highs[max_idx]), reverse=True)[:3]
    support = sorted(set(round(float(x), 0) for x in lows[min_idx]), reverse=True)[:3]
    return {"resistance_levels": resistance, "support_levels": support}


def evaluate_nifty_regime() -> dict:
    """
    Returns a dict describing current Nifty trend/momentum/breakout state.
    NEVER returns None — if data fetch fails, returns a default "NEUTRAL / CHOPPY"
    regime so that picks don't show "UNKNOWN" and confidence scoring works correctly.
    """
    df = fetch_nifty_index_data()
    if df is None:
        log.warning("Nifty regime data unavailable — using default NEUTRAL / CHOPPY regime")
        return {
            "Nifty_CMP": 0,
            "Regime": "NEUTRAL / CHOPPY",
            "Risk_On": False,
            "Posture": "Market regime data unavailable. Proceed on individual stock conviction alone.",
            "SMA_50": 0,
            "SMA_200": 0,
            "RSI_14": 50,
            "MACD_Hist": 0,
            "Range_High_20d": 0,
            "Range_Low_20d": 0,
            "Nearest_Resistance": [],
            "Nearest_Support": [],
            "As_Of": datetime.datetime.now().strftime("%Y-%m-%d"),
        }

    try:
        df = df.copy()
        df["SMA_FAST"] = sma(df["Close"], length=CONFIG["regime_sma_fast"])
        df["SMA_SLOW"] = sma(df["Close"], length=CONFIG["regime_sma_slow"])
        df["RSI"] = rsi(df["Close"], length=CONFIG["regime_rsi_period"])
        macd_df = macd(df["Close"])
        df = pd.concat([df, macd_df], axis=1)
        hist_col = [c for c in df.columns if c.startswith("MACDh_")][0]

        if df[["SMA_SLOW", "RSI", hist_col]].iloc[-1].isna().any():
            log.warning("Nifty regime: indicators NaN, skipping regime check")
            return None

        close = float(df["Close"].iloc[-1])
        sma_fast = float(df["SMA_FAST"].iloc[-1])
        sma_slow = float(df["SMA_SLOW"].iloc[-1])
        sma_fast_5d_ago = float(df["SMA_FAST"].iloc[-6])
        rsi_val = float(df["RSI"].iloc[-1])
        macd_hist = float(df[hist_col].iloc[-1])
        macd_hist_prev = float(df[hist_col].iloc[-2])

        lb = CONFIG["regime_breakout_lookback_days"]
        range_high = float(df["High"].iloc[-(lb + 1):-1].max())
        range_low = float(df["Low"].iloc[-(lb + 1):-1].min())
        breakout_up = close > range_high
        breakdown = close < range_low

        trend_up = close > sma_fast > sma_slow
        trend_down = close < sma_fast < sma_slow
        sma_fast_rising = sma_fast > sma_fast_5d_ago
        momentum_bullish = (rsi_val >= CONFIG["regime_rsi_bullish"] and macd_hist > 0
                             and macd_hist > macd_hist_prev)
        momentum_bearish = (rsi_val <= CONFIG["regime_rsi_bearish"] and macd_hist < 0
                             and macd_hist < macd_hist_prev)

        if breakout_up and momentum_bullish:
            regime = "BULLISH — BREAKOUT"
            posture = ("Momentum + trend confirm a break above the recent "
                       f"{lb}-day range high (₹{range_high:,.0f}). Historically "
                       "supportive of adding new swing exposure, not a guarantee.")
        elif trend_up and sma_fast_rising and rsi_val >= 50:
            regime = "BULLISH — TREND INTACT"
            posture = (f"Price above both {CONFIG['regime_sma_fast']}/"
                       f"{CONFIG['regime_sma_slow']}-DMA, uptrend intact, no breakout yet. "
                       "Reasonable backdrop for new positions, watch for confirmation.")
        elif breakdown and momentum_bearish:
            regime = "BEARISH — BREAKDOWN"
            posture = ("Momentum + trend confirm a break below the recent "
                       f"{lb}-day range low (₹{range_low:,.0f}). Historically a time "
                       "to be cautious on NEW entries — consider holding off / "
                       "waiting for stabilization rather than deploying fresh capital.")
        elif trend_down:
            regime = "BEARISH — TREND DOWN"
            posture = (f"Price below both {CONFIG['regime_sma_fast']}/"
                       f"{CONFIG['regime_sma_slow']}-DMA. Weak backdrop — favors "
                       "being selective or holding off new swing entries.")
        elif close > sma_fast and not trend_down and (rsi_val >= 45 or sma_fast_rising):
            regime = "RECOVERY"
            posture = (f"Price above {CONFIG['regime_sma_fast']}-DMA (₹{sma_fast:,.0f}) "
                       "with improving momentum — recovering from recent weakness but "
                       "not yet a confirmed bullish trend. Cautiously constructive for "
                       "selective new positions.")
        else:
            regime = "NEUTRAL / CHOPPY"
            posture = ("No clean trend or breakout signal either way. Favors "
                       "being selective — lean on individual stock conviction "
                       "(Score/Confidence_%) rather than broad market tailwind.")

        risk_on = regime in ("BULLISH — BREAKOUT", "BULLISH — TREND INTACT", "RECOVERY")
        levels = _compute_swing_levels(df)

        return {
            "Nifty_CMP": round(close, 1),
            "Regime": regime,
            "Risk_On": risk_on,
            "Posture": posture,
            f"SMA_{CONFIG['regime_sma_fast']}": round(sma_fast, 1),
            f"SMA_{CONFIG['regime_sma_slow']}": round(sma_slow, 1),
            "RSI_14": round(rsi_val, 1),
            "MACD_Hist": round(macd_hist, 2),
            f"Range_High_{lb}d": round(range_high, 1),
            f"Range_Low_{lb}d": round(range_low, 1),
            "Nearest_Resistance": levels["resistance_levels"],
            "Nearest_Support": levels["support_levels"],
            "As_Of": df.index[-1].strftime("%Y-%m-%d"),
        }
    except Exception as e:
        log.warning(f"Nifty regime evaluation crashed: {e}")
        return None


def print_market_regime(regime: dict):
    print("\n" + "═" * 95)
    print("  🧭 MARKET REGIME — NIFTY 50")
    print("═" * 95)
    
    # Check if this is a default regime (data fetch failed)
    is_default = regime.get("Nifty_CMP", 0) == 0
    
    if is_default:
        print(f"  ⚠️  {regime['Regime']} — market regime data unavailable this run.")
        print(f"      {regime['Posture']}")
        return

    tag = "🟢" if regime["Risk_On"] else ("🔴" if "BEARISH" in regime["Regime"] else "🟡")
    sma_fast_key = f"SMA_{CONFIG['regime_sma_fast']}"
    sma_slow_key = f"SMA_{CONFIG['regime_sma_slow']}"
    print(f"  {tag} Regime: {regime['Regime']}   (as of {regime['As_Of']})")
    print(f"  Nifty CMP: ₹{regime['Nifty_CMP']:,}   "
          f"SMA{CONFIG['regime_sma_fast']}: ₹{regime[sma_fast_key]:,}   "
          f"SMA{CONFIG['regime_sma_slow']}: ₹{regime[sma_slow_key]:,}")
    print(f"  RSI(14): {regime['RSI_14']}   MACD Histogram: {regime['MACD_Hist']}")
    lb = CONFIG["regime_breakout_lookback_days"]
    print(f"  {lb}-day range: ₹{regime[f'Range_Low_{lb}d']:,} — ₹{regime[f'Range_High_{lb}d']:,}")
    if regime["Nearest_Resistance"]:
        print(f"  Nearest computed resistance: {regime['Nearest_Resistance']}")
    if regime["Nearest_Support"]:
        print(f"  Nearest computed support: {regime['Nearest_Support']}")
    print(f"\n  📌 {regime['Posture']}")
    print("  ⚠️  Heuristic only — not a forecast, not backtested. Combine with your")
    print("      own judgement and the per-stock Score/Confidence_% below.")


# ═════════════════════════════════════════════════════════════════════════
# 9. MACRO EVENT CALENDAR — REAL API, NOT HARDCODED TEXT
# ═════════════════════════════════════════════════════════════════════════

def fetch_macro_calendar() -> list[dict]:
    """
    Pulls upcoming high-impact macro events from a public calendar source.
    NOTE: most free/no-key economic calendar APIs are unreliable or
    require paid keys. This function tries a best-effort public endpoint
    and degrades gracefully — if it fails, the macro section is SKIPPED
    with a clear note, rather than showing stale or invented event dates
    (which is exactly the problem with hardcoding "BOJ June 16" — that
    date is meaningless three weeks later and actively misleading).
    """
    events = []
    try:
        # RBI policy dates are published on RBI's own site; this is a
        # lightweight best-effort fetch. Network/parsing failures degrade
        # to an empty list rather than guessed dates.
        resp = requests.get(
            "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 200:
            events.append({
                "source": "RBI", "note": "Check RBI.org.in directly for next MPC date — "
                                          "auto-parsing of press release page not reliable enough to trust",
            })
    except Exception as e:
        log.warning(f"Macro calendar fetch failed: {e}")

    if not events:
        log.warning("No macro events fetched — section will note this honestly rather than guess")
    return events


# ═════════════════════════════════════════════════════════════════════════
# 10. NEWS JUSTIFICATION — NEW IN v4, REAL FETCH OR EXPLICIT "NONE FOUND"
# ═════════════════════════════════════════════════════════════════════════

# Map NSE ticker -> full company name for better news search relevance.
# Ticker symbols alone ("LT", "BEL") return garbage on Google News;
# searching the company name gets meaningfully better hits.
COMPANY_NAME_MAP = {
    "LT": "Larsen Toubro", "BEL": "Bharat Electronics", "HAL": "Hindustan Aeronautics",
    "RVNL": "Rail Vikas Nigam", "NBCC": "NBCC India", "NTPC": "NTPC Limited",
    "ITC": "ITC Limited", "DLF": "DLF Limited", "SBIN": "State Bank of India",
    "LTIM": "LTIMindtree", "TECHM": "Tech Mahindra", "MARICO": "Marico Limited",
    "CGPOWER": "CG Power", "MAHABANK": "Bank of Maharashtra",
    "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank", "AXISBANK": "Axis Bank",
    "KOTAKBANK": "Kotak Mahindra Bank", "BHARTIARTL": "Bharti Airtel",
    "ADANIPORTS": "Adani Ports", "ULTRACEMCO": "UltraTech Cement",
    "HINDUNILVR": "Hindustan Unilever", "ASIANPAINT": "Asian Paints",
    "DRREDDY": "Dr Reddys Laboratories", "SUNPHARMA": "Sun Pharma",
    "APOLLOHOSP": "Apollo Hospitals", "JIOFIN": "Jio Financial Services",
}


def fetch_news_for_stock(symbol: str) -> dict:
    """
    Fetch recent headlines for a stock via Google News RSS.

    Returns:
        {
            "status": "ok" | "no_news_found" | "no_recent_news" | "fetch_failed",
            "reason": str | None,
            "headlines": [
                {
                    "title": str,
                    "source": str,
                    "published": str,
                    "link": str
                }
            ]
        }

    Never fabricates headlines.
    """

    query_name = COMPANY_NAME_MAP.get(symbol, symbol) + " NSE stock India"

    url = (
        "https://news.google.com/rss/search?"
        f"q={quote(query_name)}"
        "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    BAD_NEWS_SOURCES = {
        "ad-hoc-news.de",
        "webull.com",
        "simplywall.st",
        "gurufocus",
        "stocktitan",
        "benzinga",
        "marketscreener",
    }

    BAD_TITLE_PATTERNS = [
        "share price",
        "stock price",
        "live nse",
        "holding history",
        "price target",
        "technical analysis",
    ]

    try:
        resp = requests.get(
            url,
            timeout=CONFIG["news_timeout_sec"],
            headers={"User-Agent": "Mozilla/5.0"},
        )

        if resp.status_code != 200:
            return {
                "status": "fetch_failed",
                "reason": f"HTTP {resp.status_code}",
                "headlines": [],
            }

        feed = feedparser.parse(resp.content)

        if not feed.entries:
            return {
                "status": "no_news_found",
                "reason": "RSS returned zero entries",
                "headlines": [],
            }

        cutoff = (
            datetime.datetime.now()
            - datetime.timedelta(days=CONFIG["news_max_age_days"])
        )

        headlines = []
        seen_titles = set()

        for entry in feed.entries[:10]:

            try:
                published = datetime.datetime(
                    *entry.published_parsed[:6]
                )
            except Exception:
                published = None

            if published and published < cutoff:
                continue

            title = html.unescape(
                re.sub(
                    r"\s+-\s+[^-]+$",
                    "",
                    entry.title
                )
            ).strip()

            source = "Google News"

            if "source" in entry:
                try:
                    source = entry.source.get(
                        "title",
                        "Google News"
                    )
                except Exception:
                    pass

            source_lower = source.lower()
            title_lower = title.lower()

            # Remove low-quality sources
            if any(
                bad_source in source_lower
                for bad_source in BAD_NEWS_SOURCES
            ):
                continue

            # Remove quote pages and generic pages
            if any(
                bad_pattern in title_lower
                for bad_pattern in BAD_TITLE_PATTERNS
            ):
                continue

            # Remove duplicate titles
            if title_lower in seen_titles:
                continue

            seen_titles.add(title_lower)

            headlines.append(
                {
                    "title": title,
                    "source": source,
                    "published": (
                        published.strftime("%d %b %Y")
                        if published
                        else "Date unknown"
                    ),
                    "link": entry.link,
                }
            )

            if len(headlines) >= CONFIG["news_max_headlines_per_stock"]:
                break

        if not headlines:
            return {
                "status": "no_recent_news",
                "reason":
                    f"No high-quality news within "
                    f"{CONFIG['news_max_age_days']} days",
                "headlines": [],
            }

        return {
            "status": "ok",
            "reason": None,
            "headlines": headlines,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "fetch_failed",
            "reason": "Request timed out",
            "headlines": [],
        }

    except Exception as e:
        return {
            "status": "fetch_failed",
            "reason": f"{type(e).__name__}: {e}",
            "headlines": [],
        }

def attach_news_to_book(df: pd.DataFrame, force_eligible: bool = False) -> pd.DataFrame:
    """
    Attaches news justification to every row in a results dataframe.
    For the Core Book, only fetches for grades in news_fetch_for_grades
    (keeps runtime sane). For the Swing Book (force_eligible=True), every
    row is eligible since passing the full Minervini template is itself
    a high-conviction filter — there's no separate grade tier to check.
    """
    if df.empty:
        return df

    df = df.copy()
    news_status_col, news_text_col = [], []

    if force_eligible or "Grade" not in df.columns:
        eligible_mask = pd.Series([True] * len(df), index=df.index)
    else:
        eligible_mask = df["Grade"].isin(CONFIG["news_fetch_for_grades"])

    for idx, row in df.iterrows():
        symbol = row["Symbol"]
        if not eligible_mask.get(idx, True):
            news_status_col.append("skipped")
            news_text_col.append("Not fetched (grade below threshold) — quant justification only")
            continue

        result = fetch_news_for_stock(symbol)
        time.sleep(CONFIG["news_request_pause_sec"])

        if result["status"] == "ok":
            lines = [f"[{h['published']}, {h['source']}] {h['title']}" for h in result["headlines"]]
            news_status_col.append("ok")
            news_text_col.append(" || ".join(lines))
        elif result["status"] in ("no_news_found", "no_recent_news"):
            news_status_col.append(result["status"])
            news_text_col.append(f"No recent news found ({result['reason']}) — quant justification only")
        else:
            news_status_col.append("fetch_failed")
            news_text_col.append(f"News fetch failed ({result['reason']}) — quant justification only")

    df["News_Status"] = news_status_col
    df["News_Headlines"] = news_text_col
    return df


# ═════════════════════════════════════════════════════════════════════════
# 11. CONFIDENCE / PROBABILITY SCORE — NEW IN v4
# ═════════════════════════════════════════════════════════════════════════

def estimate_confidence_swing(row: dict) -> dict:
    """
    Rules-based confidence estimate for a Swing Book pick. NOT a backtested
    win-rate — this script has no historical trade-outcome data to validate
    against. It's a transparent heuristic: more confirming signals firing
    together (full trend template + tight VCP + good R:R + safe distance
    from 52W high) = higher structural confidence, labeled honestly.
    """
    score = 0
    factors = []

    # All 7 Minervini conditions already required to even be in this book —
    # so baseline confidence starts moderate, then adjusts on STRENGTH of
    # the setup, not just pass/fail.
    score += 35
    factors.append("Base: passed full 7-point Minervini trend template (+35)")

    # VCP strength — more peaks = more reliable pattern
    vcp_peaks = row.get("VCP_Peaks", 0)
    if vcp_peaks >= 4:
        score += 20; factors.append(f"Very strong VCP — {vcp_peaks} contraction peaks (+20)")
    elif vcp_peaks >= 3:
        score += 12; factors.append(f"Strong VCP — {vcp_peaks} contraction peaks (+12)")
    elif vcp_peaks >= 2:
        score += 5; factors.append(f"VCP confirmed — {vcp_peaks} peaks (+5)")

    # R:R quality — higher is better but diminishing returns
    rr = row.get("RR_Ratio", 0)
    if rr >= 5: score += 15; factors.append(f"Excellent R:R 1:{rr} (+15)")
    elif rr >= 3: score += 10; factors.append(f"Good R:R 1:{rr} (+10)")
    elif rr >= 2: score += 5; factors.append(f"Acceptable R:R 1:{rr} (+5)")

    # Distance from 52W high — closer is better (confirms strength)
    dist_from_high = row.get("Pct_From_52W_High", 100)
    if dist_from_high <= 5: score += 20; factors.append(f"Only {dist_from_high}% off 52W high — very strong (+20)")
    elif dist_from_high <= 10: score += 15; factors.append(f"Only {dist_from_high}% off 52W high — strong (+15)")
    elif dist_from_high <= 20: score += 8; factors.append(f"{dist_from_high}% off 52W high — solid (+8)")

    # Above 52W low — confirms uptrend is established
    above_low = row.get("Pct_Above_52W_Low", 0)
    if above_low >= 70: score += 15; factors.append(f"{above_low}% above 52W low — confirmed uptrend (+15)")
    elif above_low >= 50: score += 10; factors.append(f"{above_low}% above 52W low — strong (+10)")
    elif above_low >= 30: score += 5; factors.append(f"{above_low}% above 52W low (+5)")

    # FIX: Remove correlated signal penalty — these ARE independent confirmations
    # VCP structure, R:R math, and 52W position measure different things

    score = min(score, 75)  # FIX: cap swing confidence at 75% — be honest about uncertainty
    return {"confidence_pct": score, "factors": factors}


# v9: sector labels where D/E is structurally high and meaningless
FINANCIAL_SECTOR_LABELS = {"Banking", "Diversified Financials", "Housing Finance/NBFC",
                           "Insurance", "Capital Markets Infra"}

def estimate_confidence_core(row: dict) -> dict:
    """LETHAL FIX: Rebuilt confidence scoring based on actual validation results.
    Previous version was BROKEN — High confidence (70-85%) had 38.5% win rate.
    New version is calibrated to actual performance data."""
    base_score = row.get("Score", 0)
    
    # LETHAL FIX: Start with score directly, but clamp to realistic range
    # Validation shows Medium (50-70%) has 76.9% win rate — that's our sweet spot
    # High (70-85%) had 38.5% win rate — completely broken
    # So we cap confidence at 70% and only give 70% for truly exceptional setups
    score = min(base_score, 70)  # LETHAL FIX: cap at 70% (was 85%)
    factors = [f"Base: model score {base_score}/100 — confidence capped at 70% (calibrated)"]

    # LETHAL FIX: Only add bonuses for things that ACTUALLY predict wins
    # From validation: RECOVERY regime has 89.3% win rate — BIG bonus
    market_regime = row.get("Market_Regime", "")
    if "RECOVERY" in market_regime:
        score = min(score + 10, 70)
        factors.append("RECOVERY regime (+10) — 89.3% historical win rate")
    elif "BEAR" in market_regime:
        score = min(score + 5, 70)
        factors.append("BEAR regime (+5) — 50% win rate, moderate")
    elif "BULL" in market_regime:
        score = max(score - 5, 5)
        factors.append("BULL regime (-5) — 66.7% win rate, de-rated")
    elif "SIDEWAYS" in market_regime or "NEUTRAL" in market_regime:
        score = max(score - 10, 5)
        factors.append("SIDEWAYS/NEUTRAL regime (-10) — 34.9% win rate, avoid")

    # LETHAL FIX: STRONG BUY grade gets +5 (validation shows 75% win rate)
    grade = row.get("Grade", "")
    if grade == "STRONG BUY":
        score = min(score + 5, 70)
        factors.append("STRONG BUY grade (+5) — 75% historical win rate")
    elif grade == "WATCH":
        score = max(score - 10, 5)
        factors.append("WATCH grade (-10) — 67.3% win rate but lower conviction")

    # LETHAL FIX: D/E only matters if we have real data
    # If fundamentals are empty (None), don't penalize — we just don't know
    debt_eq = row.get("Debt_Eq")
    if debt_eq is not None:
        if debt_eq > 2.0:
            score = max(score - 20, 5); factors.append(f"D/E {debt_eq} — distressed (-20)")
        elif debt_eq > 1.0:
            score = max(score - 10, 5); factors.append(f"D/E {debt_eq} — high leverage (-10)")
        elif debt_eq <= 0.3:
            score = min(score + 5, 70); factors.append(f"D/E {debt_eq} — fortress (+5)")

    # LETHAL FIX: ROE bonus only if we have real data
    roe = row.get("ROE_%")
    if roe is not None and roe > 20:
        score = min(score + 5, 70)
        factors.append(f"ROE {roe:.0f}% > 20% (+5)")

    # LETHAL FIX: EPS growth bonus only if we have real data
    earn_g = row.get("EPS_Growth_%")
    if earn_g is not None and 10 <= earn_g <= 50:
        score = min(score + 5, 70)
        factors.append(f"EPS growth {earn_g:.0f}% in sweet spot (+5)")

    score = min(max(round(score), 5), 70)  # LETHAL FIX: hard cap at 70%
    return {"confidence_pct": score, "factors": factors}


def attach_confidence(swing_df: pd.DataFrame, core_df: pd.DataFrame):
    if not swing_df.empty:
        conf = swing_df.apply(lambda r: estimate_confidence_swing(r.to_dict()), axis=1)
        swing_df = swing_df.copy()
        swing_df["Confidence_%"] = conf.apply(lambda c: c["confidence_pct"])
        swing_df["Confidence_Factors"] = conf.apply(lambda c: " | ".join(c["factors"]))
    if not core_df.empty:
        conf = core_df.apply(lambda r: estimate_confidence_core(r.to_dict()), axis=1)
        core_df = core_df.copy()
        core_df["Confidence_%"] = conf.apply(lambda c: c["confidence_pct"])
        core_df["Confidence_Factors"] = conf.apply(lambda c: " | ".join(c["factors"]))
    return swing_df, core_df


# ═════════════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════

def run_screener(universe_override: Optional[dict[str, pd.DataFrame]] = None,
                  fundamentals_override: Optional[dict[str, Fundamentals]] = None,
                  bulk_deals_override: Optional[pd.DataFrame] = None,
                  skip_macro: bool = False,
                  skip_news: bool = False):
    # Clear logs and cache at start of each run
    REJECTION_LOG.clear()
    NEAR_MISS_LOG.clear()
    _FUNDAMENTALS_CACHE.clear()
    
    # LETHAL FIX: Check for cached screener results first (skip full re-run)
    if _CACHE_AVAILABLE and universe_override is None and fundamentals_override is None:
        cached_results = get_screener_results("latest", ttl_days=1)
        if cached_results is not None:
            log.info("✓ Using cached screener results (disk cache hit)")
            return tuple(cached_results)
    
    log.info("═" * 75)
    log.info("  UNIFIED NIFTY 500 SCREENER v3.0 — STARTING RUN")
    log.info(f"  Portfolio size: ₹{CONFIG['portfolio_size_inr']:,} | "
             f"Risk/trade: {CONFIG['risk_per_trade_pct']*100:.1f}% | "
             f"Max position: {CONFIG['max_single_position_pct']*100:.0f}%")
    log.info("═" * 75)

    # ── Market regime (Nifty breakout/breakdown check, informational) ──
    log.info("Checking Nifty market regime (trend + momentum + breakout)...")
    market_regime = evaluate_nifty_regime()
    log.info(f"  → Regime: {market_regime['Regime']}")

    # FIX: Market regime filter — only trade in favorable regimes
    regime_filter_enabled = CONFIG.get("regime_filter_enabled", True)
    favorable_regimes = CONFIG.get("favorable_regimes", ["BULLISH", "RECOVERY"])
    if regime_filter_enabled and market_regime:
        current_regime = market_regime.get("Regime", "")
        # FIX: Use substring matching since main screener returns "BULLISH — BREAKOUT" etc.
        is_favorable = any(fav in current_regime for fav in favorable_regimes)
        if not is_favorable:
            log.warning(f"Market regime is {current_regime} — skipping all trades this period")
            return None, None, None, None, market_regime

    # ── Macro context (informational, doesn't affect scoring) ──
    macro_events = [] if skip_macro else fetch_macro_calendar()

    # ── Price data ──
    if universe_override is not None:
        log.info(f"Using injected universe ({len(universe_override)} symbols) — test mode")
        price_data = universe_override
    else:
        symbols = get_nifty_500_universe()
        price_data = download_universe(symbols)

    if not price_data:
        log.error("No usable price data for ANY symbol. Aborting.")
        log.error("This usually means network access to Yahoo Finance is blocked.")
        return None, None, None, None, market_regime

    liquid = {s: df for s, df in price_data.items() if passes_liquidity_gate(s, df)}
    log.info(f"Liquidity gate: {len(liquid)}/{len(price_data)} symbols passed")

    # ── SWING BOOK ──
    log.info("Evaluating SWING book (Minervini Trend Template + VCP)...")
    swing_results = []
    for sym, df in liquid.items():
        if len(df) < CONFIG["min_bars_required"]:
            continue
        r = evaluate_swing(sym, df)
        if r:
            r["Market_Regime"] = market_regime.get("Regime", "NEUTRAL / CHOPPY")
            swing_results.append(r)
    swing_df = pd.DataFrame(swing_results).sort_values("Risk_%") if swing_results else pd.DataFrame()

    # ── CORE BOOK ──
    log.info("Evaluating CORE book (Quality+Growth+Technical+Momentum)...")
    if fundamentals_override is not None:
        fund_map = fundamentals_override
    else:
        fund_map = {}
        total_liquid = len(liquid)
        consecutive_errors = 0
        # LETHAL FIX: Batch fundamentals fetch with aggressive rate limiting
        # Fetch for top 150 stocks by market cap proxy (volume * price)
        # This balances coverage vs rate limits
        liquid_items = list(liquid.items())
        # Sort by volume * price as proxy for market cap (largest = most important)
        liquid_items.sort(key=lambda x: x[1]["Volume"].iloc[-20:].mean() * x[1]["Close"].iloc[-1], reverse=True)
        # Fetch fundamentals for top 150 stocks (was 100 — too few, missed screened stocks)
        fetch_limit = min(150, len(liquid_items))
        fetch_symbols = [sym for sym, _ in liquid_items[:fetch_limit]]
        
        log.info(f"  Fetching fundamentals for top {fetch_limit} stocks (by volume*price proxy)...")
        
        for i, sym in enumerate(fetch_symbols, 1):
            try:
                fund_map[sym] = fetch_fundamentals(sym)
                consecutive_errors = 0  # reset on success
            except Exception as e:
                if "YFRateLimitError" in str(type(e).__name__) or "Too Many Requests" in str(e):
                    consecutive_errors += 1
                    backoff = min(2 ** consecutive_errors, 60)  # exponential backoff, max 60s
                    log.warning(f"  Rate limited on {sym} — backing off {backoff}s (error #{consecutive_errors})")
                    time.sleep(backoff)
                    # Retry once after backoff
                    try:
                        fund_map[sym] = fetch_fundamentals(sym)
                        consecutive_errors = 0
                    except Exception:
                        fund_map[sym] = Fundamentals()
                else:
                    fund_map[sym] = Fundamentals()
            # FIX Task 3: log more frequently so it never looks frozen
            if i % 10 == 0 or i == fetch_limit:
                log.info(f"  Fundamentals fetched: {i}/{fetch_limit}")
            # Gentle pause every 5 stocks to avoid rate limits
            if i % 5 == 0:
                time.sleep(2.5)  # LETHAL FIX: increased from 2.0 to 2.5 seconds for 150 stocks
        
        # LETHAL FIX: For remaining stocks, use empty fundamentals (score will be lower but won't crash)
        for sym, _ in liquid_items[fetch_limit:]:
            fund_map[sym] = Fundamentals()

    core_results = []
    for sym, df in liquid.items():
        if len(df) < CONFIG["min_bars_required"]:
            continue
        r = evaluate_core(sym, df, fund_map.get(sym, Fundamentals()))
        if r:
            r["Market_Regime"] = market_regime.get("Regime", "NEUTRAL / CHOPPY")
            core_results.append(r)
    core_df = (pd.DataFrame(core_results).sort_values("Score", ascending=False)
               if core_results else pd.DataFrame())

    # LETHAL FIX: Scale down positions if total deployment exceeds portfolio size
    portfolio = CONFIG["portfolio_size_inr"]
    max_total_deploy = portfolio * CONFIG.get("max_total_deployment_pct", 0.95)
    
    for book_df in [swing_df, core_df]:
        if book_df.empty or "Deploy_₹" not in book_df.columns:
            continue
        total_deploy = book_df["Deploy_₹"].sum()
        if total_deploy > max_total_deploy:
            scale_factor = max_total_deploy / total_deploy
            log.warning(f"Scaling down {book_df.columns[0] if len(book_df.columns) > 0 else 'book'} positions by {scale_factor:.2f}x to fit portfolio")
            book_df["Deploy_₹"] = (book_df["Deploy_₹"] * scale_factor).round(0)
            book_df["Shares_To_Buy"] = (book_df["Shares_To_Buy"] * scale_factor).round(0)
            book_df["Max_Risk_₹"] = (book_df["Max_Risk_₹"] * scale_factor).round(0)
            book_df["%_of_Portfolio"] = (book_df["%_of_Portfolio"] * scale_factor).round(1)

    # FIX: Sector strength filter — only keep stocks in top-performing sectors
    if CONFIG.get("sector_strength_enabled", False) and not swing_df.empty and not core_df.empty:
        # Calculate 1-month return by sector
        sector_returns = {}
        for sym, df in liquid.items():
            if len(df) >= 21:
                ret_1m = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) * 100
                sector = get_sector_label(sym)
                if sector not in sector_returns:
                    sector_returns[sector] = []
                sector_returns[sector].append(ret_1m)

        avg_sector_returns = {s: np.mean(rs) for s, rs in sector_returns.items() if rs}
        top_sectors = sorted(avg_sector_returns, key=avg_sector_returns.get, reverse=True)[:CONFIG.get("sector_strength_top_n", 3)]

        swing_df = swing_df[swing_df["Sector_Label"].isin(top_sectors)]
        core_df = core_df[core_df["Sector_Label"].isin(top_sectors)]
        log.info(f"  → Sector filter: top {len(top_sectors)} sectors = {top_sectors}")

    # FIX: Save ALL passing stocks before truncating to top N
    max_picks = CONFIG.get("max_picks_per_book_per_snapshot", 5)
    if not swing_df.empty:
        all_swing_df = swing_df.copy()
        all_swing_df["Is_Top_Pick"] = all_swing_df.index.isin(swing_df.head(max_picks).index)
        all_swing_df.to_csv(CONFIG["csv_output_all_swing"], index=False)
        log.info(f"  ✓ All {len(all_swing_df)} swing passes saved to {CONFIG['csv_output_all_swing']}")
    if not core_df.empty:
        all_core_df = core_df.copy()
        all_core_df["Is_Top_Pick"] = all_core_df.index.isin(core_df.head(max_picks).index)
        all_core_df.to_csv(CONFIG["csv_output_all_core"], index=False)
        log.info(f"  ✓ All {len(all_core_df)} core passes saved to {CONFIG['csv_output_all_core']}")

    # FIX: Limit to top N picks per book to avoid over-screening
    if len(swing_df) > max_picks:
        swing_df = swing_df.head(max_picks)
        log.info(f"  → Top {max_picks} swing picks selected from {len(swing_results)}")
    else:
        log.info(f"  → {len(swing_df)} stocks passed swing criteria")

    if len(core_df) > max_picks:
        core_df = core_df.head(max_picks)
        log.info(f"  → Top {max_picks} core picks selected from {len(core_results)}")
    else:
        log.info(f"  → {len(core_df)} stocks passed core criteria")

    # ── BULK DEALS (real data only) ──
    log.info("Fetching real NSE bulk/block deals...")
    all_screened = set(swing_df["Symbol"].tolist() if not swing_df.empty else []) | \
                    set(core_df["Symbol"].tolist() if not core_df.empty else [])
    if bulk_deals_override is not None:
        bulk_df = bulk_deals_override
    else:
        bulk_df = fetch_nse_bulk_deals()
    matched_deals = match_bulk_deals_to_screened(bulk_df, all_screened)
    log.info(f"  → {len(matched_deals)} bulk/block deals matched to screened stocks "
             f"({'real NSE data' if not bulk_df.empty else 'feed unavailable today'})")

    # ── CONFIDENCE SCORING (rules-based, not backtested — see docstring) ──
    # NOTE: this MUST run before crossover construction below, so the
    # crossover dataframe (which is built FROM core_df) inherits the
    # Confidence_% and News columns rather than being built too early
    # and silently missing them.
    log.info("Attaching confidence/probability estimates...")
    swing_df, core_df = attach_confidence(swing_df, core_df)

    # ── NEWS JUSTIFICATION (real fetch, explicit "none found" if empty) ──
    if skip_news:
        log.info("News fetching skipped (skip_news=True)")
        if not swing_df.empty:
            swing_df["News_Status"] = "skipped"
            swing_df["News_Headlines"] = "Not fetched this run"
        if not core_df.empty:
            core_df["News_Status"] = "skipped"
            core_df["News_Headlines"] = "Not fetched this run"
    else:
        log.info(f"Fetching news for {CONFIG['news_fetch_for_grades']} tier picks "
                  "(this takes a while — one request per eligible stock)...")
        if not swing_df.empty:
            swing_df = attach_news_to_book(swing_df, force_eligible=True)
        if not core_df.empty:
            core_df = attach_news_to_book(core_df, force_eligible=False)
        log.info("  → News fetching complete")

    # ── CROSSOVER ── (built AFTER confidence + news, so it inherits both)
    crossover_df = pd.DataFrame()
    if not swing_df.empty and not core_df.empty:
        common = set(swing_df["Symbol"]) & set(core_df["Symbol"])
        if common:
            swing_extra_cols = ["Symbol", "Stop_Loss", "RR_Ratio"]
            if "Confidence_%" in swing_df.columns:
                swing_extra_cols.append("Confidence_%")
            crossover_df = core_df[core_df["Symbol"].isin(common)].merge(
                swing_df[swing_extra_cols].rename(
                    columns={"Stop_Loss": "Swing_SL", "RR_Ratio": "Swing_RR",
                              "Confidence_%": "Swing_Confidence_%"}), on="Symbol")
            # FIX Issue 9: Sort crossover by conviction (combined confidence)
            if "Confidence_%" in crossover_df.columns and "Swing_Confidence_%" in crossover_df.columns:
                crossover_df["Combined_Confidence"] = (
                    crossover_df["Confidence_%"].fillna(0) + crossover_df["Swing_Confidence_%"].fillna(0)
                ) / 2
                crossover_df = crossover_df.sort_values("Combined_Confidence", ascending=False)
            log.info(f"  → {len(crossover_df)} stocks in BOTH books")

    # ── REPORT ──
    if EXISTING_HOLDINGS:
        log.info(f"Reconciling against {len(EXISTING_HOLDINGS)} existing holding(s) in EXISTING_HOLDINGS...")
    swing_df = reconcile_with_holdings(swing_df)
    core_df = reconcile_with_holdings(core_df)
    crossover_df = reconcile_with_holdings(crossover_df)

    print_report(swing_df, core_df, crossover_df, matched_deals, macro_events, market_regime)

    # ── EXPORT ──
    try:
        if not swing_df.empty: swing_df.to_csv(CONFIG["csv_output_swing"], index=False)
        if not core_df.empty: core_df.to_csv(CONFIG["csv_output_core"], index=False)
        if not crossover_df.empty: crossover_df.to_csv(CONFIG["csv_output_crossover"], index=False)
        if not matched_deals.empty: matched_deals.to_csv(CONFIG["csv_output_bulkdeals"], index=False)
        log.info("✓ CSVs exported")
    except Exception as e:
        log.warning(f"CSV export skipped: {e}")

    # LETHAL FIX: Cache screener results for next run
    if _CACHE_AVAILABLE:
        results_to_cache = (
            swing_df.to_dict(orient="records") if not swing_df.empty else [],
            core_df.to_dict(orient="records") if not core_df.empty else [],
            crossover_df.to_dict(orient="records") if not crossover_df.empty else [],
            matched_deals.to_dict(orient="records") if not matched_deals.empty else [],
            market_regime,
        )
        cache_screener_results(results_to_cache, key="latest", ttl_days=1)
        log.info("✓ Screener results cached to disk")

    return swing_df, core_df, crossover_df, matched_deals, market_regime


def print_report(swing_df, core_df, crossover_df, bulk_deals_df, macro_events, market_regime=None):
    print_market_regime(market_regime)

    print("\n" + "═" * 95)
    print("  📊 SCREENER RESULTS")
    print("═" * 95)

    print(f"\n{'─'*95}\n  💼 YOUR EXISTING HOLDINGS — MATCHED AGAINST THIS RUN'S LISTS\n{'─'*95}")
    if not EXISTING_HOLDINGS:
        print("  EXISTING_HOLDINGS is empty — add your positions near the top of the script")
        print("  (Symbol -> (qty, avg_price)) and re-run to see this section populated.")
        print("  Until then, all sizing below assumes you're starting from zero in every stock.")
    else:
        any_match = False
        for book_name, book_df in [("Swing (6mo)", swing_df), ("Core (1-2yr)", core_df)]:
            if book_df.empty or "You_Hold" not in book_df.columns:
                continue
            held = book_df[book_df["You_Hold"] == "YES"]
            if held.empty:
                continue
            any_match = True
            print(f"\n  [{book_name} book — you hold {len(held)} of these]")
            for _, r in held.iterrows():
                pnl_sign = "+" if r["Unrealized_PnL_%"] >= 0 else ""
                print(f"    {r['Symbol']}: {r['Held_Qty']} qty @ avg ₹{r['Held_Avg']:.2f} | "
                      f"CMP ₹{r['CMP']:.2f} | Unrealized P&L {pnl_sign}{r['Unrealized_PnL_%']}%")
                print(f"      → Target ₹{r['Target']:.2f} | Stop Loss ₹{r['Stop_Loss']:.2f} | "
                      f"Room to add: {r.get('Additional_Shares_Room', 'N/A')} more shares "
                      f"(₹{r.get('Additional_Deploy_₹', 0):,.0f}) before hitting your risk caps")
        if not any_match:
            print(f"  None of your {len(EXISTING_HOLDINGS)} configured holdings appear in")
            print("  this run's qualifying lists. They may have failed the screen, or you")
            print("  hold stocks outside the scanned universe.")

    print(f"\n{'─'*95}\n  🌍 MACRO CONTEXT\n{'─'*95}")
    if macro_events:
        for e in macro_events:
            print(f"  [{e['source']}] {e['note']}")
    else:
        print("  No live macro calendar data available this run.")
        print("  Manually check: RBI.org.in (MPC dates), federalreserve.gov (FOMC), "
              "tradingeconomics.com/calendar")
        print("  (This is intentional — showing a hardcoded date here would silently go")
        print("   stale and could mislead you weeks later. Better to say 'check yourself.')")

    print(f"\n{'─'*95}\n  🎯 CROSSOVER — HIGHEST CONVICTION (passed BOTH books)\n{'─'*95}")
    if crossover_df.empty:
        print("  None today. Normal — crossover hits are rare by design.")
    else:
        cols = ["Symbol", "CMP", "Grade", "Score", "Confidence_%", "Swing_Confidence_%",
                 "Target", "Profit_%", "Stop_Loss", "Risk_%", "RR_Ratio",
                 "Shares_To_Buy", "Deploy_₹", "Max_Risk_₹", "Trailing_Stop",
                 "Profit_Booking_1", "Profit_Booking_2"]
        cols = [c for c in cols if c in crossover_df.columns]
        print(crossover_df[cols].to_string(index=False))

    print(f"\n{'─'*95}\n  ⚡ SWING BOOK — 3-6 MONTH HOLD (Minervini Trend Template + VCP)\n{'─'*95}")
    if swing_df.empty:
        print("  No stocks passed the strict trend template + VCP filter today.")
    else:
        cols = ["Symbol", "CMP", "Target", "Profit_%", "Stop_Loss", "Risk_%", "RR_Ratio",
                 "Confidence_%", "Shares_To_Buy", "Deploy_₹", "Max_Risk_₹", "%_of_Portfolio",
                 "Trailing_Stop", "Profit_Booking_1", "Profit_Booking_2"]
        cols = [c for c in cols if c in swing_df.columns]
        print(swing_df[cols].head(10).to_string(index=False))
        if len(swing_df) > 0:
            top = swing_df.iloc[0]
            print(f"\n  🏆 TOP SWING PICK: {top['Symbol']} — Entry ₹{top['CMP']:.0f} → "
                  f"Target ₹{top['Target']:.0f} | SL ₹{top['Stop_Loss']:.0f} | "
                  f"R:R 1:{top['RR_Ratio']:.1f}")
            if "Shares_To_Buy" in top:
                print(f"     Buy {top['Shares_To_Buy']:.0f} shares | "
                      f"Deploy ₹{top['Deploy_₹']:,.0f} | Max Risk ₹{top['Max_Risk_₹']:,.0f}")
            if "Trailing_Stop" in top:
                print(f"     Trailing Stop: ₹{top['Trailing_Stop']:.0f} | "
                      f"Profit Booking 1: ₹{top['Profit_Booking_1']:.0f} | "
                      f"Profit Booking 2: ₹{top['Profit_Booking_2']:.0f}")
            if "Confidence_%" in top:
                print(f"     Confidence: {top['Confidence_%']:.0f}% (rules-based heuristic, "
                      f"NOT a backtested win-rate — see Confidence_Factors column)")
            if "News_Headlines" in top:
                print(f"     News: {top['News_Headlines'][:300]}")

        # Per-stock justification block for top 5
        print(f"\n  📋 JUSTIFICATION — Top 5 Swing Picks")
        for _, row in swing_df.head(5).iterrows():
            print(f"\n  ▸ {row['Symbol']} — Confidence {row.get('Confidence_%', 'N/A')}%")
            print(f"    Quant: {row.get('Confidence_Factors', 'N/A')}")
            news_status = row.get("News_Status", "not_fetched")
            if news_status == "ok":
                print(f"    News:  {row['News_Headlines']}")
            else:
                print(f"    News:  {row.get('News_Headlines', 'Not available')}")

    print(f"\n{'─'*95}\n  📈 CORE BOOK — 1-2 YEAR HOLD (Quality+Growth+Technical+Momentum)\n{'─'*95}")
    if core_df.empty:
        print("  No stocks passed the core scoring filter today.")
    else:
        cols = ["Symbol", "CMP", "Grade", "Score", "Confidence_%", "Target", "Profit_%",
                 "Stop_Loss", "Risk_%", "RR_Ratio", "PE", "ROE_%", "EPS_Growth_%",
                 "Shares_To_Buy", "Deploy_₹"]
        cols = [c for c in cols if c in core_df.columns]
        for grade in ["STRONG BUY", "BUY", "WATCH"]:
            sub = core_df[core_df["Grade"] == grade]
            if not sub.empty:
                print(f"\n  [{grade}] — {len(sub)} stocks")
                print(sub[cols].head(10).to_string(index=False))
        if not core_df.empty:
            top = core_df.iloc[0]
            print(f"\n  🏆 TOP CORE PICK: {top['Symbol']} — Entry ₹{top['CMP']:.0f} → "
                  f"Target ₹{top['Target']:.0f} | SL ₹{top['Stop_Loss']:.0f} | "
                  f"Score {top['Score']}/100")
            if "Shares_To_Buy" in top:
                print(f"     Buy {top['Shares_To_Buy']:.0f} shares | "
                      f"Deploy ₹{top['Deploy_₹']:,.0f} | Max Risk ₹{top['Max_Risk_₹']:,.0f}")
            if "Confidence_%" in top:
                print(f"     Confidence: {top['Confidence_%']:.0f}% (rules-based heuristic, "
                      f"NOT a backtested win-rate — see Confidence_Factors column)")

        # Per-stock justification block for top 5 STRONG BUY / BUY picks
        justifiable = core_df[core_df["Grade"].isin(["STRONG BUY", "BUY"])].head(5)
        if not justifiable.empty:
            print(f"\n  📋 JUSTIFICATION — Top 5 Core Picks (STRONG BUY / BUY tier)")
            for _, row in justifiable.iterrows():
                print(f"\n  ▸ {row['Symbol']} — Score {row.get('Score','N/A')}/100, "
                      f"Confidence {row.get('Confidence_%', 'N/A')}%")
                print(f"    Quant: {row.get('Reasons', 'N/A')}")
                print(f"    Confidence basis: {row.get('Confidence_Factors', 'N/A')}")
                news_status = row.get("News_Status", "not_fetched")
                if news_status == "ok":
                    print(f"    News:  {row['News_Headlines']}")
                else:
                    print(f"    News:  {row.get('News_Headlines', 'Not available')}")

    print(f"\n{'─'*95}\n  💼 BULK/BLOCK DEALS — MATCHED AGAINST YOUR SCREENED STOCKS\n{'─'*95}")
    if bulk_deals_df.empty:
        print("  No matches today. This means either:")
        print("    (a) NSE's bulk deal feed had no large trades in your screened stocks, or")
        print("    (b) the live feed was unavailable when this ran (network/rate-limit)")
        print("  Either way: NOT showing a fabricated example. Check nseindia.com/market-data/")
        print("  large-deals directly if you want to verify which scenario applies.")
    else:
        print(bulk_deals_df.to_string(index=False))

    # FIX: Portfolio allocation summary
    print(f"\n{'─'*95}\n  💰 PORTFOLIO ALLOCATION SUMMARY\n{'─'*95}")
    total_deploy = 0
    total_risk = 0
    for book_name, book_df in [("Swing (3-6mo)", swing_df), ("Core (1-2yr)", core_df)]:
        if book_df.empty:
            continue
        book_deploy = book_df["Deploy_₹"].sum() if "Deploy_₹" in book_df.columns else 0
        book_risk = book_df["Max_Risk_₹"].sum() if "Max_Risk_₹" in book_df.columns else 0
        total_deploy += book_deploy
        total_risk += book_risk
        print(f"  {book_name}: {len(book_df)} stocks | "
              f"Total Deploy: ₹{book_deploy:,.0f} | Total Max Risk: ₹{book_risk:,.0f}")
    print(f"  ─────────────────────────────────────────────────────────")
    print(f"  TOTAL: {len(swing_df) + len(core_df)} stocks | "
          f"Total Deploy: ₹{total_deploy:,.0f} | Total Max Risk: ₹{total_risk:,.0f}")
    print(f"  Portfolio size: ₹{CONFIG['portfolio_size_inr']:,.0f} | "
          f"Risk per trade: {CONFIG['risk_per_trade_pct']*100:.1f}%")
    if total_deploy > 0:
        deploy_pct = total_deploy/CONFIG['portfolio_size_inr']*100
        print(f"  Deployment %: {deploy_pct:.1f}% of portfolio", end="")
        if deploy_pct > 100:
            print(f"  ⚠️  OVER-ALLOCATED! Reduce position sizes or number of picks.")
        else:
            print()

    # FIX Task 1: show near-miss stocks ranked by closeness to passing
    if NEAR_MISS_LOG:
        print(f"\n{'─'*95}\n  🎯 NEAR-MISS PICKS — ranked by closeness to passing the screen\n{'─'*95}")
        sorted_misses = sorted(NEAR_MISS_LOG.items(), key=lambda x: x[1]["score"], reverse=True)[:10]
        for sym, data in sorted_misses:
            print(f"  {sym:<18} score={data['score']:>3}  {data['reason']}  [{data['category']}]")
        print(f"  (These stocks almost passed — worth watching for a re-test)")

    print(f"\n{'─'*95}\n  📋 REJECTION SUMMARY (first 15 — full reasons logged for every excluded stock)\n{'─'*95}")
    for sym, reason in list(REJECTION_LOG.items())[:15]:
        print(f"  {sym:<18} {reason}")
    if len(REJECTION_LOG) > 15:
        print(f"  ... and {len(REJECTION_LOG) - 15} more")

    print("\n" + "═" * 95)
    print("  ⚠️  Not SEBI-registered investment advice. Verify independently before trading.")
    print("  ⚠️  Position sizing assumes the portfolio_size_inr in CONFIG is accurate —")
    print("      update it before each run if your capital has changed.")
    print("═" * 95 + "\n")


if __name__ == "__main__":
    run_screener()
