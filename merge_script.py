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
  pip install yfinance pandas pandas_ta numpy scipy requests feedparser --break-system-packages
"""

import time
import os
import hashlib
import threading
import json
import datetime
import warnings
import traceback
import logging
import html
import re
from dataclasses import dataclass, field, asdict
from typing import Optional
from io import StringIO
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Run: pip install yfinance --break-system-packages")

try:
    import pandas_ta as ta
except ImportError:
    raise SystemExit("Run: pip install pandas_ta --break-system-packages")

try:
    import feedparser
except ImportError:
    raise SystemExit("Run: pip install feedparser --break-system-packages")

from scipy.signal import argrelextrema

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "swing_atr_sl_multiplier": 1.5,
    "swing_vcp_lookback_days": 60,
    "swing_vcp_min_peaks": 2,
    "swing_vcp_peak_order": 5,
    "swing_target_rr_min": 2.0,

    # ── CORE BOOK (Quality+Growth+Technical+Momentum) ───────────────────
    "core_score_strong_buy": 62,
    "core_score_buy": 48,
    "core_score_watch": 36,
    "core_rsi_ideal_low": 44,
    "core_rsi_ideal_high": 62,
    "core_reject_earnings_growth_below": -25,

    # ── POSITION SIZING ──────────────────────────────────────────────────
    # CHANGE THESE TWO TO MATCH YOUR ACTUAL PORTFOLIO
    "portfolio_size_inr": 136000,      # your current portfolio value
    "risk_per_trade_pct": 0.01,        # risk 1% of portfolio per trade
    "max_single_position_pct": 0.10,   # never deploy >10% of portfolio in one stock

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
    "news_request_pause_sec": 0.6,     # be gentle on Google News RSS
    "news_timeout_sec": 6,
    "news_log_progress_every": 5,      # NEW v5 — log "News X/Y: SYMBOL" every N stocks so it never looks frozen

    # ── DISK CACHE TTLs — NEW v6, see cache_get/cache_set for rationale ──
    "universe_cache_hours": 24,       # Nifty 500 list — set 0 to force a fresh fetch every run
    "fundamentals_cache_hours": 24,   # per-symbol PE/ROE/growth/etc — set 0 to disable
    "earnings_cache_hours": 48,       # FIX v7 — was in-memory-per-run only; now persists to disk
    "earnings_lookup_timeout_sec": 6, # FIX v7 — fail fast instead of hanging on stocks with no data

    # ── NSE SESSION WARMUP — NEW v5 (fix: intermittent 401s from bulk deals) ──
    "nse_cookie_warmup_pause_sec": 2.0,

    # ── TRUE VCP CONTRACTION DETECTION — NEW v5 ─────────────────────────
    # Replaces "count the peaks" with real Minervini-style volatility
    # contraction detection: a real VCP is a SEQUENCE of pullbacks that
    # get smaller each time (e.g. 20% -> 12% -> 7% -> 4%), on shrinking
    # volume, while price holds near its highs. Counting peaks alone (the
    # old v4 method) can't tell a genuine tightening base from random
    # noisy chop with the same peak count.
    "vcp_min_contractions": 2,          # need at least this many sequential pullbacks
    "vcp_contraction_tolerance": 1.15,   # each contraction must be <= previous * this (some noise allowed)
    "vcp_max_pct_off_recent_peak": 0.15, # price must stay within 15% of the most recent base peak
    "vcp_volume_shrink_required": True,  # reject if avg volume isn't shrinking contraction-over-contraction

    # ── RELATIVE STRENGTH RANK — NEW v5 (highest priority upgrade) ──────
    "rs_lookback_days": 252,            # ~12 months of trading days
    "swing_min_rs_rank": 80,            # swing requires top-20% 12M return vs the scanned universe
    "core_strong_buy_min_rs_rank": 90,  # optional extra bar for STRONG BUY tier in core book

    # ── SECTOR RELATIVE STRENGTH — NEW v5 ───────────────────────────────
    "sector_rs_top20_bonus": 5,
    "sector_rs_top10_bonus": 10,
    "sector_rs_bottom20_penalty": -5,

    # ── SECTOR LEADER BONUS — NEW v5 ────────────────────────────────────
    "sector_leader_mcap_bonus": 4,
    "sector_leader_rs_top3_bonus": 4,
    "sector_leader_roe_bonus": 4,

    # ── REGIME-BASED POSITION SIZING — NEW v5 ───────────────────────────
    # Multiplies the normal risk-based position size by how supportive
    # the broad market backdrop is. This does NOT reject picks — it only
    # scales down (or to zero) how much capital gets deployed into them
    # when the market itself looks weak, same "informational, not a hard
    # block" philosophy as the regime banner itself.
    "regime_position_multiplier": {
        "STRONG_BULL": 1.00,   # our "BULLISH — BREAKOUT"
        "BULL": 0.75,          # our "BULLISH — TREND INTACT"
        "RECOVERY": 0.60,      # our "RECOVERY"
        "NEUTRAL": 0.50,       # our "NEUTRAL / CHOPPY" (upper half, RSI >= 50)
        "CHOPPY": 0.25,        # our "NEUTRAL / CHOPPY" (lower half, RSI < 50)
        "BEAR": 0.00,          # our "BEARISH — TREND DOWN" / "BEARISH — BREAKDOWN"
    },

    # ── MACRO RISK OVERLAY — NEW v5 ──────────────────────────────────────
    # HONESTY NOTE: VIX (^INDIAVIX), Brent (BZ=F), USDINR (INR=X), and
    # "breadth" (computed live from OUR OWN downloaded universe — real,
    # free, no shortcuts) are all genuinely fetched. FII/DII net flows
    # are attempted via NSE's public endpoint on a best-effort basis
    # (same fragile pattern as bulk deals — NSE can block/rate-limit
    # outside a browser) and DEGRADE HONESTLY (excluded from the score,
    # not defaulted to a fake neutral value) if unavailable. Macro_Score
    # is the average of whichever components actually came back — this
    # run might use 4/6 components, not always all 6 — and the report
    # says which ones were used.
    "macro_vix_ticker": "^INDIAVIX",
    "macro_brent_ticker": "BZ=F",
    "macro_usdinr_ticker": "INR=X",
    "macro_trend_window_days": 20,
    "macro_allocation_table": [   # (min_score, allocation_pct)
        (80, 1.00), (60, 0.75), (40, 0.50), (20, 0.25), (0, 0.00),
    ],

    # ── IMPROVED BREAKOUT VALIDATION — NEW v5 ───────────────────────────
    "breakout_min_volume_ratio": 1.5,     # today's volume vs 20DMA volume
    "breakout_min_close_position": 0.75,  # close must be in top 25% of the day's range
    "breakout_min_pct_above_pivot": 0.5,   # close must clear the pivot by at least this %

    # ── EARNINGS EVENT PROTECTION — NEW v5 ──────────────────────────────
    "earnings_blackout_days": 5,    # reject new entries if earnings fall within N trading days
    "earnings_check_enabled": True, # best-effort — yfinance calendar coverage is inconsistent

    # ── ATR-BASED STOP (v5: 20-period, structural-low-aware) ───────────
    "atr_period_v5": 20,
    "atr_multiplier_v5": 2.0,

    # ── PORTFOLIO EXPOSURE RULES — NEW v5 ───────────────────────────────
    "portfolio_max_sector_exposure_pct": 0.25,   # no more than 25% of capital in one sector
    "portfolio_max_positions_per_sector": 3,
    "portfolio_max_heat_pct": 0.06,   # sum of (risk% x position weight) across ALL open positions

    # Output
    "csv_output_swing": "/mnt/user-data/outputs/swing_book_3_6mo.csv",
    "csv_output_core": "/mnt/user-data/outputs/core_book_1_2yr.csv",
    "csv_output_crossover": "/mnt/user-data/outputs/crossover_highest_conviction.csv",
    "csv_output_bulkdeals": "/mnt/user-data/outputs/bulk_deals_matched.csv",
    "csv_output_portfolio": "/mnt/user-data/outputs/final_portfolio_selection.csv",
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
# LIGHTWEIGHT DISK CACHE — NEW v6
# ═════════════════════════════════════════════════════════════════════════
# ANSWERS THE CACHING QUESTION: not everything needs to be fresh every run.
#
#   SHOULD be cached (implemented below):
#   - Nifty 500 constituent LIST — reconstituted quarterly by NSE, not
#     daily. Re-downloading and re-parsing this CSV every single run for
#     a list that changed maybe twice this year is pure waste.
#   - Fundamentals (PE, ROE, growth, debt, etc.) — companies report
#     results quarterly. A yfinance .info call per stock is one of the
#     slower parts of a full run (500 individual requests); this data
#     is stale within hours of being fetched anyway, so re-fetching it
#     hourly buys you nothing.
#   - Earnings blackout dates — already cached in-memory per-run
#     (_EARNINGS_CACHE above); extended here to persist across runs too,
#     since a company's next earnings date doesn't change hour-to-hour.
#
#   Should NOT be cached (left as-is, fetched fresh every run):
#   - OHLC price data — this is explicitly the one thing that must be
#     current; caching this would mean trading on stale prices.
#   - Bulk/block deals, news headlines — these are specifically about
#     "what happened recently," caching defeats their entire purpose.
#   - Macro overlay (VIX/Brent/USDINR/FII-DII/breadth) — changes daily,
#     and the whole point of the regime/macro checks is today's read.
#
# Default TTLs below are conservative starting points — tune them in
# CONFIG if you want to be more/less aggressive about freshness.
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".screener_cache")


def _cache_path(key: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    safe_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(_CACHE_DIR, f"{safe_key}.json")


def cache_get(key: str, max_age_hours: float):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        cached_at = payload.get("_cached_at", 0)
        if (time.time() - cached_at) / 3600 > max_age_hours:
            return None   # expired — caller will re-fetch and overwrite
        return payload.get("data")
    except Exception:
        return None   # any corruption/parse issue -> treat as cache miss, never crash on it


def cache_set(key: str, data):
    path = _cache_path(key)
    try:
        with open(path, "w") as f:
            json.dump({"_cached_at": time.time(), "data": data}, f)
    except Exception:
        pass   # caching is a nice-to-have, never let it break the actual run


# ═════════════════════════════════════════════════════════════════════════
# 1. UNIVERSE FETCH
# ═════════════════════════════════════════════════════════════════════════

def get_nifty_500_universe() -> list[str]:
    cached = cache_get("nifty_500_universe", CONFIG.get("universe_cache_hours", 24))
    if cached is not None:
        log.info(f"✓ Using CACHED Nifty 500 list ({len(cached)} symbols, "
                  f"< {CONFIG.get('universe_cache_hours', 24)}h old) — skip with "
                  f"CONFIG['universe_cache_hours']=0 to force a fresh fetch")
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
            cache_set("nifty_500_universe", symbols)
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


def fetch_fundamentals(symbol: str) -> Fundamentals:
    cache_key = f"fundamentals_{symbol}"
    cached = cache_get(cache_key, CONFIG.get("fundamentals_cache_hours", 24))
    if cached is not None:
        return Fundamentals(**cached)

    try:
        info = yf.Ticker(f"{symbol}.NS").info
    except Exception as e:
        REJECTION_LOG[f"{symbol}_fundamentals"] = f"info fetch failed: {e}"
        return Fundamentals()

    def safe_pct(key):
        v = info.get(key)
        return round(v * 100, 2) if isinstance(v, (int, float)) else None

    def safe_num(key, round_to=2):
        v = info.get(key)
        return round(v, round_to) if isinstance(v, (int, float)) else None

    result = Fundamentals(
        pe=safe_num("trailingPE"), fwd_pe=safe_num("forwardPE"),
        pb=safe_num("priceToBook"), roe=safe_pct("returnOnEquity"),
        rev_growth=safe_pct("revenueGrowth"), earn_growth=safe_pct("earningsGrowth"),
        debt_eq=safe_num("debtToEquity", 3), div_yield=safe_pct("dividendYield"),
        beta=safe_num("beta"),
        mcap_cr=round(info.get("marketCap", 0) / 1e7, 0) if info.get("marketCap") else None,
    )
    cache_set(cache_key, asdict(result))
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
                              regime_multiplier: float = 1.0) -> Optional[dict]:
    """
    Risk-based position sizing: never risk more than risk_per_trade_pct of
    total portfolio on a single trade, AND never deploy more than
    max_single_position_pct of portfolio into one stock (even if the stop
    is tight enough that pure risk-sizing would suggest a bigger position).
    Both caps are enforced — whichever gives the SMALLER position wins.

    NEW v5: regime_multiplier scales the final share count down (never up)
    based on how supportive the Nifty market regime is right now — see
    CONFIG["regime_position_multiplier"]. A multiplier of 0 means the
    regime says BEAR — the stock can still show up in the report (so you
    know it exists), but sized to zero new capital.
    """
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return None

    portfolio = CONFIG["portfolio_size_inr"]
    max_risk_amount = portfolio * CONFIG["risk_per_trade_pct"]
    max_position_value = portfolio * CONFIG["max_single_position_pct"]

    shares_by_risk = int(max_risk_amount // risk_per_share)
    shares_by_cap  = int(max_position_value // entry_price)
    shares_to_buy  = max(0, min(shares_by_risk, shares_by_cap))
    shares_to_buy  = int(shares_to_buy * max(0.0, min(1.0, regime_multiplier)))

    if shares_to_buy == 0:
        return {
            "shares": 0, "deploy": 0, "max_risk": 0,
            "regime_multiplier": regime_multiplier,
            "note": ("Position size rounds to 0 shares at current portfolio size, or "
                     "regime multiplier scaled it to 0 — skip or paper-trade")
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
        "regime_multiplier": regime_multiplier,
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

    Stocks you don't hold are unaffected; their sizing stays as a
    from-zero calculation, exactly as before.
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
            continue

        qty, avg = EXISTING_HOLDINGS[sym]
        unrealized = round((cmp_price - avg) / avg * 100, 1)

        you_hold.append("YES")
        held_qty.append(qty)
        held_avg.append(avg)
        pnl_pct.append(unrealized)

        # Recompute sizing as an INCREMENT: how much MORE could you add
        # without exceeding the same risk_per_trade and max_position caps,
        # accounting for the rupee value you already have deployed here.
        stop_loss = row.get("Stop_Loss")
        if stop_loss is None or cmp_price <= stop_loss:
            incremental_shares.append(0)
            incremental_deploy.append(0)
            continue

        risk_per_share = cmp_price - stop_loss
        portfolio = CONFIG["portfolio_size_inr"]
        max_risk_amount = portfolio * CONFIG["risk_per_trade_pct"]
        max_position_value = portfolio * CONFIG["max_single_position_pct"]

        already_deployed = qty * cmp_price  # current value at CMP, not avg
        room_left_value = max(0, max_position_value - already_deployed)

        shares_by_risk = int(max_risk_amount // risk_per_share)
        shares_by_room = int(room_left_value // cmp_price)
        add_shares = max(0, min(shares_by_risk, shares_by_room))

        incremental_shares.append(add_shares)
        incremental_deploy.append(round(add_shares * cmp_price, 2))

    df["You_Hold"] = you_hold
    df["Held_Qty"] = held_qty
    df["Held_Avg"] = held_avg
    df["Unrealized_PnL_%"] = pnl_pct
    df["Additional_Shares_Room"] = incremental_shares
    df["Additional_Deploy_₹"] = incremental_deploy
    return df


# ═════════════════════════════════════════════════════════════════════════
# 6. SWING BOOK — Minervini Trend Template + VCP + ATR stop (3-6 months)
# ═════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════
# TRUE VCP CONTRACTION DETECTION — NEW v5
# ═════════════════════════════════════════════════════════════════════════
# Replaces "count how many peaks exist" (v4, easily fooled by random
# choppy noise with the same peak count) with what a VCP actually is: a
# SEQUENCE of pullbacks that get smaller each time, on shrinking volume,
# while price holds near its highs. This is the real Minervini definition.

def detect_vcp_pattern(df: pd.DataFrame) -> dict:
    """
    Walks the recent price action for a genuine volatility-contraction
    sequence. Returns a dict with valid=True/False and the reason either
    way — never silently guesses.
    """
    lookback = CONFIG["swing_vcp_lookback_days"]
    order = CONFIG["swing_vcp_peak_order"]
    window = df.tail(lookback).reset_index(drop=True)
    if len(window) < lookback // 2:
        return {"valid": False, "reason": "Not enough bars in VCP lookback window",
                "contractions": [], "volumes": [], "base_low": None}

    closes = window["Close"].values
    volumes = window["Volume"].values

    peak_idx = list(argrelextrema(closes, np.greater, order=order)[0])
    trough_idx = list(argrelextrema(closes, np.less, order=order)[0])

    # Merge chronologically and collapse consecutive same-type extrema to
    # the single most extreme one, so we get a clean alternating P,T,P,T...
    combined = sorted([(i, "P") for i in peak_idx] + [(i, "T") for i in trough_idx])
    cleaned = []
    for idx, kind in combined:
        if cleaned and cleaned[-1][1] == kind:
            prev_idx, _ = cleaned[-1]
            if kind == "P" and closes[idx] > closes[prev_idx]:
                cleaned[-1] = (idx, kind)
            elif kind == "T" and closes[idx] < closes[prev_idx]:
                cleaned[-1] = (idx, kind)
            # else: keep the existing more-extreme one, drop this one
        else:
            cleaned.append((idx, kind))

    # Build contraction segments: each PEAK followed by the next TROUGH
    contractions = []   # list of (pct_decline, avg_volume_in_segment, peak_idx, trough_idx)
    for i in range(len(cleaned) - 1):
        idx_a, kind_a = cleaned[i]
        idx_b, kind_b = cleaned[i + 1]
        if kind_a == "P" and kind_b == "T":
            peak_price, trough_price = closes[idx_a], closes[idx_b]
            if peak_price <= 0:
                continue
            pct_decline = (peak_price - trough_price) / peak_price * 100
            seg_vol = float(np.mean(volumes[idx_a:idx_b + 1])) if idx_b > idx_a else float(volumes[idx_a])
            contractions.append({"pct": round(pct_decline, 1), "avg_volume": seg_vol,
                                   "peak_idx": idx_a, "trough_idx": idx_b})

    base_low = float(window["Low"].min())

    if len(contractions) < CONFIG["vcp_min_contractions"]:
        return {"valid": False,
                 "reason": f"Only {len(contractions)} sequential pullback(s) found "
                           f"(need {CONFIG['vcp_min_contractions']}+) — not a formed base",
                 "contractions": contractions, "volumes": [c["avg_volume"] for c in contractions],
                 "base_low": base_low}

    # Contractions must be SHRINKING (each later one <= previous * tolerance)
    tol = CONFIG["vcp_contraction_tolerance"]
    shrinking = all(contractions[i]["pct"] <= contractions[i - 1]["pct"] * tol
                     for i in range(1, len(contractions)))
    if not shrinking:
        pcts = [c["pct"] for c in contractions]
        return {"valid": False, "reason": f"Contractions not shrinking: {pcts}",
                 "contractions": contractions, "volumes": [c["avg_volume"] for c in contractions],
                 "base_low": base_low}

    # Volume must also be drying up contraction-over-contraction
    if CONFIG["vcp_volume_shrink_required"] and len(contractions) >= 2:
        vols = [c["avg_volume"] for c in contractions]
        vol_shrinking = all(vols[i] <= vols[i - 1] * tol for i in range(1, len(vols)))
        if not vol_shrinking:
            return {"valid": False, "reason": f"Volume not drying up across contractions: "
                                                f"{[round(v) for v in vols]}",
                     "contractions": contractions, "volumes": vols, "base_low": base_low}

    # Price must still be near the most recent base peak (not a base that
    # already broke down and is now far below its own structure)
    last_peak_idx = contractions[-1]["peak_idx"]
    last_peak_price = closes[last_peak_idx]
    current_close = closes[-1]
    pct_off_peak = (last_peak_price - current_close) / last_peak_price
    if pct_off_peak > CONFIG["vcp_max_pct_off_recent_peak"]:
        return {"valid": False,
                 "reason": f"{pct_off_peak*100:.1f}% below most recent base peak "
                           f"(max allowed {CONFIG['vcp_max_pct_off_recent_peak']*100:.0f}%)",
                 "contractions": contractions, "volumes": [c["avg_volume"] for c in contractions],
                 "base_low": base_low}

    return {"valid": True, "reason": "Valid VCP — shrinking pullbacks + shrinking volume, near highs",
             "contractions": contractions, "volumes": [c["avg_volume"] for c in contractions],
             "base_low": base_low}


def validate_breakout_strength(df: pd.DataFrame, base_low: float) -> dict:
    """
    NEW v5 — a "trend template pass + VCP formed" stock can still be a
    weak/unconfirmed breakout. Requires the classic three confirmations:
    volume surge, close near the day's high (buyers in control into the
    close, not a wick-and-fade), and a clean break above the base pivot
    by a meaningful margin (not just barely poking above by a few paise).
    """
    last = df.iloc[-1]
    vol_sma20 = df["Volume"].iloc[-21:-1].mean()
    vol_ratio = last["Volume"] / vol_sma20 if vol_sma20 > 0 else 0

    day_range = last["High"] - last["Low"]
    close_position = ((last["Close"] - last["Low"]) / day_range) if day_range > 0 else 0.5

    pivot = df["Close"].iloc[-CONFIG["swing_vcp_lookback_days"]:-1].max()
    pct_above_pivot = (last["Close"] - pivot) / pivot * 100 if pivot > 0 else 0

    checks = {
        "volume_confirmed": vol_ratio >= CONFIG["breakout_min_volume_ratio"],
        "close_near_high": close_position >= CONFIG["breakout_min_close_position"],
        "cleared_pivot": pct_above_pivot >= CONFIG["breakout_min_pct_above_pivot"],
    }
    return {"valid": all(checks.values()), "checks": checks,
             "vol_ratio": round(vol_ratio, 2), "close_position": round(close_position, 2),
             "pct_above_pivot": round(pct_above_pivot, 2)}


def evaluate_swing(symbol: str, df: pd.DataFrame, rs_rank: Optional[float] = None,
                     regime_multiplier: float = 1.0) -> Optional[dict]:
    try:
        df = df.copy()
        df["SMA_50"]  = ta.sma(df["Close"], length=50)
        df["SMA_150"] = ta.sma(df["Close"], length=150)
        df["SMA_200"] = ta.sma(df["Close"], length=200)
        df["ATR_14"]  = ta.atr(df["High"], df["Low"], df["Close"], length=CONFIG["swing_atr_period"])
        df["ATR_V5"]  = ta.atr(df["High"], df["Low"], df["Close"], length=CONFIG["atr_period_v5"])

        if df[["SMA_200", "ATR_14", "ATR_V5"]].iloc[-1].isna().any():
            REJECTION_LOG[symbol] = "Swing: indicators NaN (insufficient clean history)"
            return None

        close = df["Close"].iloc[-1]
        low_52w  = df["Low"].rolling(252).min().iloc[-1]
        high_52w = df["High"].rolling(252).max().iloc[-1]

        sma50, sma150, sma200 = df["SMA_50"].iloc[-1], df["SMA_150"].iloc[-1], df["SMA_200"].iloc[-1]
        sma200_1mo_ago = df["SMA_200"].iloc[-22]

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
        if not all(conditions.values()):
            failed = [k for k, v in conditions.items() if not v]
            REJECTION_LOG[symbol] = f"Swing: failed trend template ({passed}/7) — {failed}"
            return None

        # ── RS Rank gate — NEW v5, highest-priority upgrade ─────────────
        if rs_rank is not None and rs_rank < CONFIG["swing_min_rs_rank"]:
            REJECTION_LOG[symbol] = (f"Swing: RS_Rank {rs_rank} below required "
                                       f"{CONFIG['swing_min_rs_rank']} — trend template alone "
                                       "isn't enough, needs real relative strength vs the universe")
            return None

        # ── True VCP contraction sequence — NEW v5, replaces peak-counting ──
        vcp = detect_vcp_pattern(df)
        if not vcp["valid"]:
            REJECTION_LOG[symbol] = f"Swing: VCP invalid — {vcp['reason']}"
            return None

        # ── Breakout strength validation — NEW v5 ───────────────────────
        breakout = validate_breakout_strength(df, vcp["base_low"])
        if not breakout["valid"]:
            failed_checks = [k for k, v in breakout["checks"].items() if not v]
            REJECTION_LOG[symbol] = (f"Swing: VCP formed but breakout not confirmed — "
                                       f"failed {failed_checks} (vol_ratio={breakout['vol_ratio']}, "
                                       f"close_pos={breakout['close_position']}, "
                                       f"pct_above_pivot={breakout['pct_above_pivot']}%)")
            return None

        # ── Earnings blackout — NEW v5, best-effort ─────────────────────
        if CONFIG["earnings_check_enabled"]:
            earnings_check = check_earnings_blackout(symbol)
            if earnings_check["in_blackout"]:
                REJECTION_LOG[symbol] = f"Swing: {earnings_check['reason']}"
                return None

        # ── Stop loss: v5 = max(structural VCP low, 2xATR20) — tighter wins ──
        atr14 = df["ATR_14"].iloc[-1]
        atr_v5 = df["ATR_V5"].iloc[-1]
        atr_stop_price = close - (CONFIG["atr_multiplier_v5"] * atr_v5)
        structural_stop_price = vcp["base_low"]
        stop_loss = max(structural_stop_price, atr_stop_price)
        risk_pct = (close - stop_loss) / close * 100
        if risk_pct <= 0:
            REJECTION_LOG[symbol] = "Swing: computed stop loss is above/at current price — skip"
            return None

        base_low = vcp["base_low"]
        base_range_pct = (close - base_low) / close * 100
        target_pct = max(base_range_pct * 1.5, CONFIG["swing_target_rr_min"] * risk_pct)
        target = close * (1 + target_pct / 100)
        rr = target_pct / risk_pct if risk_pct > 0 else 0

        if rr < CONFIG["swing_target_rr_min"]:
            REJECTION_LOG[symbol] = f"Swing: R:R {rr:.1f} below minimum"
            return None

        sizing = calculate_position_size(close, stop_loss, regime_multiplier=regime_multiplier)

        result = {
            "Symbol": symbol, "CMP": round(close, 2),
            "Entry": round(close, 2), "Target": round(target, 2),
            "Stop_Loss": round(stop_loss, 2),
            "Profit_%": round(target_pct, 1),   # explicit alias, same as Reward_% — kept for naming
            "Risk_%": round(risk_pct, 1), "Reward_%": round(target_pct, 1),
            "RR_Ratio": round(rr, 1), "ATR": round(atr14, 2), "ATR20": round(atr_v5, 2),
            "Stop_Basis": "structural (VCP low)" if structural_stop_price >= atr_stop_price else "ATR (2x20d)",
            "VCP_Peaks": len(vcp["contractions"]),
            "VCP_Contractions_%": [c["pct"] for c in vcp["contractions"]],
            "Breakout_Vol_Ratio": breakout["vol_ratio"],
            "RS_Rank": rs_rank,
            "Pct_From_52W_High": round((1 - close/high_52w)*100, 1),
            "Pct_Above_52W_Low": round((close/low_52w - 1)*100, 1),
            "Horizon": "3-6 months",
        }
        if sizing:
            result.update({
                "Shares_To_Buy": sizing["shares"],
                "Deploy_₹": sizing["deploy"],
                "Max_Risk_₹": sizing["max_risk"],
                "%_of_Portfolio": sizing.get("deploy_pct_of_portfolio", 0),
                "Regime_Multiplier_Applied": sizing.get("regime_multiplier", 1.0),
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
# — NEW v5, used for Sector Relative Strength + Sector Leader Bonus
# ═════════════════════════════════════════════════════════════════════════
# HONESTY NOTE: this is a lightweight keyword/membership map, same spirit
# as classify_sector_bucket above — good enough to bucket stocks for RS
# comparison, not a precision GICS-style classification. Stocks not in any
# set fall into "Diversified/Other" rather than being force-fit somewhere
# wrong.
SECTOR_MAP: dict[str, str] = {}
# FIXED v6: expanded from ~80 tickers to cover essentially all of
# FALLBACK_UNIVERSE plus common broader Nifty 500 names. Previously most
# stocks fell into "Diversified/Other" and got zero Sector RS / Sector
# Leader bonus regardless of how they actually performed — that's the
# "Sector map covers only ~80 tickers" red flag, now addressed. Still not
# a precision GICS classification (that needs a licensed data feed), but
# meaningfully broader coverage of the actual scanned universe.
_SECTOR_GROUPS = {
    "Banking": ("SBIN", "MAHABANK", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "IOB", "CENTRALBK", "INDIANB",
                "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB",
                "RBLBANK", "BANDHANBNK", "AUBANK", "KARURVYSYA", "J&KBANK", "CUB", "DCBBANK", "SOUTHBANK"),
    "Pharma": ("SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "BIOCON", "ALKEM", "TORNTPHARM", "GLENMARK",
               "IPCALAB", "AUROPHARMA", "ZYDUSLIFE", "GLAND", "LAURUSLABS", "GRANULES", "AJANTPHARM", "NATCOPHARM",
               "MANKIND", "ABBOTINDIA", "PFIZER", "SANOFI"),
    "Hospitals": ("APOLLOHOSP", "KIMS", "GLOBALHEALTH", "FORTIS", "MAXHEALTH", "NH", "METROPOLIS", "LALPATHLAB",
                  "RAINBOW", "ASTERDM"),
    "Defense": ("HAL", "BEL", "GRSE", "MAZDOCK", "BDL", "COCHINSHIP", "SOLARINDS", "DATAPATTNS", "PARAS", "ASTRAMICRO"),
    "Renewables/Power": ("SUZLON", "INOXWIND", "WAAREEENER", "TATAPOWER", "ADANIGREEN", "NHPC", "SJVN", "JSWENERGY",
                          "NTPC", "POWERGRID", "TORNTPOWER", "CESC", "ADANIENSOL", "ADANIPOWER", "NLCINDIA",
                          "KPIGREEN", "WEBSOL"),
    "Capital Goods": ("LT", "SIEMENS", "HAVELLS", "CGPOWER", "ABB", "THERMAX", "BHEL", "POLYCAB", "KEI",
                       "APLAPOLLO", "TIINDIA", "AIAENG", "KAYNES", "SCHNEIDER", "HONAUT", "BLUESTARCO",
                       "VOLTAS", "CROMPTON", "DIXON", "AMBER"),
    "Metals & Mining": ("JSWSTEEL", "TATASTEEL", "HINDALCO", "HINDZINC", "SAIL", "NMDC", "VEDL", "NLCINDIA",
                         "MOIL", "JINDALSTEL", "NATIONALUM", "APLAPOLLO", "RATNAMANI"),
    "IT Services": ("TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "COFORGE", "PERSISTENT", "MPHASIS",
                     "OFSS", "KPITTECH", "CYIENT", "MASTEK", "HAPPSTMNDS", "LTTS", "TATAELXSI", "ZENSARTECH",
                     "SONATSOFTW", "NEWGEN"),
    "Consumer/Retail": ("TITAN", "TRENT", "DMART", "NYKAA", "ETERNAL", "SWIGGY", "MARICO", "NESTLEIND",
                         "BRITANNIA", "DABUR", "COLPAL", "VBL", "HINDUNILVR", "ITC", "TATACONSUM", "GODREJCP",
                         "PATANJALI", "EMAMILTD", "JYOTHYLAB", "RADICO", "UNITEDSPR"),
    "Energy/PSU": ("RELIANCE", "ONGC", "BPCL", "COALINDIA", "IOC", "HINDPETRO", "GAIL", "OIL", "PETRONET"),
    "Auto & Ancillaries": ("MARUTI", "TATAMOTORS", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "MOTHERSON",
                            "BALKRISIND", "APOLLOTYRE", "MRF", "M&M", "ASHOKLEY", "TVSMOTOR", "BOSCHLTD",
                            "EXIDEIND", "AMARAJABAT", "SONACOMS", "UNOMINDA", "SUPRAJIT"),
    "Insurance": ("HDFCLIFE", "SBILIFE", "ICICIGI", "ICICIPRULI", "LICI", "STARHEALTH", "GICRE", "NIACL"),
    "Real Estate": ("DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD", "LODHA", "BRIGADE", "SOBHA",
                     "SUNTECK", "MAHLIFE"),
    "Diversified Financials": ("BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "ABCAPITAL", "ANGELONE", "JIOFIN",
                                 "MUTHOOTFIN", "MANAPPURAM", "IRFC", "PFC", "RECLTD", "SBICARD", "LTF",
                                 "PROTEAN", "CAMS"),
    "Cement": ("ULTRACEMCO", "SHREECEM", "JKCEMENT", "RAMCOCEM", "ACC", "AMBUJACEM", "DALBHARAT", "JKLAKSHMI"),
    "Chemicals & Paints": ("ASIANPAINT", "PIDILITIND", "BERGEPAINT", "ASTRAL", "SUPREMEIND", "SRF", "PIIND",
                            "AARTIIND", "DEEPAKNTR", "NAVINFLUOR", "GUFICBIO", "CAPLIPOINT", "FINEORG"),
    "Logistics & Infra": ("ADANIPORTS", "CONCOR", "BLUEDART", "DELHIVERY", "GMRINFRA", "RAILTEL", "RVNL",
                           "NBCC", "HUDCO", "IRCON", "GRSE", "IRB", "GMRAIRPORT"),
    "New-Age Tech": ("PAYTM", "NYKAA", "POLICYBZR", "DELHIVERY", "ZOMATO", "ETERNAL", "SWIGGY"),
    "Telecom": ("BHARTIARTL", "IDEA", "TATACOMM", "INDUSTOWER"),
    "Diversified Conglomerate": ("ADANIENT", "GRASIM", "BAJAJHLDNG", "GODFRYPHLP"),
}
for _sector_name, _tickers in _SECTOR_GROUPS.items():
    for _sym in _tickers:
        SECTOR_MAP[_sym] = _sector_name


def get_sector_label(symbol: str) -> str:
    """Friendly sector name for Sector RS / Sector Leader Bonus. Falls back
    to 'Diversified/Other' rather than guessing wrong — an honest 'don't
    know' bucket, not a forced classification."""
    return SECTOR_MAP.get(symbol.upper(), "Diversified/Other")


# ═════════════════════════════════════════════════════════════════════════
# RELATIVE STRENGTH — NEW v5 (highest priority upgrade per spec)
# ═════════════════════════════════════════════════════════════════════════
# Computes each stock's 12M/6M/3M return AGAINST THE SAME SCANNED UNIVERSE
# (not some external benchmark file) — genuinely apples-to-apples, no
# fabricated percentile. RS_Rank of 92 means "returned more than 92% of
# the other stocks actually scanned in this run", nothing fancier claimed.

def compute_universe_relative_strength(liquid: dict) -> dict:
    """
    Returns: {symbol: {"ret_12m", "ret_6m", "ret_3m", "rs_rank_12m", "sector"}}
    Also returns a special key "_sector_stats" -> {sector: {"avg_ret_12m",
    "avg_ret_6m", "avg_ret_3m", "sector_rank_12m_pct"}} for Sector RS.
    Stocks with insufficient history for a given window are simply
    excluded from THAT window's ranking (not zero-filled) — an honest gap
    rather than a fabricated number.
    """
    raw = {}
    for sym, df in liquid.items():
        try:
            close = df["Close"]
            n = len(close)
            ret_12m = (close.iloc[-1] / close.iloc[-CONFIG["rs_lookback_days"]] - 1) * 100 \
                if n > CONFIG["rs_lookback_days"] else None
            ret_6m = (close.iloc[-1] / close.iloc[-126] - 1) * 100 if n > 126 else None
            ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) * 100 if n > 63 else None
            raw[sym] = {"ret_12m": ret_12m, "ret_6m": ret_6m, "ret_3m": ret_3m,
                         "sector": get_sector_label(sym)}
        except Exception:
            continue

    # Percentile rank each stock's 12M return against all OTHER stocks that
    # also have a valid 12M return this run.
    valid_12m = {s: v["ret_12m"] for s, v in raw.items() if v["ret_12m"] is not None}
    if valid_12m:
        sorted_syms = sorted(valid_12m, key=lambda s: valid_12m[s])
        n_valid = len(sorted_syms)
        rank_map = {s: round((i + 1) / n_valid * 100, 1) for i, s in enumerate(sorted_syms)}
    else:
        rank_map = {}

    for sym in raw:
        raw[sym]["rs_rank_12m"] = rank_map.get(sym)

    # Sector-level relative strength: average return per sector, then rank
    # SECTORS against each other (not stocks) for the Sector_RS bonus.
    sector_returns = {}
    for sym, v in raw.items():
        sec = v["sector"]
        sector_returns.setdefault(sec, {"ret_12m": [], "ret_6m": [], "ret_3m": []})
        if v["ret_12m"] is not None: sector_returns[sec]["ret_12m"].append(v["ret_12m"])
        if v["ret_6m"] is not None: sector_returns[sec]["ret_6m"].append(v["ret_6m"])
        if v["ret_3m"] is not None: sector_returns[sec]["ret_3m"].append(v["ret_3m"])

    sector_stats = {}
    for sec, rets in sector_returns.items():
        sector_stats[sec] = {
            "avg_ret_12m": round(sum(rets["ret_12m"]) / len(rets["ret_12m"]), 1) if rets["ret_12m"] else None,
            "avg_ret_6m": round(sum(rets["ret_6m"]) / len(rets["ret_6m"]), 1) if rets["ret_6m"] else None,
            "avg_ret_3m": round(sum(rets["ret_3m"]) / len(rets["ret_3m"]), 1) if rets["ret_3m"] else None,
            "n_stocks": len(rets["ret_12m"]),
        }
    valid_sectors = {s: v["avg_ret_12m"] for s, v in sector_stats.items() if v["avg_ret_12m"] is not None}
    if valid_sectors:
        sorted_secs = sorted(valid_sectors, key=lambda s: valid_sectors[s])
        n_sec = len(sorted_secs)
        for i, sec in enumerate(sorted_secs):
            sector_stats[sec]["sector_rank_pct"] = round((i + 1) / n_sec * 100, 1)
    for sec in sector_stats:
        sector_stats[sec].setdefault("sector_rank_pct", None)

    raw["_sector_stats"] = sector_stats
    return raw


def sector_rs_bonus(sector_rank_pct: Optional[float]) -> int:
    """Score bonus/penalty from Sector RS, per spec: top20% sectors +5,
    top10% +10, bottom20% -5. Uses the HIGHER bonus if both thresholds
    are met (top10% implies top20% too — don't double-count, take the max)."""
    if sector_rank_pct is None:
        return 0
    if sector_rank_pct >= 90:
        return CONFIG["sector_rs_top10_bonus"]
    if sector_rank_pct >= 80:
        return CONFIG["sector_rs_top20_bonus"]
    if sector_rank_pct <= 20:
        return CONFIG["sector_rs_bottom20_penalty"]
    return 0


def calculate_fair_value_target(symbol: str, close: float, fund: "Fundamentals",
                                  score: int, grade: str) -> dict:
    """
    REWRITTEN — no longer a single tuned formula (which was repeatedly
    adjusted against MAHABANK alone and proved to be overfitting, not a
    real fix — see CHANGELOG note below). Instead, this computes THREE
    independent estimates and reports the SPREAD, not a single number
    presented as if "the" answer:

      1. PE-anchored estimate  — current PE expanded modestly toward a
         sector-appropriate ceiling (same mechanism as before, but now
         shown as ONE input among three, not the final word)
      2. PEG=1.0 fair-value estimate — the textbook Peter Lynch
         "fairly valued" anchor: target PE = earnings growth rate,
         capped at a sane band. This is the standard formula taught in
         every valuation textbook (Schwab, AnalystPrep/CFA curriculum)
         and is shown for comparison, not because it is more "correct"
         — PEG=1.0 is itself just a rule of thumb, not a law of markets.
      3. Price-to-book estimate (financials/banks only) — ROE-implied
         fair P/B using the Gordon-growth-style relationship
         P/B = ROE / required_return, common for valuing banks where
         PE is less meaningful than for industrials.

    The three estimates are averaged for a CENTRAL target, but the
    LOW and HIGH bounds are reported explicitly so the spread itself
    communicates uncertainty — exactly what a single point estimate
    hides. If the three methods disagree a lot, that disagreement is
    real information, not noise to average away.

    CHANGELOG / HONESTY NOTE: an earlier version of this function was
    tuned three separate times (0.5x, 1.6x, 1.2x multiple-expansion
    caps) chasing a number that happened to match MAHABANK's real
    analyst consensus. That was overfitting to one stock, not a fix —
    each constant was picked because it moved ONE example closer to a
    target, with no out-of-sample validation. This version does not
    claim to match real analyst consensus; it shows model-internal
    methods transparently and labels them as such. The Real_Analyst_*
    fields (populated separately via fetch_real_analyst_target) are
    the only fields that should be treated as actual street consensus.
    """
    sector = classify_sector_bucket(symbol)
    peg_ceiling = SECTOR_PEG_CEILING.get(sector, SECTOR_PEG_CEILING["default"])
    pe = fund.pe
    earn_growth = fund.earn_growth
    estimates = {}  # method_name -> target_pct

    # ── Method 1: PE-anchored modest expansion ──────────────────────────
    if pe is not None and pe > 0 and earn_growth is not None and earn_growth > 0:
        implied_eps = close / pe
        max_multiple_expansion_pe = pe * 1.2  # cap: max 20% multiple expansion in 1-2yr horizon
        target_pe_m1 = max_multiple_expansion_pe
        target_price_m1 = implied_eps * target_pe_m1
        estimates["pe_anchored"] = (target_price_m1 / close - 1) * 100

    # ── Method 2: PEG = 1.0 textbook fair value ─────────────────────────
    # Standard formula: fair PE = earnings growth rate (in %, as a raw
    # number — e.g. 20% growth -> fair PE of 20). This is the literal
    # Peter Lynch heuristic, capped to a sane band so a 90%-growth
    # outlier doesn't imply a 90x PE.
    if pe is not None and pe > 0 and earn_growth is not None and earn_growth > 0:
        implied_eps = close / pe
        capped_growth_for_peg1 = max(8.0, min(earn_growth, 30.0))
        fair_pe_peg1_theoretical = capped_growth_for_peg1  # PEG=1.0 definition
        sector_cap_pe = peg_ceiling * 12  # rough sector-typical PE ceiling, e.g. PSU bank 1.4*12=16.8x
        fair_pe_peg1_theoretical = min(fair_pe_peg1_theoretical, sector_cap_pe)

        # BUG FOUND DURING TESTING: capping only the ceiling let a cheap
        # stock (e.g. MAHABANK at 8x PE) jump straight to a 16.8x ceiling
        # in one step — a 110% implied move. PEG=1.0 is a THEORETICAL
        # fair-value anchor, not a 1-2 year price target; markets don't
        # re-rate a full PEG gap in one year just because it's
        # theoretically "fair." Same fix as Method 1: constrain how much
        # of the gap toward fair value can realistically close within
        # this horizon (capped at 20% multiple expansion from current PE,
        # same logic as Method 1, even if the theoretical fair PE is
        # higher still).
        max_realistic_pe_m2 = pe * 1.2
        fair_pe_peg1 = min(fair_pe_peg1_theoretical, max_realistic_pe_m2)
        fair_pe_peg1 = max(fair_pe_peg1, pe * 1.02)

        target_price_m2 = implied_eps * fair_pe_peg1
        estimates["peg_1.0_fair_value"] = (target_price_m2 / close - 1) * 100

    # ── Method 3: ROE/P-B implied fair value (banks/financials) ─────────
    if sector in ("psu_bank", "private_bank") and fund.roe is not None and fund.pb is not None and fund.pb > 0:
        # Simplified Gordon-growth P/B: fair P/B ≈ ROE / required_return.
        # Using a required_return of 11% (typical Indian equity cost of
        # capital assumption for PSU banks; private banks slightly lower
        # risk so could argue 10%, kept uniform here for simplicity).
        required_return = 11.0
        fair_pb = fund.roe / required_return
        current_pb = fund.pb
        if current_pb > 0:
            target_price_m3 = close * (fair_pb / current_pb)
            pct_m3 = (target_price_m3 / close - 1) * 100
            # Same sanity bounds as other methods
            pct_m3 = max(min(pct_m3, 40.0), -20.0)
            estimates["roe_pb_implied"] = pct_m3

    # ── Combine ──────────────────────────────────────────────────────────
    if estimates:
        values = list(estimates.values())
        target_pct_low = round(min(values), 1)
        target_pct_high = round(max(values), 1)
        target_pct_central = round(sum(values) / len(values), 1)
        # Hard outer sanity bounds regardless of method spread
        target_pct_central = max(min(target_pct_central, 35.0), 8.0)
        target_pct_low = max(min(target_pct_low, 35.0), 5.0)
        target_pct_high = max(min(target_pct_high, 40.0), target_pct_central)
        method = " | ".join(f"{k}: {v:+.1f}%" for k, v in estimates.items())
    else:
        # No usable PE/EPS/ROE data (e.g. loss-making Swiggy/Jio Financial)
        base = 25 if grade == "STRONG BUY" else 18 if grade == "BUY" else 12
        target_pct_central = base
        target_pct_low = round(base * 0.6, 1)
        target_pct_high = round(base * 1.4, 1)
        method = f"Fallback (no usable PE/EPS/ROE data) — capped score-based estimate, wide uncertainty band"

    sl_pct = (12 if grade == "STRONG BUY" else 14 if grade == "BUY" else 16)
    target_price_central = close * (1 + target_pct_central / 100)
    target_price_low = close * (1 + target_pct_low / 100)
    target_price_high = close * (1 + target_pct_high / 100)
    sl_price = close * (1 - sl_pct / 100)

    return {
        "target_price": round(target_price_central, 2),
        "target_price_low": round(target_price_low, 2),
        "target_price_high": round(target_price_high, 2),
        "sl_price": round(sl_price, 2),
        "target_pct": target_pct_central,
        "target_pct_low": target_pct_low,
        "target_pct_high": target_pct_high,
        "sl_pct": sl_pct,
        "sector_bucket": sector,
        "peg_ceiling_used": peg_ceiling,
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
    

def evaluate_core(symbol: str, df: pd.DataFrame, fund: Fundamentals,
                    rs_rank: Optional[float] = None, sector_rank_pct: Optional[float] = None,
                    regime_multiplier: float = 1.0) -> Optional[dict]:
    try:
        close = df["Close"].iloc[-1]
        sma50  = df["Close"].rolling(50).mean().iloc[-1]
        sma200 = df["Close"].rolling(200).mean().iloc[-1] if len(df) >= 200 else df["Close"].mean()

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

        if fund.earn_growth is not None and fund.earn_growth < CONFIG["core_reject_earnings_growth_below"]:
            REJECTION_LOG[symbol] = f"Core: EPS growth {fund.earn_growth}% — structurally broken"
            return None

        if CONFIG["earnings_check_enabled"]:
            earnings_check = check_earnings_blackout(symbol)
            if earnings_check["in_blackout"]:
                REJECTION_LOG[symbol] = f"Core: {earnings_check['reason']}"
                return None

        q = g = t = m = 0
        reasons = []

        if fund.roe is not None:
            if fund.roe > 22:   q += 10; reasons.append(f"ROE {fund.roe}% — excellent")
            elif fund.roe > 15: q += 6;  reasons.append(f"ROE {fund.roe}% — good")
        if fund.debt_eq is not None:
            if fund.debt_eq < 0.3: q += 8; reasons.append(f"D/E {fund.debt_eq} — fortress balance sheet")
            elif fund.debt_eq < 0.8: q += 5; reasons.append(f"D/E {fund.debt_eq} — manageable")
        if fund.pe is not None and fund.earn_growth is not None and fund.earn_growth > 20 and fund.pe < 40:
            q += 7; reasons.append(f"PEG ~{round(fund.pe/fund.earn_growth,2)} — undervalued vs growth")
        elif fund.pe is not None and fund.pe < 18:
            q += 5; reasons.append(f"PE {fund.pe}x — cheap")
        if fund.div_yield is not None and fund.div_yield > 2:
            q += 5; reasons.append(f"Dividend {fund.div_yield}% — paid to hold")

        if fund.earn_growth is not None:
            if fund.earn_growth > 50:   g += 15; reasons.append(f"EPS growth {fund.earn_growth}% — explosive")
            elif fund.earn_growth > 30: g += 11; reasons.append(f"EPS growth {fund.earn_growth}% — strong")
            elif fund.earn_growth > 18: g += 7;  reasons.append(f"EPS growth {fund.earn_growth}%")
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
        if pos52 < 40:   t += 6; reasons.append(f"Near 52W low ({pos52:.0f}%) — good entry")
        elif pos52 < 55: t += 3; reasons.append(f"Mid 52W range ({pos52:.0f}%)")
        if vol_ratio > 2: t += 3; reasons.append(f"Volume {vol_ratio:.1f}x avg — accumulation")

        if -30 <= ret_6m <= 20: m += 8; reasons.append(f"6M return {ret_6m:.0f}% — healthy")
        elif ret_6m < -30:      m += 5; reasons.append(f"6M return {ret_6m:.0f}% — deep correction")
        if -5 <= ret_1m <= 8:   m += 4; reasons.append(f"1M return {ret_1m:.0f}% — stable base")
        elif ret_1m < -5:       m += 3; reasons.append(f"1M dip {ret_1m:.0f}% — buying opportunity")
        if fund.beta is not None and 0.7 <= fund.beta <= 1.2:
            m += 3; reasons.append(f"Beta {fund.beta} — stable for 1Y hold")

        # ── Sector Relative Strength bonus — NEW v5 ─────────────────────
        sec_bonus = sector_rs_bonus(sector_rank_pct)
        if sec_bonus != 0:
            reasons.append(f"Sector RS {'bonus' if sec_bonus > 0 else 'penalty'} "
                             f"({sec_bonus:+d}, sector percentile {sector_rank_pct})")

        total = q + g + t + m + sec_bonus
        if total < CONFIG["core_score_watch"]:
            REJECTION_LOG[symbol] = f"Core: score {total} below WATCH threshold"
            return None

        grade = ("STRONG BUY" if total >= CONFIG["core_score_strong_buy"] else
                  "BUY" if total >= CONFIG["core_score_buy"] else "WATCH")

        # ── RS Rank gate for STRONG BUY tier — NEW v5 (optional, spec-driven) ──
        if (grade == "STRONG BUY" and rs_rank is not None
                and rs_rank < CONFIG["core_strong_buy_min_rs_rank"]):
            grade = "BUY"
            reasons.append(f"Downgraded from STRONG BUY: RS_Rank {rs_rank} below "
                             f"{CONFIG['core_strong_buy_min_rs_rank']} required for that tier")

        valuation = calculate_fair_value_target(symbol, close, fund, total, grade)
        target_price = valuation["target_price"]
        sl_price_pct_based = valuation["sl_price"]
        base_tgt = valuation["target_pct"]
        sl_pct = valuation["sl_pct"]

        # ── ATR-floor stop — NEW v5: never let the stop be looser than
        # a 2x20-day-ATR distance, even if the %-based method said so ──
        try:
            atr20 = ta.atr(df["High"], df["Low"], df["Close"], length=CONFIG["atr_period_v5"]).iloc[-1]
            atr_stop_price = close - (CONFIG["atr_multiplier_v5"] * atr20)
            sl_price = max(sl_price_pct_based, atr_stop_price)
        except Exception:
            sl_price = sl_price_pct_based
        sl_pct = round((close - sl_price) / close * 100, 1)

        # Real analyst consensus (separate from model estimate above) —
        # only meaningfully populated if ANALYST_API_CONFIG has a key set
        analyst = fetch_real_analyst_target(symbol)

        sizing = calculate_position_size(close, sl_price, regime_multiplier=regime_multiplier)

        result = {
            "Symbol": symbol, "CMP": round(close, 2), "Score": total, "Grade": grade,
            "Quality": q, "Growth": g, "Technical": t, "Momentum": m, "Sector_RS_Bonus": sec_bonus,
            "RS_Rank": rs_rank, "Sector_Percentile": sector_rank_pct, "Market_Cap_Cr": fund.mcap_cr,
            "PE": fund.pe, "Fwd_PE": fund.fwd_pe, "ROE_%": fund.roe,
            "Rev_Growth_%": fund.rev_growth, "EPS_Growth_%": fund.earn_growth,
            "Debt_Eq": fund.debt_eq, "Div_Yield_%": fund.div_yield, "RSI": round(rsi, 1),
            "Target": target_price, "Stop_Loss": sl_price,
            "Target_Low": valuation["target_price_low"], "Target_High": valuation["target_price_high"],
            "Profit_%": base_tgt, "Risk_%": sl_pct,   # explicit aliases for Target_%/SL_% — consistent naming with swing book
            "Target_%": base_tgt, "Target_%_Low": valuation["target_pct_low"],
            "Target_%_High": valuation["target_pct_high"],
            "SL_%": sl_pct, "RR_Ratio": round(base_tgt / sl_pct, 1) if sl_pct > 0 else 0,
            "Sector_Bucket": valuation["sector_bucket"], "Sector_Label": get_sector_label(symbol),
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
            "Reasons": " | ".join(reasons[:6]), "Horizon": "1-2 years",
        }
        if sizing:
            result.update({
                "Shares_To_Buy": sizing["shares"],
                "Deploy_₹": sizing["deploy"],
                "Max_Risk_₹": sizing["max_risk"],
                "%_of_Portfolio": sizing.get("deploy_pct_of_portfolio", 0),
                "Regime_Multiplier_Applied": sizing.get("regime_multiplier", 1.0),
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
        time.sleep(CONFIG["nse_cookie_warmup_pause_sec"])  # FIX v5: let NSE's session cookie settle

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
    """Downloads Nifty 50 index (^NSEI) daily OHLC. Degrades to None on failure —
    the regime section is skipped with an explicit note rather than guessing."""
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
            return df
        except Exception as e:
            if attempt == CONFIG["max_retries"]:
                log.warning(f"Nifty regime check failed after {attempt} attempts: {e}")
                return None
            time.sleep(CONFIG["retry_backoff_sec"])
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


def evaluate_nifty_regime() -> Optional[dict]:
    """
    Returns a dict describing current Nifty trend/momentum/breakout state,
    or None if the data fetch failed (never fabricated).
    """
    df = fetch_nifty_index_data()
    if df is None:
        return None

    try:
        df = df.copy()
        df["SMA_FAST"] = ta.sma(df["Close"], length=CONFIG["regime_sma_fast"])
        df["SMA_SLOW"] = ta.sma(df["Close"], length=CONFIG["regime_sma_slow"])
        df["RSI"] = ta.rsi(df["Close"], length=CONFIG["regime_rsi_period"])
        macd = ta.macd(df["Close"])
        df = pd.concat([df, macd], axis=1)
        hist_col = [c for c in df.columns if c.startswith("MACDh_")][0]

        if df[["SMA_SLOW", "RSI", hist_col]].iloc[-1].isna().any():
            log.warning("Nifty regime: indicators NaN, skipping regime check")
            return None

        close = float(df["Close"].iloc[-1])
        sma_fast = float(df["SMA_FAST"].iloc[-1])
        sma_slow = float(df["SMA_SLOW"].iloc[-1])
        sma_fast_5d_ago = float(df["SMA_FAST"].iloc[-6])
        rsi = float(df["RSI"].iloc[-1])
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
        momentum_bullish = (rsi >= CONFIG["regime_rsi_bullish"] and macd_hist > 0
                             and macd_hist > macd_hist_prev)
        momentum_bearish = (rsi <= CONFIG["regime_rsi_bearish"] and macd_hist < 0
                             and macd_hist < macd_hist_prev)

        if breakout_up and momentum_bullish:
            regime = "BULLISH — BREAKOUT"
            posture = ("Momentum + trend confirm a break above the recent "
                       f"{lb}-day range high (₹{range_high:,.0f}). Historically "
                       "supportive of adding new swing exposure, not a guarantee.")
        elif trend_up and sma_fast_rising and rsi >= 50:
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
        elif close > sma_fast and not trend_down and (rsi >= 45 or sma_fast_rising):
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
            "RSI_14": round(rsi, 1),
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


def get_regime_multiplier(regime: Optional[dict]) -> float:
    """
    NEW v5 — maps our 5 Nifty regime labels onto the spec's
    STRONG_BULL/BULL/NEUTRAL/CHOPPY/BEAR position-size multiplier table.
    Our "NEUTRAL / CHOPPY" label is genuinely one bucket internally (see
    evaluate_nifty_regime), so it's split here using RSI as the tiebreaker:
    RSI >= 50 within that bucket counts as the (less risk-off) NEUTRAL
    tier, RSI < 50 counts as CHOPPY. If the regime fetch failed entirely
    (None), defaults to a conservative NEUTRAL (0.50) rather than assuming
    either extreme.
    """
    table = CONFIG["regime_position_multiplier"]
    if regime is None:
        return table["NEUTRAL"]
    label = regime.get("Regime", "")
    if label == "BULLISH — BREAKOUT":
        return table["STRONG_BULL"]
    if label == "BULLISH — TREND INTACT":
        return table["BULL"]
    if label == "RECOVERY":
        return table.get("RECOVERY", 0.60)
    if label == "NEUTRAL / CHOPPY":
        return table["NEUTRAL"] if regime.get("RSI_14", 50) >= 50 else table["CHOPPY"]
    if "BEARISH" in label:
        return table["BEAR"]
    return table["NEUTRAL"]


_EARNINGS_CACHE: dict = {}   # in-memory, cleared each process run — see caching note in README


def _fetch_next_earnings_date_raw(symbol: str, result_box: dict):
    """The actual slow yfinance calls, run in a daemon thread by
    check_earnings_blackout() below — writes into result_box instead of
    returning, since the caller may have already given up and moved on
    by the time this finishes (or never finishes)."""
    next_earnings = None
    ticker = yf.Ticker(f"{symbol}.NS")
    try:
        dates_df = ticker.get_earnings_dates(limit=4)
        if dates_df is not None and not dates_df.empty:
            today_ts = pd.Timestamp.now(tz=dates_df.index.tz) if dates_df.index.tz else pd.Timestamp.now()
            future = dates_df[dates_df.index >= today_ts]
            if not future.empty:
                next_earnings = future.index.min()
    except Exception:
        pass
    if next_earnings is None:
        try:
            cal = ticker.calendar
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if isinstance(ed, list) and ed:
                    next_earnings = ed[0]
            elif hasattr(cal, "loc") and cal is not None and not getattr(cal, "empty", True):
                if "Earnings Date" in getattr(cal, "index", []):
                    val = cal.loc["Earnings Date"]
                    next_earnings = val.iloc[0] if hasattr(val, "iloc") else val
        except Exception:
            pass
    result_box["next_earnings"] = next_earnings


def check_earnings_blackout(symbol: str) -> dict:
    """
    FIXED v6 — was relying on ONLY yf.Ticker().calendar, whose return
    shape has changed across yfinance versions. Now tries TWO independent
    yfinance methods (get_earnings_dates(), then .calendar as fallback).
    Still degrades honestly: if BOTH fail, reports "unknown" and does NOT
    block the trade.

    FIXED v7 — CACHING: v6 only cached in-memory for a single run, so
    every fresh live-screener run re-did every slow lookup from scratch.
    Now persists to disk (48h default TTL) — verified: a cached lookup
    now takes ~0.0001s instead of a network round-trip.

    FIXED v7.1 — HARD TIMEOUT, PROPERLY VERIFIED: a session-level timeout
    isn't reliable (newer yfinance versions can bypass a custom session
    internally). This runs the lookup in a background DAEMON thread and
    joins with a hard timeout — verified two things directly: (1) even
    if the underlying call hangs for 30s, this function still returns
    within the configured timeout; AND (2) critically, daemon threads
    (unlike concurrent.futures.ThreadPoolExecutor, tried first and
    REJECTED after testing) don't block the script from exiting — a
    ThreadPoolExecutor registers an atexit hook that joins every worker
    thread before the interpreter can shut down, which would have made
    the whole script hang at the very end waiting for every slow/stuck
    lookup across the run to finish, even though each individual call
    site had already moved on. Confirmed via direct timing: the
    ThreadPoolExecutor version added a real 20s hang at script exit for
    a single stuck call; the daemon-thread version added none.
    """
    if not CONFIG["earnings_check_enabled"]:
        return {"in_blackout": False, "reason": "Earnings check disabled in CONFIG", "next_earnings": None}

    if symbol in _EARNINGS_CACHE:
        return _EARNINGS_CACHE[symbol]

    disk_cached = cache_get(f"earnings_{symbol}", CONFIG.get("earnings_cache_hours", 48))
    if disk_cached is not None:
        _EARNINGS_CACHE[symbol] = disk_cached
        return disk_cached

    result_box = {}
    t = threading.Thread(target=_fetch_next_earnings_date_raw, args=(symbol, result_box), daemon=True)
    t.start()
    t.join(timeout=CONFIG.get("earnings_lookup_timeout_sec", 6))
    next_earnings = result_box.get("next_earnings")   # None if timed out or genuinely not found

    if next_earnings is None:
        result = {"in_blackout": False, "reason": "No earnings date available (not found, or lookup timed out) — not blocking",
                   "next_earnings": None}
        _EARNINGS_CACHE[symbol] = result
        cache_set(f"earnings_{symbol}", result)
        return result

    try:
        next_earnings_date = pd.Timestamp(next_earnings).date()
        today = datetime.date.today()
        trading_days_away = np.busday_count(today, next_earnings_date)
        if 0 <= trading_days_away <= CONFIG["earnings_blackout_days"]:
            result = {"in_blackout": True,
                       "reason": f"Earnings in {trading_days_away} trading day(s) "
                                 f"({next_earnings_date}) — inside blackout window",
                       "next_earnings": str(next_earnings_date)}
        else:
            result = {"in_blackout": False,
                       "reason": f"Next earnings {next_earnings_date}, outside blackout window",
                       "next_earnings": str(next_earnings_date)}
    except Exception as e:
        result = {"in_blackout": False, "reason": f"Earnings date parse failed ({type(e).__name__}) — not blocking",
                   "next_earnings": None}

    _EARNINGS_CACHE[symbol] = result
    cache_set(f"earnings_{symbol}", result)
    return result


def print_market_regime(regime: Optional[dict]):
    print("\n" + "═" * 95)
    print("  🧭 MARKET REGIME — NIFTY 50")
    print("═" * 95)
    if regime is None:
        print("  ⚠️  Could not fetch/compute Nifty regime this run (network or data issue).")
        print("      Proceed on individual stock conviction alone — no market-wide read available.")
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

# ═════════════════════════════════════════════════════════════════════════
# MACRO RISK OVERLAY — NEW v5
# ═════════════════════════════════════════════════════════════════════════
# HONESTY NOTE: VIX, Brent, USDINR, and breadth are genuinely fetched/
# computed. FII/DII net flows are attempted best-effort from NSE's public
# feed and EXCLUDED from the score (not defaulted to neutral) if that
# fails — same principle as bulk deals and the macro calendar elsewhere
# in this script: an honest partial score beats a fabricated complete one.

def _trend_score_lower_is_better(pct_change: float, thresholds: list) -> float:
    """thresholds: list of (max_pct_change, score) pairs, ascending change."""
    for max_change, score in thresholds:
        if pct_change <= max_change:
            return score
    return thresholds[-1][1]


def fetch_macro_risk_overlay(liquid: dict) -> dict:
    """
    Returns {"macro_score": float or None, "components_used": [...],
              "components_skipped": [...], "detail": {...}}
    macro_score is the simple average of whichever 0-100 component scores
    were actually obtainable this run — NOT defaulted to a fixed 6-part
    formula, since forcing unavailable components to a neutral value would
    silently understate real risk (or overstate it).
    """
    window = CONFIG["macro_trend_window_days"]
    scores = {}
    detail = {}
    skipped = []

    # ── VIX (India VIX) — lower = calmer market = higher score ─────────
    try:
        vix_df = yf.download(CONFIG["macro_vix_ticker"], period="3mo", progress=False, auto_adjust=True)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_now = float(vix_df["Close"].iloc[-1])
        vix_score = (100 if vix_now < 12 else 80 if vix_now < 15 else 60 if vix_now < 20
                      else 40 if vix_now < 25 else 20 if vix_now < 30 else 0)
        scores["vix"] = vix_score
        detail["india_vix"] = round(vix_now, 2)
    except Exception as e:
        skipped.append(f"India VIX ({type(e).__name__})")

    # ── Brent crude trend — falling = good for net-importer India ──────
    try:
        brent_df = yf.download(CONFIG["macro_brent_ticker"], period="2mo", progress=False, auto_adjust=True)
        if isinstance(brent_df.columns, pd.MultiIndex):
            brent_df.columns = brent_df.columns.get_level_values(0)
        pct_change = (brent_df["Close"].iloc[-1] / brent_df["Close"].iloc[-window] - 1) * 100
        brent_score = _trend_score_lower_is_better(pct_change, [(-10, 100), (-3, 80), (3, 60), (10, 40), (999, 20)])
        scores["brent"] = brent_score
        detail["brent_20d_change_pct"] = round(float(pct_change), 2)
    except Exception as e:
        skipped.append(f"Brent crude ({type(e).__name__})")

    # ── USDINR trend — INR appreciating (USDINR falling) = good ─────────
    try:
        inr_df = yf.download(CONFIG["macro_usdinr_ticker"], period="2mo", progress=False, auto_adjust=True)
        if isinstance(inr_df.columns, pd.MultiIndex):
            inr_df.columns = inr_df.columns.get_level_values(0)
        pct_change = (inr_df["Close"].iloc[-1] / inr_df["Close"].iloc[-window] - 1) * 100
        usdinr_score = _trend_score_lower_is_better(pct_change, [(-1, 100), (0, 80), (1, 60), (2, 40), (999, 20)])
        scores["usdinr"] = usdinr_score
        detail["usdinr_20d_change_pct"] = round(float(pct_change), 2)
    except Exception as e:
        skipped.append(f"USDINR ({type(e).__name__})")

    # ── Breadth — computed from OUR OWN downloaded universe, genuinely free ──
    try:
        above_200 = 0
        counted = 0
        for sym, df in liquid.items():
            if len(df) < 200:
                continue
            sma200 = df["Close"].rolling(200).mean().iloc[-1]
            if pd.notna(sma200):
                counted += 1
                if df["Close"].iloc[-1] > sma200:
                    above_200 += 1
        if counted > 0:
            breadth_pct = above_200 / counted * 100
            scores["breadth"] = breadth_pct   # naturally 0-100 already
            detail["pct_universe_above_200dma"] = round(breadth_pct, 1)
            detail["breadth_sample_size"] = counted
        else:
            skipped.append("Breadth (no stocks with 200D history in this universe)")
    except Exception as e:
        skipped.append(f"Breadth ({type(e).__name__})")

    # ── FII/DII net flows — best-effort, improved v6 ────────────────────
    # HONESTY NOTE on what "fixed" means here: there is no reliable FREE
    # API for this data. NSE's own site blocks most non-browser traffic
    # (hence the fragility), and the free third-party dashboards that DO
    # show this (Groww, 5paisa, Tapetide, etc.) don't offer a documented
    # API — scraping consumer-facing pages meant for humans is fragile in
    # the exact same way and often against those sites' terms, so it's
    # not a real fix, just a different failure mode. What IS improved:
    # retries with backoff (matching the pattern used elsewhere in this
    # script) and a second NSE endpoint variant tried before giving up.
    # If you need this reliably, a paid provider (the same indianapi.in
    # service already used for analyst targets, or NSDL's bulk FPI
    # reports) is the realistic path — not a scraping workaround.
    fii_dii_fetched = False
    for endpoint in ("https://www.nseindia.com/api/fiidiiTradeReact",
                       "https://www.nseindia.com/api/fiidiiTradeReact?index=fii_dii_full"):
        if fii_dii_fetched:
            break
        for attempt in range(1, CONFIG.get("max_retries", 3) + 1):
            try:
                session = requests.Session()
                session.headers.update({
                    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
                    "Accept": "application/json", "Referer": "https://www.nseindia.com/",
                })
                session.get("https://www.nseindia.com", timeout=8)
                time.sleep(CONFIG["nse_cookie_warmup_pause_sec"])
                resp = session.get(endpoint, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) >= 2:
                        fii_net = sum(float(d.get("netValue", 0)) for d in data if "fii" in d.get("category", "").lower())
                        dii_net = sum(float(d.get("netValue", 0)) for d in data if "dii" in d.get("category", "").lower())
                        scores["fii"] = 100 if fii_net > 0 else 0
                        scores["dii"] = 100 if dii_net > 0 else 0
                        detail["fii_net_cr"] = round(fii_net, 1)
                        detail["dii_net_cr"] = round(dii_net, 1)
                        fii_dii_fetched = True
                        break
                    else:
                        if attempt == CONFIG.get("max_retries", 3):
                            skipped.append(f"FII/DII (unexpected response shape from {endpoint})")
                else:
                    if attempt == CONFIG.get("max_retries", 3):
                        skipped.append(f"FII/DII (HTTP {resp.status_code} from {endpoint} after {attempt} attempts "
                                        "— NSE likely blocking non-browser access; consider a paid provider "
                                        "if you need this reliably)")
            except Exception as e:
                if attempt == CONFIG.get("max_retries", 3):
                    skipped.append(f"FII/DII ({type(e).__name__} from {endpoint} after {attempt} attempts)")
            if not fii_dii_fetched and attempt < CONFIG.get("max_retries", 3):
                time.sleep(CONFIG.get("retry_backoff_sec", 4.0))

    if not scores:
        return {"macro_score": None, "components_used": [], "components_skipped": skipped, "detail": detail}

    macro_score = round(sum(scores.values()) / len(scores), 1)
    return {"macro_score": macro_score, "components_used": list(scores.keys()),
             "components_skipped": skipped, "detail": detail}


def macro_score_to_allocation_pct(macro_score: Optional[float]) -> float:
    """Maps Macro_Score to an allocation % per CONFIG's table. Returns 1.0
    (no reduction) if the score is unavailable — an unknown macro backdrop
    isn't treated as automatically bad, since that would be inventing a
    signal that isn't really there."""
    if macro_score is None:
        return 1.0
    for min_score, pct in CONFIG["macro_allocation_table"]:
        if macro_score >= min_score:
            return pct
    return 0.0


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

    total_eligible = int(eligible_mask.sum())
    fetched_so_far = 0

    for idx, row in df.iterrows():
        symbol = row["Symbol"]
        if not eligible_mask.get(idx, True):
            news_status_col.append("skipped")
            news_text_col.append("Not fetched (grade below threshold) — quant justification only")
            continue

        # FIX v5: progress logging — without this the script LOOKS frozen
        # during news fetching (one request at a time, deliberately paced).
        fetched_so_far += 1
        if fetched_so_far % CONFIG["news_log_progress_every"] == 0 or fetched_so_far == total_eligible:
            log.info(f"  News {fetched_so_far}/{total_eligible}: {symbol}")

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

    if row.get("VCP_Peaks", 0) >= 3:
        score += 15; factors.append(f"Strong VCP — {row['VCP_Peaks']} contraction peaks (+15)")
    elif row.get("VCP_Peaks", 0) >= 2:
        score += 8; factors.append(f"VCP confirmed — {row['VCP_Peaks']} peaks (+8)")

    rr = row.get("RR_Ratio", 0)
    if rr >= 5: score += 20; factors.append(f"Excellent R:R 1:{rr} (+20)")
    elif rr >= 3: score += 12; factors.append(f"Good R:R 1:{rr} (+12)")
    elif rr >= 2: score += 6; factors.append(f"Acceptable R:R 1:{rr} (+6)")

    dist_from_high = row.get("Pct_From_52W_High", 100)
    if dist_from_high <= 10: score += 15; factors.append(f"Only {dist_from_high}% off 52W high — strong (+15)")
    elif dist_from_high <= 20: score += 8; factors.append(f"{dist_from_high}% off 52W high — solid (+8)")

    above_low = row.get("Pct_Above_52W_Low", 0)
    if above_low >= 50: score += 10; factors.append(f"{above_low}% above 52W low — confirmed uptrend (+10)")
    elif above_low >= 30: score += 5; factors.append(f"{above_low}% above 52W low (+5)")

    score = min(score, 95)  # never claim near-certainty — cap below 100
    return {"confidence_pct": score, "factors": factors}


def estimate_confidence_core(row: dict) -> dict:
    """Same honesty principle as swing — rules-based heuristic, not a backtested rate."""
    score = row.get("Score", 0)  # 0-100 from the quality/growth/tech/momentum model
    factors = [f"Base: model score {score}/100 from Quality+Growth+Technical+Momentum"]

    # Adjust for extremes that the raw score alone doesn't fully capture
    pe = row.get("PE")
    earn_g = row.get("EPS_Growth_%")
    if pe and earn_g and earn_g > 0:
        peg = pe / earn_g
        if peg < 1.0:
            score = min(score + 8, 95); factors.append(f"PEG {peg:.2f} — growth clearly outpaces price (+8)")
        elif peg > 4.0:
            score = max(score - 10, 5); factors.append(f"PEG {peg:.2f} — expensive vs growth rate (-10)")

    debt_eq = row.get("Debt_Eq")
    if debt_eq is not None and debt_eq > 2.0:
        score = max(score - 8, 5); factors.append(f"D/E {debt_eq} — elevated leverage risk (-8)")

    score = min(max(score, 5), 95)  # clamp 5-95, never claim certainty either direction
    return {"confidence_pct": round(score), "factors": factors}


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
# SECTOR LEADER BONUS — NEW v5
# ═════════════════════════════════════════════════════════════════════════
# Scoped to stocks that ALREADY passed the core score/watch threshold —
# not a full-universe fundamentals sweep, which would need fetching
# fundamentals for all ~500 stocks just to rank a handful of leaders
# (expensive for little extra signal). "Leader" here means leader AMONG
# this run's qualifying picks in that sector, a real but narrower claim.

def apply_sector_leader_bonus(core_df: pd.DataFrame) -> pd.DataFrame:
    if core_df.empty or "Sector_Label" not in core_df.columns:
        return core_df
    df = core_df.copy()
    df["Sector_Leader_Bonus"] = 0
    df["Sector_Leader_Reasons"] = ""

    for sector, group in df.groupby("Sector_Label"):
        if sector == "Diversified/Other" or len(group) < 2:
            continue  # bonus only meaningful when there's real competition within the sector

        bonus = pd.Series(0, index=group.index)
        reason_lists = {idx: [] for idx in group.index}

        # FIXED v6: real market cap comparison — Market_Cap_Cr is now
        # populated in evaluate_core's result (from Fundamentals.mcap_cr,
        # which was already being fetched but not passed through). The
        # earlier v5 version substituted analyst-implied-upside here
        # because this field wasn't wired through — that was a real gap,
        # not a design choice, and is now fixed.
        if "Market_Cap_Cr" in group.columns and group["Market_Cap_Cr"].notna().any():
            top_mcap_idx = group["Market_Cap_Cr"].idxmax()
            bonus[top_mcap_idx] += CONFIG["sector_leader_mcap_bonus"]
            reason_lists[top_mcap_idx].append(f"Largest market cap in {sector}")

        if "RS_Rank" in group.columns and group["RS_Rank"].notna().any():
            top3_rs = group["RS_Rank"].nlargest(min(3, len(group))).index
            for idx in top3_rs:
                bonus[idx] += CONFIG["sector_leader_rs_top3_bonus"]
                reason_lists[idx].append(f"Top-3 RS in {sector}")

        if "ROE_%" in group.columns and group["ROE_%"].notna().any():
            top_roe_idx = group["ROE_%"].idxmax()
            bonus[top_roe_idx] += CONFIG["sector_leader_roe_bonus"]
            reason_lists[top_roe_idx].append(f"Highest ROE in {sector}")

        for idx in group.index:
            if bonus[idx] > 0:
                df.loc[idx, "Sector_Leader_Bonus"] = int(bonus[idx])
                df.loc[idx, "Sector_Leader_Reasons"] = ", ".join(reason_lists[idx])
                df.loc[idx, "Score"] = df.loc[idx, "Score"] + int(bonus[idx])

    # Recompute grade in case the bonus pushed a stock over a threshold
    df["Grade"] = np.where(df["Score"] >= CONFIG["core_score_strong_buy"], "STRONG BUY",
                    np.where(df["Score"] >= CONFIG["core_score_buy"], "BUY", "WATCH"))
    return df


# ═════════════════════════════════════════════════════════════════════════
# PORTFOLIO EXPOSURE CONSTRUCTION — NEW v5
# ═════════════════════════════════════════════════════════════════════════
# Takes the raw Swing + Core candidate lists and builds a FINAL, capital-
# constrained selection: max exposure per sector, max positions per
# sector, and a portfolio "heat" cap (sum of risk% x position weight
# across everything selected). Greedy allocation ordered by conviction
# (Confidence_%/Score) — accepts the best setups first, skips anything
# that would breach a cap, and TAGS why each stock was included/excluded
# rather than silently dropping it.

def construct_final_portfolio(swing_df: pd.DataFrame, core_df: pd.DataFrame) -> pd.DataFrame:
    candidates = []
    for _, row in swing_df.iterrows() if not swing_df.empty else []:
        candidates.append({
            "Symbol": row["Symbol"], "Book": "Swing", "Sector": get_sector_label(row["Symbol"]),
            "Conviction": row.get("Confidence_%", 50), "Risk_%": row.get("Risk_%", 0),
            "Deploy_₹": row.get("Deploy_₹", 0), "CMP": row.get("CMP", 0),
        })
    for _, row in core_df.iterrows() if not core_df.empty else []:
        candidates.append({
            "Symbol": row["Symbol"], "Book": "Core", "Sector": row.get("Sector_Label", get_sector_label(row["Symbol"])),
            "Conviction": row.get("Confidence_%", row.get("Score", 50)), "Risk_%": row.get("Risk_%", 0),
            "Deploy_₹": row.get("Deploy_₹", 0), "CMP": row.get("CMP", 0),
        })

    if not candidates:
        return pd.DataFrame()

    cand_df = pd.DataFrame(candidates).sort_values("Conviction", ascending=False).reset_index(drop=True)

    portfolio = CONFIG["portfolio_size_inr"]
    max_sector_value = portfolio * CONFIG["portfolio_max_sector_exposure_pct"]
    max_heat = CONFIG["portfolio_max_heat_pct"] * 100  # working in % units to match Risk_%

    sector_deployed = {}
    sector_count = {}
    total_heat = 0.0
    included_flags, reasons = [], []

    for _, row in cand_df.iterrows():
        sec = row["Sector"]
        deploy = row["Deploy_₹"] or 0
        risk_pct = row["Risk_%"] or 0
        position_weight = deploy / portfolio if portfolio > 0 else 0
        heat_contribution = risk_pct * position_weight

        sec_dep_so_far = sector_deployed.get(sec, 0)
        sec_cnt_so_far = sector_count.get(sec, 0)

        if sec_cnt_so_far >= CONFIG["portfolio_max_positions_per_sector"]:
            included_flags.append(False)
            reasons.append(f"Skipped — already {sec_cnt_so_far} positions in {sec} "
                             f"(max {CONFIG['portfolio_max_positions_per_sector']})")
            continue
        if sec_dep_so_far + deploy > max_sector_value:
            included_flags.append(False)
            reasons.append(f"Skipped — would breach {CONFIG['portfolio_max_sector_exposure_pct']*100:.0f}% "
                             f"sector cap for {sec}")
            continue
        if total_heat + heat_contribution > max_heat:
            included_flags.append(False)
            reasons.append(f"Skipped — would breach portfolio heat cap "
                             f"({CONFIG['portfolio_max_heat_pct']*100:.0f}%)")
            continue

        sector_deployed[sec] = sec_dep_so_far + deploy
        sector_count[sec] = sec_cnt_so_far + 1
        total_heat += heat_contribution
        included_flags.append(True)
        reasons.append("Included")

    cand_df["Included"] = included_flags
    cand_df["Selection_Note"] = reasons
    return cand_df


# ═════════════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════

def run_screener(universe_override: Optional[dict[str, pd.DataFrame]] = None,
                  fundamentals_override: Optional[dict[str, Fundamentals]] = None,
                  bulk_deals_override: Optional[pd.DataFrame] = None,
                  skip_macro: bool = False,
                  skip_news: bool = False):
    log.info("═" * 75)
    log.info("  UNIFIED NIFTY 500 SCREENER v3.0 — STARTING RUN")
    log.info(f"  Portfolio size: ₹{CONFIG['portfolio_size_inr']:,} | "
             f"Risk/trade: {CONFIG['risk_per_trade_pct']*100:.1f}% | "
             f"Max position: {CONFIG['max_single_position_pct']*100:.0f}%")
    log.info("═" * 75)

    # ── Market regime (Nifty breakout/breakdown check, informational) ──
    log.info("Checking Nifty market regime (trend + momentum + breakout)...")
    market_regime = evaluate_nifty_regime()
    if market_regime:
        log.info(f"  → Regime: {market_regime['Regime']}")
    else:
        log.info("  → Regime check unavailable this run")

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

    # ── Regime-based position sizing multiplier — NEW v5 ────────────────
    regime_multiplier = get_regime_multiplier(market_regime)
    log.info(f"Regime position-size multiplier this run: {regime_multiplier:.2f}x "
              f"(regime: {market_regime['Regime'] if market_regime else 'unavailable, defaulted NEUTRAL'})")

    # ── Macro Risk Overlay — NEW v5 (VIX/Brent/USDINR real, breadth real, FII/DII best-effort) ──
    log.info("Computing macro risk overlay (VIX, Brent, USDINR, breadth, FII/DII)...")
    macro_overlay = fetch_macro_risk_overlay(liquid)
    macro_allocation_pct = macro_score_to_allocation_pct(macro_overlay["macro_score"])
    if macro_overlay["macro_score"] is not None:
        log.info(f"  → Macro_Score: {macro_overlay['macro_score']}/100 "
                  f"(from {macro_overlay['components_used']}) → allocation {macro_allocation_pct*100:.0f}%")
    else:
        log.info("  → Macro_Score unavailable this run (no components fetched) — allocation not reduced")
    if macro_overlay["components_skipped"]:
        log.info(f"  → Skipped (honest, not defaulted): {macro_overlay['components_skipped']}")

    combined_multiplier = regime_multiplier * macro_allocation_pct

    # ── Relative Strength (RS Rank + Sector RS) — NEW v5, highest priority ──
    log.info("Computing 12M relative strength ranks across the scanned universe...")
    rs_data = compute_universe_relative_strength(liquid)
    sector_stats = rs_data.get("_sector_stats", {})
    log.info(f"  → RS ranks computed for {len([k for k in rs_data if k != '_sector_stats'])} stocks "
              f"across {len(sector_stats)} sector buckets")

    # ── SWING BOOK ──
    log.info("Evaluating SWING book (Minervini Trend Template + true VCP + RS Rank gate)...")
    swing_results = []
    for sym, df in liquid.items():
        if len(df) < CONFIG["min_bars_required"]:
            continue
        rs_rank = rs_data.get(sym, {}).get("rs_rank_12m")
        r = evaluate_swing(sym, df, rs_rank=rs_rank, regime_multiplier=combined_multiplier)
        if r:
            swing_results.append(r)
    swing_df = pd.DataFrame(swing_results).sort_values("Risk_%") if swing_results else pd.DataFrame()
    log.info(f"  → {len(swing_df)} stocks passed swing criteria")

    # ── CORE BOOK ──
    log.info("Evaluating CORE book (Quality+Growth+Technical+Momentum+Sector RS)...")
    if fundamentals_override is not None:
        fund_map = fundamentals_override
    else:
        fund_map = {}
        for i, sym in enumerate(liquid.keys(), 1):
            fund_map[sym] = fetch_fundamentals(sym)
            if i % 20 == 0:
                log.info(f"  Fundamentals fetched: {i}/{len(liquid)}")
                time.sleep(1)

    core_results = []
    for sym, df in liquid.items():
        if len(df) < CONFIG["min_bars_required"]:
            continue
        sym_rs = rs_data.get(sym, {})
        rs_rank = sym_rs.get("rs_rank_12m")
        sector_rank_pct = sector_stats.get(sym_rs.get("sector", ""), {}).get("sector_rank_pct")
        r = evaluate_core(sym, df, fund_map.get(sym, Fundamentals()), rs_rank=rs_rank,
                            sector_rank_pct=sector_rank_pct, regime_multiplier=combined_multiplier)
        if r:
            core_results.append(r)
    core_df = (pd.DataFrame(core_results).sort_values("Score", ascending=False)
               if core_results else pd.DataFrame())
    log.info(f"  → {len(core_df)} stocks passed core criteria")

    # ── Sector Leader Bonus — NEW v5 ─────────────────────────────────────
    if not core_df.empty:
        core_df = apply_sector_leader_bonus(core_df)
        core_df = core_df.sort_values("Score", ascending=False)

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
            log.info(f"  → {len(crossover_df)} stocks in BOTH books")

    # ── FINAL PORTFOLIO CONSTRUCTION (sector caps + heat cap) — NEW v5 ──
    log.info("Constructing final capital-constrained portfolio selection...")
    portfolio_df = construct_final_portfolio(swing_df, core_df)
    if not portfolio_df.empty:
        n_included = int(portfolio_df["Included"].sum())
        log.info(f"  → {n_included}/{len(portfolio_df)} candidates fit within sector/heat caps")

    # ── REPORT ──
    if EXISTING_HOLDINGS:
        log.info(f"Reconciling against {len(EXISTING_HOLDINGS)} existing holding(s) in EXISTING_HOLDINGS...")
    swing_df = reconcile_with_holdings(swing_df)
    core_df = reconcile_with_holdings(core_df)
    crossover_df = reconcile_with_holdings(crossover_df)

    print_report(swing_df, core_df, crossover_df, matched_deals, macro_events, market_regime,
                  macro_overlay=macro_overlay, portfolio_df=portfolio_df, sector_stats=sector_stats)

    # ── EXPORT ──
    try:
        if not swing_df.empty: swing_df.to_csv(CONFIG["csv_output_swing"], index=False)
        if not core_df.empty: core_df.to_csv(CONFIG["csv_output_core"], index=False)
        if not crossover_df.empty: crossover_df.to_csv(CONFIG["csv_output_crossover"], index=False)
        if not matched_deals.empty: matched_deals.to_csv(CONFIG["csv_output_bulkdeals"], index=False)
        if not portfolio_df.empty: portfolio_df.to_csv(CONFIG["csv_output_portfolio"], index=False)
        log.info("✓ CSVs exported")
    except Exception as e:
        log.warning(f"CSV export skipped: {e}")

    return swing_df, core_df, crossover_df, matched_deals, market_regime


def print_report(swing_df, core_df, crossover_df, bulk_deals_df, macro_events, market_regime=None,
                   macro_overlay=None, portfolio_df=None, sector_stats=None):
    print_market_regime(market_regime)

    # ── Macro Risk Overlay — NEW v5 ──────────────────────────────────────
    print("\n" + "═" * 95)
    print("  🌡️  MACRO RISK OVERLAY")
    print("═" * 95)
    if macro_overlay is None or macro_overlay.get("macro_score") is None:
        print("  ⚠️  Macro_Score unavailable this run — no components could be fetched.")
        print("      Allocation NOT reduced (unknown backdrop isn't treated as a bad one).")
    else:
        print(f"  Macro_Score: {macro_overlay['macro_score']}/100  "
              f"(from {len(macro_overlay['components_used'])} components: {macro_overlay['components_used']})")
        for k, v in macro_overlay["detail"].items():
            print(f"    {k}: {v}")
        alloc_pct = macro_score_to_allocation_pct(macro_overlay["macro_score"])
        print(f"  → Allocation multiplier from macro backdrop: {alloc_pct*100:.0f}%")
    if macro_overlay and macro_overlay.get("components_skipped"):
        print(f"  Skipped this run (honest, not faked): {macro_overlay['components_skipped']}")

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

    print(f"\n{'─'*95}\n  🏭 SECTOR RELATIVE STRENGTH (this run's scanned universe)\n{'─'*95}")
    if not sector_stats:
        print("  Not computed this run.")
    else:
        ranked = sorted(sector_stats.items(),
                          key=lambda kv: (kv[1].get("sector_rank_pct") or 0), reverse=True)
        for sec, stats in ranked:
            if stats.get("avg_ret_12m") is None:
                continue
            pct = stats.get("sector_rank_pct")
            tag = "🟢 TOP20%" if pct and pct >= 80 else ("🔴 BOTTOM20%" if pct and pct <= 20 else "  ")
            print(f"  {tag} {sec:<20} 12M avg: {stats['avg_ret_12m']:+.1f}%  "
                  f"6M: {stats.get('avg_ret_6m', 0):+.1f}%  3M: {stats.get('avg_ret_3m', 0):+.1f}%  "
                  f"(n={stats['n_stocks']}, percentile {pct})")

    print(f"\n{'─'*95}\n  🎯 CROSSOVER — HIGHEST CONVICTION (passed BOTH books)\n{'─'*95}")
    if crossover_df.empty:
        print("  None today. Normal — crossover hits are rare by design.")
    else:
        cols = ["Symbol", "CMP", "Grade", "Score", "Confidence_%", "Swing_Confidence_%",
                 "Target", "Profit_%", "Stop_Loss", "Risk_%", "RR_Ratio",
                 "Shares_To_Buy", "Deploy_₹", "Max_Risk_₹"]
        cols = [c for c in cols if c in crossover_df.columns]
        # FIX v5: display label only — "Setup Score", internal column name
        # (Confidence_%) is untouched for backward compatibility.
        display_names = {"Confidence_%": "Setup Score", "Swing_Confidence_%": "Swing Setup Score"}
        print(crossover_df[cols].rename(columns=display_names).to_string(index=False))

    print(f"\n{'─'*95}\n  ⚡ SWING BOOK — 3-6 MONTH HOLD (Minervini Trend Template + VCP)\n{'─'*95}")
    if swing_df.empty:
        print("  No stocks passed the strict trend template + VCP filter today.")
    else:
        cols = ["Symbol", "CMP", "Target", "Profit_%", "Stop_Loss", "Risk_%", "RR_Ratio",
                 "RS_Rank", "Confidence_%", "Shares_To_Buy", "Deploy_₹", "Max_Risk_₹", "%_of_Portfolio"]
        cols = [c for c in cols if c in swing_df.columns]
        display_names = {"Confidence_%": "Setup Score"}
        print(swing_df[cols].head(10).rename(columns=display_names).to_string(index=False))
        if len(swing_df) > 0:
            top = swing_df.iloc[0]
            print(f"\n  🏆 TOP SWING PICK: {top['Symbol']} — Entry ₹{top['CMP']:.0f} → "
                  f"Target ₹{top['Target']:.0f} | SL ₹{top['Stop_Loss']:.0f} | "
                  f"R:R 1:{top['RR_Ratio']:.1f}")
            if "Shares_To_Buy" in top:
                print(f"     Buy {top['Shares_To_Buy']:.0f} shares | "
                      f"Deploy ₹{top['Deploy_₹']:,.0f} | Max Risk ₹{top['Max_Risk_₹']:,.0f}")
            if "Confidence_%" in top:
                print(f"     Setup Score: {top['Confidence_%']:.0f}% (rules-based heuristic, "
                      f"NOT a backtested win-rate — see Setup Drivers below)")
            if "News_Headlines" in top:
                print(f"     News: {top['News_Headlines'][:300]}")

        # Per-stock justification block for top 5
        print(f"\n  📋 JUSTIFICATION — Top 5 Swing Picks")
        for _, row in swing_df.head(5).iterrows():
            print(f"\n  ▸ {row['Symbol']} — Setup Score {row.get('Confidence_%', 'N/A')}%")
            print(f"    Setup Drivers: {row.get('Confidence_Factors', 'N/A')}")
            news_status = row.get("News_Status", "not_fetched")
            if news_status == "ok":
                print(f"    News:  {row['News_Headlines']}")
            else:
                print(f"    News:  {row.get('News_Headlines', 'Not available')}")

    print(f"\n{'─'*95}\n  📈 CORE BOOK — 1-2 YEAR HOLD (Quality+Growth+Technical+Momentum)\n{'─'*95}")
    if core_df.empty:
        print("  No stocks passed the core scoring filter today.")
    else:
        cols = ["Symbol", "CMP", "Grade", "Score", "RS_Rank", "Sector_Label", "Confidence_%", "Target", "Profit_%",
                 "Stop_Loss", "Risk_%", "RR_Ratio", "PE", "ROE_%", "EPS_Growth_%",
                 "Shares_To_Buy", "Deploy_₹"]
        cols = [c for c in cols if c in core_df.columns]
        display_names = {"Confidence_%": "Setup Score"}
        for grade in ["STRONG BUY", "BUY", "WATCH"]:
            sub = core_df[core_df["Grade"] == grade]
            if not sub.empty:
                print(f"\n  [{grade}] — {len(sub)} stocks")
                print(sub[cols].head(10).rename(columns=display_names).to_string(index=False))
        if not core_df.empty:
            top = core_df.iloc[0]
            print(f"\n  🏆 TOP CORE PICK: {top['Symbol']} — Entry ₹{top['CMP']:.0f} → "
                  f"Target ₹{top['Target']:.0f} | SL ₹{top['Stop_Loss']:.0f} | "
                  f"Score {top['Score']}/100")
            if "Shares_To_Buy" in top:
                print(f"     Buy {top['Shares_To_Buy']:.0f} shares | "
                      f"Deploy ₹{top['Deploy_₹']:,.0f} | Max Risk ₹{top['Max_Risk_₹']:,.0f}")
            if "Confidence_%" in top:
                print(f"     Setup Score: {top['Confidence_%']:.0f}% (rules-based heuristic, "
                      f"NOT a backtested win-rate — see Setup Drivers below)")

        # Per-stock justification block for top 5 STRONG BUY / BUY picks
        justifiable = core_df[core_df["Grade"].isin(["STRONG BUY", "BUY"])].head(5)
        if not justifiable.empty:
            print(f"\n  📋 JUSTIFICATION — Top 5 Core Picks (STRONG BUY / BUY tier)")
            for _, row in justifiable.iterrows():
                print(f"\n  ▸ {row['Symbol']} — Score {row.get('Score','N/A')}/100, "
                      f"Setup Score {row.get('Confidence_%', 'N/A')}%")
                if row.get("Sector_Leader_Bonus", 0):
                    print(f"    Sector Leader Bonus: +{row['Sector_Leader_Bonus']} ({row.get('Sector_Leader_Reasons', '')})")
                print(f"    Quant: {row.get('Reasons', 'N/A')}")
                print(f"    Setup Drivers: {row.get('Confidence_Factors', 'N/A')}")
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

    print(f"\n{'─'*95}\n  💰 FINAL PORTFOLIO SELECTION (sector caps + portfolio heat cap applied)\n{'─'*95}")
    if portfolio_df is None or portfolio_df.empty:
        print("  No candidates to allocate this run.")
    else:
        included = portfolio_df[portfolio_df["Included"]]
        excluded = portfolio_df[~portfolio_df["Included"]]
        print(f"  Rules: max {CONFIG['portfolio_max_sector_exposure_pct']*100:.0f}% capital/sector, "
              f"max {CONFIG['portfolio_max_positions_per_sector']} positions/sector, "
              f"max {CONFIG['portfolio_max_heat_pct']*100:.0f}% portfolio heat")
        print(f"\n  ✅ INCLUDED ({len(included)}):")
        if not included.empty:
            print(included[["Symbol", "Book", "Sector", "Conviction", "Deploy_₹", "Risk_%"]]
                   .to_string(index=False))
        if not excluded.empty:
            print(f"\n  ⛔ EXCLUDED ({len(excluded)}) — reason each:")
            for _, r in excluded.iterrows():
                print(f"    {r['Symbol']}: {r['Selection_Note']}")

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


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE — merged into this same file per user request.
# All functions below use the SAME evaluate_swing/evaluate_core/CONFIG/
# Fundamentals/calculate_position_size/SECTOR_MAP defined above — no
# separate file, no dynamic import, single source of truth.
# ═══════════════════════════════════════════════════════════════════════════════════════════════



# ═════════════════════════════════════════════════════════════════════════
# 1. WALK-FORWARD BACKTESTER
# ═════════════════════════════════════════════════════════════════════════

def backtest_signal(symbol: str, df: pd.DataFrame, fund: Fundamentals,
                     mode: str = "swing",
                     hold_days: int = 90,
                     step_days: int = 21) -> list[dict]:
    """
    Walk-forward backtest for a single stock.

    For each step:
      1. Slice df up to window_end (simulates "only having data up to today")
      2. Run evaluate_swing or evaluate_core on that slice
      3. If signal fires, look at actual price move over next hold_days
      4. Classify as WIN (hit target before stop), LOSS (hit stop first),
         or OPEN (neither hit within hold_days — use final price)

    Path-dependent outcome (Gemini's good idea, implemented correctly here
    with NO look-ahead bias — we use the ACTUAL forward price series, not
    shifted labels computed during training):
      WIN  = high in forward window >= target price (target hit first)
      LOSS = low in forward window <= stop loss AND low hit before high hit target
    """
    results = []
    txn_cost = CONFIG.get("transaction_cost_pct", 0.0015)
    min_bars  = CONFIG["min_bars_required"]

    for i in range(min_bars, len(df) - hold_days, step_days):
        window_end = i
        forward_start = i + 1
        forward_end   = min(i + 1 + hold_days, len(df))

        df_window  = df.iloc[:window_end].copy()
        df_forward = df.iloc[forward_start:forward_end].copy()

        if df_forward.empty:
            continue

        # Clear rejection log so we get a clean signal decision
        REJECTION_LOG.clear()

        try:
            if mode == "swing":
                signal = evaluate_swing(symbol, df_window)
            else:
                signal = evaluate_core(symbol, df_window, fund)
        except Exception:
            continue

        if signal is None:
            continue

        entry  = signal["CMP"]
        # Both evaluate_swing and evaluate_core always set "Target" — the
        # fallback below only protects against a future signal shape change,
        # it isn't expected to trigger today.
        target = signal.get("Target", entry * 1.20)
        sl     = signal["Stop_Loss"]

        if entry <= 0 or sl >= entry:
            continue

        # Path-dependent outcome check (no look-ahead — using actual future prices)
        fwd_high = df_forward["High"].values
        fwd_low  = df_forward["Low"].values
        fwd_close = df_forward["Close"].values

        outcome = "OPEN"
        for day_idx, (h, l) in enumerate(zip(fwd_high, fwd_low)):
            target_hit = h >= target
            sl_hit     = l <= sl
            if target_hit and not sl_hit:
                outcome = "WIN"; break
            elif sl_hit and not target_hit:
                outcome = "LOSS"; break
            elif target_hit and sl_hit:
                # Both hit same day — conservative: call it LOSS
                outcome = "LOSS"; break

        final_price = fwd_close[-1] if outcome == "OPEN" else (target if outcome == "WIN" else sl)
        gross_return = (final_price - entry) / entry * 100
        net_return   = gross_return - (txn_cost * 100 * 2)  # round-trip

        signal_date = df.index[window_end] if hasattr(df.index[window_end], 'date') else window_end

        results.append({
            "Symbol":      symbol,
            "Signal_Date": str(signal_date)[:10],
            "Mode":        mode,
            "Entry":       round(entry, 2),
            "Target":      round(target, 2),
            "Stop_Loss":   round(sl, 2),
            "Outcome":     outcome,
            "Gross_Return_%": round(gross_return, 2),
            "Net_Return_%":   round(net_return, 2),
            "Hold_Days":   hold_days,
        })

    return results


# ═════════════════════════════════════════════════════════════════════════
# 2. COVID-ERA STRESS TEST (Jan 2020 — Jun 2020)
# ═════════════════════════════════════════════════════════════════════════

def run_covid_stress_test(price_data: dict, fund_map: dict = None) -> pd.DataFrame:
    """
    Runs the screener's signals specifically on the Covid crash and recovery
    period (Jan 2020 to Jun 2020) — the most extreme stress period in
    recent Indian market history.

    What we expect to see (honest):
      - Swing Book signals should STOP firing around Feb-Mar 2020 as
        prices crash below all moving averages — this is CORRECT behaviour
        (the Minervini template correctly rejects downtrending stocks)
      - Some signals may fire in Apr-Jun 2020 during the recovery —
        these should show strong win rates since the rebound was sharp
      - Core Book (fundamentals) may continue to flag cheap stocks even
        during the crash — testing whether "buy cheap quality" worked
        better than "buy technical breakouts" during Covid
    """
    log.info("Running Covid-era stress test (Jan 2020 – Jun 2020)...")

    COVID_START = "2019-01-01"  # need data from a year BEFORE for indicators
    COVID_SIGNAL_START = pd.Timestamp("2020-01-01")
    COVID_SIGNAL_END   = pd.Timestamp("2020-06-30")

    all_results = []

    for sym, df_full in price_data.items():
        df_full.index = pd.to_datetime(df_full.index)
        df_covid = df_full[df_full.index >= COVID_START].copy()

        if len(df_covid) < CONFIG["min_bars_required"] + 90:
            continue

        fund = (fund_map or {}).get(sym, Fundamentals())

        for mode in ["swing", "core"]:
            results = backtest_signal(sym, df_covid, fund, mode=mode,
                                       hold_days=90, step_days=21)
            # Filter to signals that fired during the Covid window only
            covid_results = [r for r in results
                              if COVID_SIGNAL_START <= pd.Timestamp(r["Signal_Date"]) <= COVID_SIGNAL_END]
            all_results.extend(covid_results)

    if not all_results:
        log.warning("No signals fired during Covid window — check data coverage")
        return pd.DataFrame()

    df_out = pd.DataFrame(all_results)
    log.info(f"Covid stress test: {len(df_out)} signal instances across "
              f"{df_out['Symbol'].nunique()} stocks")
    return df_out


# ═════════════════════════════════════════════════════════════════════════
# 3. AGGREGATE STATISTICS
# ═════════════════════════════════════════════════════════════════════════

def compute_stats(results_df: pd.DataFrame, label: str = "Backtest") -> dict:
    """Compute hit rate, average return, expectancy, and R-multiple stats."""
    if results_df.empty:
        return {}

    wins   = results_df[results_df["Outcome"] == "WIN"]
    losses = results_df[results_df["Outcome"] == "LOSS"]
    open_  = results_df[results_df["Outcome"] == "OPEN"]

    n      = len(results_df)
    hit_rate = len(wins) / n * 100 if n > 0 else 0
    avg_win  = wins["Net_Return_%"].mean() if len(wins) > 0 else 0
    avg_loss = losses["Net_Return_%"].mean() if len(losses) > 0 else 0
    avg_ret  = results_df["Net_Return_%"].mean()

    # Expectancy = (hit_rate * avg_win) + ((1 - hit_rate) * avg_loss)
    expectancy = (hit_rate/100 * avg_win) + ((1 - hit_rate/100) * avg_loss)

    stats = {
        "label":       label,
        "n_signals":   n,
        "n_wins":      len(wins),
        "n_losses":    len(losses),
        "n_open":      len(open_),
        "hit_rate_%":  round(hit_rate, 1),
        "avg_win_%":   round(avg_win, 2),
        "avg_loss_%":  round(avg_loss, 2),
        "avg_return_%": round(avg_ret, 2),
        "expectancy_%": round(expectancy, 2),
    }
    return stats


def print_backtest_report(full_df: pd.DataFrame, covid_df: pd.DataFrame):
    sep = "═" * 90

    print(f"\n{sep}")
    print("  📊 BACKTEST REPORT — WALK-FORWARD VALIDATION")
    print(sep)
    print("\n  ⚠️  ALL NUMBERS BELOW ARE BASED ON SIMULATED DATA (sandbox has no")
    print("  live Yahoo/NSE access). Run on your own machine for real historical")
    print("  prices. The methodology is correct — only the data is simulated.\n")

    if not full_df.empty:
        for mode in ["swing", "core"]:
            sub = full_df[full_df["Mode"] == mode]
            if sub.empty:
                continue
            stats = compute_stats(sub, f"{mode.upper()} BOOK")
            print(f"  {'─'*80}")
            print(f"  {stats['label']}")
            print(f"  {'─'*80}")
            print(f"  Signals:   {stats['n_signals']}  |  Wins: {stats['n_wins']}  |  "
                  f"Losses: {stats['n_losses']}  |  Open: {stats['n_open']}")
            print(f"  Hit Rate:  {stats['hit_rate_%']}%")
            print(f"  Avg Win:   +{stats['avg_win_%']}%  |  Avg Loss: {stats['avg_loss_%']}%")
            print(f"  Avg Return (net of costs): {stats['avg_return_%']}%")
            print(f"  Expectancy per trade:      {stats['expectancy_%']}%")

    print(f"\n  {'─'*80}")
    print("  🦠 COVID-ERA STRESS TEST (Jan – Jun 2020)")
    print(f"  {'─'*80}")
    if covid_df.empty:
        print("  No signals fired during Covid window — as expected for Swing Book.")
        print("  Minervini Trend Template correctly halts all swing signals when")
        print("  prices crash below their moving averages. This is a feature.")
        print("  Core Book signals (fundamentals-based) may still fire on cheap stocks.")
    else:
        for mode in ["swing", "core"]:
            sub = covid_df[covid_df["Mode"] == mode]
            if sub.empty:
                print(f"  {mode.upper()} BOOK: No signals fired (correct for crash environment)")
                continue
            stats = compute_stats(sub, f"Covid {mode.upper()}")
            print(f"\n  {stats['label']}: {stats['n_signals']} signals | "
                  f"Hit rate {stats['hit_rate_%']}% | "
                  f"Avg return {stats['avg_return_%']}%")
            print(f"  Expectancy: {stats['expectancy_%']}%")

    print(f"\n{sep}")
    print("  HONEST INTERPRETATION GUIDE")
    print(sep)
    for line in [
        "Hit rate 55-65%: Good. This is what professional systematic traders achieve.",
        "Hit rate >70%:   Suspicious. Check for data leakage or overfitting to one period.",
        "Hit rate <45%:   Poor. Signal may be random. Widen stop or tighten entry criteria.",
        "Expectancy >0:   Positive edge. Even with losses, you make money overall.",
        "Expectancy <0:   No edge. Don't trade this signal regardless of hit rate.",
        "Covid period:    Swing signals going SILENT is CORRECT — the template is working.",
        "  Core signals firing during Covid at cheap PEs = valid contrarian entries if",
        "  fundamentals were genuinely intact (not just cheap because business was broken).",
    ]:
        print(f"  {line}")
    print(f"{sep}\n")


# ═════════════════════════════════════════════════════════════════════════
# 4. MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

def run_backtest(price_data: dict, fund_map: dict = None,
                  hold_days_swing: int = 90, hold_days_core: int = None):
    """
    Full walk-forward backtest across all provided stocks.
    Results include the Covid stress test as a sub-report.
    hold_days_core defaults to CONFIG['core_backtest_hold_days'] (365)
    after the Learning 3 fix — Core is a 1-2yr book, not 6-month.
    """
    if hold_days_core is None:
        hold_days_core = CONFIG.get("core_backtest_hold_days", 365)
    log.info(f"Starting walk-forward backtest on {len(price_data)} stocks...")
    all_results = []

    for i, (sym, df) in enumerate(price_data.items(), 1):
        fund = (fund_map or {}).get(sym, Fundamentals())

        for mode, hold in [("swing", hold_days_swing), ("core", hold_days_core)]:
            results = backtest_signal(sym, df, fund, mode=mode,
                                       hold_days=hold, step_days=21)
            all_results.extend(results)

        if i % 10 == 0:
            log.info(f"  Backtested {i}/{len(price_data)} stocks...")

    full_df  = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    covid_df = run_covid_stress_test(price_data, fund_map)
    print_backtest_report(full_df, covid_df)

    # Save
    if not full_df.empty:
        full_df.to_csv(os.path.join(_THIS_DIR, "backtest_results.csv"), index=False)
        log.info("✓ Backtest results saved to backtest_results.csv")
    if not covid_df.empty:
        covid_df.to_csv(os.path.join(_THIS_DIR, "backtest_covid.csv"), index=False)
        log.info("✓ Covid stress test results saved to backtest_covid.csv")

    return full_df, covid_df


# ═════════════════════════════════════════════════════════════════════════
# FULL WALK-FORWARD VALIDATION SUITE — NEW v5
# ═════════════════════════════════════════════════════════════════════════
# Implements the monthly-re-screen methodology: on the first trading day
# of every month, run the screener using ONLY data available up to that
# day, take the top 5 Swing + top 5 Core picks, then measure what actually
# happened 1/3/6/12 months later. This is what actually validates whether
# the strategy has real edge — everything above this point in the file
# tests "does the mechanism work correctly", this tests "does the
# STRATEGY actually make money."
#
# HONEST SCOPE NOTE: this is real, working code for the full methodology
# requested (CAGR/Sharpe/Sortino/Calmar/benchmark comparison/equity curve/
# trade log/monthly returns/sector performance/PDF report). What it CANNOT
# do is fabricate the multi-year run itself — that needs several years of
# real historical data for ~500 stocks, downloaded on a machine with
# actual internet access to Yahoo Finance/NSE, and will take a long time
# to execute (hundreds of monthly re-screens x hundreds of stocks). Run
# it on your own machine with real downloaded data; this sandbox can't
# reach yfinance to produce real numbers, and fabricating them would be
# worse than not having them.
#
# Two benchmarks (Nifty50 via ^NSEI) are genuinely fetchable via yfinance.
# Nifty500, Nifty Momentum 30, and Nifty Alpha 50 do NOT have reliable
# free yfinance tickers as of this writing — this degrades honestly
# (reports "unavailable", doesn't substitute Nifty50 silently as a stand-in).

def compute_portfolio_metrics(equity_curve: pd.Series, risk_free_rate_annual: float = 0.065) -> dict:
    """
    equity_curve: pd.Series indexed by date, value = portfolio value.
    Returns CAGR, annualized volatility, Sharpe, Sortino, Calmar, Max Drawdown.
    Standard formulas — no proprietary tuning, so results are directly
    comparable to any other strategy report using the same definitions.
    """
    if equity_curve is None or len(equity_curve) < 2:
        return {"error": "Equity curve too short to compute metrics"}

    equity_curve = equity_curve.sort_index()
    daily_returns = equity_curve.pct_change().dropna()
    n_days = len(equity_curve)
    years = max(n_days / 252, 1 / 252)

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0

    ann_vol = daily_returns.std() * np.sqrt(252)
    daily_rf = risk_free_rate_annual / 252
    excess_returns = daily_returns - daily_rf
    sharpe = (excess_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "annualized_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "calmar_ratio": round(calmar, 2),
        "years_covered": round(years, 2),
        "drawdown_series": drawdown,
    }


def compute_trade_metrics(trades_df: pd.DataFrame) -> dict:
    """
    trades_df needs columns: Outcome ('WIN'/'LOSS'), Net_Return_%, Hold_Days.
    Standard trade-level stats — same definitions used by any prop desk
    or trading-journal tool, nothing bespoke.
    """
    if trades_df is None or trades_df.empty:
        return {"error": "No trades to compute metrics from"}

    wins = trades_df[trades_df["Outcome"] == "WIN"]
    losses = trades_df[trades_df["Outcome"] == "LOSS"]
    n_total = len(trades_df)
    win_rate = len(wins) / n_total * 100 if n_total > 0 else 0

    avg_winner = wins["Net_Return_%"].mean() if not wins.empty else 0
    avg_loser = losses["Net_Return_%"].mean() if not losses.empty else 0

    gross_profit = wins["Net_Return_%"].sum() if not wins.empty else 0
    gross_loss = abs(losses["Net_Return_%"].sum()) if not losses.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    expectancy = (win_rate / 100 * avg_winner) + ((1 - win_rate / 100) * avg_loser)

    return {
        "n_trades": n_total,
        "win_rate_pct": round(win_rate, 1),
        "avg_winner_pct": round(avg_winner, 2),
        "avg_loser_pct": round(avg_loser, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf (no losing trades)",
        "expectancy_pct_per_trade": round(expectancy, 2),
        "avg_holding_period_days": round(trades_df["Hold_Days"].mean(), 1) if "Hold_Days" in trades_df.columns else None,
        "largest_winner_pct": round(trades_df["Net_Return_%"].max(), 2),
        "largest_loser_pct": round(trades_df["Net_Return_%"].min(), 2),
    }


def fetch_benchmark_returns(start_date, end_date) -> dict:
    """
    Real fetch for Nifty50. Nifty500/Momentum30/Alpha50 are explicitly
    marked unavailable rather than silently substituted — a stand-in
    benchmark would make the alpha number meaningless without saying so.
    """
    results = {}
    try:
        import yfinance as yf
        df = yf.download("^NSEI", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            total_ret = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
            years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)
            cagr = (1 + total_ret) ** (1 / years) - 1
            results["Nifty50"] = {"total_return_pct": round(total_ret * 100, 2), "cagr_pct": round(cagr * 100, 2)}
    except Exception as e:
        results["Nifty50"] = {"error": f"{type(e).__name__}: {e}"}

    for name in ("Nifty500", "Nifty_Momentum_30", "Nifty_Alpha_50"):
        results[name] = {"error": "No reliable free yfinance ticker for this index as of this writing — "
                                    "use NSE's own index data downloads if you need this comparison"}
    return results


def compute_monthly_returns_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Groups closed trades by the month they were CLOSED (entry + hold
    period) and sums net return — a simple, transparent monthly P&L view,
    not a true equity-curve resample (which needs position-level dating
    this trade log doesn't carry by default)."""
    if trades_df.empty or "Signal_Date" not in trades_df.columns:
        return pd.DataFrame()
    df = trades_df.copy()
    df["Signal_Date"] = pd.to_datetime(df["Signal_Date"])
    df["Month"] = df["Signal_Date"].dt.to_period("M")
    monthly = df.groupby("Month")["Net_Return_%"].agg(["sum", "mean", "count"]).reset_index()
    monthly.columns = ["Month", "Total_Return_%", "Avg_Return_Per_Trade_%", "N_Trades"]
    return monthly


def compute_sector_performance_table(trades_df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    """Requires a {symbol: sector} map — pass in SECTOR_MAP / get_sector_label
    from the main screener module (already loaded as `mod` in this file)."""
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["Sector"] = df["Symbol"].map(lambda s: sector_map.get(s, "Diversified/Other"))
    sector_perf = df.groupby("Sector")["Net_Return_%"].agg(["mean", "sum", "count"]).reset_index()
    sector_perf.columns = ["Sector", "Avg_Return_%", "Total_Return_%", "N_Trades"]
    return sector_perf.sort_values("Avg_Return_%", ascending=False)


def generate_pdf_report(equity_curve: pd.Series, drawdown_series: pd.Series,
                          monthly_returns: pd.DataFrame, sector_perf: pd.DataFrame,
                          portfolio_metrics: dict, trade_metrics: dict,
                          output_path: str = os.path.join(_THIS_DIR, "strategy_report.pdf")):
    """
    Real PDF via matplotlib (no extra heavyweight dependency like reportlab
    needed) — equity curve, drawdown curve, and a text summary page of all
    the metrics. Requires matplotlib (add to requirements.txt if missing).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(output_path) as pdf:
            # Page 1: summary text
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            lines = ["STRATEGY BACKTEST REPORT", "=" * 40, ""]
            lines.append("PORTFOLIO METRICS")
            for k, v in portfolio_metrics.items():
                if k == "drawdown_series":
                    continue
                lines.append(f"  {k}: {v}")
            lines.append("")
            lines.append("TRADE METRICS")
            for k, v in trade_metrics.items():
                lines.append(f"  {k}: {v}")
            ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=9, family="monospace")
            pdf.savefig(fig)
            plt.close(fig)

            # Page 2: equity curve
            if equity_curve is not None and len(equity_curve) > 1:
                fig, ax = plt.subplots(figsize=(11, 6))
                ax.plot(equity_curve.index, equity_curve.values)
                ax.set_title("Equity Curve")
                ax.set_ylabel("Portfolio Value (₹)")
                pdf.savefig(fig)
                plt.close(fig)

            # Page 3: drawdown curve
            if drawdown_series is not None and len(drawdown_series) > 1:
                fig, ax = plt.subplots(figsize=(11, 6))
                ax.fill_between(drawdown_series.index, drawdown_series.values * 100, 0, color="red", alpha=0.4)
                ax.set_title("Drawdown Curve")
                ax.set_ylabel("Drawdown (%)")
                pdf.savefig(fig)
                plt.close(fig)

            # Page 4: monthly returns table
            if monthly_returns is not None and not monthly_returns.empty:
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                ax.table(cellText=monthly_returns.round(2).values,
                          colLabels=monthly_returns.columns, loc="center")
                ax.set_title("Monthly Returns")
                pdf.savefig(fig)
                plt.close(fig)

        log.info(f"✓ PDF report saved: {output_path}")
        return output_path
    except ImportError:
        log.warning("matplotlib not installed — PDF report skipped. "
                     "Run: pip install matplotlib --break-system-packages")
        return None
    except Exception as e:
        log.warning(f"PDF report generation failed: {e}")
        return None


def run_full_walkforward_backtest(price_data: dict, fund_map: dict = None,
                                     start_date: str = "2021-01-01", end_date: str = "2026-01-01",
                                     top_n_picks: int = 5, portfolio_size_inr: float = None):
    """
    THE methodology requested: monthly re-screen -> top 5 swing + top 5
    core picks -> measure forward returns -> full metrics suite ->
    benchmark comparison -> PDF report.

    FIXED v6 — EQUITY CURVE: v5 used a crude "apply average trade return
    x 10% notional allocation" approximation, which doesn't reflect real
    capital constraints or overlapping positions. This version maintains
    an actual ledger: real starting cash, real position sizing via
    calculate_position_size() (same function the live screener uses),
    cash is actually deducted when a position opens and only returned
    when it exits (via real target/stop-loss/hold-expiry logic, checked
    month-by-month against the real price path), and equity at each
    point = cash + mark-to-market value of whatever's still open. New
    signals are skipped if there isn't enough free cash, or if the stock
    is already held — both real constraints a live account would face.

    REQUIRES real multi-year price_data (2021+) for this to mean anything.
    """
    portfolio_size_inr = portfolio_size_inr or CONFIG.get("portfolio_size_inr", 136000)
    log.info(f"Starting FULL walk-forward validation: {start_date} to {end_date}, "
              f"top {top_n_picks} picks/book/month, starting capital ₹{portfolio_size_inr:,.0f}")

    # FIX v7 — CRITICAL: check_earnings_blackout() always queries the REAL
    # current-day earnings calendar from yfinance (there's no free source
    # of point-in-time HISTORICAL earnings dates). During a backtest
    # simulating e.g. January 2020, that check is comparing today's real
    # 2026 earnings date against a simulated 2020 trading day — the
    # comparison is meaningless, and it also makes a slow live network
    # call per qualifying stock per simulated month, which is what was
    # causing the multi-minute gaps between log lines. Disabled here for
    # the duration of the backtest only; restored afterward so live
    # screening (run_screener()) keeps using it normally.
    _earnings_check_was_enabled = CONFIG.get("earnings_check_enabled", True)
    if _earnings_check_was_enabled:
        log.info("Disabling earnings-blackout check for this backtest run — it can only check "
                  "TODAY's real earnings calendar, which is meaningless against simulated "
                  "historical months, and was the main cause of the slow per-stock network calls.")
        CONFIG["earnings_check_enabled"] = False

    cash = portfolio_size_inr
    open_positions = []   # list of dicts, one per open position
    all_trades = []
    equity_dates, equity_values = [], []

    date_range = pd.date_range(start_date, end_date, freq="MS")  # month start
    for month_start in date_range:
        # ── 1. Close any position whose target/SL was hit or hold period
        #        expired, checked against the REAL price path since entry ──
        still_open = []
        for pos in open_positions:
            full_df = price_data[pos["symbol"]]
            cur_idx = min(full_df.index.get_indexer([month_start], method="nearest")[0], len(full_df) - 1)
            entry_idx = pos["entry_idx"]
            scheduled_exit_idx = pos["scheduled_exit_idx"]
            check_end_idx = min(cur_idx, scheduled_exit_idx)

            if check_end_idx <= entry_idx:
                still_open.append(pos)
                continue

            path = full_df["Close"].iloc[entry_idx:check_end_idx + 1]
            hit_target = pos["target"] and (path >= pos["target"]).any()
            hit_sl = pos["stop_loss"] and (path <= pos["stop_loss"]).any()
            expired = cur_idx >= scheduled_exit_idx

            if hit_target or hit_sl or expired:
                if hit_sl:
                    exit_price, outcome = pos["stop_loss"], "LOSS"
                elif hit_target:
                    exit_price, outcome = pos["target"], "WIN"
                else:
                    exit_price = full_df["Close"].iloc[check_end_idx]
                    outcome = "WIN" if exit_price > pos["entry_price"] else "LOSS"

                proceeds = pos["shares"] * exit_price
                cash += proceeds   # REAL cash return, not a notional approximation
                net_return = ((exit_price / pos["entry_price"] - 1) * 100
                               - CONFIG.get("transaction_cost_pct", 0.1) * 2)
                hold_days_actual = check_end_idx - entry_idx
                all_trades.append({
                    "Symbol": pos["symbol"], "Signal_Date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "Mode": pos["mode"], "Entry": pos["entry_price"], "Exit": round(exit_price, 2),
                    "Outcome": outcome, "Net_Return_%": round(net_return, 2),
                    "Hold_Days": hold_days_actual, "Shares": pos["shares"], "Deploy_₹": pos["deploy"],
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── 2. Screen fresh signals using ONLY data available before this month ──
        month_candidates = []
        for sym, df in price_data.items():
            window = df[df.index < month_start]
            if len(window) < CONFIG.get("min_bars_required", 260):
                continue
            fund = (fund_map or {}).get(sym, Fundamentals())
            for mode, hold in [("swing", 90), ("core", CONFIG.get("core_backtest_hold_days", 365))]:
                try:
                    sig = (evaluate_swing(sym, window) if mode == "swing"
                            else evaluate_core(sym, window, fund))
                except Exception:
                    sig = None
                if sig:
                    month_candidates.append((sig, mode, hold))

        held_symbols = {p["symbol"] for p in open_positions}
        for mode in ("swing", "core"):
            book = [t for t in month_candidates if t[1] == mode]
            book.sort(key=lambda t: t[0].get("Score", t[0].get("RR_Ratio", 0)), reverse=True)
            for sig, mode_, hold in book[:top_n_picks]:
                if sig["Symbol"] in held_symbols:
                    continue   # don't pyramid into an already-open position
                entry_price = sig["CMP"]
                stop_loss = sig.get("Stop_Loss")
                if stop_loss is None or entry_price <= stop_loss:
                    continue

                # REAL position sizing — same function the live screener uses
                sizing = calculate_position_size(entry_price, stop_loss)
                if not sizing or sizing.get("shares", 0) <= 0:
                    continue
                deploy = sizing["deploy"]
                if deploy > cash:
                    continue   # REAL cash constraint — can't buy what you can't afford

                full_df = price_data[sig["Symbol"]]
                entry_idx = full_df.index.get_indexer([month_start], method="nearest")[0]
                scheduled_exit_idx = min(entry_idx + hold, len(full_df) - 1)

                cash -= deploy   # REAL cash deduction
                open_positions.append({
                    "symbol": sig["Symbol"], "mode": mode_, "entry_price": entry_price,
                    "shares": sizing["shares"], "deploy": deploy,
                    "target": sig.get("Target"), "stop_loss": stop_loss,
                    "entry_idx": entry_idx, "scheduled_exit_idx": scheduled_exit_idx,
                    "entry_date": month_start,
                })
                held_symbols.add(sig["Symbol"])

        # ── 3. Mark-to-market: equity = cash + current value of open positions ──
        mtm_value = 0.0
        for pos in open_positions:
            full_df = price_data[pos["symbol"]]
            cur_idx = min(full_df.index.get_indexer([month_start], method="nearest")[0], len(full_df) - 1)
            mtm_value += pos["shares"] * full_df["Close"].iloc[cur_idx]
        equity_dates.append(month_start)
        equity_values.append(cash + mtm_value)

    # Restore the earnings-check setting so run_screener() (live use) isn't affected
    CONFIG["earnings_check_enabled"] = _earnings_check_was_enabled

    # ── Close remaining open positions at the end of the period for final metrics ──
    for pos in open_positions:
        full_df = price_data[pos["symbol"]]
        exit_price = full_df["Close"].iloc[-1]
        net_return = (exit_price / pos["entry_price"] - 1) * 100 - CONFIG.get("transaction_cost_pct", 0.1) * 2
        outcome = "WIN" if exit_price > pos["entry_price"] else "LOSS"
        all_trades.append({
            "Symbol": pos["symbol"], "Signal_Date": pos["entry_date"].strftime("%Y-%m-%d"),
            "Mode": pos["mode"], "Entry": pos["entry_price"], "Exit": round(exit_price, 2),
            "Outcome": outcome, "Net_Return_%": round(net_return, 2),
            "Hold_Days": len(full_df) - 1 - pos["entry_idx"], "Shares": pos["shares"],
            "Deploy_₹": pos["deploy"], "Note": "Force-closed at end of backtest period",
        })

    trades_df = pd.DataFrame(all_trades)
    equity_curve = pd.Series(equity_values, index=equity_dates)

    portfolio_metrics = compute_portfolio_metrics(equity_curve)
    trade_metrics = compute_trade_metrics(trades_df) if not trades_df.empty else {"error": "No trades generated"}
    benchmark = fetch_benchmark_returns(start_date, end_date)
    monthly_returns = compute_monthly_returns_table(trades_df) if not trades_df.empty else pd.DataFrame()
    sector_perf = (compute_sector_performance_table(trades_df, SECTOR_MAP)
                    if not trades_df.empty else pd.DataFrame())

    log.info("=" * 70)
    log.info("WALK-FORWARD BACKTEST COMPLETE")
    log.info(f"Final equity: ₹{equity_values[-1]:,.0f} (started ₹{portfolio_size_inr:,.0f}) "
              f"| {len(open_positions)} positions still open at cutoff")
    log.info(f"Portfolio metrics: {portfolio_metrics}")
    log.info(f"Trade metrics: {trade_metrics}")
    log.info(f"Benchmark: {benchmark}")
    log.info("=" * 70)

    drawdown_series = portfolio_metrics.get("drawdown_series")
    pdf_path = generate_pdf_report(equity_curve, drawdown_series, monthly_returns, sector_perf,
                                     portfolio_metrics, trade_metrics)

    if not trades_df.empty:
        trades_df.to_csv(os.path.join(_THIS_DIR, "walkforward_trade_log.csv"), index=False)
    if not monthly_returns.empty:
        monthly_returns.to_csv(os.path.join(_THIS_DIR, "walkforward_monthly_returns.csv"), index=False)
    if not sector_perf.empty:
        sector_perf.to_csv(os.path.join(_THIS_DIR, "walkforward_sector_performance.csv"), index=False)

    return {
        "trades_df": trades_df, "equity_curve": equity_curve,
        "portfolio_metrics": portfolio_metrics, "trade_metrics": trade_metrics,
        "benchmark": benchmark, "monthly_returns": monthly_returns,
        "sector_performance": sector_perf, "pdf_report_path": pdf_path,
        "final_cash": cash, "final_open_positions": len(open_positions),
    }


# ═════════════════════════════════════════════════════════════════════════
# TRAIN / TEST / VALIDATE — NEW v6
# ═════════════════════════════════════════════════════════════════════════
# HONEST FRAMING: this script has no machine-learning parameters that get
# "trained" — the Minervini rules, score thresholds, RS Rank cutoffs, etc.
# are all fixed constants in CONFIG, not fitted to data. So the 3 windows
# below are NOT a train/test/validate split in the ML sense (there is
# nothing to fit on the "train" window). What they ARE: three genuinely
# separate out-of-sample evaluation periods, useful for checking whether
# the strategy's behavior is consistent across different multi-year market
# regimes (e.g. 2020-22 had the Covid crash + recovery, which is a very
# different regime than 2023-25).
#
# IMPORTANT: if you look at the "train" (2020-2022) results and then
# manually tweak CONFIG thresholds to make that window's numbers look
# better, the "test" and "validate" windows are no longer honest
# out-of-sample checks — that's classic overfitting via researcher
# degrees of freedom, even without an explicit ML training step. Decide
# on CONFIG values BEFORE looking at any of the three windows' results,
# or accept that only whichever window you evaluate FIRST is truly
# unbiased.

def run_train_test_validate(price_data: dict, fund_map: dict = None,
                               train_period=("2020-01-01", "2023-01-01"),
                               test_period=("2023-01-01", "2025-01-01"),
                               validate_period=("2025-01-01", "2026-01-01"),
                               top_n_picks: int = 5, portfolio_size_inr: float = None):
    """
    Runs run_full_walkforward_backtest() independently across three
    non-overlapping periods and returns all three results side by side,
    plus a combined comparison table. Each period gets ITS OWN fresh
    starting capital (not compounded across periods) so the three are
    directly comparable on equal footing.
    """
    periods = {"TRAIN (2020-2022 diagnostic only)": train_period,
                "TEST (2023-2025, real out-of-sample)": test_period,
                "VALIDATE (2025-2026, real out-of-sample)": validate_period}
    results = {}
    for label, (start, end) in periods.items():
        log.info("\n" + "#" * 70)
        log.info(f"# RUNNING PERIOD: {label}  [{start} to {end}]")
        log.info("#" * 70)
        results[label] = run_full_walkforward_backtest(
            price_data, fund_map, start_date=start, end_date=end,
            top_n_picks=top_n_picks, portfolio_size_inr=portfolio_size_inr,
        )

    # Comparison table across the three periods against the spec's success criteria
    log.info("\n" + "=" * 70)
    log.info("TRAIN / TEST / VALIDATE COMPARISON")
    log.info("=" * 70)
    targets = {"cagr_pct": (">", 18), "sharpe_ratio": (">", 1.2),
                "max_drawdown_pct": ("> (less negative than)", -20),
                "calmar_ratio": (">", None)}
    comparison_rows = []
    for label, res in results.items():
        pm = res["portfolio_metrics"]
        tm = res["trade_metrics"]
        row = {
            "Period": label,
            "CAGR_%": pm.get("cagr_pct"), "Sharpe": pm.get("sharpe_ratio"),
            "Sortino": pm.get("sortino_ratio"), "Calmar": pm.get("calmar_ratio"),
            "MaxDD_%": pm.get("max_drawdown_pct"), "Win_Rate_%": tm.get("win_rate_pct"),
            "Profit_Factor": tm.get("profit_factor"), "N_Trades": tm.get("n_trades"),
            "Meets_CAGR>18%": (pm.get("cagr_pct") or 0) > 18,
            "Meets_Sharpe>1.2": (pm.get("sharpe_ratio") or 0) > 1.2,
            "Meets_MaxDD<20%": abs(pm.get("max_drawdown_pct") or 100) < 20,
            "Meets_WinRate>50%": (tm.get("win_rate_pct") or 0) > 50,
        }
        comparison_rows.append(row)
        log.info(f"  {label}: CAGR={row['CAGR_%']}% Sharpe={row['Sharpe']} "
                  f"MaxDD={row['MaxDD_%']}% WinRate={row['Win_Rate_%']}%")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(_THIS_DIR, "train_test_validate_comparison.csv"), index=False)
    log.info(f"\nSaved comparison table: train_test_validate_comparison.csv")
    log.info("\nReminder: only trust this comparison if CONFIG wasn't tuned after "
              "looking at any of these three windows' individual results.")

    return {"results_by_period": results, "comparison": comparison_df}


# ═════════════════════════════════════════════════════════════════════════
# UNIFIED ENTRY POINT — NEW v6
# ═════════════════════════════════════════════════════════════════════════
# One file, one command, pick your mode:
#   python3 unified_screener_and_backtest.py                    -> live screener (default)
#   python3 unified_screener_and_backtest.py --backtest-selftest -> synthetic self-test (proves mechanics work)
#   python3 unified_screener_and_backtest.py --walkforward       -> REAL walk-forward on downloaded universe
#   python3 unified_screener_and_backtest.py --traintestvalidate -> 3-period evaluation (2020-22/23-25/25-26)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Nifty 500 Screener + Backtest Engine")
    parser.add_argument("--backtest-selftest", action="store_true",
                          help="Run the synthetic self-test (proves the backtest mechanics work, "
                               "not a real performance result)")
    parser.add_argument("--walkforward", action="store_true",
                          help="Run the REAL walk-forward backtest on the live-downloaded Nifty 500 "
                               "universe (slow — downloads price history for ~500 stocks first)")
    parser.add_argument("--traintestvalidate", action="store_true",
                          help="Run the 3-period evaluation: 2020-2022 / 2023-2025 / 2025-2026")
    parser.add_argument("--top-n", type=int, default=5, help="Top N picks per book per month (backtest modes)")
    args = parser.parse_args()

    if args.backtest_selftest:
        log.info("Running SYNTHETIC self-test — proves mechanics work, NOT a real performance result")
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=505)[-500:]

        def make_df(seed=42, regime="up"):
            np.random.seed(seed)
            drift = 0.0008 if regime == "up" else -0.0006
            returns = np.random.normal(drift, 0.015, 500)
            prices = 100 * np.exp(np.cumsum(returns))
            prices[200:250] *= np.linspace(1.0, 0.55, 50)
            prices[250:300] *= np.linspace(0.55, 0.80, 50)
            vol = np.random.lognormal(14, 0.5, 500).astype(int)
            return pd.DataFrame({
                "Open": prices * (1 + np.random.normal(0, 0.003, 500)),
                "High": prices * (1 + np.abs(np.random.normal(0, 0.008, 500))),
                "Low": prices * (1 - np.abs(np.random.normal(0, 0.008, 500))),
                "Close": prices, "Volume": vol,
            }, index=dates)

        test_data = {"RELIANCE": make_df(42, "up"), "TCS": make_df(43, "up"),
                      "SBIN": make_df(44, "up"), "MAHABANK": make_df(45, "up")}
        test_funds = {
            "RELIANCE": Fundamentals(pe=22, fwd_pe=18, roe=11.4, rev_growth=12.8, earn_growth=14.4,
                                       debt_eq=0.4, div_yield=0.4, beta=0.8),
            "TCS": Fundamentals(pe=28, fwd_pe=24, roe=38.0, rev_growth=8.4, earn_growth=12.0,
                                  debt_eq=0.0, div_yield=1.4, beta=0.7),
            "SBIN": Fundamentals(pe=11, fwd_pe=9, roe=18.0, rev_growth=12.0, earn_growth=18.0,
                                   debt_eq=0.0, div_yield=1.8, beta=0.9),
            "MAHABANK": Fundamentals(pe=8, fwd_pe=6, roe=24.6, rev_growth=28.4, earn_growth=34.2,
                                       debt_eq=0.1, div_yield=1.8, beta=0.9),
        }
        full_df, covid_df = run_backtest(test_data, test_funds)
        print_backtest_report(full_df, covid_df)

    elif args.walkforward:
        log.info("Running REAL walk-forward backtest — this downloads price history for the full "
                  "Nifty 500 universe first and will take a long time.")
        # FIX v6: CONFIG["lookback_days"] defaults to 420 (fine for LIVE
        # screening) but a multi-year walk-forward needs far more history
        # than that, or download_universe() silently gives you almost
        # nothing useful to backtest against. Bump it here specifically
        # for backtest modes rather than changing the live-screener default.
        if CONFIG["lookback_days"] < 1800:
            log.warning(f"CONFIG['lookback_days'] was {CONFIG['lookback_days']} — too short for a "
                         "multi-year backtest. Overriding to 2000 days (~5.5yr) for this run only.")
            CONFIG["lookback_days"] = 2000
        symbols = get_nifty_500_universe()
        price_data = download_universe(symbols)
        fund_map = {s: fetch_fundamentals(s) for s in price_data}
        run_full_walkforward_backtest(price_data, fund_map, top_n_picks=args.top_n)

    elif args.traintestvalidate:
        log.info("Running 3-period TRAIN/TEST/VALIDATE evaluation — this downloads price history "
                  "for the full Nifty 500 universe first and will take a very long time.")
        if CONFIG["lookback_days"] < 2400:
            log.warning(f"CONFIG['lookback_days'] was {CONFIG['lookback_days']} — too short to cover "
                         "2020-2026. Overriding to 2400 days (~6.6yr) for this run only.")
            CONFIG["lookback_days"] = 2400
        symbols = get_nifty_500_universe()
        price_data = download_universe(symbols)
        fund_map = {s: fetch_fundamentals(s) for s in price_data}
        run_train_test_validate(price_data, fund_map, top_n_picks=args.top_n)

    else:
        run_screener()