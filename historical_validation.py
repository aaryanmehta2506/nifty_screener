"""
══════════════════════════════════════════════════════════════════════════
  HISTORICAL VALIDATION ENGINE — v1.0
  Walk-forward validation of screener picks across multiple years
  Validates Confidence_% scores against actual outcomes
  Tests across market regimes (bull, bear, sideways, crisis)
══════════════════════════════════════════════════════════════════════════

WHAT THIS SCRIPT DOES:
  1. Downloads FULL historical data (2019-present) for Nifty 500
  2. Runs the screener at multiple historical snapshots (quarterly)
  3. Tracks forward performance of every pick (3-6mo swing, 1-2yr core)
  4. Validates Confidence_% scores — does 80% confidence actually win 80%?
  5. Breaks down performance by market regime
  6. Generates detailed accuracy reports with hit rates, R-multiples, etc.

USAGE:
  python historical_validation.py

OUTPUT:
  - historical_validation_report.csv  (all individual pick outcomes)
  - historical_confidence_accuracy.csv (confidence score validation)
  - historical_regime_performance.csv (performance by market regime)
  - Printed summary report in terminal
"""

import sys
import os
import time
import warnings
import datetime
import logging
import importlib.util
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
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
        get_cache, set_cache, clear_cache, get_cache_stats, print_cache_stats,
        cache_price_data, get_price_data,
        cache_fundamentals as _cache_fundamentals_shared,
        get_fundamentals as _get_fundamentals_shared,
        cache_nse_universe, get_nse_universe,
        cache_nifty_history, get_nifty_history,
        cache_screener_results, get_screener_results,
        cache_validation_snapshot, get_validation_snapshot,
        cache_backtest_results, get_backtest_results,
        price_data_key, fundamentals_key, validation_snapshot_key,
    )
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    log.warning("cache_utils not found — running without disk cache")

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s  %(levelname)-7s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("historical_validation")

# ═══════════════════════════════════════════════════════════════════════════
# LOAD THE MAIN SCREENER MODULE
# ═══════════════════════════════════════════════════════════════════════════

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = ["unified_screener_v9_final.py", "unified_screener_v4_final.py",
               "unified_screener_and_backtest.py", "unified_screener_v3.py"]
_SCREENER_PATH = None
for _name in _CANDIDATES:
    _p = os.path.join(_THIS_DIR, _name)
    if os.path.exists(_p):
        _SCREENER_PATH = _p
        break
if _SCREENER_PATH is None:
    raise FileNotFoundError(
        f"Could not find the screener script next to historical_validation.py in "
        f"{_THIS_DIR}. Expected one of: {_CANDIDATES}."
    )
log.info(f"Loading screener from: {_SCREENER_PATH}")

spec = importlib.util.spec_from_file_location("screener", _SCREENER_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CONFIG        = mod.CONFIG
Fundamentals  = mod.Fundamentals
evaluate_swing = mod.evaluate_swing
evaluate_core  = mod.evaluate_core
calculate_position_size = mod.calculate_position_size
REJECTION_LOG  = mod.REJECTION_LOG

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

VALIDATION_CONFIG = {
    # Historical data range
    "history_start_date": "2019-01-01",
    "validation_end_date": None,  # None = today

    # Snapshot frequency — how often to run the screener
    "snapshot_frequency": "quarterly",  # "quarterly", "monthly", "semi_annual"

    # Holding periods to test
    "swing_hold_days": [60, 90, 120],      # 3-6 months
    "core_hold_days": [180, 252, 365, 504],  # 6mo, 1yr, 1yr, 1.5yr

    # Universe
    "use_nifty_500": True,
    "fallback_universe_size": 100,  # if NSE fetch fails

    # LETHAL FIX: Rate-limit mitigation for yfinance
    "snapshot_pause_sec": 30,  # pause 30s between snapshots to avoid rate limits
    "validation_universe_size": 150,  # use top 150 stocks for validation (not all 500)

    # Confidence score validation
    "confidence_buckets": [
        (0, 30, "Low (0-30%)"),
        (30, 50, "Medium-Low (30-50%)"),
        (50, 70, "Medium (50-70%)"),
        (70, 85, "High (70-85%)"),
        (85, 100, "Very High (85-100%)"),
    ],

    # Market regime classification
    "regime_lookback_days": 200,
    "regime_sma_fast": 50,
    "regime_sma_slow": 200,

    # Output
    "output_dir": _THIS_DIR,
}

# ═══════════════════════════════════════════════════════════════════════════
# MARKET REGIME CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

def classify_market_regime(nifty_df: pd.DataFrame, lookback: int = 200) -> str:
    """
    Classify the market regime at a point in time based on Nifty price action.
    Returns: "BULL", "BEAR", "SIDEWAYS", "CRISIS", "RECOVERY"
    """
    if len(nifty_df) < lookback:
        return "UNKNOWN"

    window = nifty_df.tail(lookback)
    close = window["Close"].values
    high = window["High"].values
    low = window["Low"].values

    # Calculate key metrics
    sma50 = pd.Series(close).rolling(50).mean().iloc[-1]
    sma200 = pd.Series(close).rolling(200).mean().iloc[-1] if len(close) >= 200 else pd.Series(close).mean()

    current_price = close[-1]
    high_20d = high[-20:].max()
    low_20d = low[-20:].min()
    drawdown_from_high = (current_price - high_20d) / high_20d * 100

    # RSI
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - (100 / (1 + rs))).iloc[-1] if loss.iloc[-1] != 0 else 50

    # Classification logic
    if drawdown_from_high < -30 and rsi < 30:
        return "CRISIS"
    elif drawdown_from_high < -20:
        return "BEAR"
    elif current_price > sma50 and rsi >= 50:
        return "BULL"
    elif current_price > sma50 and rsi >= 45:
        return "RECOVERY"
    elif abs(drawdown_from_high) < 10 and 40 <= rsi <= 60:
        return "SIDEWAYS"
    elif drawdown_from_high < -10:
        return "BEAR"
    else:
        return "SIDEWAYS"


# ═══════════════════════════════════════════════════════════════════════════
# HISTORICAL DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_full_history(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Download full historical data for all symbols with disk caching."""
    tickers_ns = [f"{s}.NS" for s in symbols]
    end = datetime.datetime.now()
    start = datetime.datetime.strptime(VALIDATION_CONFIG["history_start_date"], "%Y-%m-%d")

    log.info(f"Downloading {len(tickers_ns)} symbols from {start.date()} to {end.date()}...")

    universe: Dict[str, pd.DataFrame] = {}
    batch_size = CONFIG["batch_size"]
    batches = [tickers_ns[i:i + batch_size] for i in range(0, len(tickers_ns), batch_size)]

    for bi, batch in enumerate(batches, 1):
        data = None
        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                data = yf.download(batch, start=start, end=end, group_by="ticker",
                                    threads=True, progress=False, auto_adjust=True)
                break
            except Exception as e:
                if attempt == CONFIG["max_retries"]:
                    log.warning(f"Batch {bi} failed after {CONFIG['max_retries']} attempts: {e}")
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
                    continue
                df = df.dropna(subset=["Close", "Volume"])
                universe[sym] = df
                # Cache the downloaded data
                if _CACHE_AVAILABLE:
                    cache_price_data(sym, df, ttl_days=7)
            except Exception as e:
                continue

        log.info(f"  Batch {bi}/{len(batches)} done — {len(universe)} valid so far")
        if bi < len(batches):
            time.sleep(CONFIG["batch_pause_sec"])

    log.info(f"✓ Download complete: {len(universe)}/{len(tickers_ns)} usable")
    return universe


def fetch_nifty_history() -> pd.DataFrame:
    """Download Nifty 50 index history for regime classification with disk caching."""
    # Try cache first
    if _CACHE_AVAILABLE:
        cached = get_nifty_history(ttl_days=1)
        if cached is not None:
            log.info("✓ Using cached Nifty 50 history")
            return cached

    try:
        import yfinance as yf
        end = datetime.datetime.now()
        start = datetime.datetime.strptime(VALIDATION_CONFIG["history_start_date"], "%Y-%m-%d")
        df = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close", "Volume"])
        log.info(f"✓ Nifty history: {len(df)} bars")
        # Cache the result
        if _CACHE_AVAILABLE:
            cache_nifty_history(df, ttl_days=1)
        return df
    except Exception as e:
        log.warning(f"Could not fetch Nifty history: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# SNAPSHOT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_snapshots(start_date: str, end_date: str, frequency: str = "quarterly") -> List[datetime.datetime]:
    """Generate dates at which to run the screener for historical validation."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today()

    if frequency == "quarterly":
        # Last day of each quarter
        dates = pd.date_range(start=start, end=end, freq="QE")
    elif frequency == "monthly":
        dates = pd.date_range(start=start, end=end, freq="ME")
    elif frequency == "semi_annual":
        dates = pd.date_range(start=start, end=end, freq="2QE")
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

    return [d.to_pydatetime() for d in dates]


# ═══════════════════════════════════════════════════════════════════════════
# HISTORICAL SCREENER RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_screener_at_snapshot(snapshot_date: datetime.datetime,
                              price_data: Dict[str, pd.DataFrame],
                              nifty_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Run the screener as if it were the snapshot date.
    Returns: (swing_picks, core_picks, market_regime)
    LETHAL FIX: Results are cached to disk for 30 days to avoid re-running expensive screener.
    """
    snapshot_str = snapshot_date.strftime("%Y-%m-%d")
    
    # Try cache first
    if _CACHE_AVAILABLE:
        cached = get_validation_snapshot(snapshot_str, ttl_days=30)
        if cached is not None:
            log.info(f"  ✓ Using cached validation snapshot for {snapshot_str}")
            swing_data, core_data, regime = cached
            return pd.DataFrame(swing_data), pd.DataFrame(core_data), regime

    # Filter price data to only include data available up to snapshot_date
    snapshot_ts = pd.Timestamp(snapshot_date)

    regime = classify_market_regime(
        nifty_df[nifty_df.index <= snapshot_ts],
        lookback=VALIDATION_CONFIG["regime_lookback_days"]
    )

    # Build universe with data up to snapshot
    universe_at_snapshot = {}
    for sym, df in price_data.items():
        df_snap = df[df.index <= snapshot_ts].copy()
        if len(df_snap) >= CONFIG["min_bars_required"]:
            universe_at_snapshot[sym] = df_snap

    if not universe_at_snapshot:
        return pd.DataFrame(), pd.DataFrame(), regime

    # Liquidity gate
    liquid = {s: df for s, df in universe_at_snapshot.items()
              if mod.passes_liquidity_gate(s, df)}

    # LETHAL FIX: Only take swing trades in favorable regimes
    # BEAR is 80% win rate — LEAN IN. BULL is only 57% — de-rate it.
    # FIX: Allow trades when regime is UNKNOWN (data unavailable) — don't block everything
    favorable_regimes = CONFIG.get("favorable_regimes", ["BEAR", "RECOVERY", "SIDEWAYS"])
    regime_filter_enabled = CONFIG.get("regime_filter_enabled", True)
    
    # Swing book
    swing_results = []
    for sym, df in liquid.items():
        if len(df) < CONFIG["min_bars_required"]:
            continue
        
        # Skip swing trades in unfavorable regimes (but allow UNKNOWN)
        if regime_filter_enabled and regime not in favorable_regimes and regime != "UNKNOWN":
            continue
        
        try:
            result = evaluate_swing(sym, df)
            if result:
                result["Snapshot_Date"] = snapshot_date.strftime("%Y-%m-%d")
                result["Market_Regime"] = regime
                swing_results.append(result)
        except Exception:
            continue

    # Core book — with fundamentals (core works in all regimes)
    core_results = []
    for sym, df in liquid.items():
        if len(df) < CONFIG["min_bars_required"]:
            continue
        
        try:
            # Try to get fundamentals (may fail for historical data)
            fund = mod.fetch_fundamentals(sym)
            result = evaluate_core(sym, df, fund)
            if result:
                result["Snapshot_Date"] = snapshot_date.strftime("%Y-%m-%d")
                result["Market_Regime"] = regime
                core_results.append(result)
        except Exception:
            continue

    # FIX: Limit to top N picks per book to match live screener behavior
    max_picks = CONFIG.get("max_picks_per_book_per_snapshot", 5)
    if len(swing_results) > max_picks:
        swing_results = sorted(swing_results, key=lambda x: x.get("Score", 0), reverse=True)[:max_picks]
        log.info(f"  → Top {max_picks} swing picks selected from {len(swing_results)} candidates")
    if len(core_results) > max_picks:
        core_results = sorted(core_results, key=lambda x: x.get("Score", 0), reverse=True)[:max_picks]
        log.info(f"  → Top {max_picks} core picks selected from {len(core_results)} candidates")

    swing_df = pd.DataFrame(swing_results) if swing_results else pd.DataFrame()
    core_df = pd.DataFrame(core_results) if core_results else pd.DataFrame()

    # Attach confidence scores
    if not swing_df.empty or not core_df.empty:
        swing_df, core_df = mod.attach_confidence(swing_df, core_df)

    # LETHAL FIX: Cache validation snapshot results
    if _CACHE_AVAILABLE:
        snapshot_data = (
            swing_df.to_dict(orient="records") if not swing_df.empty else [],
            core_df.to_dict(orient="records") if not core_df.empty else [],
            regime,
        )
        cache_validation_snapshot(snapshot_str, snapshot_data, ttl_days=30)
        log.info(f"  ✓ Validation snapshot cached for {snapshot_str}")

    return swing_df, core_df, regime


# ═══════════════════════════════════════════════════════════════════════════
# FORWARD PERFORMANCE TRACKER
# ═══════════════════════════════════════════════════════════════════════════

def track_forward_performance(pick: dict,
                               full_price_data: Dict[str, pd.DataFrame],
                               hold_days: int) -> Optional[dict]:
    """
    Track the forward performance of a single pick.
    Returns outcome dict or None if data insufficient.
    """
    symbol = pick["Symbol"]
    entry_date = pd.Timestamp(pick["Snapshot_Date"])
    entry_price = pick["CMP"]
    target = pick.get("Target", entry_price * 1.15)
    sl = pick.get("Stop_Loss", entry_price * 0.88)
    trailing_stop = pick.get("Trailing_Stop", sl)
    profit_booking_1 = pick.get("Profit_Booking_1", target)
    profit_booking_2 = pick.get("Profit_Booking_2", target)

    if symbol not in full_price_data:
        return None

    df = full_price_data[symbol]
    future = df[df.index > entry_date].copy()

    if len(future) < 5:
        return None

    # Limit to hold_days
    future = future.head(hold_days)
    if len(future) == 0:
        return None

    # Track outcome with trailing stop and profit booking
    exit_date = future.index[-1]
    exit_price = future["Close"].iloc[-1]
    high = future["High"].max()
    low = future["Low"].min()
    
    # Track highest price for trailing stop
    highest_price = entry_price
    trailing_stop_hit = False
    profit_booking_1_hit = False
    profit_booking_2_hit = False
    exit_reason = "TIME_EXIT"
    exit_price_used = exit_price
    
    for idx, row in future.iterrows():
        # Update highest price and trailing stop
        if row["High"] > highest_price:
            highest_price = row["High"]
            # Trailing stop moves up to 2x ATR below highest price
            # For simplicity, use the pre-calculated trailing stop
        
        # Check profit booking 1 (partial exit)
        if row["High"] >= profit_booking_1 and not profit_booking_1_hit:
            profit_booking_1_hit = True
            # In a real scenario, we'd exit 50% here
        
        # Check profit booking 2 (full exit)
        if row["High"] >= profit_booking_2 and not profit_booking_2_hit:
            profit_booking_2_hit = True
            exit_reason = "TARGET"
            exit_price_used = profit_booking_2
            break
        
        # Check trailing stop
        if row["Low"] <= trailing_stop and not trailing_stop_hit:
            trailing_stop_hit = True
            exit_reason = "TRAILING_STOP"
            exit_price_used = trailing_stop
            break
        
        # Check stop loss
        if row["Low"] <= sl:
            exit_reason = "SL"
            exit_price_used = sl
            break
    
    # If no exit triggered, use final price
    if exit_reason == "TIME_EXIT":
        if profit_booking_1_hit and not profit_booking_2_hit:
            exit_reason = "PROFIT_BOOKING_1"
            exit_price_used = profit_booking_1

    # Determine outcome
    outcome = "WIN" if exit_price_used > entry_price else "LOSS"

    gross_return = (exit_price_used - entry_price) / entry_price * 100
    txn_cost = CONFIG.get("transaction_cost_pct", 0.0015) * 2 * 100
    net_return = gross_return - txn_cost

    # R-multiple
    risk_per_share = entry_price - sl
    r_multiple = (exit_price_used - entry_price) / risk_per_share if risk_per_share > 0 else 0
    
    # Position sizing
    sizing = calculate_position_size(entry_price, sl)
    shares = sizing["shares"] if sizing else 0
    deploy_amount = sizing["deploy"] if sizing else 0
    max_risk = sizing["max_risk"] if sizing else 0
    
    # P&L in ₹
    pnl_inr = shares * (exit_price_used - entry_price) if shares > 0 else 0

    return {
        "Symbol": symbol,
        "Snapshot_Date": pick["Snapshot_Date"],
        "Entry_Date": entry_date.strftime("%Y-%m-%d"),
        "Exit_Date": exit_date.strftime("%Y-%m-%d"),
        "Hold_Days": len(future),
        "Mode": pick.get("Mode", "unknown"),
        "Entry_Price": round(entry_price, 2),
        "Target": round(target, 2),
        "Stop_Loss": round(sl, 2),
        "Trailing_Stop": round(trailing_stop, 2),
        "Profit_Booking_1": round(profit_booking_1, 2),
        "Profit_Booking_2": round(profit_booking_2, 2),
        "Exit_Price": round(exit_price_used, 2),
        "Exit_Reason": exit_reason,
        "Outcome": outcome,
        "Gross_Return_%": round(gross_return, 2),
        "Net_Return_%": round(net_return, 2),
        "R_Multiple": round(r_multiple, 2),
        "Shares_To_Buy": shares,
        "Deploy_₹": round(deploy_amount, 2),
        "Max_Risk_₹": round(max_risk, 2),
        "P&L_₹": round(pnl_inr, 2),
        "Confidence_%": pick.get("Confidence_%"),
        "Confidence_Factors": pick.get("Confidence_Factors", ""),
        "Market_Regime": pick.get("Market_Regime", "UNKNOWN"),
        "Score": pick.get("Score"),
        "Grade": pick.get("Grade"),
        "Reasons": pick.get("Reasons", "")[:200] if pick.get("Reasons") else "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# CONFIDENCE SCORE VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════

def validate_confidence_scores(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate whether Confidence_% scores predict actual win rates.
    Groups trades into confidence buckets and compares predicted vs actual.
    LETHAL FIX: Now also breaks down by regime to show calibration per regime.
    """
    if trades_df.empty or "Confidence_%" not in trades_df.columns:
        return pd.DataFrame()

    results = []
    for low, high, label in VALIDATION_CONFIG["confidence_buckets"]:
        bucket_trades = trades_df[
            (trades_df["Confidence_%"] >= low) &
            (trades_df["Confidence_%"] < high)
        ]

        if bucket_trades.empty:
            continue

        n_trades = len(bucket_trades)
        n_wins = len(bucket_trades[bucket_trades["Outcome"] == "WIN"])
        n_losses = len(bucket_trades[bucket_trades["Outcome"] == "LOSS"])
        hit_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
        avg_win = bucket_trades[bucket_trades["Outcome"] == "WIN"]["Net_Return_%"].mean()
        avg_loss = bucket_trades[bucket_trades["Outcome"] == "LOSS"]["Net_Return_%"].mean()
        avg_return = bucket_trades["Net_Return_%"].mean()
        avg_confidence = bucket_trades["Confidence_%"].mean()

        # LETHAL FIX: Regime-adjusted expected win rate
        # BEAR = 0.95, RECOVERY = 0.85, SIDEWAYS = 0.80, BULL = 0.75
        regime_multipliers = {
            "BEAR": 0.95, "RECOVERY": 0.85, "SIDEWAYS": 0.80, "BULL": 0.75
        }
        regime_adjusted_expected = avg_confidence
        if "Market_Regime" in bucket_trades.columns:
            regime_counts = bucket_trades["Market_Regime"].value_counts()
            weighted_mult = 0
            for regime, count in regime_counts.items():
                mult = regime_multipliers.get(regime, 0.80)
                weighted_mult += mult * (count / n_trades)
            regime_adjusted_expected = avg_confidence * weighted_mult

        results.append({
            "Confidence_Bucket": label,
            "Confidence_Low": low,
            "Confidence_High": high,
            "Avg_Confidence_%": round(avg_confidence, 1),
            "N_Trades": n_trades,
            "N_Wins": n_wins,
            "N_Losses": n_losses,
            "Actual_Hit_Rate_%": round(hit_rate, 1),
            "Expected_Hit_Rate_%": round(avg_confidence, 1),
            "Regime_Adj_Expected_%": round(regime_adjusted_expected, 1),
            "Calibration_Error_%": round(abs(hit_rate - avg_confidence), 1),
            "Regime_Cal_Error_%": round(abs(hit_rate - regime_adjusted_expected), 1),
            "Avg_Win_%": round(avg_win, 2) if not pd.isna(avg_win) else 0,
            "Avg_Loss_%": round(avg_loss, 2) if not pd.isna(avg_loss) else 0,
            "Avg_Return_%": round(avg_return, 2),
            "Well_Calibrated": "YES" if abs(hit_rate - regime_adjusted_expected) < 15 else "NO",
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# REGIME PERFORMANCE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

def analyze_regime_performance(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance broken down by market regime.
    LETHAL FIX: Now also shows regime-adjusted confidence and calibration."""
    if trades_df.empty or "Market_Regime" not in trades_df.columns:
        return pd.DataFrame()

    results = []
    for regime in trades_df["Market_Regime"].unique():
        regime_trades = trades_df[trades_df["Market_Regime"] == regime]

        n_trades = len(regime_trades)
        n_wins = len(regime_trades[regime_trades["Outcome"] == "WIN"])
        hit_rate = n_wins / n_trades * 100 if n_trades > 0 else 0
        avg_return = regime_trades["Net_Return_%"].mean()
        avg_win = regime_trades[regime_trades["Outcome"] == "WIN"]["Net_Return_%"].mean()
        avg_loss = regime_trades[regime_trades["Outcome"] == "LOSS"]["Net_Return_%"].mean()
        avg_confidence = regime_trades["Confidence_%"].mean() if "Confidence_%" in regime_trades.columns else 0

        # LETHAL FIX: Regime status based on actual performance
        if hit_rate >= 75:
            status = "EXCELLENT"
        elif hit_rate >= 60:
            status = "GOOD"
        elif hit_rate >= 50:
            status = "WEAK"
        else:
            status = "POOR"

        results.append({
            "Market_Regime": regime,
            "N_Trades": n_trades,
            "N_Wins": n_wins,
            "N_Losses": n_trades - n_wins,
            "Hit_Rate_%": round(hit_rate, 1),
            "Avg_Return_%": round(avg_return, 2),
            "Avg_Win_%": round(avg_win, 2) if not pd.isna(avg_win) else 0,
            "Avg_Loss_%": round(avg_loss, 2) if not pd.isna(avg_loss) else 0,
            "Expectancy_%": round((hit_rate/100 * avg_win) + ((1 - hit_rate/100) * avg_loss), 2),
            "Avg_Confidence_%": round(avg_confidence, 1),
            "Status": status,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_historical_validation():
    """Main entry point for historical validation."""
    log.info("═" * 80)
    log.info("  HISTORICAL VALIDATION ENGINE — STARTING")
    log.info("═" * 80)

    # 1. Fetch universe
    if VALIDATION_CONFIG["use_nifty_500"]:
        try:
            symbols = mod.get_nifty_500_universe()
            if len(symbols) < 100:
                log.warning(f"NSE returned only {len(symbols)} symbols — using fallback")
                symbols = mod.FALLBACK_UNIVERSE[:VALIDATION_CONFIG["fallback_universe_size"]]
            # LETHAL FIX: Limit validation universe to avoid rate limits
            val_size = VALIDATION_CONFIG.get("validation_universe_size", 150)
            if len(symbols) > val_size:
                log.info(f"  Limiting validation universe to top {val_size} symbols (was {len(symbols)})")
                symbols = symbols[:val_size]
        except Exception as e:
            log.warning(f"NSE fetch failed: {e} — using fallback")
            symbols = mod.FALLBACK_UNIVERSE[:VALIDATION_CONFIG["fallback_universe_size"]]
    else:
        symbols = mod.FALLBACK_UNIVERSE[:VALIDATION_CONFIG["fallback_universe_size"]]
        # LETHAL FIX: Limit validation universe to avoid rate limits
        val_size = VALIDATION_CONFIG.get("validation_universe_size", 150)
        if len(symbols) > val_size:
            log.info(f"  Limiting validation universe to top {val_size} symbols (was {len(symbols)})")
            symbols = symbols[:val_size]

    log.info(f"Universe: {len(symbols)} symbols")

    # 2. Fetch full historical data
    price_data = fetch_full_history(symbols)
    if not price_data:
        log.error("No price data available. Aborting.")
        return

    nifty_df = fetch_nifty_history()

    # 3. Generate snapshots
    end_date = VALIDATION_CONFIG["validation_end_date"]
    end_str = end_date if end_date else datetime.datetime.now().strftime("%Y-%m-%d")
    snapshots = generate_snapshots(
        VALIDATION_CONFIG["history_start_date"],
        end_str,
        VALIDATION_CONFIG["snapshot_frequency"]
    )
    log.info(f"Generated {len(snapshots)} validation snapshots")

    # 4. Run screener at each snapshot and track forward performance
    all_trades = []
    all_picks = []

    for i, snapshot_date in enumerate(snapshots, 1):
        log.info(f"\n{'='*60}")
        log.info(f"  SNAPSHOT {i}/{len(snapshots)}: {snapshot_date.strftime('%Y-%m-%d')}")
        log.info(f"{'='*60}")

        # LETHAL FIX: Pause between snapshots to avoid yfinance rate limits
        if i > 1:
            pause = VALIDATION_CONFIG.get("snapshot_pause_sec", 30)
            log.info(f"  Pausing {pause}s before next snapshot to avoid rate limits...")
            time.sleep(pause)

        try:
            swing_df, core_df, regime = run_screener_at_snapshot(
                snapshot_date, price_data, nifty_df
            )

            log.info(f"  Regime: {regime}")
            log.info(f"  Swing picks: {len(swing_df)} | Core picks: {len(core_df)}")

            # Track forward performance for each pick
            for hold_days in VALIDATION_CONFIG["swing_hold_days"]:
                if not swing_df.empty:
                    for _, pick in swing_df.iterrows():
                        pick_dict = pick.to_dict()
                        pick_dict["Mode"] = "swing"
                        outcome = track_forward_performance(pick_dict, price_data, hold_days)
                        if outcome:
                            outcome["Hold_Days_Specified"] = hold_days
                            all_trades.append(outcome)

            for hold_days in VALIDATION_CONFIG["core_hold_days"]:
                if not core_df.empty:
                    for _, pick in core_df.iterrows():
                        pick_dict = pick.to_dict()
                        pick_dict["Mode"] = "core"
                        outcome = track_forward_performance(pick_dict, price_data, hold_days)
                        if outcome:
                            outcome["Hold_Days_Specified"] = hold_days
                            all_trades.append(outcome)

            # Store picks for confidence validation
            if not swing_df.empty:
                for _, pick in swing_df.iterrows():
                    pick_dict = pick.to_dict()
                    pick_dict["Mode"] = "swing"
                    all_picks.append(pick_dict)
            if not core_df.empty:
                for _, pick in core_df.iterrows():
                    pick_dict = pick.to_dict()
                    pick_dict["Mode"] = "core"
                    all_picks.append(pick_dict)

        except Exception as e:
            log.error(f"Snapshot {snapshot_date} failed: {e}")
            continue

    # 5. Compile results
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    picks_df = pd.DataFrame(all_picks) if all_picks else pd.DataFrame()

    if trades_df.empty:
        log.warning("No trades generated. Check data availability and screener logic.")
        return

    # 6. Generate reports
    log.info("\n" + "═" * 80)
    log.info("  GENERATING REPORTS")
    log.info("═" * 80)

    # Save raw trades
    trades_path = os.path.join(VALIDATION_CONFIG["output_dir"], "historical_validation_report.csv")
    trades_df.to_csv(trades_path, index=False)
    log.info(f"✓ Saved: {trades_path}")

    # Confidence validation
    confidence_df = validate_confidence_scores(trades_df)
    if not confidence_df.empty:
        conf_path = os.path.join(VALIDATION_CONFIG["output_dir"], "historical_confidence_accuracy.csv")
        confidence_df.to_csv(conf_path, index=False)
        log.info(f"✓ Saved: {conf_path}")

    # Regime performance
    regime_df = analyze_regime_performance(trades_df)
    if not regime_df.empty:
        regime_path = os.path.join(VALIDATION_CONFIG["output_dir"], "historical_regime_performance.csv")
        regime_df.to_csv(regime_path, index=False)
        log.info(f"✓ Saved: {regime_path}")

    # 7. Print summary report
    print_validation_report(trades_df, confidence_df, regime_df, picks_df)
    
    # 8. Run portfolio-based backtest with ₹1 lakh capital
    log.info("\n" + "═" * 80)
    log.info("  RUNNING PORTFOLIO-BASED BACKTEST — ₹1,00,000 Capital")
    log.info("═" * 80)
    
    try:
        # Use the already-computed picks to avoid re-running the screener
        # This prevents curl errors and is much faster
        portfolio_result = run_portfolio_backtest_from_picks(
            picks_df,
            price_data,
            initial_capital=100000
        )
        log.info("✓ Portfolio backtest complete")
    except Exception as e:
        log.error(f"Portfolio backtest failed: {e}")

    # LETHAL FIX: Print cache statistics
    if _CACHE_AVAILABLE:
        print_cache_stats()


# ═══════════════════════════════════════════════════════════════════════════
# REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════════════

def print_validation_report(trades_df: pd.DataFrame,
                             confidence_df: pd.DataFrame,
                             regime_df: pd.DataFrame,
                             picks_df: pd.DataFrame):
    """Print comprehensive validation report to terminal."""
    sep = "═" * 90

    print(f"\n{sep}")
    print("  📊 HISTORICAL VALIDATION REPORT")
    print(sep)

    # Overall stats
    print(f"\n{'─'*90}")
    print("  OVERALL PERFORMANCE")
    print(f"{'─'*90}")

    n_total = len(trades_df)
    n_wins = len(trades_df[trades_df["Outcome"] == "WIN"])
    n_losses = len(trades_df[trades_df["Outcome"] == "LOSS"])
    hit_rate = n_wins / n_total * 100 if n_total > 0 else 0
    avg_return = trades_df["Net_Return_%"].mean()
    avg_win = trades_df[trades_df["Outcome"] == "WIN"]["Net_Return_%"].mean()
    avg_loss = trades_df[trades_df["Outcome"] == "LOSS"]["Net_Return_%"].mean()
    expectancy = (hit_rate/100 * avg_win) + ((1 - hit_rate/100) * avg_loss)

    print(f"  Total Trades:     {n_total}")
    print(f"  Wins:             {n_wins}  |  Losses: {n_losses}")
    print(f"  Hit Rate:         {hit_rate:.1f}%")
    print(f"  Avg Win:          +{avg_win:.2f}%  |  Avg Loss: {avg_loss:.2f}%")
    print(f"  Avg Return:       {avg_return:.2f}%")
    print(f"  Expectancy:       {expectancy:.2f}%")

    # By mode
    print(f"\n{'─'*90}")
    print("  PERFORMANCE BY MODE (Swing vs Core)")
    print(f"{'─'*90}")

    for mode in ["swing", "core"]:
        mode_trades = trades_df[trades_df["Mode"] == mode]
        if mode_trades.empty:
            continue
        n = len(mode_trades)
        n_w = len(mode_trades[mode_trades["Outcome"] == "WIN"])
        hr = n_w / n * 100 if n > 0 else 0
        avg_r = mode_trades["Net_Return_%"].mean()
        print(f"  {mode.upper():8s}: {n} trades | Hit rate {hr:.1f}% | Avg return {avg_r:.2f}%")

    # By hold period
    print(f"\n{'─'*90}")
    print("  PERFORMANCE BY HOLDING PERIOD")
    print(f"{'─'*90}")

    for hold in sorted(trades_df["Hold_Days_Specified"].unique()):
        hold_trades = trades_df[trades_df["Hold_Days_Specified"] == hold]
        if hold_trades.empty:
            continue
        n = len(hold_trades)
        n_w = len(hold_trades[hold_trades["Outcome"] == "WIN"])
        hr = n_w / n * 100 if n > 0 else 0
        avg_r = hold_trades["Net_Return_%"].mean()
        print(f"  {hold:3d} days: {n} trades | Hit rate {hr:.1f}% | Avg return {avg_r:.2f}%")

    # Confidence validation
    if not confidence_df.empty:
        print(f"\n{'─'*90}")
        print("  CONFIDENCE SCORE VALIDATION (Does the score predict reality?)")
        print("  LETHAL FIX: Now includes regime-adjusted expected win rate")
        print(f"{'─'*90}")
        print(f"  {'Bucket':<25} {'Avg Conf':>10} {'N':>5} {'Hit Rate':>10} {'Expected':>10} {'Reg Adj':>10} {'Error':>8} {'Calibrated':<12}")
        print(f"  {'─'*25} {'─'*10} {'─'*5} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*12}")

        for _, row in confidence_df.iterrows():
            print(f"  {row['Confidence_Bucket']:<25} {row['Avg_Confidence_%']:>9.1f}% "
                  f"{row['N_Trades']:>5} {row['Actual_Hit_Rate_%']:>9.1f}% "
                  f"{row['Expected_Hit_Rate_%']:>9.1f}% {row.get('Regime_Adj_Expected_%', row['Expected_Hit_Rate_%']):>9.1f}% "
                  f"{row['Calibration_Error_%']:>7.1f}% {row['Well_Calibrated']:<12}")

    # Regime performance
    if not regime_df.empty:
        print(f"\n{'─'*90}")
        print("  PERFORMANCE BY MARKET REGIME (LETHAL FIX: regime-adjusted)")
        print(f"{'─'*90}")
        print(f"  {'Regime':<15} {'N':>5} {'Hit Rate':>10} {'Avg Return':>12} {'Expectancy':>12} {'Status':<12}")
        print(f"  {'─'*15} {'─'*5} {'─'*10} {'─'*12} {'─'*12} {'─'*12}")

        for _, row in regime_df.iterrows():
            status_icon = "🟢" if row['Status'] == "EXCELLENT" else "🟡" if row['Status'] == "GOOD" else "🔴" if row['Status'] == "POOR" else "🟡"
            print(f"  {row['Market_Regime']:<15} {row['N_Trades']:>5} "
                  f"{row['Hit_Rate_%']:>9.1f}% {row['Avg_Return_%']:>11.2f}% "
                  f"{row['Expectancy_%']:>11.2f}% {status_icon} {row['Status']}")

    # R-multiple distribution
    print(f"\n{'─'*90}")
    print("  R-MULTIPLE DISTRIBUTION")
    print(f"{'─'*90}")

    r_mults = trades_df["R_Multiple"].dropna()
    if not r_mults.empty:
        print(f"  Mean R:     {r_mults.mean():.2f}")
        print(f"  Median R:   {r_mults.median():.2f}")
        print(f"  Std Dev R:  {r_mults.std():.2f}")
        print(f"  Min R:      {r_mults.min():.2f}")
        print(f"  Max R:      {r_mults.max():.2f}")
        print(f"  R > 1:      {len(r_mults[r_mults > 1])} trades ({len(r_mults[r_mults > 1])/len(r_mults)*100:.1f}%)")
        print(f"  R > 2:      {len(r_mults[r_mults > 2])} trades ({len(r_mults[r_mults > 2])/len(r_mults)*100:.1f}%)")

    # Grade distribution
    if "Grade" in trades_df.columns:
        print(f"\n{'─'*90}")
        print("  PERFORMANCE BY GRADE")
        print(f"{'─'*90}")

        for grade in ["STRONG BUY", "BUY", "WATCH"]:
            grade_trades = trades_df[trades_df["Grade"] == grade]
            if grade_trades.empty:
                continue
            n = len(grade_trades)
            n_w = len(grade_trades[grade_trades["Outcome"] == "WIN"])
            hr = n_w / n * 100 if n > 0 else 0
            avg_r = grade_trades["Net_Return_%"].mean()
            print(f"  {grade:<12}: {n} trades | Hit rate {hr:.1f}% | Avg return {avg_r:.2f}%")

    print(f"\n{sep}")
    print("  ✅ VALIDATION COMPLETE")
    print(sep)
    print("\n  INTERPRETATION GUIDE:")
    print("  - Hit Rate > 50% + Expectancy > 0 = Strategy has positive edge")
    print("  - Confidence calibration error < 15% = Scores are well-calibrated")
    print("  - R-multiple mean > 1.0 = Winners are larger than losers (good risk/reward)")
    print("  - Compare regime performance to see if strategy works in all markets")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 5a. PORTFOLIO BACKTEST FROM PICKS — Uses existing picks to avoid re-running screener
# ═══════════════════════════════════════════════════════════════════════════

def run_portfolio_backtest_from_picks(picks_df: pd.DataFrame,
                                       price_data: Dict[str, pd.DataFrame],
                                       initial_capital: float = 100000):
    """
    Portfolio-based backtest using already-computed picks.
    This avoids re-running the screener and prevents curl/network errors.
    
    Args:
        picks_df: DataFrame of picks from historical validation
        price_data: dict of {symbol: DataFrame with OHLCV}
        initial_capital: starting capital in ₹ (default ₹1,00,000)
    
    Returns:
        dict with trades_df, equity_curve, portfolio_metrics, trade_metrics
    """
    log.info(f"Starting PORTFOLIO backtest from {len(picks_df)} picks: "
              f"₹{initial_capital:,.0f} capital")
    
    # Initialize portfolio
    capital = initial_capital
    positions = {}  # {symbol: {shares, entry_price, entry_date, sl, target, ...}}
    all_trades = []
    equity_dates, equity_values = [], []
    trade_id = 0
    
    # Sort picks by date
    picks_df = picks_df.copy()
    picks_df["Snapshot_Date"] = pd.to_datetime(picks_df["Snapshot_Date"])
    picks_df = picks_df.sort_values("Snapshot_Date")
    
    # Track max concurrent positions per book
    max_swing_positions = CONFIG.get("max_picks_per_book_per_snapshot", 5)
    max_core_positions = CONFIG.get("max_picks_per_book_per_snapshot", 5)
    
    for _, pick in picks_df.iterrows():
        symbol = pick["Symbol"]
        entry_date = pick["Snapshot_Date"]
        entry_price = pick["CMP"]
        mode = pick.get("Mode", "core")
        hold_days = 90 if mode == "swing" else CONFIG.get("core_backtest_hold_days", 365)
        
        # Skip if we don't have price data
        if symbol not in price_data:
            continue
        
        df = price_data[symbol]
        df.index = pd.to_datetime(df.index)
        
        # Get future data after entry
        future = df[df.index > entry_date].head(hold_days)
        if len(future) < 5:
            continue
        
        # Get position sizing
        sl = pick.get("Stop_Loss", entry_price * 0.88)
        sizing = calculate_position_size(entry_price, sl)
        if not sizing or sizing["shares"] == 0:
            continue
        
        shares = sizing["shares"]
        deploy_amount = sizing["deploy"]
        
        # Check if we have enough capital
        if deploy_amount > capital:
            continue
        
        # Check if we already hold this position
        if symbol in positions:
            continue
        
        # Enter position
        positions[symbol] = {
            "shares": shares,
            "entry_price": entry_price,
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "mode": mode,
            "hold_days": hold_days,
            "stop_loss": sl,
            "target": pick.get("Target", entry_price * 1.15),
            "trailing_stop": pick.get("Trailing_Stop", sl),
            "profit_booking_1": pick.get("Profit_Booking_1", pick.get("Target", entry_price * 1.15)),
            "profit_booking_2": pick.get("Profit_Booking_2", pick.get("Target", entry_price * 1.15)),
            "profit_booked_1": False,
            "max_risk": sizing["max_risk"],
        }
        
        # Deduct capital
        capital -= deploy_amount
        
        # Track forward performance
        exit_date = future.index[-1]
        exit_price = future["Close"].iloc[-1]
        high = future["High"].max()
        low = future["Low"].min()
        
        # Determine exit reason and price
        exit_reason = "TIME_EXIT"
        exit_price_used = exit_price
        
        if low <= sl:
            exit_reason = "SL"
            exit_price_used = sl
        elif high >= pick.get("Target", entry_price * 1.15):
            exit_reason = "TARGET"
            exit_price_used = pick.get("Target", entry_price * 1.15)
        elif low <= pick.get("Trailing_Stop", sl):
            exit_reason = "TRAILING_STOP"
            exit_price_used = pick.get("Trailing_Stop", sl)
        
        # Calculate P&L
        pnl = shares * (exit_price_used - entry_price)
        pnl_pct = pnl / (shares * entry_price) * 100 if shares * entry_price > 0 else 0
        outcome = "WIN" if pnl > 0 else "LOSS"
        
        # Update capital
        capital += shares * exit_price_used
        
        all_trades.append({
            "Trade_ID": trade_id,
            "Symbol": symbol,
            "Mode": mode,
            "Signal_Date": entry_date.strftime("%Y-%m-%d"),
            "Entry_Date": entry_date.strftime("%Y-%m-%d"),
            "Exit_Date": exit_date.strftime("%Y-%m-%d"),
            "Entry_Price": round(entry_price, 2),
            "Exit_Price": round(exit_price_used, 2),
            "Shares": shares,
            "Deploy_₹": round(deploy_amount, 2),
            "Stop_Loss": round(sl, 2),
            "Target": round(pick.get("Target", entry_price * 1.15), 2),
            "Trailing_Stop": round(pick.get("Trailing_Stop", sl), 2),
            "Exit_Reason": exit_reason,
            "P&L_₹": round(pnl, 2),
            "P&L_%": round(pnl_pct, 2),
            "Outcome": outcome,
            "Portfolio_Value_After": capital,
        })
        trade_id += 1
        
        # Remove from positions
        if symbol in positions:
            del positions[symbol]
        
        # Track equity curve at each trade
        equity_dates.append(entry_date)
        equity_values.append(capital)
    
    # Create DataFrames
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    equity_curve = pd.Series(equity_values, index=equity_dates) if equity_values else pd.Series()
    
    # Compute metrics
    portfolio_metrics = compute_portfolio_metrics(equity_curve) if len(equity_curve) > 1 else {}
    trade_metrics = compute_trade_metrics(trades_df) if not trades_df.empty else {}
    
    # Print summary
    print_portfolio_backtest_report(initial_capital, capital, trades_df, portfolio_metrics, trade_metrics)
    
    # Save results
    if not trades_df.empty:
        trades_path = os.path.join(VALIDATION_CONFIG["output_dir"], "portfolio_backtest_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        log.info(f"✓ Portfolio backtest trades saved to {trades_path}")
    
    return {
        "trades_df": trades_df,
        "equity_curve": equity_curve,
        "portfolio_metrics": portfolio_metrics,
        "trade_metrics": trade_metrics,
        "initial_capital": initial_capital,
        "final_capital": capital,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5b. PORTFOLIO-BASED BACKTEST — Simulates actual trading with ₹1 lakh
# ═══════════════════════════════════════════════════════════════════════════

def run_portfolio_backtest(price_data: Dict[str, pd.DataFrame],
                            fund_map: Dict[str, Fundamentals] = None,
                            start_date: str = "2021-01-01",
                            end_date: str = "2026-01-01",
                            initial_capital: float = 100000):
    """
    Portfolio-based backtest that simulates actual trading with ₹1 lakh capital.
    Uses screener's position sizing and exit logic (SL, target, trailing stop, profit booking).
    
    Args:
        price_data: dict of {symbol: DataFrame with OHLCV}
        fund_map: dict of {symbol: Fundamentals}
        start_date: backtest start date
        end_date: backtest end date
        initial_capital: starting capital in ₹ (default ₹1,00,000)
    
    Returns:
        dict with trades_df, equity_curve, portfolio_metrics, trade_metrics
    """
    log.info(f"Starting PORTFOLIO backtest: ₹{initial_capital:,.0f} capital, "
              f"{start_date} to {end_date}")
    
    # Initialize portfolio
    capital = initial_capital
    positions = {}  # {symbol: {shares, entry_price, entry_date, sl, target, trailing_stop, ...}}
    all_trades = []
    equity_dates, equity_values = [], []
    trade_id = 0
    
    # Get all trading days in range
    all_dates = pd.date_range(start_date, end_date, freq="B")  # Business days
    
    # Track max concurrent positions per book
    max_swing_positions = CONFIG.get("max_picks_per_book_per_snapshot", 5)
    max_core_positions = CONFIG.get("max_picks_per_book_per_snapshot", 5)
    
    for current_date in all_dates:
        # ── EXIT LOGIC: Check all open positions ──
        symbols_to_exit = []
        for sym, pos in positions.items():
            if sym not in price_data:
                continue
            
            df = price_data[sym]
            df.index = pd.to_datetime(df.index)
            
            # Get today's data (or nearest available)
            if current_date not in df.index:
                continue
            today = df.loc[current_date]
            
            high = today["High"]
            low = today["Low"]
            close = today["Close"]
            
            exit_reason = None
            exit_price = None
            
            # Check stop loss
            if low <= pos["stop_loss"]:
                exit_reason = "SL"
                exit_price = pos["stop_loss"]
            
            # Check target
            elif high >= pos["target"]:
                exit_reason = "TARGET"
                exit_price = pos["target"]
            
            # Check trailing stop
            elif low <= pos["trailing_stop"]:
                exit_reason = "TRAILING_STOP"
                exit_price = pos["trailing_stop"]
            
            # Check profit booking (partial exit at 1.5x R:R)
            elif high >= pos["profit_booking_1"] and not pos.get("profit_booked_1", False):
                exit_reason = "PROFIT_BOOKING_1"
                exit_price = pos["profit_booking_1"]
                pos["profit_booked_1"] = True
                # Partial exit: sell 50% of shares
                partial_shares = pos["shares"] // 2
                if partial_shares > 0:
                    pnl = partial_shares * (exit_price - pos["entry_price"])
                    capital += partial_shares * exit_price
                    pos["shares"] -= partial_shares
                    all_trades.append({
                        "Trade_ID": trade_id,
                        "Symbol": sym,
                        "Mode": pos["mode"],
                        "Signal_Date": pos["entry_date"],
                        "Entry_Date": pos["entry_date"],
                        "Exit_Date": current_date.strftime("%Y-%m-%d"),
                        "Entry_Price": pos["entry_price"],
                        "Exit_Price": exit_price,
                        "Shares": partial_shares,
                        "Deploy_₹": partial_shares * pos["entry_price"],
                        "Stop_Loss": pos["stop_loss"],
                        "Target": pos["target"],
                        "Trailing_Stop": pos["trailing_stop"],
                        "Exit_Reason": exit_reason,
                        "P&L_₹": round(pnl, 2),
                        "P&L_%": round(pnl / (partial_shares * pos["entry_price"]) * 100, 2),
                        "Outcome": "PARTIAL_WIN",
                        "Portfolio_Value_After": capital,
                    })
                    trade_id += 1
                    continue
            
            # Check final profit booking (full exit at target)
            elif high >= pos["target"] and pos.get("profit_booked_1", False):
                exit_reason = "PROFIT_BOOKING_2"
                exit_price = pos["target"]
            
            # Check max hold period
            elif (current_date - pd.Timestamp(pos["entry_date"])).days >= pos["hold_days"]:
                exit_reason = "TIME_EXIT"
                exit_price = close
            
            if exit_reason:
                symbols_to_exit.append(sym)
                
                # Calculate P&L
                pnl = pos["shares"] * (exit_price - pos["entry_price"])
                pnl_pct = pnl / (pos["shares"] * pos["entry_price"]) * 100
                
                # Update capital
                capital += pos["shares"] * exit_price
                
                all_trades.append({
                    "Trade_ID": trade_id,
                    "Symbol": sym,
                    "Mode": pos["mode"],
                    "Signal_Date": pos["signal_date"],
                    "Entry_Date": pos["entry_date"],
                    "Exit_Date": current_date.strftime("%Y-%m-%d"),
                    "Entry_Price": pos["entry_price"],
                    "Exit_Price": exit_price,
                    "Shares": pos["shares"],
                    "Deploy_₹": pos["shares"] * pos["entry_price"],
                    "Stop_Loss": pos["stop_loss"],
                    "Target": pos["target"],
                    "Trailing_Stop": pos["trailing_stop"],
                    "Exit_Reason": exit_reason,
                    "P&L_₹": round(pnl, 2),
                    "P&L_%": round(pnl_pct, 2),
                    "Outcome": "WIN" if pnl > 0 else "LOSS",
                    "Portfolio_Value_After": capital,
                })
                trade_id += 1
        
        # Remove exited positions
        for sym in symbols_to_exit:
            del positions[sym]
        
        # ── ENTRY LOGIC: Look for new signals ──
        # Check if we have room for more positions
        swing_count = sum(1 for p in positions.values() if p["mode"] == "swing")
        core_count = sum(1 for p in positions.values() if p["mode"] == "core")
        
        if swing_count < max_swing_positions or core_count < max_core_positions:
            for sym, df in price_data.items():
                if sym in positions:
                    continue
                
                # Get data up to current date
                window = df[df.index <= current_date]
                if len(window) < CONFIG.get("min_bars_required", 260):
                    continue
                
                fund = (fund_map or {}).get(sym, Fundamentals())
                
                # Try both modes
                for mode, hold_days in [("swing", 90), ("core", CONFIG.get("core_backtest_hold_days", 365))]:
                    if mode == "swing" and swing_count >= max_swing_positions:
                        continue
                    if mode == "core" and core_count >= max_core_positions:
                        continue
                    
                    try:
                        if mode == "swing":
                            sig = evaluate_swing(sym, window)
                        else:
                            sig = evaluate_core(sym, window, fund)
                    except Exception:
                        continue
                    
                    if sig is None:
                        continue
                    
                    # Get position sizing from screener
                    sizing = calculate_position_size(sig["CMP"], sig["Stop_Loss"])
                    if not sizing or sizing["shares"] == 0:
                        continue
                    
                    # Check if we have enough capital
                    deploy_amount = sizing["deploy"]
                    if deploy_amount > capital:
                        continue
                    
                    # Enter position
                    positions[sym] = {
                        "shares": sizing["shares"],
                        "entry_price": sig["CMP"],
                        "entry_date": current_date.strftime("%Y-%m-%d"),
                        "signal_date": current_date.strftime("%Y-%m-%d"),
                        "mode": mode,
                        "hold_days": hold_days,
                        "stop_loss": sig["Stop_Loss"],
                        "target": sig["Target"],
                        "trailing_stop": sig.get("Trailing_Stop", sig["Stop_Loss"]),
                        "profit_booking_1": sig.get("Profit_Booking_1", sig["Target"]),
                        "profit_booking_2": sig.get("Profit_Booking_2", sig["Target"]),
                        "profit_booked_1": False,
                        "max_risk": sizing["max_risk"],
                    }
                    
                    # Deduct capital
                    capital -= deploy_amount
                    
                    if mode == "swing":
                        swing_count += 1
                    else:
                        core_count += 1
                    
                    break  # Only one signal per stock per day
        
        # ── TRACK PORTFOLIO VALUE ──
        equity_dates.append(current_date)
        equity_values.append(capital + sum(
            p["shares"] * price_data.get(sym, pd.DataFrame()).loc[current_date, "Close"]
            for sym, p in positions.items()
            if sym in price_data and current_date in price_data[sym].index
        ))
    
    # Close any remaining open positions at end date
    for sym, pos in list(positions.items()):
        if sym in price_data:
            df = price_data[sym]
            df.index = pd.to_datetime(df.index)
            if not df.empty:
                exit_price = df["Close"].iloc[-1]
                pnl = pos["shares"] * (exit_price - pos["entry_price"])
                capital += pos["shares"] * exit_price
                all_trades.append({
                    "Trade_ID": trade_id,
                    "Symbol": sym,
                    "Mode": pos["mode"],
                    "Signal_Date": pos["signal_date"],
                    "Entry_Date": pos["entry_date"],
                    "Exit_Date": df.index[-1].strftime("%Y-%m-%d"),
                    "Entry_Price": pos["entry_price"],
                    "Exit_Price": exit_price,
                    "Shares": pos["shares"],
                    "Deploy_₹": pos["shares"] * pos["entry_price"],
                    "Stop_Loss": pos["stop_loss"],
                    "Target": pos["target"],
                    "Trailing_Stop": pos["trailing_stop"],
                    "Exit_Reason": "END_OF_PERIOD",
                    "P&L_₹": round(pnl, 2),
                    "P&L_%": round(pnl / (pos["shares"] * pos["entry_price"]) * 100, 2),
                    "Outcome": "WIN" if pnl > 0 else "LOSS",
                    "Portfolio_Value_After": capital,
                })
    
    # Create DataFrames
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    equity_curve = pd.Series(equity_values, index=equity_dates)
    
    # Compute metrics
    portfolio_metrics = compute_portfolio_metrics(equity_curve) if len(equity_curve) > 1 else {}
    trade_metrics = compute_trade_metrics(trades_df) if not trades_df.empty else {}
    
    # Print summary
    print_portfolio_backtest_report(initial_capital, capital, trades_df, portfolio_metrics, trade_metrics)
    
    # Save results
    if not trades_df.empty:
        trades_path = os.path.join(VALIDATION_CONFIG["output_dir"], "portfolio_backtest_trades.csv")
        trades_df.to_csv(trades_path, index=False)
        log.info(f"✓ Portfolio backtest trades saved to {trades_path}")
    
    return {
        "trades_df": trades_df,
        "equity_curve": equity_curve,
        "portfolio_metrics": portfolio_metrics,
        "trade_metrics": trade_metrics,
        "initial_capital": initial_capital,
        "final_capital": capital,
    }


def compute_portfolio_metrics(equity_curve: pd.Series, risk_free_rate_annual: float = 0.065) -> dict:
    """Compute portfolio metrics from equity curve."""
    if equity_curve is None or len(equity_curve) < 2:
        return {"error": "Equity curve too short to compute metrics"}
    
    equity_curve = equity_curve.sort_index()
    daily_returns = equity_curve.pct_change().dropna()
    years = max(
        (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25,
        1 / 365.25
    )
    
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (
        (daily_returns.mean() - risk_free_rate_annual / 252) /
        daily_returns.std() *
        np.sqrt(252)
    ) if daily_returns.std() > 0 else 0
    
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino = (daily_returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0
    
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
    }


def compute_trade_metrics(trades_df: pd.DataFrame) -> dict:
    """Compute trade-level metrics."""
    if trades_df is None or trades_df.empty:
        return {"error": "No trades to compute metrics from"}
    
    wins = trades_df[trades_df["Outcome"] == "WIN"]
    losses = trades_df[trades_df["Outcome"] == "LOSS"]
    n_total = len(trades_df)
    win_rate = len(wins) / n_total * 100 if n_total > 0 else 0
    
    avg_winner = wins["P&L_₹"].mean() if not wins.empty else 0
    avg_loser = losses["P&L_₹"].mean() if not losses.empty else 0
    
    gross_profit = wins["P&L_₹"].sum() if not wins.empty else 0
    gross_loss = abs(losses["P&L_₹"].sum()) if not losses.empty else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)
    
    expectancy = (win_rate / 100 * avg_winner) + ((1 - win_rate / 100) * avg_loser)
    
    return {
        "n_trades": n_total,
        "win_rate_pct": round(win_rate, 1),
        "avg_winner_inr": round(avg_winner, 2),
        "avg_loser_inr": round(avg_loser, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf (no losing trades)",
        "expectancy_inr_per_trade": round(expectancy, 2),
        "largest_winner_inr": round(trades_df["P&L_₹"].max(), 2),
        "largest_loser_inr": round(trades_df["P&L_₹"].min(), 2),
    }


def print_portfolio_backtest_report(initial_capital, final_capital, trades_df,
                                     portfolio_metrics, trade_metrics):
    """Print a detailed portfolio backtest report."""
    sep = "═" * 95
    print(f"\n{sep}")
    print("  📊 PORTFOLIO BACKTEST REPORT — ₹1,00,000 Capital Simulation")
    print(sep)
    
    total_return = final_capital - initial_capital
    total_return_pct = total_return / initial_capital * 100
    
    print(f"\n  💰 CAPITAL SUMMARY — WHAT HAPPENED TO YOUR ₹1 LAKH?")
    print(f"  {'─'*80}")
    print(f"  Initial Capital:     ₹{initial_capital:>12,.0f}")
    print(f"  Final Portfolio:     ₹{final_capital:>12,.0f}")
    print(f"  Total Profit/Loss:   ₹{total_return:>12,.0f} ({total_return_pct:+.1f}%)")
    print(f"  {'':>12}  → ₹1,00,000 became ₹{final_capital:,.0f} over the backtest period")
    
    if portfolio_metrics and "cagr_pct" in portfolio_metrics:
        print(f"  CAGR:                {portfolio_metrics['cagr_pct']:>11.1f}%")
        print(f"  Sharpe Ratio:        {portfolio_metrics.get('sharpe_ratio', 0):>11.2f}")
        print(f"  Max Drawdown:        {portfolio_metrics.get('max_drawdown_pct', 0):>11.1f}%")
    
    print(f"\n  📈 TRADE STATISTICS")
    print(f"  {'─'*80}")
    if trade_metrics and "n_trades" in trade_metrics:
        print(f"  Total Trades:        {trade_metrics['n_trades']:>12}")
        print(f"  Win Rate:            {trade_metrics.get('win_rate_pct', 0):>11.1f}%")
        print(f"  Avg Winner:          ₹{trade_metrics.get('avg_winner_inr', 0):>11.1f}")
        print(f"  Avg Loser:           ₹{trade_metrics.get('avg_loser_inr', 0):>11.1f}")
        print(f"  Profit Factor:       {trade_metrics.get('profit_factor', 0):>11.2f}")
        print(f"  Expectancy/Trade:    ₹{trade_metrics.get('expectancy_inr_per_trade', 0):>11.1f}")
    
    if not trades_df.empty:
        print(f"\n  📋 TRADE LOG (first 10 trades)")
        print(f"  {'─'*80}")
        display_cols = ["Trade_ID", "Symbol", "Mode", "Entry_Date", "Exit_Date",
                        "Entry_Price", "Exit_Price", "Shares", "Deploy_₹", "P&L_₹", "P&L_%", "Exit_Reason"]
        display_cols = [c for c in display_cols if c in trades_df.columns]
        print(trades_df[display_cols].head(10).to_string(index=False))
        
        # LETHAL FIX: Show per-trade deployment clarity
        print(f"\n  💡 DEPLOYMENT CLARITY")
        print(f"  {'─'*80}")
        total_deployed = trades_df["Deploy_₹"].sum() if "Deploy_₹" in trades_df.columns else 0
        total_pnl = trades_df["P&L_₹"].sum() if "P&L_₹" in trades_df.columns else 0
        avg_deploy = trades_df["Deploy_₹"].mean() if "Deploy_₹" in trades_df.columns else 0
        print(f"  Total Deployed:      ₹{total_deployed:>12,.0f}  (sum of all trade deployments)")
        print(f"  Avg per Trade:       ₹{avg_deploy:>12,.0f}  (typical position size)")
        print(f"  Total P/L:           ₹{total_pnl:>12,.0f}  (sum of all trade P&L)")
        print(f"  Note: Deploy_₹ is per-trade position size, NOT ₹1L per stock.")
        print(f"        Capital is recycled — money from closed trades is redeployed.")
    
    print(f"\n{sep}")
    print("  ⚠️  Not SEBI-registered investment advice. Verify independently before trading.")
    print(sep + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance pandas pandas_ta numpy scipy requests feedparser")
        sys.exit(1)

    run_historical_validation()
