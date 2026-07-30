"""
═══════════════════════════════════════════════════════════════════════════
  UNIFIED SCREENER — BACKTESTING ENGINE
  Walk-forward validation of both Swing and Core signals
  Includes: Covid-era stress test, hit-rate analysis, R-multiple stats
═══════════════════════════════════════════════════════════════════════════
"""

import sys
import time
import warnings
import datetime
import logging
import importlib.util
import hashlib

import numpy as np
import pandas as pd

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
        price_data_key, fundamentals_key, backtest_results_key,
    )
    _CACHE_AVAILABLE = True
except ImportError:
    _CACHE_AVAILABLE = False
    log.warning("cache_utils not found — running without disk cache")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s  %(levelname)-7s  %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("backtest")

# Load the main screener module so we reuse the EXACT same signal logic.
import os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = ["unified_screener_v9_final.py", "unified_screener_v4_final.py",
               "unified_screener_v3.py", "unified_screener.py"]
_SCREENER_PATH = None
for _name in _CANDIDATES:
    _p = os.path.join(_THIS_DIR, _name)
    if os.path.exists(_p):
        _SCREENER_PATH = _p
        break
if _SCREENER_PATH is None:
    raise FileNotFoundError(
        f"Could not find the screener script next to backtest_engine.py in "
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
REJECTION_LOG  = mod.REJECTION_LOG


# ═════════════════════════════════════════════════════════════════════════
# 1. WALK-FORWARD BACKTESTER
# ═════════════════════════════════════════════════════════════════════════

def backtest_signal(symbol: str, df: pd.DataFrame, fund: Fundamentals,
                     mode: str = "swing",
                     hold_days: int = 90,
                     step_days: int = 21) -> list[dict]:
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
        target = signal.get("Target", entry * 1.20)
        sl     = signal["Stop_Loss"]

        if entry <= 0 or sl >= entry:
            continue

        fwd_high = df_forward["High"].values
        fwd_low  = df_forward["Low"].values
        fwd_close = df_forward["Close"].values

        outcome = "OPEN"
        current_sl = sl
        for day_idx, (h, l, c) in enumerate(zip(fwd_high, fwd_low, fwd_close)):
            # Trailing Stop Logic: If price moves halfway to target, trail SL to breakeven
            if h >= entry + ((target - entry) * 0.5):
                current_sl = max(current_sl, entry)
            
            target_hit = h >= target
            sl_hit     = l <= current_sl
            
            if target_hit and not sl_hit:
                outcome = "WIN"; break
            elif sl_hit and not target_hit:
                outcome = "LOSS"; break
            elif target_hit and sl_hit:
                outcome = "LOSS"; break

        final_price = fwd_close[-1] if outcome == "OPEN" else (target if outcome == "WIN" else sl)
        gross_return = (final_price - entry) / entry * 100
        net_return   = gross_return - (txn_cost * 100 * 2)  

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
    log.info("Running Covid-era stress test (Jan 2020 – Jun 2020)...")
    COVID_START = "2019-01-01" 
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
            covid_results = [r for r in results
                              if COVID_SIGNAL_START <= pd.Timestamp(r["Signal_Date"]) <= COVID_SIGNAL_END]
            all_results.extend(covid_results)

    if not all_results:
        log.warning("No signals fired during Covid window — check data coverage")
        return pd.DataFrame()

    df_out = pd.DataFrame(all_results)
    log.info(f"Covid stress test: {len(df_out)} signal instances across {df_out['Symbol'].nunique()} stocks")
    return df_out

# ═════════════════════════════════════════════════════════════════════════
# 3. AGGREGATE STATISTICS
# ═════════════════════════════════════════════════════════════════════════

def compute_stats(results_df: pd.DataFrame, label: str = "Backtest") -> dict:
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
        print("  No signals fired during Covid window.")
    else:
        for mode in ["swing", "core"]:
            sub = covid_df[covid_df["Mode"] == mode]
            if sub.empty:
                continue
            stats = compute_stats(sub, f"Covid {mode.upper()}")
            print(f"\n  {stats['label']}: {stats['n_signals']} signals | Hit rate {stats['hit_rate_%']}%")
            print(f"  Expectancy: {stats['expectancy_%']}%")

# ═════════════════════════════════════════════════════════════════════════
# 4. MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════

def run_backtest(price_data: dict, fund_map: dict = None,
                  hold_days_swing: int = 90, hold_days_core: int = None):
    if hold_days_core is None:
        hold_days_core = CONFIG.get("core_backtest_hold_days",504)
    
    # LETHAL FIX: Check for cached backtest results first
    if _CACHE_AVAILABLE:
        symbols_hash = hashlib.md5(str(sorted(price_data.keys())).encode()).hexdigest()[:12]
        cache_key = backtest_results_key(symbols_hash, "all", str(hold_days_swing), "standard")
        cached = get_backtest_results(cache_key, ttl_days=30)
        if cached is not None:
            log.info("✓ Using cached backtest results (disk cache hit)")
            full_df = pd.DataFrame(cached.get("full_df", []))
            covid_df = pd.DataFrame(cached.get("covid_df", []))
            print_backtest_report(full_df, covid_df)
            return full_df, covid_df
    
    log.info(f"Starting standard walk-forward backtest on {len(price_data)} stocks...")
    all_results = []

    for i, (sym, df) in enumerate(price_data.items(), 1):
        fund = (fund_map or {}).get(sym, Fundamentals())
        for mode, hold in [("swing", hold_days_swing), ("core", hold_days_core)]:
            results = backtest_signal(sym, df, fund, mode=mode,
                                       hold_days=hold, step_days=21)
            all_results.extend(results)

    full_df  = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    covid_df = run_covid_stress_test(price_data, fund_map)
    print_backtest_report(full_df, covid_df)

    if not full_df.empty:
        full_df.to_csv(os.path.join(_THIS_DIR, "backtest_results.csv"), index=False)
    if not covid_df.empty:
        covid_df.to_csv(os.path.join(_THIS_DIR, "backtest_covid.csv"), index=False)
    
    # LETHAL FIX: Cache backtest results
    if _CACHE_AVAILABLE:
        symbols_hash = hashlib.md5(str(sorted(price_data.keys())).encode()).hexdigest()[:12]
        cache_key = backtest_results_key(symbols_hash, "all", str(hold_days_swing), "standard")
        cache_backtest_results({
            "full_df": full_df.to_dict(orient="records") if not full_df.empty else [],
            "covid_df": covid_df.to_dict(orient="records") if not covid_df.empty else [],
        }, key=cache_key, ttl_days=30)
        log.info("✓ Backtest results cached to disk")
    
    return full_df, covid_df

# ═════════════════════════════════════════════════════════════════════════
# FULL WALK-FORWARD VALIDATION SUITE — NEW v5 (RESTORED & FIXED)
# ═════════════════════════════════════════════════════════════════════════

def compute_portfolio_metrics(equity_curve: pd.Series, risk_free_rate_annual: float = 0.065) -> dict:
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

    ann_vol = daily_returns.std() * np.sqrt(12)
    monthly_rf = risk_free_rate_annual / 12
    excess_returns = daily_returns - monthly_rf
    sharpe = (
        excess_returns.mean() /
        daily_returns.std() *
        np.sqrt(12)
    ) if daily_returns.std() > 0 else 0

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
        results[name] = {"error": "Unavailable natively via free yfinance tickers."}
    return results


def compute_monthly_returns_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or "Signal_Date" not in trades_df.columns:
        return pd.DataFrame()
    df = trades_df.copy()
    df["Signal_Date"] = pd.to_datetime(df["Signal_Date"])
    df["Month"] = df["Signal_Date"].dt.to_period("M")
    monthly = df.groupby("Month")["Net_Return_%"].agg(["sum", "mean", "count"]).reset_index()
    monthly.columns = ["Month", "Total_Return_%", "Avg_Return_Per_Trade_%", "N_Trades"]
    return monthly


def compute_sector_performance_table(trades_df: pd.DataFrame, sector_map: dict) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["Sector"] = df["Symbol"].map(lambda s: sector_map.get(s, "Diversified/Other"))
    sector_perf = df.groupby("Sector")["Net_Return_%"].agg(["mean", "sum", "count"]).reset_index()
    sector_perf.columns = ["Sector", "Avg_Return_%", "Total_Return_%", "N_Trades"]
    return sector_perf.sort_values("Avg_Return_%", ascending=False)


def compute_bucket_returns(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}
    buckets = {}
    if "Hold_Days" in trades_df.columns:
        buckets["Hold_Days"] = trades_df.groupby("Hold_Days")["Net_Return_%"].mean().to_dict()
    if "Mode" in trades_df.columns:
        buckets["Regime_Mode"] = trades_df.groupby("Mode")["Net_Return_%"].mean().to_dict()
    return buckets


def run_full_walkforward_backtest(price_data: dict, fund_map: dict = None,
                                     start_date: str = "2021-01-01", end_date: str = "2026-01-01",
                                     top_n_picks: int = 5, portfolio_size_inr: float = None):
    
    portfolio_size_inr = portfolio_size_inr or CONFIG.get("portfolio_size_inr", 136000)
    log.info(f"Starting FULL walk-forward validation: {start_date} to {end_date}, "
              f"top {top_n_picks} picks/book/month")

    all_trades = []
    equity = portfolio_size_inr
    equity_dates, equity_values = [], []

    date_range = pd.date_range(start_date, end_date, freq="MS")  
    for month_start in date_range:
        month_trades_this_period = []
        for sym, df in price_data.items():
            window = df[df.index < month_start]
            if len(window) < CONFIG.get("min_bars_required", 260):
                continue
            fund = (fund_map or {}).get(sym, Fundamentals())
            core_count = 0
            swing_count = 0
            for mode, hold in [("swing", 90), ("core", CONFIG.get("core_backtest_hold_days", 365))]:
                try:
                    sig = (mod.evaluate_swing(sym, window) if mode == "swing"
                            else mod.evaluate_core(sym, window, fund))
                except Exception:
                    sig = None
                if sig is None:
                    continue
                month_trades_this_period.append(
                (sig, mode, hold, window)
                )
                if mode == "core":
                    core_count += 1
                else:
                    swing_count += 1
            
            

        for mode in ("swing", "core"):
            book_trades = [t for t in month_trades_this_period if t[1] == mode]
            book_trades.sort(key=lambda t: t[0].get("Score", t[0].get("RR_Ratio", 0)), reverse=True)
            for sig, mode_, hold, window in book_trades[:top_n_picks]:
                full_df = price_data[sig["Symbol"]]
                future_dates = full_df.index[
                    full_df.index >= month_start
                ]

                if len(future_dates) == 0:
                    continue

                entry_date = future_dates[0]
                entry_idx = full_df.index.get_loc(entry_date)

                exit_idx = min(entry_idx + hold, len(full_df) - 1)
                if exit_idx <= entry_idx:
                    continue
                entry_price = sig["CMP"]
                exit_price = full_df["Close"].iloc[exit_idx]
                target, sl = sig.get("Target"), sig.get("Stop_Loss")
                path = full_df["Close"].iloc[entry_idx:exit_idx + 1]
                hit_target = target and (path >= target).any()
                hit_sl = sl and (path <= sl).any()
                outcome = "WIN" if (hit_target and not hit_sl) else ("LOSS" if hit_sl else
                           ("WIN" if exit_price > entry_price else "LOSS"))
                net_return = (exit_price / entry_price - 1) * 100 - CONFIG.get("transaction_cost_pct",0.0015) * 2
                all_trades.append({
                    "Symbol": sig["Symbol"], "Signal_Date": month_start.strftime("%Y-%m-%d"),
                    "Mode": mode_, "Entry": entry_price, "Exit": exit_price,
                    "Outcome": outcome, "Net_Return_%": round(net_return, 2), "Hold_Days": hold,
                })

        this_month_trades = [
                        t for t in all_trades
                        if t["Signal_Date"] == month_start.strftime("%Y-%m-%d")
                    ]

        if this_month_trades:
                        allocation_per_trade = 1.0 / len(this_month_trades)

                        portfolio_return = sum(
                            allocation_per_trade * (t["Net_Return_%"] / 100)
                            for t in this_month_trades
                        )

                        equity *= (1 + portfolio_return)

        equity_dates.append(month_start)
        equity_values.append(equity)

        log.info(
                        f"{month_start.strftime('%Y-%m')} | "
                        f"Trades={len(this_month_trades)} | "
                        f"Portfolio={equity:,.0f}"
                    )

    trades_df = pd.DataFrame(all_trades)
    equity_curve = pd.Series(equity_values, index=equity_dates)

    portfolio_metrics = compute_portfolio_metrics(equity_curve)
    trade_metrics = compute_trade_metrics(trades_df) if not trades_df.empty else {"error": "No trades generated"}
    benchmark = fetch_benchmark_returns(start_date, end_date)
    
    if "Nifty50" in benchmark and "cagr_pct" in benchmark["Nifty50"]:
        portfolio_metrics["alpha_vs_nifty50_pct"] = round(portfolio_metrics["cagr_pct"] - benchmark["Nifty50"]["cagr_pct"], 2)
        
    monthly_returns = compute_monthly_returns_table(trades_df) if not trades_df.empty else pd.DataFrame()
    sector_perf = (compute_sector_performance_table(trades_df, mod.SECTOR_MAP)
                    if not trades_df.empty else pd.DataFrame())
    bucket_metrics = compute_bucket_returns(trades_df) if not trades_df.empty else {}

    return {
        "trades_df": trades_df, "equity_curve": equity_curve,
        "portfolio_metrics": portfolio_metrics, "trade_metrics": trade_metrics,
        "benchmark": benchmark, "monthly_returns": monthly_returns,
        "sector_performance": sector_perf, "bucket_metrics": bucket_metrics
    }

def evaluate_success_criteria(metrics, phase_name):
    cagr = metrics.get("cagr_pct", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    print(f"\n[{phase_name}] --- SUCCESS EVALUATION ---")
    print(f"CAGR > 18%: {'PASSED' if cagr > 18 else 'FAILED'} ({cagr}%)")
    print(f"Sharpe > 1.2: {'PASSED' if sharpe > 1.2 else 'FAILED'} ({sharpe})")


# ═══════════════════════════════════════════════════════════════════════════
# 5. PORTFOLIO-BASED BACKTEST — Simulates actual trading with ₹1 lakh
# ═══════════════════════════════════════════════════════════════════════════

def run_portfolio_backtest(price_data: dict, fund_map: dict = None,
                            start_date: str = "2021-01-01", end_date: str = "2026-01-01",
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
                            sig = mod.evaluate_swing(sym, window)
                        else:
                            sig = mod.evaluate_core(sym, window, fund)
                    except Exception:
                        continue
                    
                    if sig is None:
                        continue
                    
                    # Get position sizing from screener
                    sizing = mod.calculate_position_size(sig["CMP"], sig["Stop_Loss"])
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
        trades_df.to_csv(os.path.join(_THIS_DIR, "portfolio_backtest_trades.csv"), index=False)
        log.info(f"✓ Portfolio backtest trades saved to portfolio_backtest_trades.csv")
    
    return {
        "trades_df": trades_df,
        "equity_curve": equity_curve,
        "portfolio_metrics": portfolio_metrics,
        "trade_metrics": trade_metrics,
        "initial_capital": initial_capital,
        "final_capital": capital,
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
    
    print(f"\n  💰 CAPITAL SUMMARY")
    print(f"  {'─'*80}")
    print(f"  Initial Capital:     ₹{initial_capital:>12,.0f}")
    print(f"  Final Portfolio:     ₹{final_capital:>12,.0f}")
    print(f"  Total Profit/Loss:   ₹{total_return:>12,.0f} ({total_return_pct:+.1f}%)")
    
    if portfolio_metrics and "cagr_pct" in portfolio_metrics:
        print(f"  CAGR:                {portfolio_metrics['cagr_pct']:>11.1f}%")
        print(f"  Sharpe Ratio:        {portfolio_metrics.get('sharpe_ratio', 0):>11.2f}")
        print(f"  Max Drawdown:        {portfolio_metrics.get('max_drawdown_pct', 0):>11.1f}%")
    
    print(f"\n  📈 TRADE STATISTICS")
    print(f"  {'─'*80}")
    if trade_metrics and "n_trades" in trade_metrics:
        print(f"  Total Trades:        {trade_metrics['n_trades']:>12}")
        print(f"  Win Rate:            {trade_metrics.get('win_rate_pct', 0):>11.1f}%")
        print(f"  Avg Winner:          ₹{trade_metrics.get('avg_winner_pct', 0):>11.1f}%")
        print(f"  Avg Loser:           ₹{trade_metrics.get('avg_loser_pct', 0):>11.1f}%")
        print(f"  Profit Factor:       {trade_metrics.get('profit_factor', 0):>11.2f}")
        print(f"  Expectancy/Trade:    {trade_metrics.get('expectancy_pct_per_trade', 0):>11.2f}%")
    
    if not trades_df.empty:
        print(f"\n  📋 TRADE LOG (first 10 trades)")
        print(f"  {'─'*80}")
        display_cols = ["Trade_ID", "Symbol", "Mode", "Entry_Date", "Exit_Date",
                        "Entry_Price", "Exit_Price", "Shares", "P&L_₹", "P&L_%", "Exit_Reason"]
        display_cols = [c for c in display_cols if c in trades_df.columns]
        print(trades_df[display_cols].head(10).to_string(index=False))
    
    print(f"\n{sep}")
    print("  ⚠️  Not SEBI-registered investment advice. Verify independently before trading.")
    print(sep + "\n")


if __name__ == "__main__":
    log.info("Starting 3-fold Walk-Forward Training & Validation...")
    
    price_data = {}
    fund_map = {}

    try:
        log.info("Fetching real universe data for multi-year historical lookback...")
        symbols = mod.get_nifty_500_universe()
        
        # FIX: Temporarily force yfinance to download data starting from 2019
        # to give indicators (like 200 DMA) enough data to warm up by 2020
        import datetime
        tickers_ns = [f"{s}.NS" for s in symbols]
        end_dt = datetime.datetime.now()
        start_dt = datetime.datetime(2019, 1, 1) # <--- Force pull from Jan 2019
        
        log.info(f"Downloading historical data from {start_dt.strftime('%Y-%m-%d')} to today...")
        
        # Leverage yfinance download directly with the expanded timeline
        import yfinance as yf
        raw_data = yf.download(tickers_ns, start=start_dt, end=end_dt, group_by="ticker", threads=True, progress=False, auto_adjust=True)
        
        price_data = {}
        for t in tickers_ns:
            sym = t.replace(".NS", "")
            df = raw_data[t].copy() if len(tickers_ns) > 1 else raw_data.copy()
            df = df.dropna(how="all")
            if not df.empty:
                df = df.dropna(subset=["Close", "Volume"])
                price_data[sym] = df
                
        log.info(f"✓ Historically aligned data ready for {len(price_data)} symbols.")
        
    except Exception as e:
        log.error(f"Failed to fetch universe data. Error: {e}")
        log.info("Falling back to simulated test data...")
        
        # Test data generation fallback
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=1000)
        def make_df(seed=42, regime="up"):
            np.random.seed(seed)
            drift = 0.0008 if regime == "up" else -0.0006
            returns = np.random.normal(drift, 0.015, 1000)
            prices = 100 * np.exp(np.cumsum(returns))
            prices[400:450] *= np.linspace(1.0, 0.55, 50)
            prices[450:500] *= np.linspace(0.55, 0.80, 50)
            vol = np.random.lognormal(14, 0.5, 1000).astype(int)
            return pd.DataFrame({
                "Open":   prices * (1 + np.random.normal(0, 0.003, 1000)),
                "High":   prices * (1 + np.abs(np.random.normal(0, 0.008, 1000))),
                "Low":    prices * (1 - np.abs(np.random.normal(0, 0.008, 1000))),
                "Close":  prices,
                "Volume": vol,
            }, index=dates)

        price_data = {
            "RELIANCE": make_df(42, "up"),
            "TCS":      make_df(43, "up"),
            "SBIN":     make_df(44, "up"),
            "MAHABANK": make_df(45, "up"),
        }
        fund_map = {
            "RELIANCE": Fundamentals(pe=22, fwd_pe=18, roe=11.4, rev_growth=12.8, earn_growth=14.4, debt_eq=0.4, div_yield=0.4, beta=0.8, mcap_cr=2000000),
            "TCS":      Fundamentals(pe=28, fwd_pe=24, roe=38.0, rev_growth=8.4, earn_growth=12.0, debt_eq=0.0, div_yield=1.4, beta=0.7, mcap_cr=1400000),
            "SBIN":     Fundamentals(pe=11, fwd_pe=9, roe=18.0, rev_growth=12.0, earn_growth=18.0, debt_eq=0.0, div_yield=1.8, beta=0.9, mcap_cr=700000),
            "MAHABANK": Fundamentals(pe=8, fwd_pe=6, roe=24.6, rev_growth=28.4, earn_growth=34.2, debt_eq=0.1, div_yield=1.8, beta=0.9, mcap_cr=50000),
        }

    if price_data:
        log.info("\n" + "="*50 + "\nPHASE 1: TRAINING (2020 - 2022)\n" + "="*50)
        train_results = run_full_walkforward_backtest(
            price_data, start_date="2020-01-01", end_date="2022-12-31"
        )
        evaluate_success_criteria(train_results["portfolio_metrics"], "TRAIN")

        log.info("\n" + "="*50 + "\nPHASE 2: TESTING (2023 - 2025)\n" + "="*50)
        test_results = run_full_walkforward_backtest(
            price_data, start_date="2023-01-01", end_date="2025-01-01"
        )
        evaluate_success_criteria(test_results["portfolio_metrics"], "TEST")

        log.info("\n" + "="*50 + "\nPHASE 3: VALIDATION (2025 - 2026)\n" + "="*50)
        val_results = run_full_walkforward_backtest(
            price_data, start_date="2025-01-01", end_date="2026-01-01"
        )
        evaluate_success_criteria(val_results["portfolio_metrics"], "VALIDATION")
        
        log.info("\n" + "="*50 + "\nRUNNING STANDARD BACKTEST SUITE (COVID STRESS TEST)\n" + "="*50)
        run_backtest(price_data, fund_map)