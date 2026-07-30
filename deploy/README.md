# Nifty 500 Screener + Backtest Engine — Local Setup (VS Code)

## What's in this folder
- `unified_screener_v4_final.py` — main screener. Scans the full Nifty 500,
  produces 3 lists: **Swing Book** (3-6mo), **Core Book** (2yr), **Crossover**
  (passes both). Also prints the Nifty Market Regime banner at the top.
- `backtest_engine.py` — walk-forward backtest of the same signal logic
  against historical data, plus a Covid-era stress test.
- `requirements.txt` — Python dependencies.

Both files must stay in the **same folder** — `backtest_engine.py` auto-detects
`unified_screener_v4_final.py` sitting next to it.

---

## 1. Prerequisites
- Python 3.10+ installed (check: `python3 --version`)
- VS Code with the Python extension installed

## 2. Open the project in VS Code
1. Unzip this folder somewhere, e.g. `~/projects/nifty_screener/`
2. In VS Code: **File → Open Folder** → select that folder
3. Open a terminal inside VS Code: **Terminal → New Terminal**

## 3. Set up a virtual environment (recommended, keeps things clean)
```bash
python3 -m venv venv

# Activate it:
# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1
```
VS Code may prompt "Select Interpreter" — pick the one inside `venv/`.

## 4. Install dependencies
```bash
pip install -r requirements.txt
```
If you hit an externally-managed-environment error on Linux and skipped the
venv step, use: `pip install -r requirements.txt --break-system-packages`

## 5. (Optional) API key for real analyst price targets
Everything works with zero API keys — this step is 100% optional and only
adds one extra data column (real sell-side analyst consensus targets instead
of the model's own internal target estimate).

1. Sign up at https://analyst.indianapi.in/ (has a free tier — check their
   current pricing/limits before relying on it)
2. Get your API key from their dashboard
3. Open `unified_screener_v4_final.py`, find this block near the top
   (search for `ANALYST_API_CONFIG`):
   ```python
   ANALYST_API_CONFIG = {
       "api_key": None,   # <-- put your key here, e.g. "abc123..."
       "base_url": "https://analyst.indianapi.in",
   }
   ```
4. Replace `None` with your key as a string: `"api_key": "your_actual_key_here",`
5. Save the file.

If you skip this entirely, the script just shows `not_configured` for that
one column and works normally otherwise — nothing else depends on it.

## 6. Run the full scan (all 3 lists)
```bash
python3 unified_screener_v4_final.py
```
This will:
1. Print the **Nifty Market Regime** banner first (trend/momentum/breakout read)
2. Fetch the live Nifty 500 constituent list from NSE
3. Download price history for all ~500 stocks (this takes a while —
   yfinance rate-limits requests, expect **10-25 minutes** depending on
   your connection; it's polite/batched on purpose so NSE/Yahoo don't
   block you)
4. Print **Swing Book**, **Core Book**, and **Crossover** tables to the
   terminal, ranked and ready to read

No arguments needed — this is the "scan everything" mode.

### Faster test run (a handful of stocks, ~30 seconds)
Useful the first time, to confirm everything's wired up before committing to
a full 500-stock run:
```python
# Run this instead, e.g. in a scratch.py file or a Python REPL:
from unified_screener_v4_final import run_screener, download_universe

test_universe = download_universe(["RELIANCE.NS", "HDFCBANK.NS", "TATASTEEL.NS",
                                     "BEL.NS", "CGPOWERANDINDUSTRIALSOLUTIONS.NS"])
run_screener(universe_override=test_universe)
```

## 7. Run the backtest engine
```bash
python3 backtest_engine.py
```
As delivered, this runs its built-in **synthetic self-test** (proves the
walk-forward + Covid stress-test mechanics work correctly) — it does NOT
backtest your real Nifty 500 picks by default, since that requires wiring in
real multi-year historical data.

To backtest against real historical data instead, you'll want to call
`run_backtest()` directly with real downloaded price data (multiple years'
worth — Core Book's 365-day hold needs well over a year of bars per stock
just to open one test window), e.g.:
```python
from backtest_engine import run_backtest
from unified_screener_v4_final import download_universe

symbols = ["RELIANCE.NS", "HDFCBANK.NS", "TATASTEEL.NS"]  # or the full Nifty 500 list
price_data = download_universe(symbols)  # needs several years of history
# then pass price_data (and fundamentals if you have them) into run_backtest()
```

## Troubleshooting
- **`No usable price data for ANY symbol`** → Yahoo Finance is unreachable
  from your network (corporate firewall/VPN?), or you're rate-limited —
  wait a few minutes and retry.
- **`ModuleNotFoundError`** → you forgot to activate the venv, or skipped
  `pip install -r requirements.txt`.
- **Output looks stale/cut off** → the full 500-stock run prints a lot;
  scroll up, or redirect to a file:
  `python3 unified_screener_v4_final.py > results.txt 2>&1`
