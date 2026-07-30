# Enhancement Plan: Live Screener Output & Portfolio Backtest

## Current State Analysis

### Live Screener (v9)
The `calculate_position_size()` function already outputs:
- `shares`: number of shares to buy
- `deploy`: amount to deploy (₹)
- `max_risk`: maximum risk amount (₹)
- `deploy_pct_of_portfolio`: percentage of portfolio

The swing evaluation already outputs:
- `Entry`, `Target`, `Stop_Loss`, `Trailing_Stop`, `Profit_Booking_1`, `Profit_Booking_2`
- `Risk_%`, `Reward_%`, `RR_Ratio`

**Gap**: Need to verify this is clearly presented in the CSV output and console summary.

### Backtest Engine
Current `backtest_signal()`:
- Simulates holding for `hold_days` (default 90)
- Checks if target or stop loss is hit
- Does NOT use trailing stop or profit booking logic
- Does NOT simulate a portfolio with ₹1 lakh capital
- Does NOT log individual trade P&L in ₹ terms

Current `run_full_walkforward_backtest()`:
- Has portfolio simulation with equity tracking
- Uses equal allocation (1/n) per trade, NOT screener's position sizing
- Does NOT use trailing stop or profit booking logic
- Does NOT log individual trade P&L in ₹ terms

## Proposed Enhancements

### 1. Live Screener Output Enhancement
**Goal**: Make sure users see exactly how much to invest, how many shares, SL, target, risk, and potential profit.

**Changes**:
- Verify CSV columns include: `Shares_To_Buy`, `Deploy_₹`, `Max_Risk_₹`, `Stop_Loss`, `Target`, `Trailing_Stop`, `Profit_Booking_1`, `Profit_Booking_2`, `Risk_%`, `Reward_%`
- Add console summary showing total portfolio allocation
- Add a "Portfolio Summary" section at the end of the screener run

### 2. Portfolio-Based Backtest Function
**Goal**: Simulate a ₹1 lakh portfolio that trades exactly as the screener suggests.

**New Function**: `run_portfolio_backtest()`

**Key Features**:
- Start with ₹1,00,000 capital
- Use `calculate_position_size()` for position sizing (not equal allocation)
- Enter trades at screener's suggested entry price
- Exit based on:
  - Stop loss hit
  - Target hit
  - Trailing stop hit (2x ATR below highest price since entry)
  - Profit booking at 1.5x R:R (partial exit)
  - Max hold period (90 days for swing, 365 for core)
- Log each trade with:
  - Entry date, exit date
  - Entry price, exit price
  - Shares bought, total deploy amount
  - Stop loss, target, trailing stop
  - Profit/loss in ₹ and %
  - Outcome (WIN/LOSS/PARTIAL)
- Track portfolio value over time
- Generate equity curve
- Calculate CAGR, Sharpe, max drawdown

**Implementation Details**:
```python
def run_portfolio_backtest(price_data: dict, fund_map: dict = None,
                            start_date: str = "2021-01-01", end_date: str = "2026-01-01",
                            initial_capital: float = 100000):
    """
    Portfolio-based backtest that simulates actual trading with ₹1 lakh capital.
    Uses screener's position sizing and exit logic.
    """
```

**Trade Simulation Logic**:
1. At each signal date, calculate position size using `calculate_position_size()`
2. Check if we have enough capital to enter the trade
3. Track open positions (max 5 per book to match screener)
4. For each open position, check daily:
   - If price hits stop loss → exit at SL
   - If price hits target → exit at target
   - If price hits trailing stop → exit at trailing stop
   - If price hits profit booking level → partial exit (50% at 1.5x R:R)
   - If hold period expires → exit at market price
5. Calculate P&L in ₹ terms
6. Update portfolio value

### 3. Enhanced Trade Logging
**Goal**: Log every trade with full details for analysis.

**New CSV Output**: `portfolio_backtest_trades.csv`
Columns:
- `Trade_ID`, `Symbol`, `Mode`, `Signal_Date`, `Entry_Date`, `Exit_Date`
- `Entry_Price`, `Exit_Price`, `Shares`, `Deploy_₹`
- `Stop_Loss`, `Target`, `Trailing_Stop`
- `Exit_Reason` (SL/Target/Trailing/Time/Partial)
- `P&L_₹`, `P&L_%`, `Outcome`
- `Portfolio_Value_After`

### 4. Summary Report
**Goal**: Show final portfolio value, total profit, CAGR, etc.

**Console Output**:
```
══════════════════════════════════════════════════════════════════════════
  📊 PORTFOLIO BACKTEST REPORT — ₹1,00,000 Capital
══════════════════════════════════════════════════════════════════════════
  Initial Capital:     ₹1,00,000
  Final Portfolio:     ₹X,XX,XXX
  Total Profit:        ₹XX,XXX (XX.X%)
  CAGR:                XX.X%
  Sharpe Ratio:        X.XX
  Max Drawdown:        XX.X%
  Total Trades:        XXX
  Win Rate:            XX.X%
  Avg Win:             ₹X,XXX
  Avg Loss:            ₹X,XXX
  Profit Factor:       X.XX
══════════════════════════════════════════════════════════════════════════
```

## Implementation Order

1. **First**: Enhance live screener output (verify CSV columns, add console summary)
2. **Second**: Create `run_portfolio_backtest()` function in `backtest_engine.py`
3. **Third**: Add trade logging with ₹ P&L
4. **Fourth**: Add summary report generation
5. **Fifth**: Test with sample data

## Files to Modify

1. `unified_screener_v9_final.py` - Verify output columns, add console summary
2. `backtest_engine.py` - Add `run_portfolio_backtest()` function
3. `deploy/unified_screener_v9_final.py` - Mirror changes from v9
