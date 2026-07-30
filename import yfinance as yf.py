import yfinance as yf

for s in ["WAAREEENER", "KALYANKJIL", "LLOYDSME"]:
    info = yf.Ticker(f"{s}.NS").info
    print(s, info.get("debtToEquity"))
    