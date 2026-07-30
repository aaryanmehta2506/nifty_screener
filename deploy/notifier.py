"""
Daily picks notifier - sends top 5 swing + top 5 core via Telegram or Email.
"""
import os
import requests
import pandas as pd
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8920415512:AAFEJzNo1-GEUZaQyYtt_tHogAq9BZQdLKc")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "837949537")

def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        print("Telegram sent")
        return True
    except Exception as e:
        print(f"Telegram failed: {e}")
        return False

def main():
    swing = pd.read_csv("swing_book_3_6mo.csv").head(5)
    core = pd.read_csv("core_book_1_2yr.csv").head(5)
    today = datetime.now().strftime("%d %b %Y")
    lines = [f"*Nifty Screener - {today}*", ""]
    if not swing.empty:
        lines.append("*Swing (3-6mo):*")
        for i, row in swing.iterrows():
            lines.append(f"{i+1}. {row.get('Symbol','?')} | Score: {row.get('Score','?')} | {row.get('Market_Regime','?')}")
        lines.append("")
    if not core.empty:
        lines.append("*Core (1-2yr):*")
        for i, row in core.iterrows():
            lines.append(f"{i+1}. {row.get('Symbol','?')} | Score: {row.get('Score','?')} | {row.get('Market_Regime','?')}")
    msg = "\n".join(lines)
    send_telegram(msg)

if __name__ == "__main__":
    main()
