"""
Daily picks notifier - sends top 5 swing + top 5 core via Telegram and/or Email.
"""
import os
import smtplib
import requests
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# ── Telegram ──────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ── Email ─────────────────────────────────────────────────────────────
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT", "")


def escape_markdown(text: str) -> str:
    """Escape Telegram Markdown special characters."""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text


def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured — skipping")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    escaped_text = escape_markdown(text)
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": escaped_text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        print("Telegram sent")
        return True
    except Exception as e:
        print(f"Telegram failed: {e}")
        return False


def send_email(subject: str, html_body: str) -> bool:
    if not all([SMTP_SERVER, EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT]):
        print("Email not configured — skipping")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECIPIENT
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, [EMAIL_RECIPIENT], msg.as_string())
        print("Email sent")
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False


def build_message() -> tuple[str, str]:
    """Return (telegram_text, email_html) for today's picks."""
    swing = pd.read_csv("swing_book_3_6mo.csv").head(5)
    core = pd.read_csv("core_book_1_2yr.csv").head(5)
    today = datetime.now().strftime("%d %b %Y")

    # ── Telegram (Markdown) ────────────────────────────────────────────
    lines = [f"*Nifty Screener - {today}*", ""]
    if not swing.empty:
        lines.append("*Swing (3-6mo):*")
        for i, row in swing.iterrows():
            lines.append(
                f"{i+1}. {row.get('Symbol','?')} | Score: {row.get('Score','?')} | {row.get('Market_Regime','?')}"
            )
        lines.append("")
    if not core.empty:
        lines.append("*Core (1-2yr):*")
        for i, row in core.iterrows():
            lines.append(
                f"{i+1}. {row.get('Symbol','?')} | Score: {row.get('Score','?')} | {row.get('Market_Regime','?')}"
            )
    telegram_text = "\n".join(lines)

    # ── Email (HTML) ──────────────────────────────────────────────────
    swing_rows = ""
    for i, row in swing.iterrows():
        swing_rows += f"""
        <tr>
          <td>{i+1}</td>
          <td><strong>{row.get('Symbol','?')}</strong></td>
          <td>{row.get('Score','?')}</td>
          <td>{row.get('Market_Regime','?')}</td>
        </tr>"""

    core_rows = ""
    for i, row in core.iterrows():
        core_rows += f"""
        <tr>
          <td>{i+1}</td>
          <td><strong>{row.get('Symbol','?')}</strong></td>
          <td>{row.get('Score','?')}</td>
          <td>{row.get('Market_Regime','?')}</td>
        </tr>"""

    email_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px; }}
    .container {{ max-width: 600px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 8px; }}
    h1 {{ color: #1a73e8; }}
    h2 {{ color: #333; border-bottom: 2px solid #1a73e8; padding-bottom: 5px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th {{ background: #1a73e8; color: #fff; padding: 8px; text-align: left; }}
    td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
    .footer {{ margin-top: 20px; font-size: 12px; color: #888; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>📈 Nifty Screener - {today}</h1>
    <p>Daily stock picks generated automatically by the Nifty 500 screener.</p>

    <h2>Swing (3-6 months)</h2>
    <table>
      <tr><th>#</th><th>Symbol</th><th>Score</th><th>Regime</th></tr>
      {swing_rows if swing_rows else '<tr><td colspan="4">No picks today</td></tr>'}
    </table>

    <h2>Core (1-2 years)</h2>
    <table>
      <tr><th>#</th><th>Symbol</th><th>Score</th><th>Regime</th></tr>
      {core_rows if core_rows else '<tr><td colspan="4">No picks today</td></tr>'}
    </table>

    <div class="footer">
      Generated on {datetime.now().strftime("%Y-%m-%d %H:%M IST")} |
      <a href="https://github.com/aaryanmehta2506/nifty_screener">View on GitHub</a>
    </div>
  </div>
</body>
</html>"""

    return telegram_text, email_html


def main():
    telegram_text, email_html = build_message()

    sent_telegram = send_telegram(telegram_text)
    sent_email = send_email(f"Nifty Screener - {datetime.now().strftime('%d %b %Y')}", email_html)

    if sent_telegram or sent_email:
        print("Notifications sent successfully")
    else:
        print("No notifications sent — check configuration")


if __name__ == "__main__":
    main()
