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


def _fmt(val, fmt=".2f"):
    """Safely format a value, returning '—' if missing/NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:{fmt}}"
    return str(val)


def build_message() -> tuple[str, str]:
    """Return (telegram_text, email_html) for today's picks."""
    try:
        swing = pd.read_csv("swing_book_3_6mo.csv").head(5)
    except FileNotFoundError:
        swing = pd.DataFrame()
    try:
        core = pd.read_csv("core_book_1_2yr.csv").head(5)
    except FileNotFoundError:
        core = pd.DataFrame()
    today = datetime.now().strftime("%d %b %Y")

    # ── Telegram (Markdown) ────────────────────────────────────────────
    lines = [f"*Nifty Screener - {today}*", ""]
    if not swing.empty:
        lines.append("*Swing (3-6mo):*")
        for i, row in swing.iterrows():
            lines.append(
                f"{i+1}. {row.get('Symbol','?')} | Conf: {_fmt(row.get('Confidence_%'), '.0f')}% | "
                f"CMP: ₹{_fmt(row.get('CMP'), '.1f')} | Target: ₹{_fmt(row.get('Target'), '.1f')} | "
                f"SL: ₹{_fmt(row.get('Stop_Loss'), '.1f')} | R:R: {_fmt(row.get('RR_Ratio'), '.1f')} | "
                f"Regime: {row.get('Market_Regime','?')}"
            )
        lines.append("")
    if not core.empty:
        lines.append("*Core (1-2yr):*")
        for i, row in core.iterrows():
            lines.append(
                f"{i+1}. {row.get('Symbol','?')} | {row.get('Grade','?')} | Score: {row.get('Score','?')} | "
                f"CMP: ₹{_fmt(row.get('CMP'), '.1f')} | Target: ₹{_fmt(row.get('Target'), '.1f')} | "
                f"SL: ₹{_fmt(row.get('Stop_Loss'), '.1f')} | R:R: {_fmt(row.get('RR_Ratio'), '.1f')} | "
                f"Conf: {_fmt(row.get('Confidence_%'), '.0f')}% | Regime: {row.get('Market_Regime','?')}"
            )
    telegram_text = "\n".join(lines)

    # ── Email (HTML) ──────────────────────────────────────────────────
    def _swing_rows():
        if swing.empty:
            return '<tr><td colspan="9">No picks today</td></tr>'
        rows = []
        for i, row in swing.iterrows():
            rows.append(f"""
            <tr>
              <td>{i+1}</td>
              <td><strong>{row.get('Symbol','?')}</strong></td>
              <td>{_fmt(row.get('Confidence_%'), '.0f')}%</td>
              <td>₹{_fmt(row.get('CMP'), '.1f')}</td>
              <td>₹{_fmt(row.get('Target'), '.1f')}</td>
              <td>₹{_fmt(row.get('Stop_Loss'), '.1f')}</td>
              <td>{_fmt(row.get('Profit_%'), '.1f')}%</td>
              <td>{_fmt(row.get('RR_Ratio'), '.1f')}</td>
              <td>{row.get('Market_Regime','?')}</td>
            </tr>""")
        return "".join(rows)

    def _core_rows():
        if core.empty:
            return '<tr><td colspan="11">No picks today</td></tr>'
        rows = []
        for i, row in core.iterrows():
            rows.append(f"""
            <tr>
              <td>{i+1}</td>
              <td><strong>{row.get('Symbol','?')}</strong></td>
              <td>{row.get('Grade','?')}</td>
              <td>{_fmt(row.get('Score'), '.0f')}</td>
              <td>₹{_fmt(row.get('CMP'), '.1f')}</td>
              <td>₹{_fmt(row.get('Target'), '.1f')}</td>
              <td>₹{_fmt(row.get('Stop_Loss'), '.1f')}</td>
              <td>{_fmt(row.get('Profit_%'), '.1f')}%</td>
              <td>{_fmt(row.get('Risk_%'), '.1f')}%</td>
              <td>{_fmt(row.get('RR_Ratio'), '.1f')}</td>
              <td>{_fmt(row.get('Confidence_%'), '.0f')}%</td>
            </tr>""")
        return "".join(rows)

    email_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: Arial, sans-serif; background: #f4f4f4; padding: 20px; }}
    .container {{ max-width: 800px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 8px; }}
    h1 {{ color: #1a73e8; }}
    h2 {{ color: #333; border-bottom: 2px solid #1a73e8; padding-bottom: 5px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th {{ background: #1a73e8; color: #fff; padding: 8px; text-align: left; }}
    td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
    .footer {{ margin-top: 20px; font-size: 12px; color: #888; }}
    .details {{ margin-top: 20px; padding: 15px; background: #f9f9f9; border-radius: 5px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>📈 Nifty Screener - {today}</h1>
    <p>Daily stock picks generated automatically by the Nifty 500 screener.</p>

    <h2>Swing (3-6 months)</h2>
    <table>
      <tr>
        <th>#</th><th>Symbol</th><th>Confidence</th><th>CMP</th><th>Target</th>
        <th>Stop Loss</th><th>Profit %</th><th>R:R</th><th>Regime</th>
      </tr>
      {_swing_rows()}
    </table>

    <h2>Core (1-2 years)</h2>
    <table>
      <tr>
        <th>#</th><th>Symbol</th><th>Signal</th><th>Score</th><th>CMP</th><th>Target</th>
        <th>Stop Loss</th><th>Profit %</th><th>Risk %</th><th>R:R</th><th>Confidence</th>
      </tr>
      {_core_rows()}
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
