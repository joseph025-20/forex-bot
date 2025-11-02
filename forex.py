#!/usr/bin/env python3
import os
import time
import csv
import smtplib
import math
import logging
import requests

import numpy as np
import pandas as pd

from datetime import datetime, timedelta, timezone
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------- CONFIG ----------------
# Twelve Data API key (you provided)
TD_API_KEY = os.environ.get("TD_API_KEY", "36fc1a8e78b04087a01478ea48fc0716")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "1b9a3ec7f5874defa4de862175267e67v")

# Email defaults (edit if needed)
FROM_EMAIL = os.environ.get("FOREX_FROM_EMAIL", "kamjoselast1@gmail.com")
APP_PASSWORD = os.environ.get("FOREX_APP_PASSWORD", "xigk hqjh goae thnn")  # replace with 16-char app password (no spaces)
TO_EMAILS = os.environ.get("FOREX_TO_EMAILS", "josephwmwangi024@gmail.com,jcol64614@gmail.com").split(",")

# Money & risk
ACCOUNT_SIZE = float(os.environ.get("ACCOUNT_SIZE", "10.0"))  # $10 default as requested
RISK_PER_TRADE_PCT = float(os.environ.get("RISK_PER_TRADE_PCT", "0.01"))  # 1% default
STOP_ATR_MULT = float(os.environ.get("STOP_ATR_MULT", "1.5"))
TARGET_RR = float(os.environ.get("TARGET_RR", "2.0"))

# Pairs list (base, quote)
PAIRS = [
    ("EUR/USD", "EUR", "USD"),
    ("GBP/USD", "GBP", "USD"),
    ("USD/JPY", "USD", "JPY"),
    ("USD/CHF", "USD", "CHF"),
    ("AUD/USD", "AUD", "USD"),
    ("USD/CAD", "USD", "CAD"),
    ("XAU/USD", "XAU", "USD"),
]

LOG_FILE = os.environ.get("FOREX_LOG_FILE", "forex_trades_log.csv")

# Rate-limit friendly pauses (Twelve Data free tier generous but keep polite)
TD_SLEEP_SECONDS = float(os.environ.get("TD_SLEEP_SECONDS", "0.8"))
NEWS_SLEEP_SECONDS = float(os.environ.get("NEWS_SLEEP_SECONDS", "1.0"))

TEST_MODE = os.environ.get("TEST_MODE", "False").lower() in ("1", "true", "yes")

# Lot sizing
LOT_SIZE_UNITS = 100_000
MIN_LOT = 0.01
LOT_STEP = 0.01

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- CLIENTS ----------------
news_client = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
vader = SentimentIntensityAnalyzer()

# ---------------- TWELVE DATA HELPERS ----------------
def td_time_series(symbol: str, interval: str = "1day", outputsize: int = 500):
    """Fetch time series (daily) from Twelve Data. Returns list of dicts (newest first) or (None, error)."""
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": TD_API_KEY, "format": "JSON"}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if "values" in data:
            return data["values"], None
        # sometimes API returns a single price response or an error
        if "code" in data or "status" in data:
            return None, data
        return None, data
    except Exception as e:
        return None, {"error": str(e)}

def fetch_daily_td(base: str, quote: str):
    """Return a list of daily bars for base/quote using Twelve Data symbol format 'BASE/QUOTE'."""
    symbol = f"{base}/{quote}"
    values, err = td_time_series(symbol)
    time.sleep(TD_SLEEP_SECONDS)
    if values is None:
        # Try alternative symbol formats (some symbols may be returned as e.g. EURUSD)
        alt_symbol = f"{base}{quote}"
        values, err2 = td_time_series(alt_symbol)
        time.sleep(TD_SLEEP_SECONDS)
        if values is None:
            return None, err or err2
        return values, None
    return values, None

# ---------------- DATAFRAME & INDICATORS ----------------
def series_values_to_df(values):
    """Convert Twelve Data 'values' (list of dicts newest-first) to DataFrame sorted oldest->newest."""
    rows = []
    if not isinstance(values, list):
        return pd.DataFrame(rows)
    for v in values:
        # Twelve Data uses keys: datetime, open, high, low, close
        try:
            rows.append({
                "date": pd.to_datetime(v.get("datetime") or v.get("datetime").split(" ")[0]),
                "open": float(v.get("open")),
                "high": float(v.get("high")),
                "low": float(v.get("low")),
                "close": float(v.get("close"))
            })
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=21, adjust=False).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14, min_periods=14).mean()
    return df

# ---------------- SIGNAL SCORING ----------------
def score_row(r, df):
    score = 0
    reasons = []
    try:
        if not np.isnan(r.get("EMA9", np.nan)) and not np.isnan(r.get("EMA21", np.nan)):
            if r["EMA9"] > r["EMA21"]:
                score += 1; reasons.append("EMA9>EMA21")
            else:
                reasons.append("EMA9<EMA21")
        if not np.isnan(r.get("MACD", np.nan)) and not np.isnan(r.get("MACD_signal", np.nan)):
            if r["MACD"] > r["MACD_signal"]:
                score += 1; reasons.append("MACD>Signal")
            else:
                reasons.append("MACD<Signal")
        if not np.isnan(r.get("RSI", np.nan)):
            if r["RSI"] <= 30:
                score += 1; reasons.append(f"RSI{r['RSI']:.1f}(oversold)")
            elif 40 < r["RSI"] < 70:
                score += 1; reasons.append(f"RSI{r['RSI']:.1f}(momentum OK)")
            elif r["RSI"] >= 70:
                reasons.append(f"RSI{r['RSI']:.1f}(overbought)")
        ma50 = df["close"].rolling(50, min_periods=1).mean().iloc[-1]
        if r["close"] > ma50:
            score += 1; reasons.append("Price>50MA")
        else:
            reasons.append("Price<50MA")
        if score >= 3:
            label = "BUY"
        elif score <= 1:
            label = "SELL"
        else:
            label = "HOLD"
        return label, score, reasons
    except Exception as e:
        return "HOLD", 0, [f"scoring error: {e}"]

# ---------------- TRADE LEVELS & SIZING ----------------
def compute_trade_levels(latest_close, atr, direction):
    entry = float(latest_close)
    fallback_pct = 0.005
    if atr is None or np.isnan(atr) or atr == 0.0:
        stop_distance = max(entry * fallback_pct, 1e-8)
    else:
        stop_distance = atr * STOP_ATR_MULT
    if direction == "BUY":
        stop = entry - stop_distance
        target = entry + stop_distance * TARGET_RR
    else:
        stop = entry + stop_distance
        target = entry - stop_distance * TARGET_RR
    risk_per_unit = abs(entry - stop)
    return entry, stop, target, risk_per_unit

def suggested_lot_and_units(account_size, risk_pct, entry, stop, pair_base, pair_quote, quote_to_usd_rate=None):
    risk_amount_usd = account_size * risk_pct
    price_diff = abs(entry - stop)
    if price_diff == 0:
        return 0.0, 0
    if pair_quote == "USD":
        risk_per_base_unit_usd = price_diff
    else:
        if quote_to_usd_rate is None:
            return 0.0, 0
        risk_per_base_unit_usd = price_diff * quote_to_usd_rate
    units = risk_amount_usd / risk_per_base_unit_usd
    if units <= 0:
        return 0.0, 0
    lots = units / float(LOT_SIZE_UNITS)
    lots = math.floor(lots / LOT_STEP) * LOT_STEP
    if lots < MIN_LOT:
        return 0.0, int(max(0, int(units)))
    base_units = int(lots * LOT_SIZE_UNITS)
    return round(lots, 2), base_units

# ---------------- BACKTEST ----------------
def backtest_df(df: pd.DataFrame):
    df = df.copy().reset_index(drop=True)
    trades = []
    for i in range(50, len(df) - 1):
        row = df.iloc[i]
        label, score, reasons = score_row(row, df.iloc[:i+1])
        if label == "HOLD":
            continue
        entry = df.loc[i + 1, "open"]
        high = df.loc[i + 1, "high"]
        low = df.loc[i + 1, "low"]
        close = df.loc[i + 1, "close"]
        atr = row.get("ATR14", np.nan)
        if np.isnan(atr) or atr == 0:
            continue
        entry_price = entry
        stop_distance = atr * STOP_ATR_MULT
        if label == "BUY":
            stop = entry_price - stop_distance
            target = entry_price + stop_distance * TARGET_RR
            if high >= target:
                exit_price, exit_reason = target, "TP"
            elif low <= stop:
                exit_price, exit_reason = stop, "SL"
            else:
                exit_price, exit_reason = close, "CLOSE"
            ret = (exit_price - entry_price) / entry_price
        else:
            stop = entry_price + stop_distance
            target = entry_price - stop_distance * TARGET_RR
            if low <= target:
                exit_price, exit_reason = target, "TP"
            elif high >= stop:
                exit_price, exit_reason = stop, "SL"
            else:
                exit_price, exit_reason = close, "CLOSE"
            ret = (entry_price - exit_price) / entry_price
        trades.append({
            "date": str(df.loc[i + 1, "date"].date()),
            "label": label,
            "score": score,
            "reasons": ";".join(reasons),
            "entry": entry_price,
            "stop": stop,
            "target": target,
            "exit": exit_price,
            "exit_reason": exit_reason,
            "return": ret
        })
    if not trades:
        return {"trades": 0}
    returns = np.array([t["return"] for t in trades])
    wins = (returns > 0).sum()
    total = len(returns)
    win_rate = wins / total * 100
    avg_ret = returns.mean()
    med_ret = np.median(returns)
    cum_return = np.prod(1 + returns) - 1
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    return {
        "trades": total,
        "wins": int(wins),
        "win_rate": round(win_rate, 2),
        "avg_return": round(avg_ret * 100, 4),
        "median_return": round(med_ret * 100, 4),
        "cumulative_return_pct": round(cum_return * 100, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "sample_trades": trades[:5]
    }

# ---------------- NEWS SENTIMENT ----------------
def fetch_news_for_keyword(keyword: str, hours: int = 24, page_size: int = 6):
    if news_client is None:
        return None, []
    try:
        to_dt = datetime.utcnow()
        from_dt = to_dt - timedelta(hours=hours)
        resp = news_client.get_everything(
            q=keyword,
            language="en",
            from_param=from_dt.isoformat(timespec='seconds') + "Z",
            to=to_dt.isoformat(timespec='seconds') + "Z",
            sort_by="publishedAt",
            page_size=page_size
        )
        articles = resp.get("articles", [])
        if not articles:
            return None, []
        scores = []
        headlines = []
        for a in articles:
            title = (a.get("title") or "").strip()
            desc = (a.get("description") or "").strip()
            text = f"{title} {desc}".strip()
            comp = vader.polarity_scores(text)["compound"]
            scores.append(comp)
            headlines.append((title, comp, a.get("url", "")))
        avg = sum(scores) / len(scores) if scores else None
        return avg, headlines
    except Exception as e:
        return None, [("NewsAPI error", 0, str(e))]

def aggregate_pair_sentiment(base: str, quote: str):
    keywords = [f"{base}/{quote}", f"{base}{quote}", base, quote,
                "Federal Reserve", "interest rates", "inflation", "central bank"]
    if base == "XAU" or quote == "XAU":
        keywords = ["gold", "XAU", "gold price", "inflation", "Federal Reserve"]
    scores = []
    headlines = []
    pos = neu = neg = 0
    for kw in keywords:
        avg, arts = fetch_news_for_keyword(kw, hours=24, page_size=6)
        time.sleep(NEWS_SLEEP_SECONDS)
        if avg is not None:
            scores.append(avg)
        for t, comp, url in arts[:3]:
            try:
                headlines.append((t, comp, url))
                if comp >= 0.2: pos += 1
                elif comp <= -0.2: neg += 1
                else: neu += 1
            except Exception:
                continue
        if len(headlines) >= 8:
            break
    overall = (sum(scores) / len(scores)) if scores else None
    counts = {"pos": pos, "neu": neu, "neg": neg}
    return overall, counts, headlines[:8]

# ---------------- COMBINE SIGNALS ----------------
def interpret_sentiment(compound):
    if compound is None:
        return "No data"
    if compound >= 0.2:
        return f"Bullish (+{compound:.2f})"
    if compound <= -0.2:
        return f"Bearish ({compound:.2f})"
    return f"Neutral ({compound:.2f})"

def combine_signal(tech_label: str, sentiment_score):
    if sentiment_score is None:
        if tech_label == "BUY":
            return "BUY (no news)"
        if tech_label == "SELL":
            return "SELL (no news)"
        return "HOLD (no news)"
    if tech_label == "BUY":
        if sentiment_score >= 0.2:
            return "STRONG BUY 🔥"
        if sentiment_score <= -0.2:
            return "CAUTION ⚠️ (tech BUY, news bearish)"
        return "BUY 👍"
    if tech_label == "SELL":
        if sentiment_score <= -0.2:
            return "STRONG SELL 🔥"
        if sentiment_score >= 0.2:
            return "CAUTION ⚠️ (tech SELL, news bullish)"
        return "SELL 👎"
    if sentiment_score >= 0.2:
        return "WEAK BUY (news bullish)"
    if sentiment_score <= -0.2:
        return "WEAK SELL (news bearish)"
    return "HOLD ➡️"

# ---------------- CSV logging ----------------
def log_trade_to_csv(path, trade):
    header = ["date","pair","signal","score","reasons","entry","stop","target","exit","exit_reason","return"]
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow({k: trade.get(k,"") for k in header})
    except Exception as e:
        logging.error("CSV log failed: %s", e)

# ---------------- ANALYZE & REPORT ----------------
def analyze_pair(pair_name, base, quote):
    # get daily bars from Twelve Data
    values, err = fetch_daily_td(base, quote)
    if values is None:
        logging.warning("%s: data error — %s", pair_name, err)
        return f"{pair_name}: data error — {err}", None

    df = series_values_to_df(values)
    if df.empty or len(df) < 60:
        msg = f"{pair_name}: not enough data ({len(df)} rows)"
        logging.warning(msg)
        return msg, None

    df = compute_indicators(df)
    latest = df.iloc[-1]
    tech_label, score, reasons = score_row(latest, df)

    entry, stop, target, risk_unit = compute_trade_levels(latest["close"], latest.get("ATR14", np.nan),
                                                          tech_label if tech_label in ("BUY","SELL") else "BUY")

    # if quote != USD -> fetch quote->USD to convert risk into USD
    quote_to_usd = None
    if quote != "USD":
        conv_vals, conv_err = fetch_daily_td(quote, "USD")
        if conv_vals is not None:
            conv_df = series_values_to_df(conv_vals)
            if not conv_df.empty:
                quote_to_usd = conv_df.iloc[-1]["close"]
        time.sleep(TD_SLEEP_SECONDS)

    suggested_lots, suggested_units = suggested_lot_and_units(ACCOUNT_SIZE, RISK_PER_TRADE_PCT, entry, stop,
                                                              base, quote, quote_to_usd)
    sentiment_score, counts, headlines = aggregate_pair_sentiment(base, quote)
    final = combine_signal(tech_label, sentiment_score)
    stats = backtest_df(df)
    return {
        "pair_name": pair_name,
        "price": latest["close"],
        "date": str(latest["date"].date()),
        "tech_label": tech_label,
        "score": score,
        "reasons": reasons,
        "sentiment_score": sentiment_score,
        "sentiment_counts": counts,
        "headlines": headlines,
        "final": final,
        "entry": entry,
        "stop": stop,
        "target": target,
        "suggested_qty": suggested_units,
        "suggested_lots": suggested_lots,
        "stats": stats
    }, None

def build_report():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"📊 Forex Market Report + News Sentiment — {ts}\n\n"]
    for pair_name, base, quote in PAIRS:
        try:
            result, _ = analyze_pair(pair_name, base, quote)
            if isinstance(result, str):
                lines.append(result + "\n\n")
                continue
            lines.append(f"🔹 {result['pair_name']} ({result['date']})\n")
            price = result['price']
            try:
                price_str = f"{price:.5f}"
            except Exception:
                price_str = str(price)
            lines.append(f"  Price: {price_str}\n")
            lines.append(f"  Technical: {result['tech_label']} (Score {result['score']}/4) — {', '.join(result['reasons'])}\n")
            lines.append(f"  News Sentiment: {interpret_sentiment(result['sentiment_score'])}\n")
            lines.append(f"  Final Decision: {result['final']}\n")
            lines.append("  Trade Plan:\n")
            try:
                lines.append(f"    Entry (approx): {result['entry']:.5f}\n")
                lines.append(f"    Stop Loss:       {result['stop']:.5f} (ATR×{STOP_ATR_MULT})\n")
                lines.append(f"    Target Profit:   {result['target']:.5f} (RR {TARGET_RR}:1)\n")
            except Exception:
                lines.append(f"    Entry: {result['entry']}\n")
                lines.append(f"    Stop: {result['stop']}\n")
                lines.append(f"    Target: {result['target']}\n")
            lines.append(f"    Suggested lots: {result['suggested_lots']} (lot step {LOT_STEP}, min {MIN_LOT})\n")
            lines.append(f"    Suggested units (base): {result['suggested_qty']}\n")
            lines.append(f"    Account size used: ${ACCOUNT_SIZE}, risk per trade: {RISK_PER_TRADE_PCT*100}%\n")
            if result['headlines']:
                lines.append("  Top headlines (recent):\n")
                for title, comp, url in result['headlines'][:4]:
                    lines.append(f"    - {title} ({comp:+.2f})\n")
            stats = result.get("stats", {})
            if stats and stats.get("trades", 0) > 0:
                lines.append("  Backtest (1-day hold with SL/TP simulation):\n")
                lines.append(f"    Trades: {stats['trades']}, Wins: {stats['wins']}, Win rate: {stats['win_rate']}%\n")
                lines.append(f"    Avg return: {stats['avg_return']}% , Median: {stats['median_return']}%\n")
                lines.append(f"    Cumulative return: {stats['cumulative_return_pct']}% , Max drawdown: {stats['max_drawdown_pct']}%\n")
                for t in stats.get("sample_trades", []):
                    log_trade = {
                        "date": t["date"], "pair": result['pair_name'], "signal": t["label"],
                        "score": t["score"], "reasons": t["reasons"], "entry": t["entry"],
                        "stop": t["stop"], "target": t["target"], "exit": t["exit"],
                        "exit_reason": t["exit_reason"], "return": t["return"]
                    }
                    try:
                        log_trade_to_csv(LOG_FILE, log_trade)
                    except Exception:
                        pass
            lines.append("\n")
        except Exception as e:
            logging.exception("Error during analysis for %s: %s", pair_name, e)
            lines.append(f"{pair_name}: error during analysis: {e}\n\n")
        time.sleep(0.5)
    return "".join(lines)

def send_email(body: str):
    if TEST_MODE:
        print("TEST_MODE enabled — email suppressed. REPORT:\n")
        print(body)
        return
    msg = MIMEMultipart()
    msg["From"] = FROM_EMAIL
    msg["To"] = ", ".join(TO_EMAILS)
    msg["Subject"] = "📊 Daily Forex Market Report + News Sentiment"
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(FROM_EMAIL, APP_PASSWORD)
            s.send_message(msg)
        print("✅ Email sent.")
    except smtplib.SMTPAuthenticationError as e:
        print("❌ SMTP Authentication failed:", e)
    except Exception as e:
        print("❌ Sending email failed:", e)

# ---------------- RUN ----------------
if __name__ == "__main__":
    report = build_report()
    print(report)
    send_email(report)
