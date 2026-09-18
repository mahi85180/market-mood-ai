# score_tracker.py
import pandas as pd
import os
from datetime import datetime

SCORES_DIR = "daily_predictions"
SCORES_FILE = os.path.join(SCORES_DIR, "score_history.csv")


def save_daily_scores(site_scores):
    os.makedirs(SCORES_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for site_name, s in site_scores.items():
        rows.append({
            "date": today, "site": site_name, "score": s["score"],
            "open": int(s["open"]), "close": int(s["close"]),
            "jodi": int(s["jodi"]), "opanna": int(s["opanna"]),
            "cpanna": int(s["cpanna"]),
        })
    df = pd.DataFrame(rows)
    if os.path.exists(SCORES_FILE):
        old = pd.read_csv(SCORES_FILE)
        old = old[old["date"] != today]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(SCORES_FILE, index=False)
    return df


def load_score_history():
    if os.path.exists(SCORES_FILE):
        return pd.read_csv(SCORES_FILE)
    return pd.DataFrame()


def get_daily_summary():
    df = load_score_history()
    if df.empty:
        return pd.DataFrame()
    daily = df.groupby("date").agg({"score": "sum", "site": "count"}).reset_index()
    daily.columns = ["date", "total_hits", "sites"]
    daily["max_possible"] = daily["sites"] * 5
    daily["hit_rate"] = (daily["total_hits"] / daily["max_possible"] * 100).round(1)
    return daily.sort_values("date")


def get_site_history(site_name):
    df = load_score_history()
    if df.empty:
        return pd.DataFrame()
    return df[df["site"] == site_name].sort_values("date")


def get_rolling_accuracy(days=7):
    """Har site ka recent N-day accuracy"""
    df = load_score_history()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return pd.DataFrame()
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    recent = df[df["date"] > cutoff]
    if recent.empty:
        return pd.DataFrame()
    agg = recent.groupby("site").agg({
        "score": ["sum", "count"],
        "open": "sum", "close": "sum", "jodi": "sum",
        "opanna": "sum", "cpanna": "sum",
    }).reset_index()
    agg.columns = ["Site", "Total_Hits", "Days", "Open", "Close", "Jodi", "OPanna", "CPanna"]
    agg["Max_Possible"] = agg["Days"] * 5
    agg["Accuracy_%"] = (agg["Total_Hits"] / agg["Max_Possible"] * 100).round(1)
    agg = agg.sort_values("Accuracy_%", ascending=False).reset_index(drop=True)
    return agg
