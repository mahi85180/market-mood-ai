# daily_predictor.py - Daily prediction generator + tracker
import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime
import os
import re

from scraper import load_history
from config import SITES

PRED_DIR = "daily_predictions"
os.makedirs(PRED_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(PRED_DIR, "win_loss_history.csv")


def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9]", "_", s)


def get_sequence(df):
    """Har row ka last number — sequential draw"""
    seq = []
    for nums in df["numbers"]:
        if nums and len(nums) >= 3:
            seq.append(int(nums[-1]))
    return seq


def build_markov(seq):
    """Transition matrix from all history"""
    trans = {}
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        trans.setdefault(a, Counter())[b] += 1
    return trans


def build_ngram(seq, n=2):
    """N-gram pattern - last n numbers ka context"""
    gram = {}
    for i in range(len(seq) - n):
        key = tuple(seq[i:i+n])
        gram.setdefault(key, Counter())[seq[i+n]] += 1
    return gram


def predict_numbers(site, top_n=5):
    """Combined prediction using Markov + N-gram + frequency"""
    df = load_history(site)
    if df.empty or len(df) < 50:
        return None

    seq = get_sequence(df)
    if len(seq) < 30:
        return None

    last = seq[-1]
    last2 = tuple(seq[-2:]) if len(seq) >= 2 else None

    # 1. Markov (last 1 number)
    trans = build_markov(seq)
    markov_scores = Counter()
    if last in trans:
        total = sum(trans[last].values())
        for num, cnt in trans[last].items():
            markov_scores[num] = cnt / total

    # 2. N-gram (last 2 numbers)
    ngram = build_ngram(seq, 2)
    ngram_scores = Counter()
    if last2 and last2 in ngram:
        total = sum(ngram[last2].values())
        for num, cnt in ngram[last2].items():
            ngram_scores[num] = cnt / total

    # 3. Recent frequency (last 20)
    recent = seq[-20:]
    freq = Counter(recent)
    freq_scores = Counter()
    total_recent = len(recent)
    for num in range(10):
        freq_scores[num] = freq.get(num, 0) / total_recent

    # 4. Cold number bonus (numbers not seen recently)
    last_seen = {}
    for i, num in enumerate(seq):
        last_seen[num] = i
    gap_scores = Counter()
    max_gap = 0
    gaps = {}
    for num in range(10):
        gap = len(seq) - last_seen.get(num, 0)
        gaps[num] = gap
        max_gap = max(max_gap, gap)
    for num in range(10):
        gap_scores[num] = gaps[num] / max_gap if max_gap > 0 else 0

    # Combined score (weighted)
    final_scores = Counter()
    for num in range(10):
        final_scores[num] = (
            markov_scores.get(num, 0) * 0.50 +
            ngram_scores.get(num, 0) * 0.25 +
            freq_scores.get(num, 0) * 0.15 +
            gap_scores.get(num, 0) * 0.10
        )

    # Top N
    top = final_scores.most_common(top_n)

    # Confidence normalization
    total_score = sum(s for _, s in top) if top else 1
    result = []
    for num, score in top:
        pct = round((score / total_score) * 100, 1) if total_score > 0 else 0
        result.append({"number": num, "score": round(score, 4), "confidence": pct})

    return {
        "site": site,
        "last_number": last,
        "top_predictions": result,
        "total_history": len(seq),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_all_predictions(top_n=5):
    """Saare sites ka prediction generate karo"""
    all_preds = {}
    for site in SITES:
        try:
            pred = predict_numbers(site, top_n)
            if pred:
                all_preds[site] = pred
        except Exception as e:
            print(f"⚠️ {site}: {e}")
    return all_preds


def save_predictions_to_csv(preds):
    """Daily prediction file me save"""
    today = datetime.now().strftime("%Y-%m-%d")
    file = os.path.join(PRED_DIR, f"predictions_{today}.csv")

    rows = []
    for site, pred in preds.items():
        for i, p in enumerate(pred["top_predictions"]):
            rows.append({
                "date": today,
                "site": site,
                "rank": i + 1,
                "predicted_number": p["number"],
                "confidence": p["confidence"],
                "last_seen_number": pred["last_number"],
                "generated_at": pred["generated_at"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(file, index=False)
    return file, df


# ============ WIN/LOSS TRACKING ============

def load_history_file():
    if os.path.exists(HISTORY_FILE):
        return pd.read_csv(HISTORY_FILE)
    return pd.DataFrame(columns=[
        "date", "site", "predicted_top1", "predicted_top3",
        "actual_number", "top1_hit", "top3_hit", "verified_at"
    ])


def save_history_file(df):
    df.to_csv(HISTORY_FILE, index=False)


def log_daily_prediction(site, top1, top3_list):
    """Aaj ki prediction log karo"""
    hist = load_history_file()
    today = datetime.now().strftime("%Y-%m-%d")
    row = {
        "date": today,
        "site": site,
        "predicted_top1": top1,
        "predicted_top3": ",".join(map(str, top3_list)),
        "actual_number": -1,
        "top1_hit": -1,
        "top3_hit": -1,
        "verified_at": "",
    }
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    save_history_file(hist)
    return len(hist)


def verify_predictions():
    """Jab actual result aaye, verify karo"""
    hist = load_history_file()
    if hist.empty:
        return 0

    # FIXED: converted dtypes so string values can be written
    hist["verified_at"] = hist["verified_at"].astype(object)
    hist["actual_number"] = hist["actual_number"].astype(int)
    hist["top1_hit"] = hist["top1_hit"].astype(int)
    hist["top3_hit"] = hist["top3_hit"].astype(int)

    verified_count = 0
    for idx, row in hist.iterrows():
        if int(row["top1_hit"]) != -1:
            continue

        site = row["site"]
        date = row["date"]

        df = load_history(site)
        if df.empty:
            continue

        seq = get_sequence(df)
        if not seq:
            continue

        # Latest number = actual
        actual = seq[-1]

        top1 = int(row["predicted_top1"])
        top3 = [int(x) for x in str(row["predicted_top3"]).split(",")]

        hist.at[idx, "actual_number"] = int(actual)
        hist.at[idx, "top1_hit"] = int(top1 == actual)
        hist.at[idx, "top3_hit"] = int(actual in top3)
        hist.at[idx, "verified_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        verified_count += 1

    save_history_file(hist)
    return verified_count


def verify_single_site(site_name):
    """Ek specific site ka pending prediction verify karo"""
    hist = load_history_file()
    if hist.empty:
        return 0, "No history file"

    # Fix dtypes
    hist["verified_at"] = hist["verified_at"].astype(object)
    hist["actual_number"] = hist["actual_number"].astype(int)
    hist["top1_hit"] = hist["top1_hit"].astype(int)
    hist["top3_hit"] = hist["top3_hit"].astype(int)

    # Sirf is site ke pending predictions
    site_pending = hist[(hist["site"] == site_name) & (hist["top1_hit"] == -1)]

    if site_pending.empty:
        return 0, f"{site_name}: koi pending prediction nahi hai"

    # Fresh data fetch
    df = load_history(site_name)
    if df.empty:
        return 0, f"{site_name}: data nahi mila"

    seq = get_sequence(df)
    if not seq:
        return 0, f"{site_name}: koi sequence nahi"

    actual = seq[-1]
    verified_count = 0

    current_len = len(seq)
    for idx, row in site_pending.iterrows():
        # Sirf tab verify karo jab naya data aaya ho
        if "last_seq_length" in hist.columns:
            try:
                prev_len = int(row["last_seq_length"])
                if current_len <= prev_len:
                    continue  # Naya data nahi aaya
            except (ValueError, TypeError):
                pass

        top1 = int(row["predicted_top1"])
        top3 = [int(x) for x in str(row["predicted_top3"]).split(",")]

        hist.at[idx, "actual_number"] = int(actual)
        hist.at[idx, "top1_hit"] = int(top1 == actual)
        hist.at[idx, "top3_hit"] = int(actual in top3)
        hist.at[idx, "verified_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        verified_count += 1

    save_history_file(hist)

    if verified_count == 0:
        return 0, f"{site_name}: naya result nahi aaya (abhi bhi {actual})"
    else:
        return verified_count, f"{site_name}: {verified_count} prediction verified! (actual={actual})"


def get_pending_predictions():
    """Saari pending predictions return karo"""
    hist = load_history_file()
    if hist.empty:
        return pd.DataFrame()
    pending = hist[hist["top1_hit"] == -1].copy()
    return pending


def show_performance(site=None):
    """Performance summary dikhao"""
    hist = load_history_file()
    if hist.empty:
        print("⚠️ Abhi koi prediction history nahi hai.")
        return

    verified = hist[hist["top1_hit"] != -1].copy()
    if verified.empty:
        print("⚠️ Abhi koi verified prediction nahi hai.")
        return

    if site:
        verified = verified[verified["site"] == site]

    if verified.empty:
        print(f"⚠️ {site} ke liye koi data nahi.")
        return

    total = len(verified)
    top1_wins = int(verified["top1_hit"].sum())
    top3_wins = int(verified["top3_hit"].sum())

    print("=" * 60)
    print("🎯 AI PERFORMANCE REPORT")
    print("=" * 60)
    if site:
        print(f"Site: {site}")
    print(f"Total predictions: {total}")
    print(f"Top-1 wins: {top1_wins}  ({round(top1_wins/total*100, 1)}%)")
    print(f"Top-3 wins: {top3_wins}  ({round(top3_wins/total*100, 1)}%)")
    print(f"Random baseline: Top-1 10%, Top-3 30%")
    print(f"Edge: Top-1 {round(top1_wins/total*100 - 10, 1)}%, "
          f"Top-3 {round(top3_wins/total*100 - 30, 1)}%")
    print("=" * 60)

    print("\n📊 Last 10 predictions:")
    print(f"{'Date':<12} {'Site':<16} {'Pred':<8} {'Actual':<8} {'Result'}")
    print("-" * 60)
    for _, r in verified.tail(10).iterrows():
        if r["top1_hit"] == 1:
            res = "✅ Top-1"
        elif r["top3_hit"] == 1:
            res = "🟡 Top-3"
        else:
            res = "❌ Miss"
        print(f"{r['date']:<12} {r['site']:<16} {int(r['predicted_top1']):<8} "
              f"{int(r['actual_number']):<8} {res}")


# ============ MAIN ============

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "predict":
            print("🎯 Generating predictions...\n")
            preds = generate_all_predictions(top_n=5)
            file, df = save_predictions_to_csv(preds)
            print(f"✅ Saved to {file}\n")

            for site, pred in preds.items():
                print(f"📊 {site}")
                print(f"   Last number: {pred['last_number']}")
                print(f"   Top 5 predictions:")
                for p in pred["top_predictions"]:
                    print(f"     {p['number']}  (confidence: {p['confidence']}%)")
                print()

        elif cmd == "verify":
            n = verify_predictions()
            print(f"✅ {n} predictions verified")

        elif cmd == "report":
            site = sys.argv[2] if len(sys.argv) > 2 else None
            show_performance(site)

        elif cmd == "log":
            # Manual log: python daily_predictor.py log "Kalyan" 5 "5,3,7"
            if len(sys.argv) >= 5:
                site = sys.argv[2]
                top1 = int(sys.argv[3])
                top3 = [int(x) for x in sys.argv[4].split(",")]
                n = log_daily_prediction(site, top1, top3)
                print(f"✅ Logged. Total entries: {n}")

    else:
        print("Usage:")
        print("  python daily_predictor.py predict     → Generate today's predictions")
        print("  python daily_predictor.py verify      → Verify old predictions")
        print("  python daily_predictor.py report      → Show performance")
        print("  python daily_predictor.py report <site>")
        print("  python daily_predictor.py log <site> <top1> <top3>")
