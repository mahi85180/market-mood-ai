# nested_test.py — Proper nested validation
import numpy as np
from collections import Counter
from config import SITES
from panna_predictor import build_all_sequences

def find_best_lags(seq, max_lag=15, top_k=4):
    """Find best lags using only the given seq (train only)."""
    if len(seq) < 300:
        return None
    # Compute autocorrelation for each lag
    acs = {}
    for lag in range(1, max_lag+1):
        if len(seq) < lag + 100: continue
        a = np.array(seq[:-lag])
        b = np.array(seq[lag:])
        if len(a) > 50:
            acs[lag] = np.corrcoef(a, b)[0, 1]
    # Pick top positive and top negative
    sorted_lags = sorted(acs.items(), key=lambda x: x[1])
    negatives = [(l, c) for l, c in sorted_lags if c < -0.02][:2]
    positives = [(l, c) for l, c in sorted_lags if c > 0.02][-2:]
    return {"negatives": [l for l, _ in negatives],
            "positives": [l for l, _ in positives]}


def make_predictor(lags):
    def fn(past):
        if len(past) < 15: return None
        scores = {d: 0.5 for d in range(10)}
        for l in lags["positives"]:
            if len(past) >= l:
                scores[past[-l]] += 0.8
        for l in lags["negatives"]:
            if len(past) >= l:
                scores[past[-l]] -= 0.5
        return max(scores.items(), key=lambda x: x[1])[0]
    return fn


def test_range(seq, fn, start, end):
    h, t = 0, 0
    for i in range(start, end):
        past = seq[:i]
        if len(past) < 100: continue
        pred = fn(past)
        if pred is None: continue
        t += 1
        if pred == seq[i]: h += 1
    return (round(h/t*100, 2) if t else None), t


print("=" * 78)
print("🔬 NESTED VALIDATION — Select on 30%, test on 70%")
print("=" * 78)
print()

in_vals, oos_vals = [], []
for site in SITES:
    data = build_all_sequences(site)
    if not data or data["total_days"] < 500: continue
    seq = data["open_seq"]
    
    # Select lags on FIRST 30% only
    train_end = int(len(seq) * 0.3)
    train_seq = seq[:train_end]
    lags = find_best_lags(train_seq)
    if not lags or (not lags["positives"] and not lags["negatives"]):
        print(f"{site:<20} SKIP (no lags found)")
        continue
    
    # Test on LAST 70% (never seen by lag selector)
    fn = make_predictor(lags)
    pct, n = test_range(seq, fn, train_end, len(seq)-1)
    if pct is None:
        continue
    
    oos_vals.append(pct)
    # Also compute in-sample on same range but with these lags (should be ~same since lags from train only)
    print(f"{site:<20} pos_lags={lags['positives']} neg_lags={lags['negatives']}  OOS={pct:.2f}%  n={n}")

print()
print("=" * 78)
print(f"OOS AVERAGE: {np.mean(oos_vals):.2f}%  (std {np.std(oos_vals):.2f})")
print(f"Random: 10.00%")
print(f"Edge: {np.mean(oos_vals)-10:+.2f}%")

se = np.std(oos_vals) / np.sqrt(len(oos_vals))
t = (np.mean(oos_vals) - 10) / se if se > 0 else 0
print(f"t-stat: {t:.2f}")
if t > 2.0:
    print("✅ REAL EDGE (t > 2.0) — selection bias ruled out")
elif t > 1.5:
    print("⚠️ WEAK SIGNAL (1.5 < t < 2.0)")
else:
    print("❌ NOISE (t < 1.5) — previous result was selection bias")
