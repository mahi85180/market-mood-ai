# bias_predictor.py — Test bias-aware predictors walk-forward
import numpy as np
from collections import Counter
from config import SITES
from panna_predictor import build_all_sequences

TEST_SIZE = 500
MIN_HIST = 100


def p_bias_v1(past):
    """Hot10 + slight boost to digit 0."""
    recent = past[-30:]
    f = Counter(recent)
    scores = {}
    for d in range(10):
        s = f.get(d, 0) / len(recent)
        if d == 0:
            s += 0.007  # +0.7% bias for digit 0
        scores[d] = s
    return max(scores.items(), key=lambda x: x[1])[0]


def p_bias_v2(past):
    """Hot10 + repeat bonus when last digit is repeated recently."""
    recent = past[-30:]
    f = Counter(recent)
    last = past[-1]
    # Check if last digit appeared more than once in recent
    recent_tail = past[-5:]
    last_count = recent_tail.count(last)
    scores = {}
    for d in range(10):
        s = f.get(d, 0) / len(recent)
        # If last digit shows repeat tendency, boost it
        if d == last and last_count >= 2:
            s += 0.01
        scores[d] = s
    return max(scores.items(), key=lambda x: x[1])[0]


def p_bias_v3(past):
    """Combined: hot10 + 0-boost + repeat bonus + cold penalty."""
    recent = past[-30:]
    f = Counter(recent)
    last = past[-1]
    recent_tail = past[-5:]
    last_count = recent_tail.count(last)
    scores = {}
    for d in range(10):
        s = f.get(d, 0) / len(recent)
        if d == 0:
            s += 0.007
        if d == 5:
            s -= 0.005
        if d == last and last_count >= 2:
            s += 0.01
        scores[d] = s
    return max(scores.items(), key=lambda x: x[1])[0]


def p_hot10_baseline(past):
    """Just hot10 — for comparison."""
    recent = past[-30:]
    f = Counter(recent)
    return max(f.items(), key=lambda x: x[1])[0]


def seq_hits(seq, fn, n=TEST_SIZE):
    if len(seq) < MIN_HIST + n + 20: return None, 0
    hits, total = 0, 0
    for i in range(len(seq)-n, len(seq)-1):
        past = seq[:i]
        if len(past) < MIN_HIST: continue
        pred = fn(past)
        if pred is None: continue
        total += 1
        if pred == seq[i]: hits += 1
    return (round(hits/total*100, 2) if total else None), total


print("=" * 75)
print("🎯 BIAS-AWARE PREDICTOR — Walk-forward test")
print("=" * 75)
print()

site_seqs = {}
for site in SITES:
    data = build_all_sequences(site)
    if data and data["total_days"] >= 300:
        site_seqs[site] = data["open_seq"]

PREDICTORS = {
    "hot30_baseline":  p_hot10_baseline,
    "bias_v1 (+0)":    p_bias_v1,
    "bias_v2 (repeat)": p_bias_v2,
    "bias_v3 (all)":   p_bias_v3,
}

results = {name: [] for name in PREDICTORS}
print(f"{'Site':<20}", end="")
for name in PREDICTORS:
    print(f" {name[:14]:<14}", end="")
print()
print("-" * 90)

for site, seq in site_seqs.items():
    print(f"{site:<20}", end="")
    for name, fn in PREDICTORS.items():
        pct, _ = seq_hits(seq, fn)
        if pct is not None:
            results[name].append(pct)
            print(f" {pct:>6.2f}%      ", end="")
        else:
            print(f"   --         ", end="")
    print()

print()
print("=" * 75)
print("📊 AVERAGE")
print("=" * 75)
for name, vals in results.items():
    if vals:
        avg = np.mean(vals)
        std = np.std(vals)
        edge = avg - 10.0
        print(f"{name:<20} {avg:>6.2f}%  (std {std:.2f})  edge {edge:+.2f}%")

print()
print("Random baseline: 10.00%")
print("Real edge if avg > 11.5% and std < 2")
