# edge_predictor.py — Test autocorr + sum-constraint predictor
import numpy as np
from collections import Counter
from config import SITES
from panna_predictor import build_all_sequences

TEST_SIZE = 500
MIN_HIST = 100

def p_lag_penalty(past):
    """Avoid digits from lag 2, prefer digits from lag 8."""
    if len(past) < 10: return None
    # Start with frequency scores
    scores = Counter(range(10))
    # Base: uniform
    base = {d: 0.5 for d in range(10)}
    # Negative: lag 2 (avoid)
    lag2 = past[-2]
    base[lag2] -= 0.5
    # Negative: lag 6 (mild)
    lag6 = past[-6]
    base[lag6] -= 0.3
    # Negative: lag 10 (mild)
    lag10 = past[-10]
    base[lag10] -= 0.3
    # Positive: lag 8
    lag8 = past[-8]
    base[lag8] += 0.8
    # Positive: lag 5 (mild from test)
    lag5 = past[-5]
    base[lag5] += 0.4
    # Positive: lag 3
    lag3 = past[-3]
    base[lag3] += 0.3
    return max(base.items(), key=lambda x: x[1])[0]


def p_sum_constraint(past, window=20):
    """If recent sum is high, prefer low digits, and vice versa."""
    if len(past) < window: return None
    recent = past[-window:]
    s = sum(recent)
    target = window * 4.5  # ideal mean
    diff = s - target  # positive = we've had high digits, need low
    # Score each digit: prefer opposite direction of excess
    scores = {}
    for d in range(10):
        # If diff > 0 (excess high), want low digits → d=0 best
        # If diff < 0 (excess low), want high digits → d=9 best
        # Simple: score = (4.5 - d) * sign(diff) + baseline
        scores[d] = -diff * (d - 4.5) / 10
    return max(scores.items(), key=lambda x: x[1])[0]


def p_combined(past):
    """Combine lag penalty + sum constraint + hot10."""
    if len(past) < 20: return None
    scores = {d: 0.0 for d in range(10)}
    # Lag signals
    scores[past[-2]] -= 0.5
    scores[past[-6]] -= 0.3
    scores[past[-10]] -= 0.3
    scores[past[-8]] += 0.8
    scores[past[-5]] += 0.4
    scores[past[-3]] += 0.3
    # Sum constraint
    recent20 = past[-20:]
    diff = sum(recent20) - 20 * 4.5
    for d in range(10):
        scores[d] -= diff * (d - 4.5) / 50
    # Hot30 mild
    recent30 = past[-30:]
    f = Counter(recent30)
    for d in range(10):
        scores[d] += f.get(d, 0) / len(recent30) * 0.5
    return max(scores.items(), key=lambda x: x[1])[0]


def p_autocorr_only(past):
    """Only lag 8 positive + lag 2 negative — the strongest signals."""
    if len(past) < 10: return None
    scores = {d: 0.0 for d in range(10)}
    scores[past[-8]] += 1.0  # strongest positive signal
    scores[past[-2]] -= 1.0  # strongest negative signal
    return max(scores.items(), key=lambda x: x[1])[0]


def seq_hits(seq, fn, n=TEST_SIZE):
    if len(seq) < MIN_HIST + n + 20: return None
    hits, total = 0, 0
    for i in range(len(seq)-n, len(seq)-1):
        past = seq[:i]
        if len(past) < MIN_HIST: continue
        pred = fn(past)
        if pred is None: continue
        total += 1
        if pred == seq[i]: hits += 1
    return round(hits/total*100, 2) if total else None


print("=" * 78)
print("🎯 EDGE PREDICTOR — Testing autocorr + sum-constraint signals")
print("=" * 78)
print()

site_seqs = {}
for site in SITES:
    data = build_all_sequences(site)
    if data and data["total_days"] >= 300:
        site_seqs[site] = data["open_seq"]

PREDICTORS = {
    "autocorr_only": p_autocorr_only,
    "lag_penalty": p_lag_penalty,
    "sum_constraint": p_sum_constraint,
    "combined": p_combined,
}

results = {k: [] for k in PREDICTORS}
print(f"{'Site':<20}", end="")
for name in PREDICTORS:
    print(f" {name[:14]:<14}", end="")
print()
print("-" * 85)

for site, seq in site_seqs.items():
    print(f"{site:<20}", end="")
    for name, fn in PREDICTORS.items():
        pct = seq_hits(seq, fn)
        if pct is not None:
            results[name].append(pct)
            print(f" {pct:>6.2f}%      ", end="")
        else:
            print(f"   --          ", end="")
    print()

print()
print("=" * 78)
print("📊 AVERAGE")
print("=" * 78)
for name, vals in results.items():
    if vals:
        avg = np.mean(vals)
        std = np.std(vals)
        edge = avg - 10.0
        marker = "🔥" if edge >= 1.5 else ("⚠️" if edge >= 0.5 else "·")
        print(f"{name:<20} {avg:>6.2f}%  (std {std:.2f})  edge {edge:+.2f}%  {marker}")

print()
print("Random: 10.00%  |  Real edge: >11.5%")
