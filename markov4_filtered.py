# markov4_filtered.py
from collections import Counter
from config import SITES
from panna_predictor import build_all_sequences

def test_markov_filtered(seq, order, min_count, n=500):
    """Only predict when context seen >= min_count times."""
    if len(seq) < order + n + 20: return None, 0
    hits, total = 0, 0
    for i in range(len(seq)-n, len(seq)-1):
        past, actual = seq[:i], seq[i]
        key = tuple(past[-order:])
        trans = Counter()
        for j in range(len(past)-order):
            if tuple(past[j:j+order]) == key:
                trans[past[j+order]] += 1
        # Only predict if context has enough historical examples
        if sum(trans.values()) < min_count:
            continue
        if not trans: continue
        total += 1
        if trans.most_common(1)[0][0] == actual: hits += 1
    return (round(hits/total*100, 2) if total else None), total

print("=" * 75)
print(f"{'Site':<20} {'m4 all':<10} {'m4 >=3':<10} {'m4 >=5':<10} {'Samples(>=5)'}")
print("=" * 75)

agg = {"all": [], "c3": [], "c5": []}
for site in SITES:
    data = build_all_sequences(site)
    if not data or data["total_days"] < 300: continue
    seq = data["open_seq"]
    a, _ = test_markov_filtered(seq, 4, 1)
    b, _ = test_markov_filtered(seq, 4, 3)
    c, sc = test_markov_filtered(seq, 4, 5)
    if a: agg["all"].append(a)
    if b: agg["c3"].append(b)
    if c: agg["c5"].append(c)
    print(f"{site:<20} {str(a):<10} {str(b):<10} {str(c):<10} {sc}")

import numpy as np
print("=" * 75)
print(f"{'AVERAGE':<20} {np.mean(agg['all']):<10.2f} {np.mean(agg['c3']):<10.2f} {np.mean(agg['c5']):<10.2f}")
print(f"{'STDDEV':<20} {np.std(agg['all']):<10.2f} {np.std(agg['c3']):<10.2f} {np.std(agg['c5']):<10.2f}")
print()
print("Random baseline = 10%")
print("If >=5 filtered still ~10% → no real signal, all noise")
print("If >=5 filtered shows 14%+ with LOW stddev → REAL EDGE")
