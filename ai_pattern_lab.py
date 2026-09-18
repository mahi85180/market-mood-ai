# ai_pattern_lab.py — Let AI test every pattern type
import numpy as np
from collections import Counter, defaultdict
from config import SITES
from panna_predictor import build_all_sequences

TEST_SIZE = 500  # last 500 predictions per site
MIN_HIST = 100   # need this much history before predicting

def seq_hits(seq, predictor_fn, n=TEST_SIZE):
    """Generic walk-forward tester. predictor_fn(past) -> top-1 digit or None."""
    if len(seq) < MIN_HIST + n + 20: return None, 0
    hits, total = 0, 0
    for i in range(len(seq)-n, len(seq)-1):
        past = seq[:i]
        if len(past) < MIN_HIST: continue
        pred = predictor_fn(past)
        if pred is None: continue
        total += 1
        if pred == seq[i]: hits += 1
    return (round(hits/total*100, 2) if total else None), total


# ============================================================
# PATTERN PREDICTORS
# ============================================================

def p_markov(order):
    def fn(past):
        key = tuple(past[-order:])
        trans = Counter()
        for j in range(len(past)-order):
            if tuple(past[j:j+order]) == key:
                trans[past[j+order]] += 1
        return trans.most_common(1)[0][0] if trans else None
    return fn

def p_freq(window):
    def fn(past):
        recent = past[-window:]
        f = Counter(recent)
        return max(f.items(), key=lambda x: x[1])[0]
    return fn

def p_recency(decay):
    def fn(past):
        scores = Counter()
        for i, n in enumerate(past):
            scores[n] += decay ** (len(past)-i-1)
        return max(scores.items(), key=lambda x: x[1])[0]
    return fn

def p_gap(past):
    last_seen = {v:k for k,v in enumerate(past)}
    gaps = {d: len(past)-last_seen.get(d, 0) for d in range(10)}
    return max(gaps.items(), key=lambda x: x[1])[0]

def p_hot(window):
    def fn(past):
        recent = past[-window:]
        older = past[-2*window:-window]
        rc, oc = Counter(recent), Counter(older)
        diff = {d: rc.get(d,0)/window - oc.get(d,0)/window for d in range(10)}
        return max(diff.items(), key=lambda x: x[1])[0]
    return fn

def p_cold(window):
    def fn(past):
        recent = past[-window:]
        f = Counter(recent)
        scores = {d: -f.get(d, 0) for d in range(10)}
        return max(scores.items(), key=lambda x: x[1])[0]
    return fn

def p_weekly(past):
    cands = [past[-7*k] for k in [1,2,3,4] if len(past) >= 7*k]
    return Counter(cands).most_common(1)[0][0] if cands else None

def p_modulo(n):
    """Predict based on pattern at position mod n."""
    def fn(past):
        L = len(past)
        group = [past[i] for i in range(L) if i % n == L % n]
        return Counter(group).most_common(1)[0][0] if group else None
    return fn

def p_ngram_top(n):
    """Top-N most frequent n-grams, predict next."""
    def fn(past):
        if len(past) < n+3: return None
        key = tuple(past[-n:])
        nxt = Counter()
        for j in range(len(past)-n):
            if tuple(past[j:j+n]) == key:
                nxt[past[j+n]] += 1
        return nxt.most_common(1)[0][0] if nxt else None
    return fn

def p_mirror(past):
    """Predict the number 2 positions back (mirror)."""
    return past[-3] if len(past) >= 3 else None

def p_next_repeat(past):
    """Predict last number repeat (streak continuation)."""
    return past[-1]

def p_anti_repeat(past):
    """Predict last number won't repeat — return previous different."""
    return past[-2] if len(past) >= 2 and past[-1] == past[-2] else None

def p_sum_last(past):
    """Sum of last 3 mod 10."""
    if len(past) < 3: return None
    return sum(past[-3:]) % 10

def p_diff_last(past):
    """Diff between last 2."""
    if len(past) < 2: return None
    return abs(past[-1] - past[-2]) % 10


# ============================================================
# RUN TESTS
# ============================================================

print("=" * 80)
print("🤖 AI PATTERN LAB — Testing 20+ pattern types")
print("=" * 80)

# Collect data
site_seqs = {}
for site in SITES:
    data = build_all_sequences(site)
    if data and data["total_days"] >= 300:
        site_seqs[site] = data["open_seq"]

print(f"Loaded {len(site_seqs)} sites\n")

PATTERNS = {
    "markov1": p_markov(1),
    "markov2": p_markov(2),
    "markov3": p_markov(3),
    "markov4": p_markov(4),
    "freq5": p_freq(5),
    "freq10": p_freq(10),
    "freq30": p_freq(30),
    "freq100": p_freq(100),
    "recency0.90": p_recency(0.90),
    "recency0.97": p_recency(0.97),
    "recency0.99": p_recency(0.99),
    "gap": p_gap,
    "hot5": p_hot(5),
    "hot10": p_hot(10),
    "hot20": p_hot(20),
    "cold10": p_cold(10),
    "cold30": p_cold(30),
    "weekly7": p_weekly,
    "mod6": p_modulo(6),
    "mod7": p_modulo(7),
    "mod30": p_modulo(30),
    "ngram2": p_ngram_top(2),
    "ngram3": p_ngram_top(3),
    "mirror3": p_mirror,
    "streak": p_next_repeat,
    "sum3": p_sum_last,
    "diff2": p_diff_last,
}

results = defaultdict(list)
for name, fn in PATTERNS.items():
    for site, seq in site_seqs.items():
        pct, samples = seq_hits(seq, fn)
        if pct is not None and samples >= 200:
            results[name].append(pct)

print("=" * 80)
print(f"{'Pattern':<16} {'Avg %':<10} {'Min %':<10} {'Max %':<10} {'Std':<8} {'Sites'}")
print("=" * 80)

# Sort by average descending
ranked = sorted(results.items(), key=lambda x: -np.mean(x[1]))
for name, vals in ranked:
    avg = np.mean(vals)
    std = np.std(vals)
    edge = avg - 10.0
    marker = ""
    if edge >= 2.0 and std < 3.0:
        marker = " 🔥 REAL EDGE?"
    elif edge >= 1.5:
        marker = " ⚠️ Weak"
    else:
        marker = " ·"
    print(f"{name:<16} {avg:>6.2f}%    {min(vals):>6.2f}%    {max(vals):>6.2f}%    "
          f"{std:>5.2f}    {len(vals)}{marker}")

print()
print("=" * 80)
print("📊 VERDICT")
print("=" * 80)
print("""
Random baseline: 10%

Interpretation:
  - Avg 10.0-11.5%: NOISE (random)
  - Avg 11.5-13.0%: Weak signal (needs confirmation)
  - Avg 13%+ with std < 3: REAL EDGE (only then worth using)
  - Avg 13%+ with std > 5: NOISE (few sites lucky)
  - Any pattern with max-min > 5%: unstable, don't trust

Look for: high average, LOW stddev (consistent across sites)
""")
