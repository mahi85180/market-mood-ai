# pattern_hunt.py
import numpy as np
from collections import Counter, defaultdict
from config import SITES
from panna_predictor import build_all_sequences

def test_markov(seq, order, n=500):
    if len(seq) < order + n + 20: return None
    hits, total = 0, 0
    for i in range(len(seq)-n, len(seq)-1):
        past, actual = seq[:i], seq[i]
        key = tuple(past[-order:])
        trans = Counter()
        for j in range(len(past)-order):
            if tuple(past[j:j+order]) == key:
                trans[past[j+order]] += 1
        if not trans: continue
        total += 1
        if trans.most_common(1)[0][0] == actual: hits += 1
    return round(hits/total*100, 2) if total else None

def test_triple(seq, n=500):
    """Test most-common triple sequences."""
    if len(seq) < 500: return None
    hits, total = 0, 0
    for i in range(len(seq)-n, len(seq)-1):
        past = seq[:i]
        key = tuple(past[-2:])
        nxt = Counter()
        for j in range(len(past)-2):
            if tuple(past[j:j+2]) == key:
                nxt[past[j+2]] += 1
        if not nxt: continue
        total += 1
        if nxt.most_common(1)[0][0] == seq[i]: hits += 1
    return round(hits/total*100, 2) if total else None

def test_open_eq_prevclose(data, n=500):
    op, cl = data["open_seq"], data["close_seq"]
    if len(op) < n+20: return None
    hits, tot = 0, 0
    for i in range(len(op)-n, len(op)-1):
        tot += 1
        if cl[i] == op[i+1]: hits += 1
    return round(hits/tot*100, 2) if tot else None

def test_weekly(seq, n=500):
    if len(seq) < n+50: return None
    hits, tot = 0, 0
    for i in range(len(seq)-n, len(seq)-1):
        past = seq[:i]
        cands = [past[-7*k] for k in [1,2,3] if len(past) >= 7*k]
        if not cands: continue
        tot += 1
        if Counter(cands).most_common(1)[0][0] == seq[i]: hits += 1
    return round(hits/tot*100, 2) if tot else None

def test_gap(seq, n=500):
    if len(seq) < n+50: return None
    hits, tot = 0, 0
    for i in range(len(seq)-n, len(seq)-1):
        past = seq[:i]
        ls = {v:k for k,v in enumerate(past)}
        gaps = {d: len(past)-ls.get(d, 0) for d in range(10)}
        tot += 1
        if max(gaps.items(), key=lambda x:x[1])[0] == seq[i]: hits += 1
    return round(hits/tot*100, 2) if tot else None

print("=" * 70)
print("PATTERN HUNT — Full history walk-forward")
print("=" * 70)

results = defaultdict(list)
for site in SITES:
    data = build_all_sequences(site)
    if not data or data["total_days"] < 300: continue
    seq = data["open_seq"]
    print(f"Testing {site}...", end=" ", flush=True)
    m1 = test_markov(seq, 1)
    m2 = test_markov(seq, 2)
    m3 = test_markov(seq, 3)
    m4 = test_markov(seq, 4)
    t3 = test_triple(seq)
    wk = test_weekly(seq)
    gp = test_gap(seq)
    oc = test_open_eq_prevclose(data)
    for k, v in [("markov1",m1),("markov2",m2),("markov3",m3),("markov4",m4),
                 ("triple",t3),("weekly",wk),("gap",gp),("open=prevclose",oc)]:
        if v is not None: results[k].append(v)
    print("done")

print()
print("=" * 70)
print(f"{'Test':<20} {'Avg %':<10} {'Min':<8} {'Max':<8} {'Sites'}")
print("=" * 70)
for k, vals in sorted(results.items(), key=lambda x: -np.mean(x[1])):
    print(f"{k:<20} {np.mean(vals):>6.2f}%   {min(vals):>5.2f}%   {max(vals):>5.2f}%   {len(vals)}")
print()
print("Random baseline: 10% Top-1")
print("If test > 12%: real edge. If ~10%: noise.")
