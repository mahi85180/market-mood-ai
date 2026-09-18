# transition_wf.py — Walk-forward transition matrix test
import numpy as np
from collections import Counter
from config import SITES
from panna_predictor import build_all_sequences

TEST_SIZE = 500
MIN_HIST = 200

def predict_transition(past):
    """If last digit is X, what's most common next digit in past?"""
    if len(past) < MIN_HIST: return None
    last = past[-1]
    nxt = Counter()
    for i in range(len(past)-1):
        if past[i] == last:
            nxt[past[i+1]] += 1
    if not nxt:
        return None
    return nxt.most_common(1)[0][0]


def predict_transition_recent(past, window=300):
    """Same but only last 'window' transitions."""
    if len(past) < MIN_HIST: return None
    sub = past[-window:]
    last = sub[-1]
    nxt = Counter()
    for i in range(len(sub)-1):
        if sub[i] == last:
            nxt[sub[i+1]] += 1
    if not nxt:
        return None
    return nxt.most_common(1)[0][0]


def test_seq(seq, fn, n=TEST_SIZE):
    h, t = 0, 0
    for i in range(len(seq)-n, len(seq)-1):
        past = seq[:i]
        if len(past) < MIN_HIST: continue
        pred = fn(past)
        if pred is None: continue
        t += 1
        if pred == seq[i]: h += 1
    return round(h/t*100, 2) if t else None


print("=" * 78)
print("🔬 TRANSITION MATRIX — Walk-Forward Test")
print("=" * 78)
print()

results = {"all": [], "recent300": []}
print(f"{'Site':<20} {'Transition (all hist)':<22} {'Transition (recent 300)'}")
print("-" * 70)

for site in SITES:
    data = build_all_sequences(site)
    if not data or data["total_days"] < 500: continue
    seq = data["open_seq"]
    
    p1 = test_seq(seq, predict_transition)
    p2 = test_seq(seq, predict_transition_recent)
    
    if p1 is not None: results["all"].append(p1)
    if p2 is not None: results["recent300"].append(p2)
    
    print(f"{site:<20} {str(p1)+'%':<22} {str(p2)+'%'}")

print()
print("=" * 78)
print(f"{'AVERAGE (all hist)':<30} {np.mean(results['all']):.2f}%")
print(f"{'AVERAGE (recent 300)':<30} {np.mean(results['recent300']):.2f}%")
print(f"{'Random baseline':<30} 10.00%")
print()

for name in ["all", "recent300"]:
    vals = results[name]
    edge = np.mean(vals) - 10
    t = edge / (np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0
    print(f"{name}: edge {edge:+.2f}%, t-stat {t:.2f}")
