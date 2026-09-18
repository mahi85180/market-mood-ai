# oos_test.py — Out-of-sample validation
# Use oldest 70% to detect pattern, test on newest 30%
import numpy as np
from collections import Counter
from config import SITES
from panna_predictor import build_all_sequences

def p_lag_penalty(past):
    if len(past) < 10: return None
    base = {d: 0.5 for d in range(10)}
    base[past[-2]] -= 0.5
    base[past[-6]] -= 0.3
    base[past[-10]] -= 0.3
    base[past[-8]] += 0.8
    base[past[-5]] += 0.4
    base[past[-3]] += 0.3
    return max(base.items(), key=lambda x: x[1])[0]


def test_split(seq, split_ratio=0.7):
    """Test on last 30% using walk-forward."""
    if len(seq) < 500: return None, None
    split = int(len(seq) * split_ratio)
    # In-sample: seq[:split], Out-of-sample: seq[split:]
    def hits(start, end):
        h, t = 0, 0
        for i in range(start, end):
            past = seq[:i]
            if len(past) < 100: continue
            pred = p_lag_penalty(past)
            if pred is None: continue
            t += 1
            if pred == seq[i]: h += 1
        return round(h/t*100, 2) if t else None, t
    
    in_pct, in_n = hits(int(len(seq)*0.3), split)
    oos_pct, oos_n = hits(split, len(seq)-1)
    return in_pct, oos_pct


print("=" * 78)
print("🔬 OUT-OF-SAMPLE TEST — lag_penalty predictor")
print("=" * 78)
print("Train window: first 70%  |  OOS window: last 30%")
print()

print(f"{'Site':<20} {'In-Sample %':<15} {'OOS %':<15} {'Delta':<10}")
print("-" * 65)

in_vals, oos_vals = [], []
for site in SITES:
    data = build_all_sequences(site)
    if not data or data["total_days"] < 500: continue
    seq = data["open_seq"]
    in_pct, oos_pct = test_split(seq)
    if in_pct is None or oos_pct is None: continue
    in_vals.append(in_pct)
    oos_vals.append(oos_pct)
    delta = oos_pct - in_pct
    marker = "🔥" if oos_pct >= 11.5 else ("⚠️" if oos_pct >= 11 else "·")
    print(f"{site:<20} {in_pct:>6.2f}%        {oos_pct:>6.2f}%        {delta:>+6.2f}%   {marker}")

print()
print("=" * 78)
print(f"{'AVERAGE':<20} {np.mean(in_vals):>6.2f}%        {np.mean(oos_vals):>6.2f}%        "
      f"{np.mean(oos_vals)-np.mean(in_vals):>+6.2f}%")
print(f"{'STDDEV':<20} {np.std(in_vals):>6.2f}         {np.std(oos_vals):>6.2f}")
print(f"{'SITES':<20} {len(in_vals)}")
print()
print(f"OOS vs Random (10%): {np.mean(oos_vals)-10:+.2f}%")
print()
print("VERDICT:")
avg_oos = np.mean(oos_vals)
std_oos = np.std(oos_vals)
se = std_oos / np.sqrt(len(oos_vals))
t_stat = (avg_oos - 10) / se if se > 0 else 0
print(f"  OOS avg:  {avg_oos:.2f}%")
print(f"  Std err:  {se:.2f}")
print(f"  t-stat:   {t_stat:.2f}")
if t_stat > 2.0:
    print(f"  ✅ REAL EDGE (t > 2.0)")
elif t_stat > 1.5:
    print(f"  ⚠️ WEAK SIGNAL (1.5 < t < 2.0)")
else:
    print(f"  ❌ NOISE (t < 1.5)")
