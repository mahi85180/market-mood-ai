# bias_hunter.py — Test for design biases in Satta Matka data
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import chisquare
from config import SITES
from panna_predictor import build_all_sequences

print("=" * 78)
print("🔬 BIAS HUNTER — Kya site ka design balanced hai?")
print("=" * 78)
print()

# ---- COLLECT ALL DIGITS ----
all_open, all_close = [], []
site_seqs = {}
for site in SITES:
    data = build_all_sequences(site)
    if not data or data["total_days"] < 300:
        continue
    site_seqs[site] = data
    all_open.extend(data["open_seq"])
    all_close.extend(data["close_seq"])

print(f"Total sites: {len(site_seqs)}")
print(f"Total Open digits: {len(all_open):,}")
print(f"Total Close digits: {len(all_close):,}")
print()

# ============================================================
# TEST 1: OVERALL DISTRIBUTION (Chi-square uniformity)
# ============================================================
print("=" * 78)
print("TEST 1: Kya har digit 10% pe hai? (Chi-square test)")
print("=" * 78)

freq = Counter(all_open)
n = len(all_open)
print(f"\n{'Digit':<8} {'Count':<10} {'%':<10} {'Expected':<10} {'Diff'}")
print("-" * 50)
for d in range(10):
    pct = freq.get(d, 0) / n * 100
    exp = n / 10
    diff = freq.get(d, 0) - exp
    marker = "⚠️" if abs(pct - 10) > 0.5 else ""
    print(f"{d:<8} {freq.get(d, 0):<10} {pct:>7.2f}%   {exp:>8.0f}   {diff:>+7.0f} {marker}")

obs = [freq.get(d, 0) for d in range(10)]
exp_arr = [n / 10] * 10
chi2, p = chisquare(obs, exp_arr)
print(f"\nChi-square: {chi2:.2f}")
print(f"P-value:    {p:.6f}")
if p > 0.05:
    print("✅ P > 0.05 → Distribution UNIFORM (no bias, pure random)")
else:
    print(f"⚠️ P < 0.05 → NOT uniform! Some digits favoured")
    top = sorted([(d, freq.get(d, 0)/n*100) for d in range(10)], key=lambda x: -x[1])
    print(f"   Top digit: {top[0][0]} at {top[0][1]:.2f}%")
    print(f"   Bottom:    {top[-1][0]} at {top[-1][1]:.2f}%")

# ============================================================
# TEST 2: REPEAT RATE (consecutive same digit)
# ============================================================
print()
print("=" * 78)
print("TEST 2: Same digit lagatar aata hai? (Expected random: 10%)")
print("=" * 78)
print()
print(f"{'Site':<20} {'Repeats':<10} {'Total':<10} {'Repeat %':<10}")
print("-" * 55)

total_reps, total_pairs = 0, 0
for site, data in site_seqs.items():
    seq = data["open_seq"]
    reps = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i-1])
    pairs = len(seq) - 1
    total_reps += reps
    total_pairs += pairs
    pct = reps / pairs * 100 if pairs else 0
    marker = "⚠️" if abs(pct - 10) > 1.5 else ""
    print(f"{site:<20} {reps:<10} {pairs:<10} {pct:>7.2f}% {marker}")

overall = total_reps / total_pairs * 100
print(f"\n{'OVERALL':<20} {total_reps:<10} {total_pairs:<10} {overall:>7.2f}%")
print(f"Expected random: 10.00%")
if abs(overall - 10) < 0.5:
    print("✅ Repeat rate normal → no anti-repeat bias")
elif overall < 10:
    print(f"⚠️ {overall:.2f}% < 10% → Anti-repeat bias (site avoids same digit twice)")
elif overall > 10:
    print(f"⚠️ {overall:.2f}% > 10% → Pro-repeat bias (site favours repeat)")

# ============================================================
# TEST 3: CONSECUTIVE PAIRS (5→6, 7→8)
# ============================================================
print()
print("=" * 78)
print("TEST 3: Consecutive pairs zyada aate hain? (5→6, 6→7, etc.)")
print("=" * 78)

# Count actual transitions
consec_actual, consec_expected, total_trans = 0, 0, 0
for site, data in site_seqs.items():
    seq = data["open_seq"]
    for i in range(1, len(seq)):
        total_trans += 1
        if abs(seq[i] - seq[i-1]) == 1:  # consecutive
            consec_actual += 1
    consec_expected += (len(seq) - 1) * 18 / 100  # 18/100 pairs are consecutive

print(f"\nConsecutive transitions: {consec_actual:,} / {total_trans:,}")
print(f"Rate: {consec_actual/total_trans*100:.2f}%")
print(f"Random expected: 18% (e.g., 5→4 or 5→6, 2 possibilities per digit)")
if abs(consec_actual/total_trans*100 - 18) < 1:
    print("✅ Normal — no consecutive bias")
else:
    print(f"⚠️ Bias detected!")

# ============================================================
# TEST 4: CORRECTION / MEAN REVERSION
# ============================================================
print()
print("=" * 78)
print("TEST 4: 'Due' number theory — agar 7 kam aaya, to zyada aayega?")
print("=" * 78)
print("Logic: Last 100 me jo digit kam aaya, next 50 me zyada aaya?")

underpred_correct, total_windows = 0, 0
for site, data in site_seqs.items():
    seq = data["open_seq"]
    for i in range(100, len(seq) - 50):
        past = seq[i-100:i]
        future = seq[i:i+50]
        past_freq = Counter(past)
        # Find least-frequent digit in past 100
        least_digit = min(range(10), key=lambda d: past_freq.get(d, 0))
        # How many times in next 50?
        next_count = future.count(least_digit)
        expected = 50 / 10  # 5 times
        if next_count > expected:
            underpred_correct += 1
        total_windows += 1

if total_windows:
    pct = underpred_correct / total_windows * 100
    print(f"\nWindows tested: {total_windows:,}")
    print(f"Least-frequent digit came MORE than expected: {pct:.2f}%")
    print(f"If random: ~50%")
    if pct > 55:
        print(f"✅ {pct:.1f}% > 55% → MEAN REVERSION detected! Site balances digits")
    elif pct < 45:
        print(f"⚠️ {pct:.1f}% < 45% → Momentum (recent trend continues)")
    else:
        print(f"❌ {pct:.1f}% ≈ 50% → No correction mechanism (pure random)")

# ============================================================
# VERDICT
# ============================================================
print()
print("=" * 78)
print("🎯 FINAL VERDICT")
print("=" * 78)
print(f"""
Distribution uniformity: {('UNIFORM' if p > 0.05 else 'BIASED')}
Repeat rate: {overall:.2f}% (random: 10%)
Consecutive rate: {consec_actual/total_trans*100:.2f}% (random: 18%)
Mean reversion: {underpred_correct/total_windows*100:.2f}% (random: 50%)

Agar sab ~random hai → RNG based, no exploitable pattern
Agar koi bias >2% hai → usko exploit karke improve kar sakte hain
""")
