# clean_edge_test.py - Panna ignore, only cross-row signal
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
from scraper import load_history

print("=" * 70)
print("🎯 CLEAN EDGE TEST — Only Cross-Row Signal")
print("=" * 70)

for site in ["Kalyan", "Milan Day", "Main Bazar", "Rajdhani Night"]:
    df = load_history(site)
    if df.empty: continue
    print(f"\n{'='*70}")
    print(f"📊 {site}")
    print('='*70)

    # Har row ka SIRF ek number - last one
    # Yeh panna structure ignore karta hai
    seq = []
    for nums in df["numbers"]:
        if nums and len(nums) >= 3:
            seq.append(int(nums[-1]))

    if len(seq) < 100:
        print(f"  Skip — sirf {len(seq)} entries")
        continue

    print(f"Sequential length: {len(seq)}")

    # Test 1: Sequential chi-square
    print("\n[Test 1] Sequential transitions:")
    trans = {}
    for i in range(len(seq) - 1):
        trans.setdefault(seq[i], Counter())[seq[i+1]] += 1

    total_chi = 0
    sig = 0
    for num in range(10):
        if num not in trans: continue
        row = trans[num]
        n = sum(row.values())
        if n < 10: continue
        obs = [row.get(j, 0) for j in range(10)]
        exp = [n/10]*10
        chi2, p = stats.chisquare(obs, exp)
        total_chi += chi2
        if p < 0.05: sig += 1

    print(f"  Total chi2: {total_chi:.2f}")
    print(f"  Significant: {sig}/10")
    print(f"  Random baseline chi2: ~90 (for 10 categories)")
    if total_chi > 200:
        print(f"  ⚠️ STRONG PATTERN — real edge!")
    elif total_chi > 120:
        print(f"  🟡 WEAK PATTERN — some edge")
    else:
        print(f"  ✅ Random-ish")

    # Test 2: Runs test on sequential
    median = np.median(seq)
    binary = [1 if x > median else 0 for x in seq]
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]: runs += 1
    n1 = sum(binary); n2 = len(binary) - n1
    exp_runs = (2*n1*n2)/(n1+n2) + 1
    var = (2*n1*n2*(2*n1*n2-n1-n2))/((n1+n2)**2*(n1+n2-1))
    z = (runs - exp_runs) / np.sqrt(var) if var > 0 else 0
    p_runs = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"\n[Test 2] Runs test:")
    print(f"  Observed: {runs}, Expected: {exp_runs:.1f}, z={z:.2f}, p={p_runs:.4f}")
    if p_runs < 0.05:
        print(f"  ⚠️ Non-random sequencing!")
    else:
        print(f"  ✅ Random sequencing")

    # Test 3: Predictive edge (simple Markov)
    # Agar last number = X tha, next me kaunsa number zyada aata hai?
    print(f"\n[Test 3] Top predictive pairs:")
    all_pairs = []
    for i in range(len(seq) - 1):
        all_pairs.append((seq[i], seq[i+1]))

    # Har pair ka count
    pair_counts = Counter(all_pairs)
    total_pairs = len(all_pairs)

    # Top 5 pairs jo random se zyada baar aate hain
    # Random expectation: total_pairs/100 per pair
    expected_per_pair = total_pairs / 100
    top_pairs = pair_counts.most_common(5)

    print(f"  Expected per pair (random): {expected_per_pair:.1f}")
    print(f"  Top pairs observed:")
    for (a, b), count in top_pairs:
        lift = count / expected_per_pair
        print(f"    {a} → {b}: {count} times  (lift: {lift:.2f}x)")

print("\n" + "=" * 70)
print("🎯 FINAL SUMMARY")
print("=" * 70)
print("Agar multiple sites me chi2 > 200 aur runs test fail hua")
print("→ Real edge hai, exploit kar sakte hain")
print("Agar sab random ho → within-row artifact tha")
print("=" * 70)
