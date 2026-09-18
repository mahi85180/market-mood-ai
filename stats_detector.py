# stats_detector.py - Real quantitative analysis
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from scraper import load_history

print("=" * 70)
print("🔬 STATISTICAL EDGE DETECTOR")
print("=" * 70)

site = input("\nSite name (default: Kalyan): ").strip() or "Kalyan"
df = load_history(site)
if df.empty:
    print(f"❌ {site} ka data nahi mila. Pehle fetch karo.")
    exit()

flat = [int(n) for nums in df["numbers"] for n in nums if str(n).isdigit()]
n = len(flat)
print(f"\n📊 Analyzing {n} numbers from {site}\n")

# ============ TEST 1: RUNS TEST ============
print("=" * 70)
print("TEST 1: RUNS TEST (Wald-Wolfowitz)")
print("=" * 70)
median = np.median(flat)
binary = [1 if x > median else 0 for x in flat]
runs = 1
for i in range(1, len(binary)):
    if binary[i] != binary[i-1]:
        runs += 1
n1 = sum(binary); n2 = len(binary) - n1
exp_runs = (2*n1*n2)/(n1+n2) + 1
var_runs = (2*n1*n2*(2*n1*n2-n1-n2))/((n1+n2)**2 * (n1+n2-1))
z = (runs - exp_runs) / np.sqrt(var_runs)
p = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"Observed runs: {runs}")
print(f"Expected runs: {exp_runs:.1f}")
print(f"Z-score: {z:.3f}")
print(f"P-value: {p:.4f}")
print(f"Verdict: {'⚠️ NON-RANDOM' if p < 0.05 else '✅ Random'}\n")

# ============ TEST 2: AUTOCORRELATION ============
print("=" * 70)
print("TEST 2: AUTOCORRELATION (does past predict future?)")
print("=" * 70)
arr = np.array(flat)
sig_autocorr = False
for lag in range(1, 11):
    a, b = arr[:-lag], arr[lag:]
    corr = np.corrcoef(a, b)[0,1]
    t = corr * np.sqrt((len(a)-2)/(1-corr**2)) if abs(corr) < 1 else 0
    p_ac = 2 * (1 - stats.t.cdf(abs(t), len(a)-2))
    flag = "⚠️ SIGNIFICANT" if p_ac < 0.05 else ""
    if p_ac < 0.05: sig_autocorr = True
    print(f"Lag {lag}: corr={corr:+.4f}  p={p_ac:.4f}  {flag}")
print()

# ============ TEST 3: SERIAL DEPENDENCE ============
print("=" * 70)
print("TEST 3: SERIAL DEPENDENCE (chi-square per preceding number)")
print("=" * 70)
transitions = {}
for i in range(len(flat) - 1):
    a, b = flat[i], flat[i+1]
    transitions.setdefault(a, Counter())[b] += 1
sig_serial = 0
for num in range(10):
    if num not in transitions: continue
    row = transitions[num]
    total = sum(row.values())
    if total < 10: continue
    obs = [row.get(j, 0) for j in range(10)]
    exp = [total/10]*10
    chi2, p_chi = stats.chisquare(obs, exp)
    flag = "⚠️" if p_chi < 0.05 else ""
    if p_chi < 0.05: sig_serial += 1
    print(f"After {num}: chi2={chi2:.2f}  p={p_chi:.4f}  {flag}")
print(f"\nSignificant transitions: {sig_serial}/10")
print()

# ============ TEST 4: ROLLING CHI-SQUARE ============
print("=" * 70)
print("TEST 4: ROLLING WINDOW CHI-SQUARE (is distribution stable?)")
print("=" * 70)
window = 200
biased_windows = 0
total_windows = 0
for i in range(0, len(flat) - window, window):
    wd = flat[i:i+window]
    freq = Counter(wd)
    obs = [freq.get(j, 0) for j in range(10)]
    exp = [len(wd)/10]*10
    chi2, p_w = stats.chisquare(obs, exp)
    total_windows += 1
    if p_w < 0.05:
        biased_windows += 1
        print(f"Window {i}-{i+window}: chi2={chi2:.2f}  p={p_w:.4f}  ⚠️ BIASED")
print(f"\nBiased windows: {biased_windows}/{total_windows}")
print(f"Expected by chance: {total_windows*0.05:.1f}")
print(f"Verdict: {'⚠️ STRUCTURE' if biased_windows > total_windows*0.1 else '✅ Stable/Uniform'}\n")

# ============ TEST 5: MEAN REVERSION (GAP ANALYSIS) ============
print("=" * 70)
print("TEST 5: MEAN REVERSION (does cold number come back faster?)")
print("=" * 70)
gaps = {}
last_seen = {}
for i, num in enumerate(flat):
    if num in last_seen:
        gaps.setdefault(num, []).append(i - last_seen[num])
    last_seen[num] = i
print(f"Expected gap (if random): 10")
for num in range(10):
    if num in gaps and len(gaps[num]) > 5:
        mean_gap = np.mean(gaps[num])
        std_gap = np.std(gaps[num])
        flag = "⚠️" if abs(mean_gap - 10) > 2 else ""
        print(f"Number {num}: mean_gap={mean_gap:.2f}  std={std_gap:.2f}  n={len(gaps[num])}  {flag}")
print()

# ============ TEST 6: VARIANCE RATIO ============
print("=" * 70)
print("TEST 6: VARIANCE RATIO (random walk or mean-reverting?)")
print("=" * 70)
returns = np.diff(arr)
var_1 = np.var(returns)
sig_vr = False
for q in [2, 5, 10, 20]:
    if q >= len(arr): break
    q_ret = arr[q:] - arr[:-q]
    var_q = np.var(q_ret)
    vr = var_q / (q * var_1) if var_1 > 0 else 0
    n_ret = len(returns)
    z_vr = (vr - 1) / np.sqrt(2*(2*q-1)*(q-1)/(3*q*n_ret))
    p_vr = 2 * (1 - stats.norm.cdf(abs(z_vr)))
    flag = "⚠️" if p_vr < 0.05 else ""
    if p_vr < 0.05: sig_vr = True
    print(f"Lag {q}: VR={vr:.4f}  z={z_vr:.3f}  p={p_vr:.4f}  {flag}")
print()

# ============ TEST 7: HOT/COLD SIGNIFICANCE ============
print("=" * 70)
print("TEST 7: HOT/COLD STATISTICAL SIGNIFICANCE (last 50)")
print("=" * 70)
recent = flat[-50:]
freq = Counter(recent)
expected = 5  # 50/10
sig_hot = 0
for num in range(10):
    obs = freq.get(num, 0)
    p_b = stats.binomtest(obs, 50, 0.1).pvalue
    flag = "⚠️" if p_b < 0.05 else ""
    if p_b < 0.05: sig_hot += 1
    print(f"Number {num}: count={obs}  expected={expected}  p={p_b:.4f}  {flag}")
print(f"\nSignificant hot/cold: {sig_hot}/10")
print(f"Expected by chance: 0.5")
print()

# ============ FINAL VERDICT ============
print("=" * 70)
print("🎯 FINAL VERDICT — KYA EDGE HAI?")
print("=" * 70)
checks = {
    "Runs Test": p < 0.05,
    "Autocorrelation": sig_autocorr,
    "Serial Dependence": sig_serial >= 2,
    "Rolling Chi-square": biased_windows > total_windows * 0.1,
    "Variance Ratio": sig_vr,
    "Hot/Cold": sig_hot >= 2,
}
passed = sum(checks.values())
print(f"\nTests passed (edge detected): {passed}/6\n")
for test, result in checks.items():
    print(f"  {'⚠️ EDGE' if result else '✅ No edge'}  — {test}")

print()
if passed >= 3:
    print("🎯 VERDICT: STRUCTURE DETECTED!")
    print("   Data me kuch patterns hain — exploitable ho sakte hain")
elif passed >= 1:
    print("🎯 VERDICT: WEAK SIGNALS")
    print("   1-2 tests pass hue — noise ya weak edge ho sakta hai")
else:
    print("🎯 VERDICT: TRULY RANDOM")
    print("   Saare tests fail — koi exploitable edge nahi")
print("=" * 70)
