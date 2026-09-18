# deep_test.py
# Walk-forward deep test of panna_predictor methods
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import json, os

from config import SITES
from panna_predictor import (
    build_all_sequences, all_method_scores, voting_predict,
    combine, load_weights, METHODS
)

RESULTS_FILE = "deep_test_results.json"


def predict_single_method(seq, method, actual):
    """Return True if method's top-1 == actual."""
    if len(seq) < 20:
        return None
    ms = all_method_scores(seq)
    sc = ms.get(method)
    if not sc:
        return None
    top = max(sc.items(), key=lambda x: x[1])[0]
    return top == actual


def predict_voting(seq, weights, top_n=5):
    """Return top-N from voting."""
    return [p["number"] for p in voting_predict(seq, weights, top_n)]


def predict_weighted(seq, weights, top_n=5):
    """Return top-N from weighted average."""
    if len(seq) < 20:
        return []
    combined = combine(all_method_scores(seq), weights)
    if not combined:
        return []
    return [n for n, _ in combined.most_common(top_n)]


def test_site(site, min_history=150, sample_size=300):
    """Walk-forward test on a single site's open sequence."""
    data = build_all_sequences(site)
    if not data:
        return None
    seq = data["open_seq"]
    if len(seq) < min_history + 50:
        return None

    weights = load_weights(site)

    # Test range: from min_history to len(seq)-1
    start = max(min_history, len(seq) - sample_size)
    end = len(seq) - 1
    if end - start < 30:
        return None

    results = {
        "site": site,
        "total_predictions": 0,
        "methods": defaultdict(lambda: {"top1": 0, "top3": 0, "top5": 0, "total": 0}),
        "voting": {"top1": 0, "top3": 0, "top5": 0, "total": 0},
        "weighted": {"top1": 0, "top3": 0, "top5": 0, "total": 0},
        "random": {"top1": 0, "top3": 0, "top5": 0, "total": 0},
    }

    for i in range(start, end):
        past = seq[:i]
        actual = seq[i]
        if len(past) < 20:
            continue

        results["total_predictions"] += 1

        # Each method's top-1
        ms = all_method_scores(past)
        for method, sc in ms.items():
            if not sc:
                continue
            top1 = max(sc.items(), key=lambda x: x[1])[0]
            top3 = [n for n, _ in sorted(sc.items(), key=lambda x: x[1], reverse=True)[:3]]
            top5 = [n for n, _ in sorted(sc.items(), key=lambda x: x[1], reverse=True)[:5]]
            m = results["methods"][method]
            m["total"] += 1
            if top1 == actual: m["top1"] += 1
            if actual in top3: m["top3"] += 1
            if actual in top5: m["top5"] += 1

        # Voting ensemble
        v_top5 = predict_voting(past, weights, 5)
        v = results["voting"]
        v["total"] += 1
        if v_top5 and v_top5[0] == actual: v["top1"] += 1
        if actual in v_top5[:3]: v["top3"] += 1
        if actual in v_top5: v["top5"] += 1

        # Weighted average
        w_top5 = predict_weighted(past, weights, 5)
        w = results["weighted"]
        w["total"] += 1
        if w_top5 and w_top5[0] == actual: w["top1"] += 1
        if actual in w_top5[:3]: w["top3"] += 1
        if actual in w_top5: w["top5"] += 1

        # Random (seed fixed for reproducibility)
        rng = np.random.default_rng(i)
        r_top5 = list(rng.choice(range(10), size=5, replace=False))
        r = results["random"]
        r["total"] += 1
        if r_top5[0] == actual: r["top1"] += 1
        if actual in r_top5[:3]: r["top3"] += 1
        if actual in r_top5: r["top5"] += 1

    return results


def pct(hit, total):
    return round(hit / total * 100, 2) if total else 0


def main():
    print("=" * 75)
    print("🔬 DEEP TEST — Walk-Forward Validation")
    print("=" * 75)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_results = []
    method_agg = defaultdict(lambda: {"top1": 0, "top3": 0, "top5": 0, "total": 0})
    voting_agg = {"top1": 0, "top3": 0, "top5": 0, "total": 0}
    weighted_agg = {"top1": 0, "top3": 0, "top5": 0, "total": 0}
    random_agg = {"top1": 0, "top3": 0, "top5": 0, "total": 0}

    for site in SITES:
        print(f"🔄 Testing: {site}...", end=" ", flush=True)
        r = test_site(site)
        if not r:
            print("SKIP (data kam)")
            continue
        all_results.append(r)
        print(f"{r['total_predictions']} predictions")

        # Aggregate
        for m, s in r["methods"].items():
            method_agg[m]["top1"] += s["top1"]
            method_agg[m]["top3"] += s["top3"]
            method_agg[m]["top5"] += s["top5"]
            method_agg[m]["total"] += s["total"]
        for key in ["top1", "top3", "top5", "total"]:
            voting_agg[key] += r["voting"][key]
            weighted_agg[key] += r["weighted"][key]
            random_agg[key] += r["random"][key]

    print()
    print("=" * 75)
    print("📊 OVERALL RESULTS (all sites combined)")
    print("=" * 75)
    print()
    print(f"{'Method':<15} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'Samples':<10}")
    print("-" * 60)

    rows = []
    for m in METHODS:
        s = method_agg[m]
        if s["total"] == 0:
            continue
        rows.append({
            "Method": m,
            "Top1": pct(s["top1"], s["total"]),
            "Top3": pct(s["top3"], s["total"]),
            "Top5": pct(s["top5"], s["total"]),
            "Samples": s["total"],
        })
    rows.sort(key=lambda x: -x["Top1"])
    for r in rows:
        print(f"{r['Method']:<15} {r['Top1']:>5.2f}%    {r['Top3']:>5.2f}%    {r['Top5']:>5.2f}%    {r['Samples']}")

    print()
    print("=" * 75)
    print("🎯 ENSEMBLES vs RANDOM")
    print("=" * 75)
    print()
    print(f"{'Approach':<20} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10}")
    print("-" * 55)
    for name, agg in [("Voting (current)", voting_agg), ("Weighted avg", weighted_agg), ("Random baseline", random_agg)]:
        if agg["total"] == 0:
            continue
        print(f"{name:<20} {pct(agg['top1'], agg['total']):>5.2f}%    "
              f"{pct(agg['top3'], agg['total']):>5.2f}%    "
              f"{pct(agg['top5'], agg['total']):>5.2f}%")

    print()
    print("Expected random:  Top-1 = 10.00%  Top-3 = 30.00%  Top-5 = 50.00%")

    # Save
    save_data = {
        "generated_at": datetime.now().isoformat(),
        "per_site": all_results,
        "methods_agg": {k: dict(v) for k, v in method_agg.items()},
        "voting_agg": voting_agg,
        "weighted_agg": weighted_agg,
        "random_agg": random_agg,
    }
    # Convert defaultdict to dict for JSON
    for r in save_data["per_site"]:
        r["methods"] = dict(r["methods"])
    with open(RESULTS_FILE, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print()
    print(f"💾 Saved: {RESULTS_FILE}")

    # Per-site best method
    print()
    print("=" * 75)
    print("🏆 PER-SITE: Best Method")
    print("=" * 75)
    print()
    print(f"{'Site':<20} {'Best Method':<15} {'Top-1':<10}")
    print("-" * 55)
    for r in all_results:
        best_m, best_a = None, 0
        for m, s in r["methods"].items():
            a = s["top1"] / s["total"] * 100 if s["total"] else 0
            if a > best_a:
                best_a = a
                best_m = m
        print(f"{r['site']:<20} {best_m or '—':<15} {best_a:>5.2f}%")

    print()
    print("=" * 75)
    print(f"✅ Done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)


if __name__ == "__main__":
    main()
