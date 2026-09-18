# panna_predictor.py - MEGA VERSION
# 10 methods + priority sites + cross-pair + recent shift
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import os
import json

from scraper import load_history
from datetime import datetime, timedelta

# ============ CONFIG ============
TOP_N_OPEN = 5
TOP_N_CLOSE = 5
TOP_N_JODI = 10
TOP_N_PANNA = 20
CONFIDENCE_THRESHOLD = 0.0

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
WEIGHTS_FILE = os.path.join(MODELS_DIR, "method_weights.json")

# Top-performing sites (from deep analysis)
PRIORITY_SITES = {
    "Kalyan": 1.5,        # 19.19% markov2
    "Time Bazar": 1.3,    # 14.14%
    "Kalyan Night": 1.3,  # 14.14%
    "Madhur Day": 1.15,   # 12.12%
    "Supreme Day": 1.15,  # 12.12%
}

# Cross-correlated pairs (from deep analysis)
CROSS_PAIRS = {
    "Supreme Day": "Madhur Night",
    "Supreme Night": "Rajdhani Night",
    "Madhur Day": "Milan Night",
    "Kalyan": "Rajdhani Night",
}

METHODS = ["markov1", "markov2", "markov3", "freq30", "freq100",
           "recency", "gap", "weekday", "shift"]

DEFAULT_WEIGHTS = {
    "markov1": 0.12, "markov2": 0.15, "markov3": 0.13,
    "freq30": 0.08, "freq100": 0.04,
    "recency": 0.12, "gap": 0.10, "weekday": 0.08,
    "shift": 0.12,
}


def load_weights(site=None):
    if site:
        safe = site.replace(" ", "_").replace("/", "_")
        sf = os.path.join(MODELS_DIR, f"weights_{safe}.json")
        if os.path.exists(sf):
            try:
                with open(sf) as f:
                    return json.load(f)
            except Exception:
                pass
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_WEIGHTS.copy()


def save_weights(w):
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(w, f, indent=2)


# ============ PARSER ============
def parse_row_to_days(numbers):
    days = []
    i = 0
    while i + 8 <= len(numbers):
        chunk = [str(x) for x in numbers[i:i+8]]
        op, jd, cp = chunk[0:3], chunk[3:5], chunk[5:8]
        try:
            open_digit = sum(int(x) for x in op) % 10
            close_digit = sum(int(x) for x in cp) % 10
            days.append({
                "open_panna": "".join(op),
                "jodi": f"{open_digit}{close_digit}",
                "close_panna": "".join(cp),
                "open": open_digit,
                "close": close_digit,
            })
        except (ValueError, IndexError):
            pass
        i += 8
    return days


def parse_with_dates(numbers, week_start_str, is_current_week=False):
    """Parse with date tracking + auto-shift when week has closed days at start.
    is_current_week=True → auto-detect start gaps and shift dates forward.
    """
    from datetime import datetime, timedelta
    import re
    match = re.search(r"(\d{2})/(\d{2})/(\d{4})", str(week_start_str))
    if not match:
        return parse_row_to_days(numbers)
    try:
        d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        start = datetime(y, m, d)
    except Exception:
        return parse_row_to_days(numbers)

    days = []
    i = 0
    while i + 8 <= len(numbers):
        chunk = [str(x) for x in numbers[i:i+8]]
        op, jd, cp = chunk[0:3], chunk[3:5], chunk[5:8]
        try:
            open_digit = sum(int(x) for x in op) % 10
            close_digit = sum(int(x) for x in cp) % 10
            days.append({
                "open_panna": "".join(op),
                "jodi": f"{open_digit}{close_digit}",
                "close_panna": "".join(cp),
                "open": open_digit,
                "close": close_digit,
            })
        except (ValueError, IndexError):
            pass
        i += 8

    N = len(days)
    # Auto-shift: only for current (in-progress) week
    offset = 0
    if is_current_week and N > 0:
        today = datetime.now()
        days_elapsed = (today - start).days  # how many days since week started
        # If site has fewer days than days elapsed, some days were closed (likely start)
        if N < days_elapsed:
            offset = 1

    for idx, d_dict in enumerate(days):
        day_date = start + timedelta(days=idx + offset)
        d_dict["date"] = day_date.strftime("%d/%m/%Y")
        d_dict["iso_date"] = day_date.strftime("%Y-%m-%d")

    return days


def build_all_sequences(site):
    df = load_history(site)
    if df.empty:
        return None
    seqs = {"open_seq": [], "close_seq": [], "jodi_seq": [],
            "open_panna_seq": [], "close_panna_seq": [], "all_days": []}
    last_row_idx = df.index[-1]
    for idx, row in df.iterrows():
        nums = row["numbers"]
        date_str = row.get("date", "")
        is_current = (idx == last_row_idx)
        for d in parse_with_dates(nums, date_str, is_current_week=is_current):
            seqs["open_seq"].append(d["open"])
            seqs["close_seq"].append(d["close"])
            seqs["jodi_seq"].append(d["jodi"])
            seqs["open_panna_seq"].append(d["open_panna"])
            seqs["close_panna_seq"].append(d["close_panna"])
            seqs["all_days"].append(d)
    seqs["total_days"] = len(seqs["all_days"])
    return seqs


# ============ 10 METHODS ============
def m_markov(seq, order=1):
    if len(seq) < order + 5:
        return Counter()
    key = tuple(seq[-order:]) if order > 1 else seq[-1]
    trans = Counter()
    if order == 1:
        for i in range(len(seq) - 1):
            if seq[i] == key:
                trans[seq[i+1]] += 1
    else:
        for i in range(len(seq) - order):
            if tuple(seq[i:i+order]) == key:
                trans[seq[i+order]] += 1
    total = sum(trans.values())
    return Counter({k: v/total for k, v in trans.items()}) if total else Counter()


def m_frequency(seq, window):
    w = min(window, len(seq))
    if w == 0:
        return Counter()
    recent = seq[-w:]
    freq = Counter(recent)
    return Counter({n: freq.get(n, 0)/w for n in range(10)})


def m_recency(seq, decay=0.97):
    scores = Counter()
    for i, n in enumerate(seq):
        scores[n] += decay ** (len(seq) - i - 1)
    total = sum(scores.values())
    if total:
        for k in scores:
            scores[k] /= total
    for n in range(10):
        if n not in scores:
            scores[n] = 0
    return scores


def m_gap(seq):
    if not seq:
        return Counter()
    last_seen = {}
    for i, n in enumerate(seq):
        last_seen[n] = i
    L = len(seq)
    scores = Counter()
    max_gap = max((L - last_seen.get(n, 0) for n in range(10)), default=1)
    for n in range(10):
        scores[n] = (L - last_seen.get(n, 0)) / max_gap
    total = sum(scores.values())
    if total:
        for k in scores:
            scores[k] /= total
    return scores


def m_weekday(seq):
    if len(seq) < 42:
        return Counter()
    scores = Counter()
    for offset in [6, 12, 18, 24, 30]:
        if offset < len(seq):
            scores[seq[-offset]] += 1
    total = sum(scores.values())
    return Counter({k: v/total for k, v in scores.items()}) if total else Counter()


# ============ NEW: RECENT SHIFT BOOST ============
def m_recent_shift(seq, window=90):
    """Numbers jo recent me zyada aane lage hain"""
    if len(seq) < window * 2:
        return Counter()
    recent = seq[-window:]
    older = seq[-window*2:-window]
    r_freq = Counter(recent)
    o_freq = Counter(older)
    scores = Counter()
    for n in range(10):
        r_pct = r_freq.get(n, 0) / len(recent)
        o_pct = o_freq.get(n, 0) / len(older) if older else 0
        # Positive shift gets boost, negative gets penalty
        shift = r_pct - o_pct
        scores[n] = max(0, r_pct + shift)  # Base + boost
    total = sum(scores.values())
    if total:
        for k in scores:
            scores[k] /= total
    return scores


# ============ NEW: CROSS-PAIR SIGNAL ============
def m_cross_pair(seq, pair_seq):
    """Correlated site se signal"""
    if not pair_seq or len(seq) < 10 or len(pair_seq) < 10:
        return Counter()
    # Check if pair site's recent pattern matches our history
    if len(seq) < 20:
        return Counter()
    # Look for similar 3-number windows in history
    recent_pair = tuple(pair_seq[-3:])
    scores = Counter()
    for i in range(len(seq) - 3):
        if len(pair_seq) > i + 3:
            pass  # Skip complex match
    # Simple: pair's last number gives weight to our next
    last_pair = pair_seq[-1]
    scores[last_pair] = 1.0
    return scores


# ============ COMBINE ALL ============
def all_method_scores(seq, pair_seq=None):
    scores = {
        "markov1": m_markov(seq, 1),
        "markov2": m_markov(seq, 2),
        "markov3": m_markov(seq, 3),
        "freq30": m_frequency(seq, 30),
        "freq100": m_frequency(seq, 100),
        "recency": m_recency(seq),
        "gap": m_gap(seq),
        "weekday": m_weekday(seq),
        "shift": m_recent_shift(seq, 90),
    }
    return scores


def combine(scores_dict, weights):
    final = Counter()
    for method, scores in scores_dict.items():
        w = weights.get(method, 0)
        if not scores or w <= 0:
            continue
        mx = max(scores.values()) if scores else 0
        if mx == 0:
            continue
        for n in range(10):
            final[n] += (scores.get(n, 0) / mx) * w
    return final


def voting_predict(seq, weights, top_n=5):
    """Each method votes for its top-3, count votes"""
    if len(seq) < 20:
        return []
    ms = all_method_scores(seq)
    votes = Counter()
    for method, sc in ms.items():
        if not sc or len(sc) < 3:
            continue
        w = weights.get(method, 0)
        top3 = sorted(sc.items(), key=lambda x: x[1], reverse=True)[:3]
        for i, (num, s) in enumerate(top3):
            # Rank 1 = 3 votes, Rank 2 = 2 votes, Rank 3 = 1 vote
            rank_weight = (3 - i) * w
            votes[num] += rank_weight
    if not votes:
        return []
    top = votes.most_common(top_n)
    total = sum(v for _, v in top) or 1
    return [{"number": n, "confidence": round((v/total)*100, 1)} for n, v in top]


def predict_single(seq, weights, top_n, min_conf=0.0, pair_seq=None):
    if len(seq) < 20:
        return []
    combined = combine(all_method_scores(seq, pair_seq), weights)
    if not combined:
        return []
    top = combined.most_common(top_n)
    total = sum(s for _, s in top) or 1
    result = []
    for n, s in top:
        conf = (s/total) * 100
        if conf < min_conf:
            continue
        result.append({"number": n, "confidence": round(conf, 1)})
    return result


def pannas_for_digit(digit, hist, top_n=20):
    possible = [f"{a}{b}{c}" for a in range(10) for b in range(10) for c in range(10)
                if (a+b+c) % 10 == digit]
    hf = Counter(hist)
    scored = [(p, hf.get(p, 0) + 1) for p in possible]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_n]
    total = sum(s for _, s in top) or 1
    return [{"panna": p, "confidence": round((s/total)*100, 1)} for p, s in top]


def derive_pannas(digit_preds, panna_hist, top_n=20):
    out = []
    for dp in digit_preds:
        for p in pannas_for_digit(dp["number"], panna_hist, 10):
            comb = (dp["confidence"] * p["confidence"]) / 100
            out.append({"panna": p["panna"], "digit": dp["number"], "confidence": round(comb, 2)})
    out.sort(key=lambda x: x["confidence"], reverse=True)
    seen, final = set(), []
    for p in out:
        if p["panna"] in seen:
            continue
        seen.add(p["panna"])
        final.append(p)
        if len(final) >= top_n:
            break
    return final


def combine_jodi(open_preds, close_preds, top_n=10):
    combos = []
    for op in open_preds:
        for cp in close_preds:
            conf = (op["confidence"] * cp["confidence"]) / 100
            combos.append({"jodi": f"{op['number']}{cp['number']}", "confidence": conf,
                           "open_digit": op["number"], "close_digit": cp["number"]})
    combos.sort(key=lambda x: x["confidence"], reverse=True)
    top = combos[:top_n]
    total = sum(c["confidence"] for c in top) or 1
    return [{"jodi": c["jodi"], "confidence": round((c["confidence"]/total)*100, 1),
             "open_digit": c["open_digit"], "close_digit": c["close_digit"]} for c in top]


def predict_site_full(site, top_n=5):
    data = build_all_sequences(site)
    if not data or data["total_days"] < 20:
        return None

    # Cross-pair data
    pair_seq = None
    if site in CROSS_PAIRS:
        pair_site = CROSS_PAIRS[site]
        pair_data = build_all_sequences(pair_site)
        if pair_data:
            pair_seq = pair_data["open_seq"]

    w = load_weights(site)
    # Use voting method (better than weighted avg)
    op = voting_predict(data["open_seq"], w, TOP_N_OPEN)
    cp = voting_predict(data["close_seq"], w, TOP_N_CLOSE)
    last = data["all_days"][-1] if data["all_days"] else {}

    # Calculate next prediction date
    last_date_str = last.get("date", "-")
    predict_for_date = "-"
    if last_date_str and last_date_str != "-":
        try:
            dt = datetime.strptime(last_date_str, "%d/%m/%Y")
            next_dt = dt + timedelta(days=1)
            predict_for_date = next_dt.strftime("%d/%m/%Y")
        except Exception:
            pass

    return {
        "site": site, "total_days": data["total_days"],
        "predict_for_index": data["total_days"],
        "last_date": last_date_str,
        "predict_for_date": predict_for_date,
        "is_priority": site in PRIORITY_SITES,
        "priority_weight": PRIORITY_SITES.get(site, 1.0),
        "pair_site": CROSS_PAIRS.get(site, "-"),
        "last_open": last.get("open", "-"),
        "last_close": last.get("close", "-"),
        "last_jodi": last.get("jodi", "-"),
        "last_open_panna": last.get("open_panna", "-"),
        "last_close_panna": last.get("close_panna", "-"),
        "open_prediction": op, "close_prediction": cp,
        "jodi_prediction": combine_jodi(op, cp, TOP_N_JODI),
        "open_panna_prediction": derive_pannas(op, data["open_panna_seq"], TOP_N_PANNA),
        "close_panna_prediction": derive_pannas(cp, data["close_panna_seq"], TOP_N_PANNA),
    }


def predict_all_sites_full(sites_dict, top_n=5):
    out = {}
    for s in sites_dict:
        try:
            p = predict_site_full(s, top_n)
            if p:
                out[s] = p
                try:
                    save_snapshot(s, p)
                except Exception:
                    pass
        except Exception as e:
            print(f"⚠️ {s}: {e}")
    return out


# ============ BACKTEST / LEARN ============

# ============ SNAPSHOT SYSTEM ============
SNAPSHOT_DIR = os.path.join("daily_predictions", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def save_snapshot(site, pred):
    """Save prediction as JSON so tomorrow we can compare with actual result."""
    pd_str = pred.get("predict_for_date", "-")
    if pd_str == "-":
        return
    fname = f"{site.replace(' ', '_')}_{pd_str.replace('/', '_')}.json"
    fpath = os.path.join(SNAPSHOT_DIR, fname)

    # Ek din me ek hi snapshot — overwrite nahi karo
    if os.path.exists(fpath):
        return

    def _default(o):
        try:
            return o.item()
        except Exception:
            return str(o)

    with open(fpath, "w") as f:
        json.dump(pred, f, default=_default)


def load_snapshot(site, date_str):
    """Load a saved snapshot for a given site + date (DD/MM/YYYY)."""
    fname = f"{site.replace(' ', '_')}_{date_str.replace('/', '_')}.json"
    fpath = os.path.join(SNAPSHOT_DIR, fname)
    if not os.path.exists(fpath):
        return None
    try:
        with open(fpath) as f:
            return json.load(f)
    except Exception:
        return None
# ============ /SNAPSHOT SYSTEM ============


def backtest_methods(site, sample_size=100):
    data = build_all_sequences(site)
    if not data or data["total_days"] < sample_size + 50:
        return None
    seq = data["open_seq"]

    pair_seq = None
    if site in CROSS_PAIRS:
        pair_data = build_all_sequences(CROSS_PAIRS[site])
        if pair_data:
            pair_seq = pair_data["open_seq"]

    start = len(seq) - sample_size
    hits = {m: 0 for m in METHODS}
    total = 0

    for i in range(start, len(seq) - 1):
        past = seq[:i]
        actual = seq[i]
        if len(past) < 20:
            continue

        past_pair = pair_seq[:i] if pair_seq and len(pair_seq) >= i else None
        ms = all_method_scores(past, past_pair)
        total += 1

        for method, sc in ms.items():
            if not sc:
                continue
            top_n = max(sc.items(), key=lambda x: x[1])[0]
            if top_n == actual:
                hits[method] += 1

    if total == 0:
        return None
    return {m: hits[m] / total for m in METHODS}


def learn_weights(sites_list):
    all_acc = defaultdict(list)
    for site in sites_list:
        accs = backtest_methods(site, sample_size=120)
        if accs:
            for m, a in accs.items():
                all_acc[m].append(a)

    if not all_acc:
        print("❌ Backtest fail")
        return DEFAULT_WEIGHTS

    avg = {m: float(np.mean(all_acc[m])) for m in METHODS if m in all_acc}
    total = sum(avg.values())
    if total == 0:
        return DEFAULT_WEIGHTS

    new_w = {m: round(avg[m] / total, 4) for m in avg}
    print("=" * 70)
    print("🧠 GLOBAL AUTO-LEARNING")
    print("=" * 70)
    print(f"\nTested on {len(sites_list)} sites\n")
    print(f"{'Method':<15} {'Accuracy':<12} {'New Weight':<12}")
    print("-" * 42)
    for m in METHODS:
        if m in avg:
            print(f"{m:<15} {avg[m]*100:>6.2f}%      {new_w[m]*100:>6.2f}%")
    save_weights(new_w)
    print(f"\n[OK] Global weights saved")

    # Per-site
    print("\n" + "=" * 70)
    print("🎯 PER-SITE AUTO-LEARNING")
    print("=" * 70)
    for site in sites_list:
        saccs = backtest_methods(site, sample_size=120)
        if not saccs:
            continue
        stot = sum(saccs.values())
        if stot == 0:
            continue
        sw = {m: round(saccs[m] / stot, 4) for m in saccs}
        safe = site.replace(" ", "_").replace("/", "_")
        sf = os.path.join(MODELS_DIR, f"weights_{safe}.json")
        with open(sf, "w") as f:
            json.dump(sw, f, indent=2)
        best = max(saccs.items(), key=lambda x: x[1])
        print(f"  {site:<20} best: {best[0]:<10} ({best[1]*100:.1f}%)")

    print("\n[OK] Per-site weights saved")
    return new_w


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "learn":
        print("🧠 Learning weights from backtest...")
        # Focus on top-performing sites
        learn_weights(list(PRIORITY_SITES.keys()) + ["Milan Day", "Rajdhani Night", "Main Bazar", "Sridevi"])
    else:
        print("=" * 70)
        print("🎯 MEGA PREDICTOR TEST")
        print("=" * 70)
        w = load_weights()
        print("\nCurrent weights:")
        for k, v in w.items():
            print(f"  {k:<10} {v*100:>5.2f}%")
        for site in list(PRIORITY_SITES.keys()):
            p = predict_site_full(site)
            if not p:
                continue
            print(f"\n{'='*70}")
            print(f"📊 {site} ({p['total_days']} days) {'⭐ PRIORITY' if p['is_priority'] else ''}")
            print(f"{'='*70}")
            print(f"Last: Open={p['last_open']} Close={p['last_close']} Jodi={p['last_jodi']}")
            print(f"\n🎯 OPEN (top 5):")
            for x in p["open_prediction"]:
                print(f"   {x['number']} — {x['confidence']}%")
            print(f"🎯 CLOSE (top 5):")
            for x in p["close_prediction"]:
                print(f"   {x['number']} — {x['confidence']}%")
