# redesign_app.py
import os, shutil, py_compile

# ===== 1. Backup =====
if not os.path.exists("app_old.py"):
    shutil.copy("app.py", "app_old.py")
    print("✅ Backup: app_old.py")

# ===== 2. New clean app.py =====
NEW_APP = '''# app.py — CLEAN REDESIGN
import streamlit as st
import pandas as pd
from config import SITES
from scraper import update_all_sites
from panna_predictor import predict_all_sites_full, load_snapshot

st.set_page_config(page_title="Market Mood AI", page_icon="🧠", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def get_predictions():
    try:
        return predict_all_sites_full(SITES, top_n=5)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return {}


def hit_check(actual, preds, key="number"):
    if actual is None or str(actual) == "-":
        return None
    return str(actual) in [str(p.get(key)) for p in preds]


def label(h):
    if h is True:
        return "✅ HIT"
    if h is False:
        return "❌ Miss"
    return "—"


def show_site_card(site_name, pred, expanded=False):
    last_date = pred.get("last_date", "-")
    predict_date = pred.get("predict_for_date", "-")

    with st.expander(f"📊 {site_name}  —  Next: {predict_date}  (based on {last_date})", expanded=expanded):
        # ---- Kal ka result ----
        snap = load_snapshot(site_name, last_date) if last_date != "-" else None

        if snap:
            st.markdown(f"##### 📊 Kal Ka Result — {last_date}")
            ao = pred.get("last_open"); ac = pred.get("last_close")
            aj = pred.get("last_jodi")
            aop = pred.get("last_open_panna"); acp = pred.get("last_close_panna")

            po = snap.get("open_prediction", [])
            pc = snap.get("close_prediction", [])
            pj = snap.get("jodi_prediction", [])
            pop = snap.get("open_panna_prediction", [])
            pcp = snap.get("close_panna_prediction", [])

            c1, c2, c3, c4, c5 = st.columns(5)
            for col, name, actual, preds, key in [
                (c1, "Open", ao, po, "number"),
                (c2, "Close", ac, pc, "number"),
                (c3, "Jodi", aj, pj, "jodi"),
                (c4, "Open Panna", aop, pop, "panna"),
                (c5, "Close Panna", acp, pcp, "panna"),
            ]:
                col.markdown(f"**{name}** {label(hit_check(actual, preds, key))}")
                col.caption(f"Actual: {actual}")

            with st.expander("🔍 Kal kya diya tha?"):
                st.write(f"**Open:** {', '.join(str(p['number']) for p in po)}")
                st.write(f"**Close:** {', '.join(str(p['number']) for p in pc)}")
                st.write(f"**Jodi:** {', '.join(str(p['jodi']) for p in pj)}")
                st.write(f"**Open Panna:** {', '.join(str(p['panna']) for p in pop)}")
                st.write(f"**Close Panna:** {', '.join(str(p['panna']) for p in pcp)}")
        else:
            st.caption(f"📌 {last_date} ka snapshot nahi tha — kal se comparison shuru hoga.")

        st.markdown("---")

        # ---- Aaj ka prediction ----
        st.markdown(f"##### 🎯 Prediction for {predict_date}")
        st.caption(f"Based on {last_date} → Open {pred.get('last_open_panna')} ({pred.get('last_open')}) | Jodi {pred.get('last_jodi')} | Close {pred.get('last_close_panna')} ({pred.get('last_close')})")

        cols = st.columns(5)
        sections = [
            ("🎯 OPEN (5)", pred["open_prediction"][:5], "number"),
            ("🎯 CLOSE (5)", pred["close_prediction"][:5], "number"),
            ("🎯 JODI (10)", pred["jodi_prediction"][:10], "jodi"),
            ("🎯 OPEN PANNA (20)", pred["open_panna_prediction"][:20], "panna"),
            ("🎯 CLOSE PANNA (20)", pred["close_panna_prediction"][:20], "panna"),
        ]
        for col, (header, items, key) in zip(cols, sections):
            col.markdown(f"**{header}**")
            for p in items:
                col.write(f"{p[key]} — {p['confidence']}%")


# ============ MAIN ============
st.title("🧠 Market Mood AI")

c1, c2 = st.columns([1, 4])
with c1:
    if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
        with st.spinner("Saari sites fetch ho rahi hain..."):
            update_all_sites()
            st.cache_data.clear()
            st.rerun()
with c2:
    search = st.text_input(
        "search", key="global_site_search",
        placeholder="🔍 Site search — Kalyan, Milan, Night, Bazar... (khaali chhodo = saari sites)",
        label_visibility="collapsed",
    )

with st.spinner("Predictions load ho rahi hain..."):
    all_preds = get_predictions()

if not all_preds:
    st.error("Koi prediction generate nahi hui. 'Refresh Data' dabao.")
    st.stop()

if search.strip():
    q = search.strip().lower()
    filtered = [s for s in all_preds.keys() if q in s.lower()]
    if not filtered:
        st.warning(f"'{search}' se koi site match nahi hui.")
        st.caption(f"Available: {', '.join(all_preds.keys())}")
        st.stop()
else:
    filtered = list(all_preds.keys())


# ===== OVERALL SCORE =====
if not search.strip():
    total_hits, total_possible, site_scores = 0, 0, []
    for s, p in all_preds.items():
        snap = load_snapshot(s, p.get("last_date", "-"))
        if not snap:
            continue
        h = 0
        total_possible += 5
        for actual, k_pred, k_field in [
            (p.get("last_open"), "open_prediction", "number"),
            (p.get("last_close"), "close_prediction", "number"),
            (p.get("last_jodi"), "jodi_prediction", "jodi"),
            (p.get("last_open_panna"), "open_panna_prediction", "panna"),
            (p.get("last_close_panna"), "close_panna_prediction", "panna"),
        ]:
            if hit_check(actual, snap.get(k_pred, []), k_field):
                h += 1
        total_hits += h
        site_scores.append({"Site": s, "Hits": h})

    if total_possible > 0:
        st.markdown("### 📊 Kal Ka Overall Score")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Hits", f"{total_hits}/{total_possible}")
        m2.metric("Hit Rate", f"{round(total_hits / total_possible * 100, 1)}%")
        best = max(site_scores, key=lambda x: x["Hits"]) if site_scores else {"Site": "—", "Hits": 0}
        m3.metric("Best Site", best["Site"], f"{best['Hits']}/5")
        st.markdown("---")
    else:
        st.info("📌 Aaj se snapshots shuru. Kal se overall score dikhega.")
        st.markdown("---")


# ===== QUICK TABLE =====
if not search.strip():
    st.markdown("### 📅 Aaj Ke Saare Predictions")
    rows = []
    for s in filtered:
        p = all_preds[s]
        op = p["open_prediction"]; cp = p["close_prediction"]; jd = p["jodi_prediction"]
        snap = load_snapshot(s, p.get("last_date", "-"))

        if snap:
            yo = "✅" if hit_check(p.get("last_open"), snap.get("open_prediction", []), "number") else "❌"
            yc = "✅" if hit_check(p.get("last_close"), snap.get("close_prediction", []), "number") else "❌"
            yj = "✅" if hit_check(p.get("last_jodi"), snap.get("jodi_prediction", []), "jodi") else "❌"
        else:
            yo = yc = yj = "—"

        rows.append({
            "Site": s,
            "Kal Open": yo,
            "Kal Close": yc,
            "Kal Jodi": yj,
            "Aaj Open Top-3": " / ".join(f"{p['number']}({p['confidence']}%)" for p in op[:3]),
            "Aaj Close Top-3": " / ".join(f"{p['number']}({p['confidence']}%)" for p in cp[:3]),
            "Aaj Jodi Top-3": " / ".join(p["jodi"] for p in jd[:3]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=500)
    st.markdown("---")
    st.caption("💡 Tip: Upar search box me site ka naam likho — poora panel expand ho jayega.")


# ===== DETAILED CARDS =====
if search.strip():
    st.markdown(f"### 🎯 Search Result — {len(filtered)} site match")
    auto_expand = True
else:
    st.markdown("### 📋 Detailed View")
    auto_expand = False

for s in filtered:
    show_site_card(s, all_preds[s], expanded=auto_expand)
'''

with open("app.py", "w") as f:
    f.write(NEW_APP)
print("✅ app.py rewritten (clean redesign)")

# ===== 3. Snapshot dedup in panna_predictor =====
with open("panna_predictor.py") as f:
    src = f.read()

old = '''    fname = f"{site.replace(' ', '_')}_{pd_str.replace('/', '_')}.json"
    fpath = os.path.join(SNAPSHOT_DIR, fname)

    def _default(o):'''

new = '''    fname = f"{site.replace(' ', '_')}_{pd_str.replace('/', '_')}.json"
    fpath = os.path.join(SNAPSHOT_DIR, fname)

    # Ek din me ek hi snapshot — overwrite nahi karo
    if os.path.exists(fpath):
        return

    def _default(o):'''

if old in src:
    src = src.replace(old, new, 1)
    with open("panna_predictor.py", "w") as f:
        f.write(src)
    print("✅ panna_predictor.py: snapshot dedup added")
else:
    print("⚠️ snapshot pattern not found")

# ===== 4. Archive junk =====
arch = "scripts/archive"
os.makedirs(arch, exist_ok=True)

junk = [
    "accuracy_boost.py","add_buttons_ui.py","add_panna_pred.py","add_refresh_btn.py",
    "add_score_banner.py","add_single_verify.py","add_voting.py","clean_edge_test.py",
    "daily_auto.py","diag_sridevi.py","diag_timebazar.py","dost_ke_liye.py",
    "expand_predictions.py","final_test.py","fix_both.py","fix_cross.py",
    "fix_date_label.py","fix_datetime_import.py","fix_dup_key.py","fix_import_v2.py",
    "fix_loop_and_remove.py","fix_prediction_date.py","fix_time_bazar.py","fix_verify.py",
    "live_test.py","make_snapshot.py","patch_app.py","patch_clean_result_logic.py",
    "patch_daily.py","patch_full_ui.py","patch_full_ui_v2.py","patch_global_search.py",
    "patch_marks.py","patch_snapshot_result.py","panna_predictor_backup.py",
    "real_edge_test.py","set_config.py","stats_detector.py","test_alts.py",
    "test_kalyan_new.py","test_sites.py","ultra_patch.py",
]
moved = 0
for f in junk:
    if os.path.exists(f):
        shutil.move(f, os.path.join(arch, f))
        moved += 1
print(f"✅ {moved} junk files → {arch}/")

# ===== 5. Syntax =====
py_compile.compile("app.py", doraise=True)
py_compile.compile("panna_predictor.py", doraise=True)
print("✅ Syntax OK")
print()
print("Ab chalao:")
print("  Ctrl+C")
print("  streamlit run app.py")
