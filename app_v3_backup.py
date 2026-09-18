# app.py — v3 with manual results + close prediction flow
import streamlit as st
import pandas as pd
import random
import os
from datetime import datetime
from config import SITES
from panna_predictor import predict_all_sites_full, load_snapshot

st.set_page_config(page_title="Market Mood AI", page_icon="🧠", layout="wide")

MANUAL_RESULTS_FILE = "daily_predictions/manual_results.csv"


def load_manual_results():
    if os.path.exists(MANUAL_RESULTS_FILE):
        try:
            return pd.read_csv(MANUAL_RESULTS_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["site", "date", "open", "close", "jodi",
                                 "open_panna", "close_panna", "entered_at"])


def save_manual_result(site, date, open_d, close_d, open_p, close_p):
    df = load_manual_results()
    df = df[~((df["site"] == site) & (df["date"] == date))]
    jodi = f"{open_d}{close_d}"
    row = {
        "site": site, "date": date,
        "open": int(open_d), "close": int(close_d), "jodi": jodi,
        "open_panna": str(open_p), "close_panna": str(close_p),
        "entered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    os.makedirs("daily_predictions", exist_ok=True)
    df.to_csv(MANUAL_RESULTS_FILE, index=False)
    return df


def get_manual_result(site, date):
    df = load_manual_results()
    m = df[(df["site"] == site) & (df["date"] == date)]
    if m.empty:
        return None
    return m.iloc[0].to_dict()


@st.cache_data(ttl=300, show_spinner=False)
def get_predictions():
    try:
        return predict_all_sites_full(SITES, top_n=5)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return {}


def hit(actual, preds, key):
    if actual is None:
        return None
    s = str(actual).strip()
    if s in ["", "-", "nan", "None"]:
        return None
    return s in [str(p.get(key)) for p in preds]


def lbl(h):
    if h is True: return "✅"
    if h is False: return "❌"
    return "—"


# ============ TITLE ============
st.title("🧠 Market Mood AI")

# ============ RANDOM 4 NUMBERS ============
rc1, rc2 = st.columns([1, 4])
with rc1:
    if st.button("🎲 4 Random Numbers", use_container_width=True, type="primary"):
        st.session_state["rand4"] = sorted(random.sample(range(10), 4))
with rc2:
    if "rand4" in st.session_state:
        nums = st.session_state["rand4"]
        st.success(f"**Random 4:** {' · '.join(map(str, nums))}")

st.markdown("---")


# ============ MANUAL RESULT ENTRY ============
st.markdown("## 📝 Aaj Ka Result Daalein")

ec1, ec2 = st.columns(2)
with ec1:
    entry_site = st.selectbox("Site chuno", list(SITES.keys()), key="entry_site")
with ec2:
    entry_date = st.text_input("Date (DD/MM/YYYY)",
                                value=datetime.now().strftime("%d/%m/%Y"),
                                key="entry_date_input")

st.markdown("### 🔓 Open Result")
oc1, oc2 = st.columns(2)
with oc1:
    open_d = st.number_input("Open Digit (0-9)", 0, 9, 0, key="in_open_d")
with oc2:
    open_p = st.text_input("Open Panna (3 digits)", value="000", max_chars=3, key="in_open_p")

# ============ CLOSE PREDICTION (shows after site selected) ============
st.markdown("---")
st.markdown("### 🎯 Is Site Ka Close Prediction")
preds = get_predictions()
pred = preds.get(entry_site)

if pred:
    last_open = pred.get("last_open")
    last_opanna = pred.get("last_open_panna")
    st.caption(f"Last known result: Open {last_open} ({last_opanna}) · "
               f"Based on {pred.get('last_date')}")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**CLOSE (5)**")
        for x in pred["close_prediction"][:5]:
            st.write(f"{x['number']} — {x['confidence']}%")
    with cc2:
        st.markdown("**CLOSE PANNA (20)**")
        for x in pred["close_panna_prediction"][:20]:
            st.write(f"{x['panna']} — {x['confidence']}%")
else:
    st.info("Is site ka data nahi mila — pehle refresh karo.")

st.markdown("---")

# ============ CLOSE RESULT ENTRY ============
st.markdown("### 🔒 Close Result")
cl1, cl2 = st.columns(2)
with cl1:
    close_d = st.number_input("Close Digit (0-9)", 0, 9, 0, key="in_close_d")
with cl2:
    close_p = st.text_input("Close Panna (3 digits)", value="000", max_chars=3, key="in_close_p")

if st.button("💾 Result Save Karo", type="primary"):
    save_manual_result(entry_site, entry_date, int(open_d), int(close_d),
                       open_p, close_p)
    st.success(f"✅ Saved: {entry_site} — {entry_date} — "
               f"Open {open_d}({open_p}) | Close {close_d}({close_p})")
    st.rerun()


# ============ COMPARISON ============
st.markdown("---")
st.markdown("## 📊 Result vs Prediction")

cmp1, cmp2 = st.columns(2)
with cmp1:
    comp_site = st.selectbox("Site", list(SITES.keys()), key="comp_site")
with cmp2:
    comp_date = st.text_input("Date (DD/MM/YYYY)",
                               value=datetime.now().strftime("%d/%m/%Y"),
                               key="comp_date")

manual = get_manual_result(comp_site, comp_date)

if not manual:
    st.info(f"⏳ {comp_site} ka {comp_date} ka result abhi enter nahi hua.")
else:
    snap = load_snapshot(comp_site, comp_date)
    if not snap:
        st.warning(f"⚠️ {comp_site} ka {comp_date} ka snapshot nahi mila — "
                   f"is din ki prediction save nahi hui thi.")
    else:
        ao = manual.get("open")
        ac = manual.get("close")
        aj = manual.get("jodi")
        aop = manual.get("open_panna")
        acp = manual.get("close_panna")

        st.markdown(f"**Actual Result ({comp_date}):**  "
                    f"Open {aop} ({ao}) · Close {acp} ({ac}) · Jodi {aj}")

        po = snap.get("open_prediction", [])
        pc = snap.get("close_prediction", [])
        pj = snap.get("jodi_prediction", [])
        pop = snap.get("open_panna_prediction", [])
        pcp = snap.get("close_panna_prediction", [])

        st.markdown("#### Hit / Miss")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Open", lbl(hit(ao, po, "number")))
        m2.metric("Close", lbl(hit(ac, pc, "number")))
        m3.metric("Jodi", lbl(hit(aj, pj, "jodi")))
        m4.metric("Open Panna", lbl(hit(aop, pop, "panna")))
        m5.metric("Close Panna", lbl(hit(acp, pcp, "panna")))

        with st.expander("🔍 Comparison Details"):
            st.write(f"**Open** — Diya tha: {', '.join(str(p['number']) for p in po)} → Actual: {ao}")
            st.write(f"**Close** — Diya tha: {', '.join(str(p['number']) for p in pc)} → Actual: {ac}")
            st.write(f"**Jodi** — Diya tha: {', '.join(str(p['jodi']) for p in pj)} → Actual: {aj}")
            st.write(f"**Open Panna** — Diya tha: {', '.join(str(p['panna']) for p in pop[:10])}... → Actual: {aop}")
            st.write(f"**Close Panna** — Diya tha: {', '.join(str(p['panna']) for p in pcp[:10])}... → Actual: {acp}")


# ============ HISTORY ============
st.markdown("---")
with st.expander("📋 Manual Results History"):
    hist = load_manual_results()
    if hist.empty:
        st.info("Abhi koi result nahi daala.")
    else:
        st.dataframe(hist.sort_values("entered_at", ascending=False),
                     use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear All Results"):
            try:
                os.remove(MANUAL_RESULTS_FILE)
            except Exception:
                pass
            st.rerun()


# ============ SEARCH ============
st.markdown("---")
st.markdown("## 🔍 Search Predictions")
search = st.text_input("search", key="search_box",
                       placeholder="Site ka naam likho — Kalyan, Milan, Night...",
                       label_visibility="collapsed")

if search.strip():
    q = search.strip().lower()
    matches = [s for s in preds.keys() if q in s.lower()]
    if not matches:
        st.warning(f"'{search}' se koi site match nahi hui.")
    else:
        for s in matches:
            p = preds[s]
            last_date = p.get("last_date", "-")
            predict_date = p.get("predict_for_date", "-")
            with st.expander(f"📊 {s} — Next: {predict_date} (based on {last_date})",
                             expanded=True):
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.markdown("**OPEN (5)**")
                    for x in p["open_prediction"][:5]:
                        st.write(f"{x['number']} — {x['confidence']}%")
                with c2:
                    st.markdown("**CLOSE (5)**")
                    for x in p["close_prediction"][:5]:
                        st.write(f"{x['number']} — {x['confidence']}%")
                with c3:
                    st.markdown("**JODI (10)**")
                    for x in p["jodi_prediction"][:10]:
                        st.write(f"{x['jodi']} — {x['confidence']}%")
                with c4:
                    st.markdown("**OPEN PANNA (20)**")
                    for x in p["open_panna_prediction"][:20]:
                        st.write(f"{x['panna']} — {x['confidence']}%")
                with c5:
                    st.markdown("**CLOSE PANNA (20)**")
                    for x in p["close_panna_prediction"][:20]:
                        st.write(f"{x['panna']} — {x['confidence']}%")
