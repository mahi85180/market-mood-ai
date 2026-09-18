# app.py — v4 with ML tab + manual results
import streamlit as st
import pandas as pd
import random
import os
from datetime import datetime
from config import SITES
from scraper import update_site, load_history, update_all_sites
from panna_predictor import predict_all_sites_full, load_snapshot
from analyzer import MarketAnalyzer
from ml_predictor import MLPredictor, TF_AVAILABLE

st.set_page_config(
    page_title="Market Mood AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================== MOBILE-FRIENDLY CSS ====================
st.markdown("""
<style>
/* Mobile-friendly adjustments */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        padding-top: 0.5rem !important;
    }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.1rem !important; }
    h4 { font-size: 1rem !important; }
    /* Bigger tap targets */
    .stButton button {
        min-height: 48px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    /* Number inputs bigger */
    .stNumberInput input, .stTextInput input {
        font-size: 18px !important;
        min-height: 44px !important;
    }
    /* Tabs bigger */
    .stTabs [data-baseweb="tab"] {
        padding: 10px 12px !important;
        font-size: 14px !important;
    }
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    /* Expander header */
    .streamlit-expanderHeader {
        font-size: 15px !important;
        padding: 12px !important;
    }
    /* Selectbox bigger */
    .stSelectbox div[data-baseweb="select"] {
        min-height: 44px !important;
    }
}

/* Desktop tweaks */
.stButton button {
    border-radius: 8px;
}

/* Make prediction lists compact */
.compact-list {
    line-height: 1.6;
    font-size: 14px;
}

/* Tabs sticky on mobile */
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        overflow-x: auto;
        flex-wrap: nowrap;
    }
}
</style>
""", unsafe_allow_html=True)


def is_mobile():
    """Detect mobile via user agent."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        if headers:
            ua = headers.get("User-Agent", "").lower()
            return any(x in ua for x in ["mobile", "android", "iphone", "ipad"])
    except Exception:
        pass
    return False

MANUAL_RESULTS_FILE = "daily_predictions/manual_results.csv"


# ==================== HELPERS ====================
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
    if h is True: return "✅ HIT"
    if h is False: return "❌ Miss"
    return "—"


# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Controls")
sel_site = st.sidebar.selectbox("Site chuno", list(SITES.keys()))
if st.sidebar.button("📥 Fetch Site", type="primary"):
    with st.spinner(f"{sel_site} fetch ho rahi hai..."):
        _, msg = update_site(sel_site)
        st.sidebar.success(msg)

if st.sidebar.button("🌐 Fetch All Sites"):
    with st.spinner("Saari sites fetch ho rahi hain..."):
        update_all_sites()
        st.cache_data.clear()
        st.sidebar.success("Done!")
        st.rerun()

if TF_AVAILABLE:
    st.sidebar.success("✅ TensorFlow ready")
else:
    st.sidebar.warning("⚠️ TF missing — sirf RF chalega")

st.sidebar.caption("Statistical tool. Prediction guarantee nahi.")


# ==================== TITLE ====================
st.title("🧠 Market Mood AI")

# Random button
if st.button("🎲 4 Random Numbers", use_container_width=True, type="primary"):
    st.session_state["rand4"] = sorted(random.sample(range(10), 4))
if "rand4" in st.session_state:
    st.success(f"**Random 4:** {' · '.join(map(str, st.session_state['rand4']))}")

st.markdown("---")


# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Result Entry", "🤖 ML Training", "🔍 Search", "📊 History"
])


# ==================== TAB 1: MANUAL RESULT ====================
with tab1:
    st.markdown("### 📝 Aaj Ka Result Daalein")

    ec1, ec2 = st.columns(2)
    with ec1:
        entry_site = st.selectbox("Site", list(SITES.keys()), key="entry_site")
    with ec2:
        entry_date = st.text_input("Date (DD/MM/YYYY)",
                                    value=datetime.now().strftime("%d/%m/%Y"),
                                    key="entry_date_input")

    st.markdown("#### 🔓 Open Result")
    oc1, oc2 = st.columns(2)
    with oc1:
        open_d = st.number_input("Open Digit", 0, 9, 0, key="in_open_d")
    with oc2:
        open_p = st.text_input("Open Panna (3 digits)", value="000", max_chars=3,
                                key="in_open_p")

    st.markdown("---")
    st.markdown("### 🎯 Is Site Ka Close Prediction")

    preds = get_predictions()
    pred = preds.get(entry_site)
    if pred:
        st.caption(f"Based on {pred.get('last_date')}: Open {pred.get('last_open')} "
                   f"({pred.get('last_open_panna')})")
        if is_mobile():
            st.markdown("**CLOSE (5)**")
            st.write(" · ".join(f"{x['number']}({x['confidence']}%)" for x in pred["close_prediction"][:5]))
            st.markdown("**CLOSE PANNA (20)**")
            st.write(" · ".join(f"{x['panna']}({x['confidence']}%)" for x in pred["close_panna_prediction"][:20]))
        else:
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
        st.info("Is site ka data nahi mila — sidebar se fetch karo.")

    st.markdown("---")
    st.markdown("#### 🔒 Close Result")
    cl1, cl2 = st.columns(2)
    with cl1:
        close_d = st.number_input("Close Digit", 0, 9, 0, key="in_close_d")
    with cl2:
        close_p = st.text_input("Close Panna (3 digits)", value="000", max_chars=3,
                                 key="in_close_p")

    if st.button("💾 Result Save Karo", type="primary"):
        save_manual_result(entry_site, entry_date, int(open_d), int(close_d),
                           open_p, close_p)
        st.success(f"✅ Saved: {entry_site} — {entry_date} — "
                   f"Open {open_d}({open_p}) | Close {close_d}({close_p})")
        st.rerun()


# ==================== TAB 2: ML TRAINING ====================
with tab2:
    st.markdown("### 🤖 ML Model Training")
    st.caption("Selected site ke data pe RF + LSTM + GRU + Transformer train karo.")

    df = load_history(sel_site)
    if df.empty:
        st.warning(f"{sel_site} ka data nahi hai. Sidebar se fetch karo.")
    else:
        analyzer = MarketAnalyzer(df)
        st.info(f"**{sel_site}** — {len(df)} rows, "
                f"{len(analyzer.all_numbers)} numbers")

        c1, c2, c3 = st.columns(3)
        window = c1.number_input("RF Window", 3, 20, 5, key="rf_w")
        seq_len = c2.number_input("Seq Length", 5, 30, 10, key="seq_l")
        epochs = c3.number_input("Epochs", 10, 100, 30, key="ep")

        if st.button("🚀 Train All Models", type="primary"):
            predictor = MLPredictor(df)
            with st.spinner("Random Forest training..."):
                _, msg = predictor.train_rf(window)
                st.success(msg)

            if TF_AVAILABLE:
                with st.spinner("LSTM training (30-60s)..."):
                    _, msg = predictor.train_lstm(seq_len, epochs)
                    st.info(msg)
                with st.spinner("GRU training..."):
                    _, msg = predictor.train_gru(seq_len, epochs)
                    st.info(msg)
                with st.spinner("Transformer training..."):
                    _, msg = predictor.train_transformer(seq_len, epochs)
                    st.info(msg)
            else:
                st.warning("TensorFlow nahi hai — sirf RF chalega.")

            st.session_state.predictor = predictor
            st.session_state.ml_trained = True
            st.session_state.ml_site = sel_site

        if st.session_state.get("ml_trained") and st.session_state.get("ml_site") == sel_site:
            predictor = st.session_state.predictor
            st.markdown("#### 📊 Model Accuracy")
            summary = predictor.model_summary()
            sdf = pd.DataFrame([{"Model": k, "Accuracy %": v}
                                 for k, v in summary.items() if v is not None])
            if not sdf.empty:
                st.dataframe(sdf, use_container_width=True, hide_index=True)

                import plotly.express as px
                st.plotly_chart(px.bar(sdf, x="Model", y="Accuracy %",
                                       color="Accuracy %",
                                       color_continuous_scale="Viridis",
                                       text="Accuracy %"),
                                use_container_width=True)

            st.markdown("#### 🔮 Combined Prediction (Top 5)")
            combined = predictor.combined_prediction(window, seq_len)
            if combined:
                for k, v in list(combined.items())[:5]:
                    st.write(f"**{k}** — {v}%")


# ==================== TAB 3: SEARCH ====================
with tab3:
    st.markdown("### 🔍 Site Search — Full Predictions")

    search = st.text_input("search", key="search_box",
                            placeholder="Site ka naam likho — Kalyan, Milan...",
                            label_visibility="collapsed")

    preds = get_predictions()

    if search.strip():
        q = search.strip().lower()
        matches = [s for s in preds.keys() if q in s.lower()]
        if not matches:
            st.warning(f"'{search}' se koi site match nahi hui.")
        else:
            for s in matches:
                p = preds[s]
                with st.expander(
                    f"📊 {s} — Next: {p.get('predict_for_date')} "
                    f"(based on {p.get('last_date')})", expanded=True):
                    _mob = is_mobile()
                    if _mob:
                        # Stack: Open+Close row, Jodi row, Panna rows
                        r1c1, r1c2 = st.columns(2)
                        with r1c1:
                            st.markdown("**OPEN (5)**")
                            for x in p["open_prediction"][:5]:
                                st.write(f"{x['number']} — {x['confidence']}%")
                        with r1c2:
                            st.markdown("**CLOSE (5)**")
                            for x in p["close_prediction"][:5]:
                                st.write(f"{x['number']} — {x['confidence']}%")
                        st.markdown("**JODI (10)**")
                        st.write(" · ".join(f"{x['jodi']}({x['confidence']}%)" for x in p["jodi_prediction"][:10]))
                        st.markdown("**OPEN PANNA (20)**")
                        st.write(" · ".join(f"{x['panna']}({x['confidence']}%)" for x in p["open_panna_prediction"][:20]))
                        st.markdown("**CLOSE PANNA (20)**")
                        st.write(" · ".join(f"{x['panna']}({x['confidence']}%)" for x in p["close_panna_prediction"][:20]))
                    else:
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
    else:
        st.info("👆 Upar site ka naam likho — poora panel dikhega.")


# ==================== TAB 4: HISTORY ====================
with tab4:
    st.markdown("### 📊 Comparison — Actual vs Prediction")

    cmp1, cmp2 = st.columns(2)
    with cmp1:
        comp_site = st.selectbox("Site", list(SITES.keys()), key="comp_site")
    with cmp2:
        comp_date = st.text_input("Date (DD/MM/YYYY)",
                                   value=datetime.now().strftime("%d/%m/%Y"),
                                   key="comp_date")

    manual = get_manual_result(comp_site, comp_date)

    if not manual:
        st.info(f"⏳ {comp_site} — {comp_date} ka result abhi enter nahi hua.")
    else:
        snap = load_snapshot(comp_site, comp_date)
        if not snap:
            st.warning(f"⚠️ {comp_date} ka snapshot nahi mila — "
                       f"is din prediction save nahi hui thi.")
        else:
            ao, ac = manual.get("open"), manual.get("close")
            aj = manual.get("jodi")
            aop, acp = manual.get("open_panna"), manual.get("close_panna")

            st.markdown(f"**Actual ({comp_date}):** "
                        f"Open {aop} ({ao}) · Close {acp} ({ac}) · Jodi {aj}")

            po = snap.get("open_prediction", [])
            pc = snap.get("close_prediction", [])
            pj = snap.get("jodi_prediction", [])
            pop = snap.get("open_panna_prediction", [])
            pcp = snap.get("close_panna_prediction", [])

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Open", lbl(hit(ao, po, "number")))
            m2.metric("Close", lbl(hit(ac, pc, "number")))
            m3.metric("Jodi", lbl(hit(aj, pj, "jodi")))
            m4.metric("Open Panna", lbl(hit(aop, pop, "panna")))
            m5.metric("Close Panna", lbl(hit(acp, pcp, "panna")))

            with st.expander("🔍 Details"):
                st.write(f"**Open** diya: {', '.join(str(p['number']) for p in po)}")
                st.write(f"**Close** diya: {', '.join(str(p['number']) for p in pc)}")
                st.write(f"**Jodi** diya: {', '.join(str(p['jodi']) for p in pj)}")

    st.markdown("---")
    with st.expander("📋 Manual Results History"):
        hist = load_manual_results()
        if hist.empty:
            st.info("Abhi koi result nahi daala.")
        else:
            st.dataframe(hist.sort_values("entered_at", ascending=False),
                         use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear All"):
                try:
                    os.remove(MANUAL_RESULTS_FILE)
                except Exception:
                    pass
                st.rerun()
