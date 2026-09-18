# patch_mobile.py
import py_compile

with open("app.py") as f:
    src = f.read()

# ===== 1. Add mobile CSS + detection right after set_page_config =====
old_after = '''st.set_page_config(page_title="Market Mood AI", page_icon="🧠", layout="wide")
'''
new_after = '''st.set_page_config(
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
'''
assert old_after in src
src = src.replace(old_after, new_after, 1)

# ===== 2. Change 5-column layout to responsive =====
# In Search tab:
old_search_cols = '''                    c1, c2, c3, c4, c5 = st.columns(5)
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
                            st.write(f"{x['panna']} — {x['confidence']}%")'''

new_search_cols = '''                    _mob = is_mobile()
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
                                st.write(f"{x['panna']} — {x['confidence']}%")'''

assert old_search_cols in src, "search cols block not found"
src = src.replace(old_search_cols, new_search_cols, 1)

# ===== 3. Close Prediction section in Result Entry — mobile stack =====
old_close = '''        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**CLOSE (5)**")
            for x in pred["close_prediction"][:5]:
                st.write(f"{x['number']} — {x['confidence']}%")
        with cc2:
            st.markdown("**CLOSE PANNA (20)**")
            for x in pred["close_panna_prediction"][:20]:
                st.write(f"{x['panna']} — {x['confidence']}%")'''

new_close = '''        if is_mobile():
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
                    st.write(f"{x['panna']} — {x['confidence']}%")'''

assert old_close in src, "close pred block not found"
src = src.replace(old_close, new_close, 1)

# ===== 4. Random button + title layout — better on mobile =====
old_random = '''rc1, rc2 = st.columns([1, 4])
with rc1:
    if st.button("🎲 4 Random Numbers", use_container_width=True, type="primary"):
        st.session_state["rand4"] = sorted(random.sample(range(10), 4))
with rc2:
    if "rand4" in st.session_state:
        st.success(f"**Random 4:** {' · '.join(map(str, st.session_state['rand4']))}")'''

new_random = '''if st.button("🎲 4 Random Numbers", use_container_width=True, type="primary"):
    st.session_state["rand4"] = sorted(random.sample(range(10), 4))
if "rand4" in st.session_state:
    st.success(f"**Random 4:** {' · '.join(map(str, st.session_state['rand4']))}")'''

assert old_random in src, "random block not found"
src = src.replace(old_random, new_random, 1)

with open("app.py", "w") as f:
    f.write(src)

py_compile.compile("app.py", doraise=True)
print("✅ Mobile-friendly patch applied")
print("✅ Syntax OK")
print()
print("Ab chalao:")
print("  git add . && git commit -m 'Mobile-friendly UI' && git push")
print("Phir Streamlit auto-redeploy karega (1-2 min)")
