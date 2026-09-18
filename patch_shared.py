# patch_shared.py — Add refresh button to both apps
import py_compile


def add_refresh(filename, key_suffix):
    with open(filename) as f:
        src = f.read()

    old = '''with tab4:
    st.markdown("### 📊 Comparison — Actual vs Prediction")'''

    new = f'''with tab4:
    _r1, _r2 = st.columns([1, 4])
    with _r1:
        if st.button("🔄 Refresh", key="refresh_{key_suffix}", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with _r2:
        st.caption("💡 Doosri app me result save kiya? Ye button dabao.")
    st.markdown("### 📊 Comparison — Actual vs Prediction")'''

    if old in src:
        src = src.replace(old, new, 1)
        print(f"✅ {filename}: refresh button added")
    else:
        print(f"⚠️ {filename}: pattern not found — check manually")
        return

    # Also add refresh to Result Entry tab so saved results visible everywhere
    old2 = '''    st.markdown("---")
    with st.expander("📋 Manual Results History"):'''

    new2 = '''    st.markdown("---")
    with st.expander("📋 Manual Results History", expanded=False):
        if st.button("🔄 Refresh History", key="refresh_hist_{key_suffix}"):
            st.cache_data.clear()
            st.rerun()'''

    if old2 in src:
        src = src.replace(old2, new2, 1)
        print(f"✅ {filename}: history refresh added")

    with open(filename, "w") as f:
        f.write(src)

    py_compile.compile(filename, doraise=True)
    print(f"✅ {filename}: syntax OK")


add_refresh("app.py", "rf")
add_refresh("app_tf.py", "tf")
print()
print("Ab chalao:")
print("  Ctrl+C dono terminals me")
print("  streamlit run app.py                    (8501)")
print("  streamlit run app_tf.py --server.port 8503")
