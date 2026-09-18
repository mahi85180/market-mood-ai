# patch_stale_warning.py
with open("app.py", "r") as f:
    src = f.read()

# Add helper function after imports
old_helpers = '''def label(h):
    if h is True:
        return "✅ HIT"
    if h is False:
        return "❌ Miss"
    return "—"
'''
new_helpers = '''def label(h):
    if h is True:
        return "✅ HIT"
    if h is False:
        return "❌ Miss"
    return "—"


def days_behind(last_date_str):
    """Returns how many days behind today's date. None if unparseable."""
    try:
        from datetime import datetime
        ld = datetime.strptime(last_date_str, "%d/%m/%Y").date()
        return (datetime.now().date() - ld).days
    except Exception:
        return None


def stale_badge(last_date_str):
    """Return warning string if stale, else empty."""
    n = days_behind(last_date_str)
    if n is None:
        return ""
    if n >= 2:
        return f"  ⚠️ DATA {n} DAYS OLD"
    if n == 1:
        return "  ·  1 day old"
    return ""
'''
assert old_helpers in src, "helper block not found"
src = src.replace(old_helpers, new_helpers, 1)

# Update header line in show_site_card
old_header = '''    with st.expander(f"📊 {site_name}  —  Next: {predict_date}  (based on {last_date})", expanded=expanded):'''
new_header = '''    _badge = stale_badge(last_date)
    _header_txt = f"📊 {site_name}  —  Next: {predict_date}  (based on {last_date}){_badge}"
    with st.expander(_header_txt, expanded=expanded):'''
assert old_header in src, "header line not found"
src = src.replace(old_header, new_header, 1)

# Add stale notice inside card, before "Kal Ka Result"
old_inner = '''        # ---- Kal ka result ----
        snap = load_snapshot(site_name, last_date) if last_date != "-" else None'''
new_inner = '''        # Stale warning
        _n = days_behind(last_date)
        if _n is not None and _n >= 2:
            st.warning(
                f"⚠️ **{site_name}** ka website data **{_n} din purana** hai. "
                f"Latest result {last_date} ka hai. "
                f"Ye predictions purane data pe based hain — accuracy low ho sakti hai. "
                f"Jab site update hogi, 'Refresh Data' dabao."
            )

        # ---- Kal ka result ----
        snap = load_snapshot(site_name, last_date) if last_date != "-" else None'''
assert old_inner in src, "inner block not found"
src = src.replace(old_inner, new_inner, 1)

# Update quick table to show stale badge
old_table = '''        rows.append({
            "Site": s,
            "Kal Open": yo,
            "Kal Close": yc,
            "Kal Jodi": yj,'''
new_table = '''        _n = days_behind(p.get("last_date", "-"))
        _stale = f"⚠️ {_n}d" if (_n is not None and _n >= 2) else ""
        rows.append({
            "Site": s,
            "Status": _stale,
            "Kal Open": yo,
            "Kal Close": yc,
            "Kal Jodi": yj,'''
assert old_table in src, "table block not found"
src = src.replace(old_table, new_table, 1)

with open("app.py", "w") as f:
    f.write(src)

import py_compile
py_compile.compile("app.py", doraise=True)
print("✅ app.py patched — stale warning added")
print()
print("Ab chalao: Ctrl+C  →  streamlit run app.py")
