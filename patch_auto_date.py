# patch_auto_date.py
import py_compile

with open("panna_predictor.py", "r") as f:
    src = f.read()

# --- Replace parse_with_dates with auto-shift version ---
old_fn = '''def parse_with_dates(numbers, week_start_str):
    """Parse with date tracking. Week start like '14/09/2026 to 19/09/2026'"""
    from datetime import datetime, timedelta
    import re
    match = re.search(r"(\\d{2})/(\\d{2})/(\\d{4})", str(week_start_str))
    if not match:
        return parse_row_to_days(numbers)
    try:
        d, m, y = int(match.group(1)), int(match.group(2)), int(match.group(3))
        start = datetime(y, m, d)
    except Exception:
        return parse_row_to_days(numbers)
    days = []
    i = 0
    day_index = 0
    while i + 8 <= len(numbers):
        chunk = [str(x) for x in numbers[i:i+8]]
        op, jd, cp = chunk[0:3], chunk[3:5], chunk[5:8]
        try:
            open_digit = sum(int(x) for x in op) % 10
            close_digit = sum(int(x) for x in cp) % 10
            day_date = start + timedelta(days=day_index)
            days.append({
                "open_panna": "".join(op),
                "jodi": f"{open_digit}{close_digit}",
                "close_panna": "".join(cp),
                "open": open_digit,
                "close": close_digit,
                "date": day_date.strftime("%d/%m/%Y"),
                "iso_date": day_date.strftime("%Y-%m-%d"),
            })
        except (ValueError, IndexError):
            pass
        i += 8
        day_index += 1
    return days'''

new_fn = '''def parse_with_dates(numbers, week_start_str, is_current_week=False):
    """Parse with date tracking + auto-shift when week has closed days at start.
    is_current_week=True → auto-detect start gaps and shift dates forward.
    """
    from datetime import datetime, timedelta
    import re
    match = re.search(r"(\\d{2})/(\\d{2})/(\\d{4})", str(week_start_str))
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

    return days'''

if old_fn not in src:
    print("❌ Old parse_with_dates not found. Manually check panna_predictor.py")
    exit(1)

src = src.replace(old_fn, new_fn, 1)

# --- Modify build_all_sequences to mark last row as current week ---
old_build = '''    for _, row in df.iterrows():
        nums = row["numbers"]
        date_str = row.get("date", "")
        for d in parse_with_dates(nums, date_str):'''

new_build = '''    last_row_idx = df.index[-1]
    for idx, row in df.iterrows():
        nums = row["numbers"]
        date_str = row.get("date", "")
        is_current = (idx == last_row_idx)
        for d in parse_with_dates(nums, date_str, is_current_week=is_current):'''

if old_build not in src:
    print("❌ build_all_sequences loop not found")
    exit(1)

src = src.replace(old_build, new_build, 1)

with open("panna_predictor.py", "w") as f:
    f.write(src)

py_compile.compile("panna_predictor.py", doraise=True)
print("✅ panna_predictor.py patched — auto date shift added")
print("✅ Syntax OK")
print()
print("Ab chalao: Ctrl+C  →  streamlit run app.py")
