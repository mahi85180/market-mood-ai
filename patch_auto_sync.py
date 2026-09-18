# patch_auto_sync.py — Auto-refresh both apps
import py_compile


def patch_file(filename):
    with open(filename) as f:
        src = f.read()

    # 1. Add import after existing imports
    old_imp = "import streamlit as st\nimport pandas as pd"
    new_imp = ("import streamlit as st\n"
               "import pandas as pd\n"
               "from streamlit_autorefresh import st_autorefresh")

    if "streamlit_autorefresh" not in src:
        if old_imp in src:
            src = src.replace(old_imp, new_imp, 1)
            print(f"✅ {filename}: import added")
        else:
            print(f"❌ {filename}: import anchor not found")
            return
    else:
        print(f"⏭️ {filename}: import already there")

    # 2. Add auto-refresh counter at top of app (after title)
    old_title = 'st.title("🧠 Market Mood AI")\n\n# Random button'
    new_title = ('st.title("🧠 Market Mood AI")\n\n'
                 '# ===== AUTO-SYNC (har 5 sec) =====\n'
                 '_sync_count = st_autorefresh(interval=5000, key="autosync")\n'
                 '# ===== /AUTO-SYNC =====\n\n'
                 '# Random button')

    if "AUTO-SYNC" not in src:
        if old_title in src:
            src = src.replace(old_title, new_title, 1)
            print(f"✅ {filename}: auto-refresh added (title)")
        else:
            # Try alt anchor for app_tf
            old_title_tf = 'st.title("🤖 Market Mood AI — TF Version")'
            new_title_tf = ('st.title("🤖 Market Mood AI — TF Version")\n\n'
                            '# ===== AUTO-SYNC (har 5 sec) =====\n'
                            '_sync_count = st_autorefresh(interval=5000, key="autosync")\n'
                            '# ===== /AUTO-SYNC =====')
            if old_title_tf in src:
                src = src.replace(old_title_tf, new_title_tf, 1)
                print(f"✅ {filename}: auto-refresh added (tf title)")
            else:
                print(f"⚠️ {filename}: title anchor not found — add manually")
                return
    else:
        print(f"⏭️ {filename}: auto-refresh already there")

    with open(filename, "w") as f:
        f.write(src)

    py_compile.compile(filename, doraise=True)
    print(f"✅ {filename}: syntax OK\n")


print("=" * 60)
print("PATCHING BOTH APPS")
print("=" * 60)
patch_file("app.py")
patch_file("app_tf.py")
print("=" * 60)
print("Ab chalao:")
print("  Ctrl+C dono terminals me")
print("  streamlit run app.py                    (8501)")
print("  streamlit run app_tf.py --server.port 8503")
print("=" * 60)

