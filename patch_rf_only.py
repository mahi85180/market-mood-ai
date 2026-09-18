# patch_rf_only.py
import py_compile

with open("app.py") as f:
    src = f.read()

# Find the ML training section and replace with RF-only
old_block = '''        if st.button("🚀 Train All Models", type="primary"):
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
            st.session_state.ml_site = sel_site'''

new_block = '''        st.info("💡 Ye RF-only version hai (cloud ke liye). Full TF version ke liye `app_tf.py` chalao.")

        if st.button("🚀 Train RF Model", type="primary"):
            predictor = MLPredictor(df)
            with st.spinner("🌲 Random Forest training..."):
                _, msg = predictor.train_rf(window)
                st.success(msg)

            st.session_state.predictor = predictor
            st.session_state.ml_trained = True
            st.session_state.ml_site = sel_site
            st.success("✅ RF trained!")'''

if old_block in src:
    src = src.replace(old_block, new_block, 1)
    print("✅ ML tab → RF only")
else:
    print("❌ Old ML block not found — structure alag hai")

# Also update the summary section — only show RF
old_summary = '''            st.markdown("#### 📊 Model Accuracy")
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
                                use_container_width=True)'''

new_summary = '''            st.markdown("#### 📊 RF Accuracy")
            summary = predictor.model_summary()
            sdf = pd.DataFrame([{"Model": k, "Accuracy %": v}
                                 for k, v in summary.items() if v is not None])
            if not sdf.empty:
                st.dataframe(sdf, use_container_width=True, hide_index=True)'''

if old_summary in src:
    src = src.replace(old_summary, new_summary, 1)
    print("✅ Summary → RF only")
else:
    print("⚠️ Summary block not found (skip)")

# Sidebar — always show RF-only message
old_sb = '''if TF_AVAILABLE:
    st.sidebar.success("✅ TensorFlow ready")
else:
    st.sidebar.warning("⚠️ TF missing — sirf RF chalega")'''

new_sb = '''st.sidebar.info("🌲 RF-only version (cloud-compatible)")'''

if old_sb in src:
    src = src.replace(old_sb, new_sb, 1)
    print("✅ Sidebar → RF-only message")
else:
    print("⚠️ Sidebar block not found (skip)")

with open("app.py", "w") as f:
    f.write(src)

py_compile.compile("app.py", doraise=True)
print("✅ app.py patched")
print("✅ Syntax OK")
print()
print("Ab:")
print("  8501 (app.py)      → RF only")
print("  8502 (app_tf.py)   → Full TF")
