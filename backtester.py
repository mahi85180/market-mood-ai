# backtester.py - OPTIMIZED VERSION
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, Input, LayerNormalization,
        MultiHeadAttention, GlobalAveragePooling1D, Add
    )
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class Backtester:
    def __init__(self, df):
        self.df = df.copy()
        self.flat = [int(n) for nums in self.df["numbers"] for n in nums if str(n).isdigit()]

    def backtest_rf(self, initial_train=200, window=5, step_size=20, max_steps=100):
        results = []
        if len(self.flat) < initial_train + window + 5:
            return None
        i = initial_train
        steps_done = 0
        while i < len(self.flat) and steps_done < max_steps:
            train = self.flat[:i]
            X, y = [], []
            for j in range(len(train) - window):
                X.append(train[j:j + window])
                y.append(train[j + window])
            if len(X) < 20:
                i += step_size
                continue
            model = RandomForestClassifier(n_estimators=50, max_depth=8, n_jobs=-1, random_state=42)
            model.fit(X, y)
            for k in range(step_size):
                idx = i + k
                if idx >= len(self.flat):
                    break
                last = self.flat[idx - window:idx]
                if len(last) < window:
                    continue
                probs = model.predict_proba([last])[0]
                classes = model.classes_
                top3 = [int(classes[j]) for j in np.argsort(probs)[::-1][:3]]
                actual = self.flat[idx]
                results.append({
                    "step": idx,
                    "predicted_top1": int(classes[np.argmax(probs)]),
                    "predicted_top3": top3,
                    "actual": actual,
                    "top1_hit": int(int(classes[np.argmax(probs)]) == actual),
                    "top3_hit": int(actual in top3),
                })
            i += step_size
            steps_done += 1
        return pd.DataFrame(results)

    def _make_seq(self, arr, seq_len):
        X, y = [], []
        for i in range(len(arr) - seq_len):
            X.append(arr[i:i + seq_len])
            y.append(arr[i + seq_len])
        return (np.array(X).reshape(-1, seq_len, 1) / 9.0,
                to_categorical(np.array(y), num_classes=10))

    def _build_model(self, kind, seq_len):
        if kind == "LSTM":
            m = Sequential([
                LSTM(32, return_sequences=True, input_shape=(seq_len, 1)),
                Dropout(0.2), LSTM(16), Dense(10, activation="softmax")
            ])
        elif kind == "GRU":
            m = Sequential([
                GRU(32, return_sequences=True, input_shape=(seq_len, 1)),
                Dropout(0.2), GRU(16), Dense(10, activation="softmax")
            ])
        else:
            inp = Input(shape=(seq_len, 1))
            x = Dense(16)(inp)
            pos = tf.range(0, seq_len)
            x = x + tf.keras.layers.Embedding(seq_len, 16)(pos)
            attn = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
            x = LayerNormalization()(Add()([x, attn]))
            ff = Dense(32, activation="relu")(x)
            x = LayerNormalization()(Add()([x, Dense(16)(ff)]))
            x = GlobalAveragePooling1D()(x)
            out = Dense(10, activation="softmax")(x)
            m = Model(inp, out)
        m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return m

    def backtest_neural(self, kind="LSTM", initial_train=300, seq_len=10,
                        epochs=10, step_size=30, max_steps=40):
        if not TF_AVAILABLE:
            return None
        results = []
        if len(self.flat) < initial_train + seq_len + 5:
            return None
        i = initial_train
        steps_done = 0
        while i < len(self.flat) and steps_done < max_steps:
            train = self.flat[:i]
            X, y = self._make_seq(train, seq_len)
            if len(X) < 30:
                i += step_size
                continue
            model = self._build_model(kind, seq_len)
            model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
            for k in range(step_size):
                idx = i + k
                if idx >= len(self.flat):
                    break
                seq = self.flat[idx - seq_len:idx]
                if len(seq) < seq_len:
                    continue
                probs = model.predict(np.array(seq).reshape(1, seq_len, 1) / 9.0, verbose=0)[0]
                top3 = list(np.argsort(probs)[::-1][:3])
                actual = self.flat[idx]
                results.append({
                    "step": idx,
                    "predicted_top1": int(np.argmax(probs)),
                    "predicted_top3": [int(x) for x in top3],
                    "actual": actual,
                    "top1_hit": int(int(np.argmax(probs)) == actual),
                    "top3_hit": int(actual in top3),
                })
            i += step_size
            steps_done += 1
        return pd.DataFrame(results)

    def summarize(self, bt_df, model_name):
        if bt_df is None or bt_df.empty:
            return {"Model": model_name, "Status": "No data"}
        return {
            "Model": model_name,
            "Total Predictions": len(bt_df),
            "Top-1 Accuracy %": round(bt_df["top1_hit"].mean() * 100, 2),
            "Top-3 Accuracy %": round(bt_df["top3_hit"].mean() * 100, 2),
            "Random Baseline %": 10.0,
            "Edge over Random": round((bt_df["top1_hit"].mean() * 100) - 10.0, 2),
        }

    def monte_carlo_baseline(self, n_sims=500):
        if not self.flat:
            return 10.0
        hits = []
        arr = np.array(self.flat)
        for _ in range(n_sims):
            picks = np.random.choice(range(10), size=len(arr))
            hits.append(np.mean(picks == arr))
        return round(np.mean(hits) * 100, 2)
