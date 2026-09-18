# ml_predictor.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# TensorFlow optional - agar install nahi hai to skip
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


class MLPredictor:
    def __init__(self, df):
        self.df = df.copy()
        self.flat = [int(n) for nums in self.df["numbers"] for n in nums if str(n).isdigit()]
        self.rf_model = None
        self.lstm_model = None
        self.gru_model = None
        self.transformer_model = None
        self.rf_accuracy = None
        self.lstm_accuracy = None
        self.gru_accuracy = None
        self.transformer_accuracy = None

    # ---------- RANDOM FOREST ----------
    def _make_rf_dataset(self, window=5):
        X, y = [], []
        for i in range(len(self.flat) - window):
            X.append(self.flat[i:i + window])
            y.append(self.flat[i + window])
        return np.array(X), np.array(y)

    def train_rf(self, window=5):
        X, y = self._make_rf_dataset(window)
        if len(X) < 30:
            return None, "Data bahut kam (30+ chahiye)"
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        self.rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
        self.rf_model.fit(X_tr, y_tr)
        self.rf_accuracy = round(accuracy_score(y_te, self.rf_model.predict(X_te)) * 100, 2)
        joblib.dump(self.rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
        return self.rf_model, f"RF: {self.rf_accuracy}%"

    def predict_rf(self, last_n):
        if self.rf_model is None:
            return {}
        probs = self.rf_model.predict_proba([last_n])[0]
        classes = self.rf_model.classes_
        top = np.argsort(probs)[::-1][:3]
        return {int(classes[i]): round(probs[i] * 100, 1) for i in top}

    # ---------- SEQUENCE DATA ----------
    def _make_seq_dataset(self, seq_len=10):
        X, y = [], []
        for i in range(len(self.flat) - seq_len):
            X.append(self.flat[i:i + seq_len])
            y.append(self.flat[i + seq_len])
        X = np.array(X).reshape(-1, seq_len, 1) / 9.0
        y_cat = to_categorical(np.array(y), num_classes=10)
        return X, y_cat

    # ---------- LSTM ----------
    def train_lstm(self, seq_len=10, epochs=30):
        if not TF_AVAILABLE:
            return None, "TensorFlow install nahi hai"
        X, y = self._make_seq_dataset(seq_len)
        if len(X) < 50:
            return None, "50+ entries chahiye"
        split = int(len(X) * 0.8)
        self.lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
            Dropout(0.2),
            LSTM(32), Dropout(0.2),
            Dense(10, activation="softmax")
        ])
        self.lstm_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.lstm_model.fit(X[:split], y[:split], epochs=epochs, batch_size=16,
                            verbose=0, validation_split=0.1)
        _, acc = self.lstm_model.evaluate(X[split:], y[split:], verbose=0)
        self.lstm_accuracy = round(acc * 100, 2)
        self.lstm_model.save(os.path.join(MODEL_DIR, "lstm.keras"))
        return self.lstm_model, f"LSTM: {self.lstm_accuracy}%"

    def predict_lstm(self, seq):
        if self.lstm_model is None:
            return {}
        p = self.lstm_model.predict(np.array(seq).reshape(1, len(seq), 1) / 9.0, verbose=0)[0]
        top = np.argsort(p)[::-1][:3]
        return {int(i): round(p[i] * 100, 1) for i in top}

    # ---------- GRU ----------
    def train_gru(self, seq_len=10, epochs=30):
        if not TF_AVAILABLE:
            return None, "TensorFlow install nahi hai"
        X, y = self._make_seq_dataset(seq_len)
        if len(X) < 50:
            return None, "50+ entries chahiye"
        split = int(len(X) * 0.8)
        self.gru_model = Sequential([
            GRU(64, return_sequences=True, input_shape=(seq_len, 1)),
            Dropout(0.2),
            GRU(32), Dropout(0.2),
            Dense(10, activation="softmax")
        ])
        self.gru_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        self.gru_model.fit(X[:split], y[:split], epochs=epochs, batch_size=16,
                           verbose=0, validation_split=0.1)
        _, acc = self.gru_model.evaluate(X[split:], y[split:], verbose=0)
        self.gru_accuracy = round(acc * 100, 2)
        self.gru_model.save(os.path.join(MODEL_DIR, "gru.keras"))
        return self.gru_model, f"GRU: {self.gru_accuracy}%"

    def predict_gru(self, seq):
        if self.gru_model is None:
            return {}
        p = self.gru_model.predict(np.array(seq).reshape(1, len(seq), 1) / 9.0, verbose=0)[0]
        top = np.argsort(p)[::-1][:3]
        return {int(i): round(p[i] * 100, 1) for i in top}

    # ---------- TRANSFORMER ----------
    def _build_transformer(self, seq_len):
        inputs = Input(shape=(seq_len, 1))
        x = Dense(32)(inputs)
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=32)(positions)
        x = x + pos_emb
        attn = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = Add()([x, attn])
        x = LayerNormalization()(x)
        ff = Dense(64, activation="relu")(x)
        ff = Dense(32)(ff)
        x = Add()([x, ff])
        x = LayerNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(10, activation="softmax")(x)
        model = Model(inputs, outputs)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def train_transformer(self, seq_len=10, epochs=30):
        if not TF_AVAILABLE:
            return None, "TensorFlow install nahi hai"
        X, y = self._make_seq_dataset(seq_len)
        if len(X) < 50:
            return None, "50+ entries chahiye"
        split = int(len(X) * 0.8)
        self.transformer_model = self._build_transformer(seq_len)
        self.transformer_model.fit(X[:split], y[:split], epochs=epochs, batch_size=16,
                                   verbose=0, validation_split=0.1)
        _, acc = self.transformer_model.evaluate(X[split:], y[split:], verbose=0)
        self.transformer_accuracy = round(acc * 100, 2)
        self.transformer_model.save(os.path.join(MODEL_DIR, "transformer.keras"))
        return self.transformer_model, f"Transformer: {self.transformer_accuracy}%"

    def predict_transformer(self, seq):
        if self.transformer_model is None:
            return {}
        p = self.transformer_model.predict(np.array(seq).reshape(1, len(seq), 1) / 9.0, verbose=0)[0]
        top = np.argsort(p)[::-1][:3]
        return {int(i): round(p[i] * 100, 1) for i in top}

    # ---------- COMBINED ----------
    def combined_prediction(self, window=5, seq_len=10):
        result = {}
        if self.rf_model is not None:
            for k, v in self.predict_rf(self.flat[-window:]).items():
                result.setdefault(k, {})["RF"] = v
        if self.lstm_model is not None:
            for k, v in self.predict_lstm(self.flat[-seq_len:]).items():
                result.setdefault(k, {})["LSTM"] = v
        if self.gru_model is not None:
            for k, v in self.predict_gru(self.flat[-seq_len:]).items():
                result.setdefault(k, {})["GRU"] = v
        if self.transformer_model is not None:
            for k, v in self.predict_transformer(self.flat[-seq_len:]).items():
                result.setdefault(k, {})["Transformer"] = v
        final = {num: round(sum(s.values()) / len(s), 1) for num, s in result.items()}
        return dict(sorted(final.items(), key=lambda x: -x[1]))

    def model_summary(self):
        return {
            "Random Forest": self.rf_accuracy,
            "LSTM": self.lstm_accuracy,
            "GRU": self.gru_accuracy,
            "Transformer": self.transformer_accuracy,
        }
