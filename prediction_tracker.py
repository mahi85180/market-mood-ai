# prediction_tracker.py
import pandas as pd
import os
import re
from datetime import datetime

TRACKER_DIR = "predictions"
os.makedirs(TRACKER_DIR, exist_ok=True)


def _safe(name):
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


class PredictionTracker:
    def __init__(self, site_name):
        self.site_name = site_name
        self.file = os.path.join(TRACKER_DIR, f"{_safe(site_name)}.csv")

    def _empty(self):
        return pd.DataFrame(columns=[
            "id", "timestamp", "model", "predict_for_index",
            "predicted_top1", "predicted_top3", "confidence",
            "actual", "top1_hit", "top3_hit", "verified"
        ])

    def load(self):
        if os.path.exists(self.file):
            try:
                return pd.read_csv(self.file)
            except Exception:
                return self._empty()
        return self._empty()

    def save(self, df):
        df.to_csv(self.file, index=False)

    def log(self, model_name, predicted_top1, predicted_top3,
            predict_for_index, confidence=None):
        df = self.load()
        new_id = int(df["id"].max()) + 1 if not df.empty else 1
        row = {
            "id": new_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "predict_for_index": int(predict_for_index),
            "predicted_top1": int(predicted_top1),
            "predicted_top3": ",".join(map(str, predicted_top3)),
            "confidence": float(confidence) if confidence is not None else 0.0,
            "actual": -1,
            "top1_hit": -1,
            "top3_hit": -1,
            "verified": False,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self.save(df)
        return new_id

    def verify(self, flat_numbers):
        df = self.load()
        if df.empty:
            return 0
        verified = 0
        for i, row in df.iterrows():
            if row["verified"]:
                continue
            idx = int(row["predict_for_index"])
            if idx < len(flat_numbers):
                actual = int(flat_numbers[idx])
                top1 = int(row["predicted_top1"])
                try:
                    top3 = [int(x) for x in str(row["predicted_top3"]).split(",") if x.strip() != ""]
                except Exception:
                    top3 = []
                df.at[i, "actual"] = actual
                df.at[i, "top1_hit"] = int(top1 == actual)
                df.at[i, "top3_hit"] = int(actual in top3)
                df.at[i, "verified"] = True
                verified += 1
        self.save(df)
        return verified

    def summary_all_models(self):
        df = self.load()
        df = df[df["verified"] == True]
        if df.empty:
            return pd.DataFrame()
        rows = []
        for model in df["model"].unique():
            mdf = df[df["model"] == model]
            rows.append({
                "Model": model,
                "Total": len(mdf),
                "Top-1 Correct": int(mdf["top1_hit"].sum()),
                "Top-1 Accuracy %": round(mdf["top1_hit"].mean() * 100, 2),
                "Top-3 Correct": int(mdf["top3_hit"].sum()),
                "Top-3 Accuracy %": round(mdf["top3_hit"].mean() * 100, 2),
            })
        return pd.DataFrame(rows).sort_values("Top-1 Accuracy %", ascending=False)

    def pending_count(self):
        df = self.load()
        return int((~df["verified"].astype(bool)).sum()) if not df.empty else 0

    def clear(self):
        if os.path.exists(self.file):
            os.remove(self.file)
