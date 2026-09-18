# auto_daily_learn.py
# Roz ye chalao - weights auto-update
import sys
import os
from datetime import datetime

print(f"🔄 Auto-Learning started: {datetime.now()}")
print("=" * 60)

# Import and run learn
try:
    from panna_predictor import learn_weights, PRIORITY_SITES
    sites = list(PRIORITY_SITES.keys()) + ["Milan Day", "Rajdhani Night", "Main Bazar", "Sridevi"]
    print(f"Training on {len(sites)} sites...")
    learn_weights(sites)
    print(f"\n✅ Auto-learning complete: {datetime.now()}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
