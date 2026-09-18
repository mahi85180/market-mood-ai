#!/bin/bash
# Daily routine — chalao subah 10 AM pe
cd "$(dirname "$0")"
source venv/bin/activate
python3 auto_daily_learn.py >> logs/auto_learn.log 2>&1
