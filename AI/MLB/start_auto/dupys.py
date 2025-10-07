#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dedupe_odds_by_game.py

What it does (no CLI):
1) Load whitelist dates (YYYY-MM-DD) from CSV.
2) Load odds CSV and keep only rows on those dates.
3) For the same game that appears across multiple sportsbooks, keep just ONE row
   per (game, side), using the latest snapshot and a bookmaker-priority tiebreak.
4) Writes:
   - mlb_odds_filtered.csv   : only whitelisted dates (pre-dedupe)
   - mlb_odds_dedup.csv      : deduped by (game, side)
   - mlb_odds_dropped.csv    : rows that were removed by the deduper (for audit)
"""

import pandas as pd
from datetime import datetime

# ================== CONFIG ==================
DATES_CSV        = "home_games_with_point.csv"   # CSV containing dates to KEEP (YYYY-MM-DD) – one per line/col

df = pd.read_csv(DATES_CSV)
print(df['Point'].head(10))


df
