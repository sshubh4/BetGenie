#!/usr/bin/env python3
import pandas as pd
import numpy as np

HOME_PATH = "jordan_final.csv"
UNIQ_PATH = "odds_unique_games.csv"
OUT_PATH  = "home_games_with_point.csv"

# --- Load ---
home = pd.read_csv(HOME_PATH)
odds_first = pd.read_csv(UNIQ_PATH)  # _date, pair_a, pair_b, game_in_day, Point

# --- Normalize home dates ---
home["_date"] = pd.to_datetime(home["Date_Parsed"], errors="coerce").dt.strftime("%Y-%m-%d")

# --- Normalize home team codes to same convention as odds (defensive) ---
home_fix = {
    "ATH":"OAK", "CHW":"CWS", "KCR":"KC", "SDP":"SD", "SFG":"SF", "TBR":"TB", "WSN":"WSH"
}
home["Tm"]  = home["Tm"].astype(str).str.strip().str.upper().replace(home_fix)
home["Opp"] = home["Opp"].astype(str).str.strip().str.upper().replace(home_fix)

# --- Order-agnostic pair ---
def sort_pair(a, b):
    if pd.isna(a) or pd.isna(b): return (None, None)
    return tuple(sorted([str(a).strip(), str(b).strip()], key=lambda x: x.lower()))

home[["pair_a","pair_b"]] = home.apply(lambda r: pd.Series(sort_pair(r["Tm"], r["Opp"])), axis=1)

# --- Home DH index from Date "(n)" ---
if "Date" in home.columns:
    gi = home["Date"].astype(str).str.extract(r"\((\d+)\)")[0].fillna("1").astype(int)
else:
    gi = 1
home["game_in_day"] = gi

# --- Build lookup & map WITHOUT merge ---
lookup = odds_first.set_index(["_date","pair_a","pair_b","game_in_day"])["Point"]
home["Point"] = [
    lookup.get((d, a, b, g), np.nan)
    for d, a, b, g in zip(home["_date"], home["pair_a"], home["pair_b"], home["game_in_day"])
]

# --- Save ---
home.to_csv(OUT_PATH, index=False)

print(f"[DONE] filled={home['Point'].notna().sum():,}  missing={home['Point'].isna().sum():,}")
print(f"Saved: {OUT_PATH}")
