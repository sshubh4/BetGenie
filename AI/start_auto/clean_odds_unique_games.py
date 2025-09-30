#!/usr/bin/env python3
import pandas as pd

IN_PATH  = "backup_odds.csv"
OUT_PATH = "odds_unique_games.csv"

# --- Load ---
odds = pd.read_csv(IN_PATH)

# --- Normalize dates ---
odds["_date"] = pd.to_datetime(odds["game_date_et"], errors="coerce").dt.strftime("%Y-%m-%d")

# --- Team code cleanup (defensive, but your file may be already fine) ---
# If you still had oddball codes, add them here to normalize
code_fix = {
    "ATH":"OAK", "CHW":"CWS", "KCR":"KC", "SDP":"SD", "SFG":"SF", "TBR":"TB", "WSN":"WSH"
}
odds["Tm"]  = odds["Tm"].astype(str).str.strip().str.upper().replace(code_fix)
odds["Opp"] = odds["Opp"].astype(str).str.strip().str.upper().replace(code_fix)

# --- Order-agnostic pair ---
def sort_pair(a, b):
    if pd.isna(a) or pd.isna(b): return (None, None)
    return tuple(sorted([str(a).strip(), str(b).strip()], key=lambda x: x.lower()))

odds[["pair_a","pair_b"]] = odds.apply(lambda r: pd.Series(sort_pair(r["Tm"], r["Opp"])), axis=1)

# --- Parse time for ordering ---
odds["commence_time"] = pd.to_datetime(odds["commence_time"], errors="coerce")

# --- Keep 1 row per unique GAME (drop bookmaker & Over/Under dupes) ---
# Use event_id inside each (_date, pair) and keep earliest time
uniq_games = (
    odds.dropna(subset=["_date","pair_a","pair_b","event_id"])
        .sort_values(["_date","pair_a","pair_b","event_id","commence_time"])
        .drop_duplicates(subset=["_date","pair_a","pair_b","event_id"], keep="first")
)

# --- Assign true DH index by time within each date+pair ---
uniq_games = uniq_games.sort_values(["_date","pair_a","pair_b","commence_time"])
uniq_games["game_in_day"] = uniq_games.groupby(["_date","pair_a","pair_b"]).cumcount() + 1

# --- Choose total line (POINT) for each unique game ---
point_col = "Point" if "Point" in uniq_games.columns else ("point" if "point" in uniq_games.columns else None)
if point_col is None:
    raise KeyError("No Point/point column in odds.")

out = (uniq_games[["_date","pair_a","pair_b","game_in_day", point_col]]
       .rename(columns={point_col:"Point"})
       .dropna(subset=["_date","pair_a","pair_b"]))

# --- Save ---
out.to_csv(OUT_PATH, index=False)

print(f"[odds] in={len(odds):,}  unique_games={len(out):,}")
print(f"Saved: {OUT_PATH}")
