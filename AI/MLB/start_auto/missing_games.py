import re
import pandas as pd
import numpy as np
import glob

# ---------- LOAD ----------
all_files = glob.glob("../teamgamelogs/*.csv")
df_list = [pd.read_csv(f) for f in all_files]
for i, df in enumerate(df_list):
    df.columns = df.columns.str.strip()
    # game_in_day
    df['game_in_day'] = df['Date'].astype(str).str.extract(r'\((\d+)\)').fillna('1').astype(int)
    # parse date
    df['Date_clean'] = df['Date'].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True).str.strip()
    df['Date_clean'] = df['Date_clean'].str.replace(r'^\w+,\s*', '', regex=True)  # drop day name
    df['full_date_str'] = df['Date_clean'] + ' ' + df['Season'].astype(str)
    df['Date_Parsed'] = pd.to_datetime(df['full_date_str'], errors='coerce')
    df_list[i] = df.sort_values(['Tm', 'Date_Parsed'])

all_games_df = pd.concat(df_list, ignore_index=True)
print(f"[all_games_df] shape: {all_games_df.shape}")

home_games = all_games_df[all_games_df['At'] != '@'].copy()
home_games.to_csv('home_games.csv', index=False)
print(f"[home_games] shape: {home_games.shape}")

# ---------- ODDS (use played_date instead of Date_Parsed) ----------
odds = pd.read_csv('./backup_odds.csv')
odds['Date_Parsed'] = odds['game_date_et']
# rows where Opp is missing (debug)
missing_opp = odds[odds['Opp'].isna()]
print(missing_opp)

print(f"[odds] shape: {odds.shape}")

# make sure both are YYYY-MM-DD strings for matching
home_games["_date"] = pd.to_datetime(home_games["Date_Parsed"], errors="coerce").dt.strftime('%Y-%m-%d')

# Prefer played_date; fallback to Date_Parsed if played_date missing
if 'played_date' in odds.columns:
    odds["_date"] = pd.to_datetime(odds["played_date"], errors="coerce").dt.strftime('%Y-%m-%d')
else:
    odds["_date"] = pd.to_datetime(odds["Date_Parsed"], errors="coerce").dt.strftime('%Y-%m-%d')

def unordered_key(date_val, tm, opp):
    a, b = sorted([tm, opp])
    return f"{date_val}|{a}|{b}"

# keys for home_games
def safe_unordered_key(date_val, tm, opp):
    # Guard against missing values
    if pd.isna(date_val) or pd.isna(tm) or pd.isna(opp):
        return np.nan
    a, b = sorted([str(tm), str(opp)])
    return f"{date_val}|{a}|{b}"

# keys for home_games
home_games["_key"] = home_games.apply(
    lambda r: safe_unordered_key(r["_date"], r["Tm"], r["Opp"]), axis=1
)

# keys for odds (now using played_date-driven _date)
odds["_key"] = odds.apply(
    lambda r: safe_unordered_key(r["_date"], r["Tm"], r["Opp"]), axis=1
)

home_keys = set(home_games["_key"].dropna().unique())
odds_keys  = set(odds["_key"].dropna().unique())

# games that exist in logs but missing odds
missing_in_odds_keys = home_keys - odds_keys
missing_in_odds = home_games[home_games["_key"].isin(missing_in_odds_keys)].copy()

# odds rows that don’t match any game
extra_in_odds_keys = odds_keys - home_keys
extra_in_odds = odds[odds["_key"].isin(extra_in_odds_keys)].copy()

print(f"Total unique games (logs): {len(home_keys)}")
print(f"Total unique games (odds): {len(odds_keys)}")
print(f"Missing in odds (rows): {missing_in_odds.shape[0]}")
print(f"Extra in odds (rows): {extra_in_odds.shape[0]}")

# Per-date gaps (counts)
miss_counts = (
    missing_in_odds.groupby("_date")
    .size()
    .sort_values(ascending=False)
)

print("\nTop 20 missing-by-date:")
print(miss_counts.head(20).to_string())  # preview
print("\nTotal dates with missing games:", miss_counts.shape[0])

# Save detailed missing rows for review
missing_in_odds.to_csv("missing_rows.csv", index=False)

# Save just the unique missing dates (YYYY-MM-DD)
missing_dates = (
    missing_in_odds[["_date"]]
    .dropna()
    .drop_duplicates()
    .sort_values("_date")
    .rename(columns={"_date": "date"})
)
missing_dates.to_csv("missing_dates.csv", index=False)

# (Optional) also save the per-date counts as CSV
miss_counts.rename("missing_count").to_frame().reset_index().to_csv(
    "missing_by_date_counts.csv", index=False
)
