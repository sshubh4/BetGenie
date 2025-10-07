import re
import pandas as pd
import numpy as np

# ---------- LOAD ----------
df = pd.read_csv('./mlb_odds.csv')

# ---------- NAME NORMALIZATION (legacy teams) ----------
df['home_team'] = df['home_team'].replace({
    'Cleveland Indians': 'Cleveland Guardians',
    'Tampa Bay Devil Rays': 'Tampa Bay Rays',
    'Florida Marlins': 'Miami Marlins',
    'Oakland Athletics': 'Athletics',  # if your odds file sometimes drops 'Oakland'
})
df['away_team'] = df['away_team'].replace({
    'Cleveland Indians': 'Cleveland Guardians',
    'Tampa Bay Devil Rays': 'Tampa Bay Rays',
    'Florida Marlins': 'Miami Marlins',
    'Oakland Athletics': 'Athletics',
})

# ---------- TEAM MAPS ----------
team_map = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SDP": "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants",
    "STL": "St Louis Cardinals",
    "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSN": "Washington Nationals",
}

aliases = {
    "Los Angeles Angels of Anaheim": "LAA",
    "Anaheim Angels": "LAA",
    "California Angels": "LAA",
    "Devil Rays": "Tampa Bay Rays",
    "Tampa Bay Devil Rays": "Tampa Bay Rays",
    "Florida Marlins": "Miami Marlins",
    "Oakland Athletics": "Athletics",
}
df.drop(columns=['price'], inplace=True)
df.rename(columns={'game_date':'Date_Parsed','away_team':'Opp','home_team':'Tm'}, inplace=True)
team_name_to_abbrev = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN",
    "Cleveland Indians": "CLE", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Anaheim Angels": "LAA", "California Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Florida Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY", "Athletics": "ATH",
    "Oakland Athletics": "ATH", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Tampa Bay Devil Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN"
}
bad_pat = re.compile(r'^\s*(american league|national league)\s*,?\s*$', re.IGNORECASE)

def drop_league_rows(df):
    def is_bad(x):
        return bad_pat.match(str(x)) is not None
    mask_tm  = df['Tm'].apply(is_bad)  if 'Tm'  in df.columns else False
    mask_opp = df['Opp'].apply(is_bad) if 'Opp' in df.columns else False
    mask = mask_tm | mask_opp
    n_before = len(df)
    out = df.loc[~mask].copy()
    print(f"Dropped {mask.sum()} rows (from {n_before} → {len(out)}).")
    return out

# Example on your frames:
df = drop_league_rows(df)
df['Tm'] = df['Tm'].map(team_name_to_abbrev)
df['Opp'] = df['Opp'].map(team_name_to_abbrev)

# ---------- VIG ----------
def american_to_prob(odds):
    if pd.isna(odds): return np.nan
    odds = float(odds)
    return 100.0/(odds+100.0) if odds > 0 else (-odds)/(-odds+100.0)
# ---- commence_time -> local day/night (drop-in) ----
# expects: odds_df with columns ['commence_time', 'Tm']
# --- commence_time (UTC) -> local date + day/night (no tz-aware column kept) ---
odds_df = df.copy()

cutoff_hour = 17  # <17 -> day, else night

# 1) parse commence time as UTC
odds_df['commence_dt_utc'] = pd.to_datetime(odds_df['commence_time'], utc=True, errors='coerce')

# 2) normalize team code and map to home timezone
odds_df['Tm'] = odds_df['Tm'].astype(str).str.upper().str.strip()
tz_map = {
    'ARI':'America/Phoenix','ATL':'America/New_York','BAL':'America/New_York','BOS':'America/New_York',
    'CHC':'America/Chicago','CHW':'America/Chicago','CIN':'America/New_York','CLE':'America/New_York',
    'COL':'America/Denver','DET':'America/New_York','HOU':'America/Chicago','KCR':'America/Chicago',
    'LAA':'America/Los_Angeles','LAD':'America/Los_Angeles','MIA':'America/New_York','MIL':'America/Chicago',
    'MIN':'America/Chicago','NYM':'America/New_York','NYY':'America/New_York','ATH':'America/Los_Angeles',
    'PHI':'America/New_York','PIT':'America/New_York','SDP':'America/Los_Angeles','SEA':'America/Los_Angeles',
    'SFG':'America/Los_Angeles','STL':'America/Chicago','TBR':'America/New_York','TEX':'America/Chicago',
    'TOR':'America/Toronto','WSN':'America/New_York'
}
odds_df['_tz'] = odds_df['Tm'].map(tz_map)

# 3) prepare outputs as stable dtypes (avoid tz-aware column altogether)
odds_df['local_date'] = pd.Series(pd.NA, index=odds_df.index, dtype='string')
odds_df['local_hour'] = pd.Series(pd.NA, index=odds_df.index, dtype='Int64')

# 4) convert per-timezone group, then immediately store strings/ints
for tz in odds_df['_tz'].dropna().unique():
    m = odds_df['_tz'].eq(tz)
    s_local = odds_df.loc[m, 'commence_dt_utc'].dt.tz_convert(tz)
    odds_df.loc[m, 'local_date'] = s_local.dt.strftime('%Y-%m-%d').astype('string').to_numpy()
    odds_df.loc[m, 'local_hour'] = s_local.dt.hour.astype('Int64').to_numpy()

# 5) flags for merging
odds_df['is_daytime'] = (odds_df['local_hour'] < cutoff_hour).astype('Int64')  # 1=day, 0=night, <NA> if missing
# 6) clean helper cols (keep commence_dt_utc if you want)
odds_df.drop(columns=['_tz'], inplace=True, errors='ignore')
 # backup before vig removal

# --- dups when day/night is the SAME (using Commence) ---
keys = ['commence_time','Tm','Opp','is_daytime']


# optional: stricter — only flag if SAME day/night has >1 distinct start times
dups_same_dn_times = (odds_df.groupby(keys, dropna=False)['commence_dt_utc']
                      .nunique().reset_index(name='start_times')
                      .query('start_times > 1'))
print(odds_df.shape)
# 1) Treat each (game, book) as a group
game_keys = ['commence_time', 'home_team', 'away_team', 'bookmaker']

# 2) Flag half-points (like 7.5, 8.5, ...)
is_half = np.isclose(np.mod(df['point'].astype(float), 1.0), 0.5, atol=1e-9)
df['_prefer'] = ~is_half  # False for half-points → sorts first

# 3) (Optional but nice) collapse Over/Under rows to one row per point so we can tie-break smartly
#    Pivot to put over/under prices on the same row
dups = (odds_df.groupby(['commence_time','Tm','Opp','is_daytime'])
        .size().reset_index(name='n')).query('n>1')
print("[odds] multi rows per (date, Tm, Opp, day?):", len(dups))
print(dups.head(3))
print(odds_df.shape)
game_key = ['commence_time','Tm','Opp']
conflicts = (odds_df.groupby(game_key)['point'].nunique()
             .reset_index(name='n_points').query('n_points > 1'))
print(f"Games with multiple totals: {len(conflicts)}")
print("[ODDS] same day/night but multiple start times:", len(dups_same_dn_times))
odds_df['point'] = odds_df['point'].apply(lambda x: x - 0.5 if x % 1.0 == 0 else x)
odds_df = odds_df.drop_duplicates(subset=["commence_time", "Tm", "Opp", "is_daytime"], keep="first")
# pattern: case-insensitive, allow spaces and an optional trailing comma
print(odds_df.shape)
odds_df.to_csv('backup_odds.csv', index=False) 
