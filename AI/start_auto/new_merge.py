import glob
import pandas as pd
import numpy as np
import re
import warnings

pd.set_option("display.max_columns", None)

# -------------------- CONFIG --------------------
csv_files = glob.glob("../teamgamelogs/*.csv")
df_list = [pd.read_csv(file) for file in csv_files]

batter_stat_names = [
    'WPA','aLI','WPA+','WPA-','cWPA','acLI','RE24',
    'R','RBI','BB','SO','BA','OBP','SLG','OPS',
    'Pit','Str','PA','H','AB','PO','A'
]
pitcher_stat_names = [
    'IP','H','R','ER','BB','HR','ERA','BF','Pit','Str','StS','StL',
    'GB','FB','LD','Unk','aLI','cWPA','RE24','acLI','SO','Ctct','GSc','IR','WPA','IS'
]

# -------------------- PREP --------------------
for i, df in enumerate(df_list):
    df.columns = df.columns.str.strip()
    # game_in_day
    df['game_in_day'] = df['Date'].astype(str).str.extract(r'\((\d+)\)').fillna('1').astype(int)
    # parse date
    df['Date_clean'] = df['Date'].astype(str).str.replace(r'\s*\(\d+\)', '', regex=True).str.strip()
    df['Date_clean'] = df['Date_clean'].str.replace(r'^\w+,\s*', '', regex=True)  # drop day name
    df['full_date_str'] = df['Date_clean'] + ' ' + df['Season'].astype(str)
    df['Date_parsed'] = pd.to_datetime(df['full_date_str'], errors='coerce')
    df_list[i] = df.sort_values(['Tm', 'Date_parsed'])

all_games_df = pd.concat(df_list, ignore_index=True)
if isinstance(all_games_df.columns, pd.MultiIndex):
    all_games_df.columns = ['_'.join(map(str, c)).strip('_') for c in all_games_df.columns]
all_games_df.columns = all_games_df.columns.str.strip()

# home/away
all_games_df['is_home'] =  (all_games_df['At'].to_numpy() != '@').astype('int8')

# Replace missing/empty values in batter/pitcher name columns with 'N/A'
for i in range(1, 16):
    col = f"batter_{i}_Batting"
    if col in all_games_df.columns:
        all_games_df[col] = all_games_df[col].fillna('N/A')
        all_games_df.loc[all_games_df[col].astype(str).str.strip() == '', col] = 'N/A'
for i in range(1, 10):
    col = f"pitcher_{i}_Pitching"
    if col in all_games_df.columns:
        all_games_df[col] = all_games_df[col].fillna('N/A')
        all_games_df.loc[all_games_df[col].astype(str).str.strip() == '', col] = 'N/A'

# ---------- NEW: simple, safe name cleaner (no heavy regex) ----------
position_codes = [
    'p','c','1b','2b','3b','ss','lf','cf','rf','dh',
    'ph','pr','ut','sf','fb','gf',
    'lm','rm','lb','cfr','cf-lf','1b-rf',
    'sp','rp','cl','jr','sr','4f','5f','6f','7f'
]
POS_CODES = {c.lower() for c in position_codes}

def clean_player_name(name):
    """Trim, keep text before comma, drop one trailing (...) group,
    then remove trailing position tokens by word (case-insensitive)."""
    if pd.isna(name):
        return 'N/A'
    s = str(name).strip()
    if not s:
        return 'N/A'

    # keep only text before any comma
    if ',' in s:
        s = s.split(',', 1)[0].strip()

    # drop a single trailing (...) group if present
    s = re.sub(r'\s*\([^)]*\)\s*$', '', s)

    # normalize separators and remove trailing position tokens
    t = s.replace('-', ' ')
    parts = t.split()
    while parts and parts[-1].lower() in POS_CODES:
        parts.pop()

    out = ' '.join(parts).strip()
    return out if out else 'N/A'

def clean_column_percentages(df):
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            if df[col].astype(str).str.contains('%').any():
                df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

all_games_df = clean_column_percentages(all_games_df)

########################################
# Normalize At -> 0/1 and sort
all_games_df['At'] = all_games_df['At'].replace({'@': 0, '': 1, np.nan: 1}).astype(int)
all_games_df = all_games_df.sort_values(by="At", ascending=True).reset_index(drop=True)

# Robust name cleanup (no dtype traps)
def clean_name_col(col: pd.Series) -> pd.Series:
    s = col.copy()
    s = s.where(~s.isna(), 'N/A').astype('string').str.strip()
    bad_tokens = {'', 'nan', 'na', 'none'}
    mask_bad = s.str.lower().isin(bad_tokens).fillna(True)
    s = s.mask(mask_bad, 'N/A')
    return s

# safety wrapper when applying cleaner
def safe_apply_clean(series: pd.Series) -> pd.Series:
    def _safe(x):
        try:
            return clean_player_name(x)
        except Exception:
            return 'N/A'
    return series.apply(_safe)

batter_cols = [f"batter_{i}_Batting" for i in range(1, 17) if f"batter_{i}_Batting" in all_games_df.columns]
pitcher_cols = [f"pitcher_{i}_Pitching" for i in range(1, 10) if f"pitcher_{i}_Pitching" in all_games_df.columns]
for col in batter_cols:
    all_games_df[col] = safe_apply_clean(clean_name_col(all_games_df[col]))
for col in pitcher_cols:
    all_games_df[col] = safe_apply_clean(clean_name_col(all_games_df[col]))

# 0 out numeric NaNs
def fill_missing_numeric_with_zero(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].fillna(0)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].fillna(0.0)
    return df
all_games_df = fill_missing_numeric_with_zero(all_games_df)

# Remove entries that start with a number in name columns
def remove_numeric_start_entries(df):
    cols = [c for c in df.columns if (c.startswith('batter_') and c.endswith('_Batting')) or
                                     (c.startswith('pitcher_') and c.endswith('_Pitching'))]
    for col in cols:
        def filter_numeric_start(val):
            if pd.isna(val):
                return 'N/A'
            val_str = str(val).strip()
            if re.match(r'^\d+(\.\d+)?', val_str):
                return 'N/A'
            return val_str
        df[col] = df[col].apply(filter_numeric_start)
    return df
all_games_df = remove_numeric_start_entries(all_games_df)

# Keep only text before any comma in name columns (idempotent with our cleaner, but fine)
def keep_before_comma(df):
    cols = [c for c in df.columns if (c.startswith('batter_') and c.endswith('_Batting')) or
                                     (c.startswith('pitcher_') and c.endswith('_Pitching'))]
    for col in cols:
        df[col] = df[col].apply(lambda v: v if pd.isna(v) else str(v).split(',', 1)[0].strip())
    return df
all_games_df = keep_before_comma(all_games_df)
all_games_df = all_games_df.sort_values(['Tm', 'Date_parsed', 'is_home']).reset_index(drop=True)

# ================== NEW: TEAM REST + OVERALL FORM (computed BEFORE split) ==================
all_games_df = all_games_df.sort_values(['Tm','Date_parsed'])
prev_date = all_games_df.groupby('Tm')['Date_parsed'].shift(1)
rest = (all_games_df['Date_parsed'] - prev_date).dt.days
all_games_df['rest_days'] = rest.clip(lower=0).fillna(3).astype(float)   # default 3 for first game
all_games_df['b2b'] = (all_games_df['rest_days'] <= 1).astype(int)

# Overall rolling runs regardless of H/A (pre-game, leak-proof)
all_games_df['rolling_10_R_overall'] = (
    all_games_df.groupby('Tm', group_keys=False)['R']
      .apply(lambda x: x.rolling(window=10, min_periods=1).mean().shift(1))
)

# -------------------- HELPERS --------------------
def leakproof_rolling_mean(series, window=10):
    return series.rolling(window=window, min_periods=1).mean().shift(1)

def build_batter_rolls(df, window=10):
    name_exists = any(f'batting_{p}_name' in df.columns or f'batter_{p}_name' in df.columns for p in range(1, 10))
    if name_exists:
        batters_long = pd.DataFrame()
        for pos in range(1, 10):
            name_col = next((c for c in [f'batting_{pos}_name', f'batter_{pos}_name'] if c in df.columns), None)
            if name_col is None:
                continue
            cols_pref_a = [c for c in df.columns if c.startswith(f'batting_{pos}_')]
            cols_pref_b = [c for c in df.columns if c.startswith(f'batter_{pos}_')]
            cols = sorted(set(cols_pref_a + cols_pref_b + [name_col]))
            temp = df[['Date_parsed','Tm','is_home'] + cols].copy()
            temp.rename(columns={name_col: 'Player_name'}, inplace=True)
            temp['batting_position'] = pos
            rename_map = {}
            for stat in batter_stat_names:
                if f'batting_{pos}_{stat}' in temp.columns:
                    rename_map[f'batting_{pos}_{stat}'] = stat
                elif f'batter_{pos}_{stat}' in temp.columns:
                    rename_map[f'batter_{pos}_{stat}'] = stat
            temp.rename(columns=rename_map, inplace=True)
            for stat in batter_stat_names:
                if stat in temp.columns:
                    temp[stat] = pd.to_numeric(temp[stat], errors='coerce')
            batters_long = pd.concat([batters_long, temp], ignore_index=True)

        if not batters_long.empty:
            batters_long.sort_values(['Player_name','Date_parsed'], inplace=True)
            for stat in batter_stat_names:
                if stat in batters_long.columns:
                    batters_long[f'rolling_10_{stat}'] = (
                        batters_long.groupby('Player_name', group_keys=False)[stat]
                                    .apply(leakproof_rolling_mean, window=window)
                    )
            for pos in range(1, 10):
                subset = batters_long[batters_long['batting_position'] == pos]
                if subset.empty:
                    continue
                keep_cols = ['Date_parsed','Tm','is_home','Player_name'] + \
                            [f'rolling_10_{s}' for s in batter_stat_names if f'rolling_10_{s}' in subset.columns]
                subset = subset[keep_cols].copy()
                subset.rename(columns=lambda c: (f'rolling_10_batter_{pos}_{c[11:]}' if c.startswith('rolling_10_') else c),
                              inplace=True)
                player_name_col = next((c for c in [f'batting_{pos}_name', f'batter_{pos}_name'] if c in df.columns), None)
                if player_name_col is None:
                    continue
                df = df.merge(
                    subset,
                    how='left',
                    left_on=['Date_parsed','Tm','is_home', player_name_col],
                    right_on=['Date_parsed','Tm','is_home','Player_name']
                ).drop(columns=['Player_name'], errors='ignore')
        return df
    else:
        for pos in range(1, 10):
            for stat in batter_stat_names:
                src = f'batting_{pos}_{stat}' if f'batting_{pos}_{stat}' in df.columns else \
                      f'batter_{pos}_{stat}'  if f'batter_{pos}_{stat}'  in df.columns else None
                if src is None:
                    continue
                df[src] = pd.to_numeric(df[src], errors='coerce')
                outcol = f'rolling_10_batter_{pos}_{stat}'
                df[outcol] = (
                    df.groupby(['Tm','is_home'], group_keys=False)[src]
                      .apply(leakproof_rolling_mean, window=window)
                )
        return df

def build_pitcher_rolls(df, window=10):
    if 'pitcher_1_name' in df.columns:
        p = df[['Date_parsed','pitcher_1_name']].copy()
        for stat in pitcher_stat_names:
            col = f'pitcher_1_{stat}'
            if col in df.columns:
                p[stat] = pd.to_numeric(df[col], errors='coerce')
        p.sort_values(['pitcher_1_name','Date_parsed'], inplace=True)
        for stat in pitcher_stat_names:
            if stat in p.columns:
                p[f'rolling_10_{stat}'] = (
                    p.groupby('pitcher_1_name', group_keys=False)[stat]
                     .apply(leakproof_rolling_mean, window=window)
                )
        keep = ['Date_parsed','pitcher_1_name'] + [f'rolling_10_{s}' for s in pitcher_stat_names if f'rolling_10_{s}' in p.columns]
        p = p[keep].rename(columns=lambda c: f'rolling_10_pitcher_1_{c[11:]}' if c.startswith('rolling_10_') else c)
        df = df.merge(p, how='left', on=['Date_parsed','pitcher_1_name'])
        return df
    else:
        for stat in pitcher_stat_names:
            col = f'pitcher_1_{stat}'
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                outcol = f'rolling_10_pitcher_1_{stat}'
                df[outcol] = (
                    df.groupby(['Tm','is_home'], group_keys=False)[col]
                      .apply(leakproof_rolling_mean, window=10)
                )
        return df
    

# -------------------- BUILD STARTER ROLLING --------------------
all_games_df = build_batter_rolls(all_games_df, window=10)
all_games_df = all_games_df.copy()
all_games_df = build_pitcher_rolls(all_games_df, window=10)
all_games_df = all_games_df.copy()

# -------------------- DROP RAW PER-BATTER/PITCHER + JUNK --------------------
batter_stats_esc  = [re.escape(s) for s in batter_stat_names]
pitcher_stats_esc = [re.escape(s) for s in pitcher_stat_names]

batter_pattern  = rf"^bat(?:ter|ting)_(?:[1-9]|1[0-8])_(?:{'|'.join(batter_stats_esc)})$"
pitcher_pattern = rf"^pitcher_(?:[1-9]|1[0-3])_(?:{'|'.join(pitcher_stats_esc)})$"

drop_cols = all_games_df.filter(regex=batter_pattern).columns.tolist() + \
            all_games_df.filter(regex=pitcher_pattern).columns.tolist()
junk_cols = [c for c in all_games_df.columns if c.startswith('Unnamed:')]

all_games_df.drop(columns=drop_cols + junk_cols, inplace=True, errors='ignore')
columns_to_drop = ['Boxscore','Boxscore URL','Orig. Scheduled','cLI','Num_Batters','Num_Pitchers','Win','Loss',"Save","Attendance","umpire_HP",
                   "umpire_1B","umpire_2B","umpire_3B","umpire_LF","umpire_RF","time_of_game","attendance",
                   "weather_temp","weather_wind","weather_sky","Time","weather_precip","num_Batters","num_Pitchers","field_condition"] + [f'pitcher_{i}_Pitching' for i in range(2, 12)]

all_games_df.drop(columns=columns_to_drop,inplace=True,errors='ignore')

# Streak -> signed integer (but we'll drop later to avoid leakage)
s = all_games_df['Streak'].fillna('').astype(str)
mask = s.str.fullmatch(r'[+-]+')
sign = s.str[0].map({'+': 1, '-': -1})
lengths = s.str.len()
all_games_df['Streak'] = np.where(mask, sign.to_numpy() * lengths.to_numpy(), np.nan)
all_games_df['Streak'] = all_games_df['Streak'].astype("Int64")
print(f"[all_games_df] shape: {all_games_df.shape}")

# -------------------- SPLIT, MERGE HOME/AWAY --------------------
home_df = all_games_df[all_games_df['is_home'] == 1].copy()
away_df = all_games_df[all_games_df['is_home'] == 0].copy()

# keep rest & overall form for suffixing on merge
keep_extra = ['rest_days','b2b','rolling_10_R_overall']
for c in keep_extra:
    if c not in home_df.columns: home_df[c] = np.nan
    if c not in away_df.columns: away_df[c] = np.nan

home_df['merge_key'] = (home_df['Date_parsed'].dt.strftime('%Y-%m-%d') + '_' +
                        home_df['Tm'] + '_' + home_df['Opp'] + '_' +
                        home_df['game_in_day'].astype(str))
away_df['merge_key'] = (away_df['Date_parsed'].dt.strftime('%Y-%m-%d') + '_' +
                        away_df['Opp'] + '_' + away_df['Tm'] + '_' +
                        away_df['game_in_day'].astype(str))
print("home rows:", len(home_df), "unique keys:", home_df['merge_key'].nunique())
print("away rows:", len(away_df), "unique keys:", away_df['merge_key'].nunique())
final_df = pd.merge(home_df, away_df, on='merge_key', how='inner', suffixes=('_home','_away'), validate='one_to_one')

print(f"[final_df initial merge] shape: {final_df.shape}")

# tidy rename (ONCE)
final_df.rename(columns={
    'Date_home': 'Date',
    'Tm_home': 'Tm',
    'Opp_home': 'Opp',
    'Time_home': 'Time',
    'D/N_home': 'D/N',
    'Date_parsed_home': 'Date_Parsed',
    'Season_home': 'Season'
}, inplace=True)
final_df = final_df.copy()
# Expose rest & overall rolling with clear names
final_df.rename(columns={
    'rest_days_home':'rest_home',
    'rest_days_away':'rest_away',
    'b2b_home':'b2b_home',
    'b2b_away':'b2b_away',
    'rolling_10_R_overall_home':'rolling_10_R_overall_home',
    'rolling_10_R_overall_away':'rolling_10_R_overall_away'
}, inplace=True)
final_df = final_df.copy()
final_df['delta_rest'] = (pd.to_numeric(final_df.get('rest_home'), errors='coerce') -
                          pd.to_numeric(final_df.get('rest_away'), errors='coerce'))
final_df['is_daytime'] = final_df['D/N'].apply(lambda x: 1 if x == 'D' else 0 )
# ================== END ODDS MERGE ==================
if {'rolling_10_R_home','rolling_10_R_away'}.issubset(final_df.columns):
    final_df['exp_runs_recent'] = pd.to_numeric(final_df['rolling_10_R_home'], errors='coerce') + \
                                  pd.to_numeric(final_df['rolling_10_R_away'], errors='coerce')
    # only compute if a total_line exists (you said not to, so this will usually skip)
    if 'total_line' in final_df.columns:
        final_df['line_minus_recent'] = final_df['total_line'] - final_df['exp_runs_recent']

era_h = pd.to_numeric(final_df.get('rolling_10_pitcher_1_ERA_home'), errors='coerce')
era_a = pd.to_numeric(final_df.get('rolling_10_pitcher_1_ERA_away'), errors='coerce')
final_df['starter_era_sum'] = era_h + era_a

for c in ['exp_runs_recent','line_minus_recent','starter_era_sum',
          'total_line_std','total_line_range','price_over','price_under','vig',
          'rest_home','rest_away','delta_rest','rolling_10_R_overall_home','rolling_10_R_overall_away']:
    if c in final_df.columns:
        q = final_df[c].astype(float)
        ql, qh = q.quantile([0.001, 0.999])
        final_df[c] = pd.to_numeric(final_df[c], errors='coerce').clip(lower=ql, upper=qh)

# -------------------- PARK FACTORS --------------------
try:
    parkfactors_df = pd.read_csv('../parkfactors/combined_parkfactors.csv')
    final_df = final_df.merge(
        parkfactors_df,
        left_on=['Tm','Season'],
        right_on=['Loc','Season'],
        how='left',
        suffixes=('', '_park')
    )
    for c in ['Loc','Year']:
        if c in final_df.columns:
            final_df.drop(columns=c, inplace=True)
except FileNotFoundError:
    print("⚠️ parkfactors/combined_parkfactors.csv not found; skipping park merge.")

# -------------------- SAVE TIDY BEFORE PRUNE (drop batter details, minor columns) --------------------
batter_bd_cols = final_df.filter(
    regex=r'^batter_(?:[1-9]|1[0-7])_(?:Batting|Details)_(?:home|away)$'
).columns.tolist()

print(f"Dropping {len(batter_bd_cols)} batter Batting/Details columns (1–17, home/away).")
final_df.drop(columns=batter_bd_cols, inplace=True, errors='ignore')

columns_to_drop = ['Date_away','Inn_away','Inn_home', "Date"]
final_df.drop(columns=columns_to_drop,inplace=True,errors='ignore')

## weather
weather_cols = [
    "datetime_utc", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
    "wind_gusts_10m", "apparent_temperature", "Original_Loc", "Date"
]
weather_df = pd.read_csv("../combined_weather_data.csv", usecols=weather_cols)

weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors="coerce").dt.strftime('%Y-%m-%d')
weather_df['Original_Loc'] = pd.Series(weather_df['Original_Loc'], dtype="string")
final_df['Date_Parsed'] = pd.to_datetime(final_df['Date_Parsed'], errors='coerce').dt.strftime('%Y-%m-%d')
final_df['Tm'] = final_df['Tm'].astype(str)

weather_vars = [
    "temperature_2m", "relative_humidity_2m", 
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
    "wind_gusts_10m", "apparent_temperature"
]

weather_merge_df = (
    weather_df[["Date", "Original_Loc"] + weather_vars]
    .drop_duplicates(subset=["Date", "Original_Loc"])
)

old_rows = len(final_df)

final_df = pd.merge(
    final_df,
    weather_merge_df,
    how='left',
    left_on=['Date_Parsed', 'Tm'],
    right_on=['Date', 'Original_Loc']
)

assert len(final_df) == old_rows, "Row count increased after weather merge!"
final_df.drop(columns=['Original_Loc', 'Date'], inplace=True, errors='ignore')

# Wind components (add to FINAL_DF too)
if {'wind_speed_10m','wind_direction_10m'}.issubset(final_df.columns):
    spd10_f = pd.to_numeric(final_df['wind_speed_10m'], errors='coerce')
    dir10_f = pd.to_numeric(final_df['wind_direction_10m'], errors='coerce')
    rad10_f = np.deg2rad(dir10_f)
    final_df['wind_u10'] = (spd10_f * np.cos(rad10_f)).astype('float32')
    final_df['wind_v10'] = (spd10_f * np.sin(rad10_f)).astype('float32')
if {'wind_speed_100m','wind_direction_100m'}.issubset(final_df.columns):
    spd100_f = pd.to_numeric(final_df['wind_speed_100m'], errors='coerce')
    dir100_f = pd.to_numeric(final_df['wind_direction_100m'], errors='coerce')
    rad100_f = np.deg2rad(dir100_f)
    final_df['wind_u100'] = (spd100_f * np.cos(rad100_f)).astype('float32')
    final_df['wind_v100'] = (spd100_f * np.sin(rad100_f)).astype('float32')

# Reorder to move weather vars to the end (robust)
non_weather = [c for c in final_df.columns if c not in weather_vars]
present_weather = [w for w in weather_vars if w in final_df.columns]
final_df = final_df[non_weather + present_weather]

# Ensure final_df is sorted
final_df = final_df.sort_values(['Tm', 'Date_Parsed'])

# rolling_10_R_home: avg last 10 home games per team (pre-game)
final_df['rolling_10_R_home'] = (
    final_df.groupby('Tm', group_keys=False)['R_home']
    .apply(lambda x: x.rolling(window=10, min_periods=1).mean().shift(1))
    .fillna(0)
)

# rolling_10_R_away: avg last 10 away games per away team
final_df['rolling_10_R_away'] = (
    final_df.groupby('Opp', group_keys=False)['R_away']
    .apply(lambda x: x.rolling(window=10, min_periods=1).mean().shift(1))
    .fillna(0)
)

# ================== PA-WEIGHTED LINEUP AGGREGATES (player-by-name) ==================
def _pa_weighted(m, w):
    w = np.where(np.isfinite(w), w, 0.0)
    m = np.where(np.isfinite(m), m, np.nan)
    num = np.nansum(m * w, axis=1)
    den = np.nansum(w, axis=1)
    out = np.divide(num, den, out=np.zeros_like(num), where=den>0)
    return out

def _build_lineup_agg_pw(df, side):
    stats = ["OPS","OBP","SLG","RE24","RBI","R"]
    pa_cols = [f"rolling_10_batter_{p}_PA_{side}" for p in range(1,10)]
    pa_cols = [c for c in pa_cols if c in df.columns]
    out = pd.DataFrame(index=df.index)

    for stat in stats:
        cols = [f"rolling_10_batter_{p}_{stat}_{side}" for p in range(1,10)]
        cols = [c for c in cols if c in df.columns]
        if not cols or not pa_cols:
            continue
        k = min(len(cols), len(pa_cols))
        M = df[cols[:k]].astype(float).to_numpy()
        W = df[pa_cols[:k]].astype(float).to_numpy()
        out[f"lineup_{stat}_pwmean_{side}"] = _pa_weighted(M, W)

    ops_cols = [f"rolling_10_batter_{p}_OPS_{side}" for p in range(1,10)]
    ops_cols = [c for c in ops_cols if c in df.columns]
    if ops_cols:
        M = df[ops_cols].astype(float).to_numpy()
        k = int(min(3, M.shape[1]))
        if k > 0:
            top3 = np.sort(M, axis=1)[:, -k:]
            out[f"lineup_OPS_top3mean_{side}"] = np.nanmean(top3, axis=1)
        else:
            out[f"lineup_OPS_top3mean_{side}"] = np.nan

    if ops_cols:
        Mops = df[ops_cols].astype(float)
        out[f"lineup_hot_ops_cnt_{side}"] = (Mops > 0.85).sum(axis=1).astype(float)

    return out

agg_home_pw = _build_lineup_agg_pw(final_df, "home")
agg_away_pw = _build_lineup_agg_pw(final_df, "away")
final_df = pd.concat([final_df, agg_home_pw, agg_away_pw], axis=1)

# ================== FEATURE PRUNE (right before save) ==================
pat_batter_slots = re.compile(r"^rolling_10_batter_[1-9]_.+_(home|away)$")
drop_batter_slot_cols = [c for c in final_df.columns if pat_batter_slots.match(c)]

# blatant leaks (game outcomes / standings-ish)
leak_exact = {
    'W/L_home','W/L_away','RA_home','RA_away',
    'W-L_home','W-L_away','Rank_home','Rank_away','GB_home','GB_away',
    'Streak_home','Streak_away',
    'R','H','1B','2B','3B','HR','BB','SO','PA'
}
drop_leak = [c for c in final_df.columns if c in leak_exact]

# redundant merges / id cols we don't need
redundant_exact = {
    'At_home','At_away','is_home_home','is_home_away','D/N_away',
    'pitcher_1_Pitching_away'
}
drop_redundant = [c for c in final_df.columns if c in redundant_exact]

# keep only a tight starter set
keep_pitcher_metrics = {'IP','ERA','BB','SO','HR','BF'}
pat_pitcher_keep = re.compile(rf"^rolling_10_pitcher_1_({'|'.join(sorted(keep_pitcher_metrics))})_(home|away)$")
pat_pitcher_all  = re.compile(r"^rolling_10_pitcher_1_.+_(home|away)$")
drop_pitcher_noise = [c for c in final_df.columns if pat_pitcher_all.match(c) and not pat_pitcher_keep.match(c)]

# ---------- DROP bullpen rolling-10 performance blocks completely ----------
drop_pen_rolling = [c for c in final_df.columns if re.match(r"^rolling_pen_.*_mean_10(_home|_away)?$", c)]

# build candidate_drop
candidate_drop = set(drop_batter_slot_cols) | set(drop_leak) | set(drop_redundant) | set(drop_pitcher_noise) | set(drop_pen_rolling)

# whitelist must-survive columns (no bullpen rolling-10 here)
whitelist_exact = {
    'over_result','over_under_label',
    'total_line','total_line_std','total_line_range','price_over','price_under','vig',
    'Park Factor',
    'temperature_2m','wind_u10','wind_u100','park_x_temp','park_x_wind10',
    'rolling_10_R_home','rolling_10_R_away',
    'rolling_10_R_overall_home','rolling_10_R_overall_away',
    'lineup_OPS_pwmean_home','lineup_OPS_pwmean_away',
    'lineup_OBP_pwmean_home','lineup_OBP_pwmean_away',
    'lineup_SLG_pwmean_home','lineup_SLG_pwmean_away',
    'lineup_RE24_pwmean_home','lineup_RE24_pwmean_away',
    'lineup_RBI_pwmean_home','lineup_RBI_pwmean_away',
    'lineup_R_pwmean_home','lineup_R_pwmean_away',
    'starter_era_sum',
    'rest_home','rest_away','delta_rest','b2b_home','b2b_away',
    'pen_IP_3day_sum_home','pen_IP_3day_sum_away',
    'pen_used_yday_home','pen_used_yday_away',
    'is_half_total','frac_from_half',
    'temp_sq','wind_speed10_sq',
    'Date_Parsed','is_daytime'
}
whitelist_prefixes = (
    'lineup_',                # our PA-weighted lineup aggregates
    'rolling_10_pitcher_1_',  # starter rolling set (we pruned noise above)
    'temperature_2m','relative_humidity_2m',
    'wind_speed_10m','wind_speed_100m',
    'wind_direction_10m','wind_direction_100m',
    'wind_gusts_10m','apparent_temperature'
)

# protect whitelist
for c in list(candidate_drop):
    if c in whitelist_exact or any(c.startswith(p) for p in whitelist_prefixes):
        candidate_drop.discard(c)

before_cols = set(final_df.columns)
final_df.drop(columns=list(candidate_drop), inplace=True, errors='ignore')
after_cols = set(final_df.columns)
print(f"[Prune] Dropped {len(before_cols - after_cols)} columns; kept {len(after_cols)}.")

# --- EXTRA FEATURE ENGINEERING (kept from your draft) ---
if 'temperature_2m' in final_df.columns:
    final_df['temp_sq'] = pd.to_numeric(final_df['temperature_2m'], errors='coerce')**2

if {'wind_u10','wind_v10'}.issubset(final_df.columns):
    wu = pd.to_numeric(final_df['wind_u10'], errors='coerce')
    wv = pd.to_numeric(final_df['wind_v10'], errors='coerce')
    final_df['wind_speed10_sq'] = (wu**2 + wv**2)

# validate required (only enforce core game fields now)
missing_req = [c for c in ['R_home','R_away','Date_Parsed'] if c not in final_df.columns]
if missing_req:
    print("⚠️ Missing required columns after prune:", missing_req)

# --- SAVE FULL DATASET ---
final_df.to_csv('jordan_final.csv', index=False)
print(f"✅ Saved pruned dataset to jordan_final.csv with shape: {final_df.shape}")
