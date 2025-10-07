import glob
import pandas as pd
import numpy as np
import re

pd.set_option("display.max_columns", None)

# -------------------- CONFIG --------------------
csv_files = glob.glob("./teamgamelogs/*.csv")
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
all_games_df['is_home'] = all_games_df['At'].apply(lambda x: 0 if x == '@' else 1)

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

# update names (strip positions / commas / parens)
position_codes = [
    'p','c','1b','2b','3b','ss','lf','cf','rf','dh',
    'ph','pr','ut','sf','fb','gf',
    'lm','rm','lb','cfr','cf-lf','1b-rf',
    'sp','rp','cl','jr','sr','4f','5f','6f','7f'
]
pos_pattern = re.compile(
    r"(\s*(" + '|'.join(position_codes) + r")(?:-(" + '|'.join(position_codes) + r"))*\s*)+$",
    re.IGNORECASE
)
paren_pattern = re.compile(r"\s*\([^)]*\)$")
comma_pattern = re.compile(r"\s*,\s*$")
comma_and_after_pattern = re.compile(r",.*$")  # Remove comma and everything after

def clean_player_name(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    name = comma_and_after_pattern.sub("", name).strip()
    name = paren_pattern.sub("", name).strip()
    name = comma_pattern.sub("", name).strip()
    while True:
        new_name = pos_pattern.sub("", name).strip()
        if new_name == name:
            break
        name = new_name
    return name

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

batter_cols = [f"batter_{i}_Batting" for i in range(1, 17) if f"batter_{i}_Batting" in all_games_df.columns]
pitcher_cols = [f"pitcher_{i}_Pitching" for i in range(1, 10) if f"pitcher_{i}_Pitching" in all_games_df.columns]
for col in batter_cols:
    all_games_df[col] = clean_name_col(all_games_df[col]).apply(clean_player_name)
for col in pitcher_cols:
    all_games_df[col] = clean_name_col(all_games_df[col]).apply(clean_player_name)

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

# Keep only text before any comma in name columns
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
    

# -------------------- BUILD ROLLING --------------------
all_games_df = build_batter_rolls(all_games_df, window=10)
all_games_df = build_pitcher_rolls(all_games_df, window=10)

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
print("all_games_df:", all_games_df.shape)

# -------------------- SPLIT, MERGE HOME/AWAY --------------------
home_df = all_games_df[all_games_df['is_home'] == 1].copy()
away_df = all_games_df[all_games_df['is_home'] == 0].copy()

# NEW: keep rest & overall form columns so they suffix on merge
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

duph = home_df['merge_key'].value_counts()
dupa = away_df['merge_key'].value_counts()
print("home dup keys:", (duph > 1).sum(), "away dup keys:", (dupa > 1).sum())
final_df = pd.merge(home_df, away_df, on='merge_key', how='inner', suffixes=('_home','_away'), validate='one_to_one')

print(final_df.shape)

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

# Expose rest & overall rolling with clear names
final_df.rename(columns={
    'rest_days_home':'rest_home',
    'rest_days_away':'rest_away',
    'b2b_home':'b2b_home',
    'b2b_away':'b2b_away',
    'rolling_10_R_overall_home':'rolling_10_R_overall_home',
    'rolling_10_R_overall_away':'rolling_10_R_overall_away'
}, inplace=True)
final_df['delta_rest'] = (pd.to_numeric(final_df.get('rest_home'), errors='coerce') -
                          pd.to_numeric(final_df.get('rest_away'), errors='coerce'))

# ================== ODDS MERGE (TOTAL LINE + DISPERSION & PRICES) ==================
mlb_odds_df = pd.read_csv('cleaned_mlb_total_odds.csv')

# Normalize date
if 'game_date' in mlb_odds_df.columns:
    mlb_odds_df['game_date'] = pd.to_datetime(mlb_odds_df['game_date'], errors='coerce').dt.strftime('%Y-%m-%d')
elif 'commence_time' in mlb_odds_df.columns:
    mlb_odds_df['game_date'] = pd.to_datetime(mlb_odds_df['commence_time'], errors='coerce').dt.strftime('%Y-%m-%d')
else:
    raise ValueError("mlb_total_odds.csv must have 'game_date' or 'commence_time'.")

team_name_to_abbrev = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW", "Cincinnati Reds": "CIN",
    "Cleveland Indians": "CLE", "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Anaheim Angels": "LAA", "California Angels": "LAA",
    "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA", "Florida Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN",
    "New York Mets": "NYM", "New York Yankees": "NYY",
    "Oakland Athletics": "OAK", "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP", "San Francisco Giants": "SFG", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Tampa Bay Devil Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR", "Washington Nationals": "WSN"
}
mlb_odds_df['home_team_abbrev'] = mlb_odds_df['home_team'].map(team_name_to_abbrev)
mlb_odds_df['away_team_abbrev'] = mlb_odds_df['away_team'].map(team_name_to_abbrev)

# points (totals) from any O/U rows
name_col = 'name' if 'name' in mlb_odds_df.columns else None
point_src = mlb_odds_df
if name_col:
    point_src = mlb_odds_df[mlb_odds_df[name_col].str.contains('over|under', case=False, na=False)].copy()
point_src['point'] = pd.to_numeric(point_src.get('point'), errors='coerce')

g_cols = ['game_date','home_team_abbrev','away_team_abbrev']
tot_med = point_src.groupby(g_cols, as_index=False)['point'].median().rename(columns={'point':'total_line'})
tot_std = point_src.groupby(g_cols, as_index=False)['point'].std().rename(columns={'point':'total_line_std'})
tot_rng = point_src.groupby(g_cols, as_index=False)['point'].agg(lambda s: s.max()-s.min()
                    ).rename(columns={'point':'total_line_range'})

# prices (robust)
def american_to_prob(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return (100.0/(odds+100.0)) if odds>0 else ((-odds)/(-odds+100.0))

price_over = None; price_under = None
if name_col and 'price' in mlb_odds_df.columns:
    over_rows  = mlb_odds_df[mlb_odds_df[name_col].str.contains('over',  case=False, na=False)]
    under_rows = mlb_odds_df[mlb_odds_df[name_col].str.contains('under', case=False, na=False)]
    po = over_rows.groupby(g_cols, as_index=False)['price'].median().rename(columns={'price':'price_over'})
    pu = under_rows.groupby(g_cols, as_index=False)['price'].median().rename(columns={'price':'price_under'})
    price_over, price_under = po, pu
else:
    # look for any columns like '*over*odds' and '*under*odds/price*'
    over_cols  = [c for c in mlb_odds_df.columns if re.search(r'over.*(odds|price)', c, re.I)]
    under_cols = [c for c in mlb_odds_df.columns if re.search(r'under.*(odds|price)', c, re.I)]
    if over_cols:
        po = mlb_odds_df.groupby(g_cols, as_index=False)[over_cols].median()
        po = po.rename(columns={over_cols[0]:'price_over'})[[*g_cols,'price_over']]
        price_over = po
    if under_cols:
        pu = mlb_odds_df.groupby(g_cols, as_index=False)[under_cols].median()
        pu = pu.rename(columns={under_cols[0]:'price_under'})[[*g_cols,'price_under']]
        price_under = pu

# assemble odds bundle
odds_bundle = tot_med.merge(tot_std, on=g_cols, how='left').merge(tot_rng, on=g_cols, how='left')
if price_over is not None:
    odds_bundle = odds_bundle.merge(price_over, on=g_cols, how='left')
if price_under is not None:
    odds_bundle = odds_bundle.merge(price_under, on=g_cols, how='left')

# compute vig if we have prices
if {'price_over','price_under'}.issubset(odds_bundle.columns):
    p_over  = odds_bundle['price_over' ].apply(american_to_prob)
    p_under = odds_bundle['price_under'].apply(american_to_prob)
    odds_bundle['vig'] = p_over + p_under - 1.0

# Merge odds
final_df['Date_Parsed'] = pd.to_datetime(final_df['Date_Parsed'], errors='coerce').dt.strftime('%Y-%m-%d')
pre_rows = len(final_df)
final_df = final_df.merge(
    odds_bundle,
    how='left',
    left_on=['Date_Parsed', 'Tm', 'Opp'],
    right_on=g_cols,
    validate='m:1'
)
assert len(final_df) == pre_rows, "Row count changed after odds merge!"
final_df.drop(columns=g_cols, inplace=True, errors='ignore')

# ================== OVER/UNDER LABELS + PRO FEATURES ==================
# ensure numeric
final_df['total_line'] = pd.to_numeric(final_df['total_line'], errors='coerce')
R_home = pd.to_numeric(final_df.get('R_home'), errors='coerce')
R_away = pd.to_numeric(final_df.get('R_away'), errors='coerce')
total_runs = R_home + R_away
line = final_df['total_line']

# masks
m_valid = line.notna()

# binary outcome: 1=Over, 0=Under, <NA>=Push or missing line
over_result = pd.Series(pd.NA, index=final_df.index, dtype='Int64')
over_result.loc[m_valid & (total_runs > line)] = 1
over_result.loc[m_valid & (total_runs < line)] = 0
final_df['over_result'] = over_result

# human-readable label
labels = pd.Series(pd.NA, index=final_df.index, dtype='string')
labels.loc[m_valid & (total_runs > line)] = 'Over'
labels.loc[m_valid & (total_runs < line)] = 'Under'
labels.loc[m_valid & (total_runs == line)] = 'Push'
final_df['over_under_label'] = labels

# Recent form (market vs form gap)
if {'rolling_10_R_home','rolling_10_R_away'}.issubset(final_df.columns):
    final_df['exp_runs_recent'] = pd.to_numeric(final_df['rolling_10_R_home'], errors='coerce') + \
                                  pd.to_numeric(final_df['rolling_10_R_away'], errors='coerce')
    final_df['line_minus_recent'] = final_df['total_line'] - final_df['exp_runs_recent']

# Starters’ recent run suppression context
era_h = pd.to_numeric(final_df.get('rolling_10_pitcher_1_ERA_home'), errors='coerce')
era_a = pd.to_numeric(final_df.get('rolling_10_pitcher_1_ERA_away'), errors='coerce')
final_df['starter_era_sum'] = era_h + era_a

# NEW: market-shape features
frac = (final_df['total_line'] - np.floor(final_df['total_line'])).abs()
final_df['is_half_total'] = (np.isclose(frac, 0.5, atol=0.02)).astype(int)
final_df['frac_from_half'] = (final_df['total_line'] - np.round(final_df['total_line']*2)/2).abs()

# Gentle clip to tame outliers (optional)
for c in ['total_line','exp_runs_recent','line_minus_recent','starter_era_sum',
          'total_line_std','total_line_range','price_over','price_under','vig',
          'rest_home','rest_away','delta_rest','rolling_10_R_overall_home','rolling_10_R_overall_away']:
    if c in final_df.columns:
        q = final_df[c].astype(float)
        ql, qh = q.quantile([0.001, 0.999])
        final_df[c] = pd.to_numeric(final_df[c], errors='coerce').clip(lower=ql, upper=qh)

# drop merge junk
drop_merge_cols = [
    'Date_clean_home', 'Date_clean_away',
    'game_in_day_home', 'game_in_day_away',
    'full_date_str_home', 'full_date_str_away',
    'merge_key', 'Boxscore'
]
final_df.drop(columns=[c for c in drop_merge_cols if c in final_df.columns], inplace=True, errors='ignore')

# -------------------- PARK FACTORS --------------------
try:
    parkfactors_df = pd.read_csv('./parkfactors/combined_parkfactors.csv')
    # Expecting columns ['Loc','Season', 'Park Factor', ...]
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

# -------------------- VERIFY ROLLING COLUMNS EXIST FOR HOME & AWAY --------------------
missing = []
for pos in range(1, 9+1):
    for stat in batter_stat_names:
        base = f"rolling_10_batter_{pos}_{stat}"
        h = f"{base}_home"
        a = f"{base}_away"
        if h not in final_df.columns:
            missing.append(h)
        if a not in final_df.columns:
            missing.append(a)
for stat in pitcher_stat_names:
    base = f"rolling_10_pitcher_1_{stat}"
    h = f"{base}_home"
    a = f"{base}_away"
    if h not in final_df.columns:
        missing.append(h)
    if a not in final_df.columns:
        missing.append(a)

if missing:
    print("❌ Missing rolling columns (home/away) that you asked for:")
    for m in missing[:20]:
        print("   -", m)
    if len(missing) > 20:
        print("   ...")
else:
    print("✅ All requested rolling columns exist for both HOME and AWAY.")

# -------------------- SAVE --------------------
batter_bd_cols = final_df.filter(
    regex=r'^batter_(?:[1-9]|1[0-7])_(?:Batting|Details)_(?:home|away)$'
).columns.tolist()

print(f"Dropping {len(batter_bd_cols)} batter Batting/Details columns (1–17, home/away).")
final_df.drop(columns=batter_bd_cols, inplace=True, errors='ignore')

# drop minor extras
columns_to_drop = ['Date_away','Inn_away','Inn_home', "Date"]
final_df.drop(columns=columns_to_drop,inplace=True,errors='ignore')

## weather
weather_cols = [
    "datetime_utc", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
    "wind_gusts_10m", "apparent_temperature", "Original_Loc", "Date"
]
weather_df = pd.read_csv("combined_weather_data.csv", usecols=weather_cols)

weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors="coerce").dt.strftime('%Y-%m-%d')
weather_df['Original_Loc'] = weather_df['Original_Loc'].astype(str)
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

# Reorder to move weather vars to the end (robust – no 'SO' dependency)
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

# simple interactions (optional, cheap)
if 'Park Factor' in final_df.columns and {'lineup_OPS_pwmean_home','lineup_OPS_pwmean_away'}.issubset(final_df.columns):
    final_df['park_x_ops_pw'] = final_df['Park Factor'] * (final_df['lineup_OPS_pwmean_home'] + final_df['lineup_OPS_pwmean_away'])
if 'wind_speed_10m' in final_df.columns and {'lineup_OPS_pwmean_home','lineup_OPS_pwmean_away'}.issubset(final_df.columns):
    final_df['wind_x_ops_pw'] = final_df['wind_speed_10m'] * (final_df['lineup_OPS_pwmean_home'] + final_df['lineup_OPS_pwmean_away'])
if 'temperature_2m' in final_df.columns and {'lineup_OPS_pwmean_home','lineup_OPS_pwmean_away'}.issubset(final_df.columns):
    final_df['temp_x_ops_pw'] = final_df['temperature_2m'] * (final_df['lineup_OPS_pwmean_home'] + final_df['lineup_OPS_pwmean_away'])
# NEW: more park-weather interactions
if 'Park Factor' in final_df.columns:
    if 'temperature_2m' in final_df.columns:
        final_df['park_x_temp'] = final_df['Park Factor'] * final_df['temperature_2m']
    if 'wind_speed_10m' in final_df.columns:
        final_df['park_x_wind10'] = final_df['Park Factor'] * final_df['wind_speed_10m']

# -------------------- TEAM NAME TO INDEX (after merges) --------------------
mlb_teams = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
    'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
    'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
]
team_to_index = {team: idx+1 for idx, team in enumerate(mlb_teams)}
def map_team_to_idx(team):
    if pd.isna(team):
        return 0
    team = str(team).strip().upper()
    return team_to_index.get(team, 0)
for col in ['Tm', 'Opp', 'Loc']:
    if col in final_df.columns:
        final_df[col] = final_df[col].apply(map_team_to_idx)

final_df['is_daytime'] = final_df['D/N'].apply(lambda x: 1 if x == 'D' else 0 )
final_df.drop(columns=['pitcher_1_Pitching_home','pitcher_1_Pitching_home','D/N'],inplace=True,errors='ignore')

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
        k = np.minimum(3, M.shape[1])
        top3 = np.sort(M, axis=1)[:, -k:] if k > 0 else np.full((len(df), 1), np.nan)
        out[f"lineup_OPS_top3mean_{side}"] = np.nanmean(top3, axis=1)

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

leak_exact = {
    'W/L_home','W/L_away','RA_home','RA_away',
    'W-L_home','W-L_away','Rank_home','Rank_away','GB_home','GB_away',
    'Streak_home','Streak_away',
    'R','H','1B','2B','3B','HR','BB','SO','PA'
}
drop_leak = [c for c in final_df.columns if c in leak_exact]

redundant_exact = {
    'At_home','At_away','is_home_home','is_home_away','D/N_away',
    'pitcher_1_Pitching_away'
}
drop_redundant = [c for c in final_df.columns if c in redundant_exact]

keep_pitcher_metrics = {'IP','H','R','ER','BB','HR','ERA','BF','Pit','Str','SO','GSc'}
pat_pitcher_keep = re.compile(rf"^rolling_10_pitcher_1_({'|'.join(sorted(keep_pitcher_metrics))})_(home|away)$")
pat_pitcher_all  = re.compile(r"^rolling_10_pitcher_1_.+_(home|away)$")
drop_pitcher_noise = [c for c in final_df.columns if pat_pitcher_all.match(c) and not pat_pitcher_keep.match(c)]

whitelist_exact = {
    'Tm','Opp','Season','Venue','Park Factor','is_daytime',
    'rolling_10_R_home','rolling_10_R_away',
    'rolling_10_R_overall_home','rolling_10_R_overall_away',  # NEW overall form
    'lineup_OPS_pwmean_home','lineup_OPS_pwmean_away',
    'lineup_OBP_pwmean_home','lineup_OBP_pwmean_away',
    'lineup_SLG_pwmean_home','lineup_SLG_pwmean_away',
    'lineup_RE24_pwmean_home','lineup_RE24_pwmean_away',
    'lineup_RBI_pwmean_home','lineup_RBI_pwmean_away',
    'lineup_R_pwmean_home','lineup_R_pwmean_away',
    'lineup_OPS_top3mean_home','lineup_OPS_top3mean_away',
    'lineup_hot_ops_cnt_home','lineup_hot_ops_cnt_away',
    'park_x_ops_pw','wind_x_ops_pw','temp_x_ops_pw',
    # odds+labels
    'total_line','total_line_std','total_line_range','price_over','price_under','vig',
    'over_result','over_under_label',
    # recency gap & starters
    'exp_runs_recent','line_minus_recent','starter_era_sum',
    # rest & market shape
    'rest_home','rest_away','delta_rest','b2b_home','b2b_away','is_half_total','frac_from_half',
    # park-weather extras
    'park_x_temp','park_x_wind10'
}
whitelist_prefixes = (
    'temperature_2m','relative_humidity_2m','wind_speed_10m','wind_speed_100m',
    'wind_direction_10m','wind_direction_100m','wind_gusts_10m','apparent_temperature',
    'wOBACon','xwOBACon','BACON','xBACON','HardHit','lineup_',
    'rolling_10_pitcher_1_'
)

candidate_drop = []
protect = {'Date_Parsed','Date_parsed_away'}
for c in final_df.columns:
    if c in protect:
        continue
    if (c in drop_leak) or (c in drop_redundant) or (c in drop_batter_slot_cols) or (c in drop_pitcher_noise):
        candidate_drop.append(c)
        continue
    if c in whitelist_exact:
        continue
    if any(c.startswith(p) for p in whitelist_prefixes):
        continue
    if any(tok in c for tok in ['Rank_', 'GB_', 'W/L_', 'W-L_']):
        candidate_drop.append(c)

before_cols = set(final_df.columns)
final_df.drop(columns=list(set(candidate_drop)), inplace=True, errors='ignore')
after_cols = set(final_df.columns)
print(f"[Prune] Dropped {len(before_cols - after_cols)} columns; kept {len(after_cols)}.")

# ================== LEAN FEATURE MATRIX (baseline) ==================
lean_cols = []
for col in ['Season','Park Factor','is_daytime']:
    if col in final_df.columns: lean_cols.append(col)
for col in ['rolling_10_R_home','rolling_10_R_away','rolling_10_R_overall_home','rolling_10_R_overall_away']:
    if col in final_df.columns: lean_cols.append(col)
for side in ('home','away'):
    for m in ['ERA','SO','BB','HR','IP','BF']:
        c = f'rolling_10_pitcher_1_{m}_{side}'
        if c in final_df.columns: lean_cols.append(c)
for col in ['temperature_2m','relative_humidity_2m','wind_speed_10m','wind_speed_100m',
            'wind_direction_10m','wind_direction_100m','wind_gusts_10m','apparent_temperature',
            'total_line','total_line_std','total_line_range','price_over','price_under','vig',
            'rest_home','rest_away','delta_rest','is_half_total','frac_from_half']:
    if col in final_df.columns: lean_cols.append(col)

lean_df = final_df[lean_cols].copy()

# Calendar features from Date_Parsed
_dt = pd.to_datetime(final_df.get('Date_Parsed', pd.NaT), errors='coerce')
month = _dt.dt.month.astype('float32')
doy = _dt.dt.dayofyear.astype('float32')
lean_df['month'] = month
lean_df['doy_sin'] = np.sin(2 * np.pi * (doy / 366.0))
lean_df['doy_cos'] = np.cos(2 * np.pi * (doy / 366.0))

# Wind components (if available)
if {'wind_speed_10m','wind_direction_10m'}.issubset(final_df.columns):
    spd10 = pd.to_numeric(final_df['wind_speed_10m'], errors='coerce')
    dir10 = pd.to_numeric(final_df['wind_direction_10m'], errors='coerce')
    rad10 = np.deg2rad(dir10)
    lean_df['wind_u10'] = (spd10 * np.cos(rad10)).astype('float32')
    lean_df['wind_v10'] = (spd10 * np.sin(rad10)).astype('float32')
if {'wind_speed_100m','wind_direction_100m'}.issubset(final_df.columns):
    spd100 = pd.to_numeric(final_df['wind_speed_100m'], errors='coerce')
    dir100 = pd.to_numeric(final_df['wind_direction_100m'], errors='coerce')
    rad100 = np.deg2rad(dir100)
    lean_df['wind_u100'] = (spd100 * np.cos(rad100)).astype('float32')
    lean_df['wind_v100'] = (spd100 * np.sin(rad100)).astype('float32')

# Median fill & gentle clip
for c in lean_df.columns:
    if not np.issubdtype(lean_df[c].dtype, np.number):
        lean_df[c] = pd.to_numeric(lean_df[c], errors='coerce')
    med = lean_df[c].median()
    lean_df[c] = lean_df[c].fillna(0.0 if pd.isna(med) else med)

q_low = lean_df.quantile(0.001)
q_hi  = lean_df.quantile(0.999)
lean_df = lean_df.clip(lower=q_low, upper=q_hi, axis=1)

lean_path = 'jordan_final_lean.csv'
lean_df.to_csv(lean_path, index=False)
print(f"[Lean] Saved lean feature matrix to {lean_path} with shape {lean_df.shape}")
# ================== /LEAN FEATURE MATRIX ==================

# --- SAVE FULL DATASET ---
final_df.drop(columns=['Gm#_home','Gm#_away','Opp','Tm','Season','Season_away','wOBACon','xwOBACon','BACON','xBACON','HardHit','OBP','Season'],inplace=True, errors='ignore')
to_drop = [
    'rolling_10_pitcher_1_IP_home',
    'rolling_10_pitcher_1_H_home',
    'rolling_10_pitcher_1_R_home',
    'rolling_10_pitcher_1_ER_home',
    'rolling_10_pitcher_1_Pit_home',
    'rolling_10_pitcher_1_Str_home',
    'rolling_10_pitcher_1_GSc_home',
    'Tm_away',
    'Opp_away',
    'Date_parsed_away',
    'rolling_10_pitcher_1_H_away',
    'rolling_10_pitcher_1_R_away',
    'rolling_10_pitcher_1_ER_away',
    'rolling_10_pitcher_1_Pit_away',
    'rolling_10_pitcher_1_Str_away',
    'rolling_10_pitcher_1_SO_away',
    'rolling_10_pitcher_1_GSc_away',
    'Team',
    'Venue',
    'lineup_OPS_std_home',
    'lineup_OBP_std_home',
    'lineup_SLG_std_home',
    'lineup_BA_std_home',
    'lineup_PA_std_home',
    'lineup_R_std_home',
    'lineup_RBI_std_home',
    'lineup_BB_std_home',
    'lineup_SO_std_home',
    'lineup_WPA_std_home',
    'lineup_RE24_std_home',
    'lineup_OPS_std_away',
    'lineup_OBP_std_away',
    'lineup_SLG_std_away',
    'lineup_BA_std_away',
    'lineup_PA_std_away',
    'lineup_R_std_away',
    'lineup_RBI_std_away',
    'lineup_BB_std_away',
    'lineup_SO_std_away',
    'lineup_WPA_std_away',
    'lineup_RE24_std_away',
] + [
    'Date_parsed_away',
    'rolling_10_pitcher_1_IP_away',
    'rolling_10_pitcher_1_BF_home', 'rolling_10_pitcher_1_BF_away',
    'wind_speed_100m', 'wind_direction_10m', 'wind_direction_100m', 'apparent_temperature',
    'lineup_OPS_q75_home', 'lineup_OPS_top3mean_home',
    'lineup_OBP_q75_home', 'lineup_OBP_top3mean_home',
    'lineup_SLG_q75_home', 'lineup_SLG_top3mean_home',
    'lineup_BA_q75_home',  'lineup_BA_top3mean_home',
    'lineup_PA_q75_home',  'lineup_PA_top3mean_home',
    'lineup_R_q75_home',   'lineup_R_top3mean_home',
    'lineup_RBI_q75_home', 'lineup_RBI_top3mean_home',
    'lineup_BB_q75_home',  'lineup_BB_top3mean_home',
    'lineup_SO_q75_home',  'lineup_SO_top3mean_home',
    'lineup_WPA_mean_home','lineup_WPA_q75_home','lineup_WPA_top3mean_home',
    'lineup_RE24_q75_home','lineup_RE24_top3mean_home',
    'lineup_hot_ops_cnt_home',
    'lineup_OPS_q75_away', 'lineup_OPS_top3mean_away',
    'lineup_OBP_q75_away', 'lineup_OBP_top3mean_away',
    'lineup_SLG_q75_away', 'lineup_SLG_top3mean_away',
    'lineup_BA_q75_away',  'lineup_BA_top3mean_away',
    'lineup_PA_q75_away',  'lineup_PA_top3mean_away',
    'lineup_R_q75_away',   'lineup_R_top3mean_away',
    'lineup_RBI_q75_away', 'lineup_RBI_top3mean_away',
    'lineup_BB_q75_away',  'lineup_BB_top3mean_away',
    'lineup_SO_q75_away',  'lineup_SO_top3mean_away',
    'lineup_WPA_mean_away','lineup_WPA_q75_away','lineup_WPA_top3mean_away',
    'lineup_RE24_q75_away','lineup_RE24_top3mean_away',
    'lineup_hot_ops_cnt_away'
]

present = [c for c in to_drop if c in final_df.columns]
missing = [c for c in to_drop if c not in final_df.columns]

final_df.drop(columns=present, inplace=True, errors='ignore')
print(f"[Drop] Removed {len(present)} columns. {len(missing)} not found.")
if missing:
    print("[Drop] Not found:", missing[:20], "..." if len(missing) > 20 else "")
final_df.to_csv('jordan_final.csv', index=False)
print("Final merged dataset shape:", final_df.shape)
print(list(final_df.columns))
