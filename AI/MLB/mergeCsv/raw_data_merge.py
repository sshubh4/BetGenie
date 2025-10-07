import pandas as pd
import numpy as np
import glob
import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

csv_files = glob.glob("./teamgamelogs/*.csv")

df_list = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(df_list, ignore_index=True)

merged_df.drop(columns=[
    "W-L",
    "Gm#",
    "GB",
    "Boxscore",
    "Boxscore URL",
    "Inn",
    "Save",
    "Orig. Scheduled",
    "Num_Batters",
    "umpire_HP",
    "umpire_1B",
    "umpire_2B",
    "umpire_3B",
    "umpire_LF",
    "umpire_RF",
    "time_of_game",
    "attendance",
    "field_condition",
    "weather_temp",
    "weather_wind",
    "weather_sky",
    "weather_precip",
    "batter_1_Details",
    "batter_2_Details",
    "batter_3_Details",
    "batter_4_Details",
    "batter_5_Details",
    "batter_6_Details",
    "batter_7_Details",
    "batter_8_Details",
    "batter_9_Details",
    "batter_10_Details",
    "batter_11_Details",
    "batter_12_Details",
    "batter_13_Details",
    "batter_14_Details",
    "batter_15_Details",
    "batter_16_Details",
    "pitcher_1_IR",
    "pitcher_1_IS",
    "pitcher_1_WPA",
    "pitcher_1_acLI",
    "pitcher_1_GSc",
    "pitcher_2_GSc",
    "pitcher_2_IR",
    "pitcher_2_IS",
    "pitcher_2_WPA",
    "pitcher_2_acLI",
    "pitcher_1_SO",
    "pitcher_2_SO",
    "pitcher_3_SO",
    "pitcher_4_SO",
    "pitcher_5_SO",
    "pitcher_6_SO",
    "pitcher_7_SO",
    "pitcher_8_SO",
    "pitcher_9_SO",
    "pitcher_1_Ctct",
    "pitcher_2_Ctct",
    "pitcher_3_Ctct",
    "pitcher_4_Ctct",
    "pitcher_5_Ctct",
    "pitcher_6_Ctct",
    "pitcher_7_Ctct",
    "pitcher_8_Ctct",
    "pitcher_9_Ctct"
], inplace=True)

# Remove all columns batter_10_* to batter_16_* and pitcher_2_* to pitcher_9_*
cols_to_drop = [col for col in merged_df.columns
                if any(
                    col.startswith(f"batter_{i}_") for i in range(10,17)
                ) or any(
                    col.startswith(f"pitcher_{i}_") for i in range(2,10)
                )]
merged_df.drop(columns=cols_to_drop, inplace=True)




final_df = merged_df.copy()

final_df["Date"] = final_df["Date"].str.replace(r"\s*\([^)]*\)", "", regex=True).str.strip()

final_df["Date"] = final_df["Date"].str.replace(r"^\w+,\s*", "", regex=True).str.strip()

final_df["full_date_str"] = final_df["Date"] + " " + final_df["Season"].astype(str)

final_df["Date"] = pd.to_datetime(final_df["full_date_str"], format="%b %d %Y").dt.strftime("%Y-%m-%d")

final_df.drop(columns=["full_date_str"], inplace=True)

#loc:locations
final_df["Loc"] = final_df.apply(lambda row: row["Opp"] if row["At"] == "@" else row["Tm"], axis=1)


cols = list(final_df.columns)
opp_index = cols.index("Opp")
cols.insert(opp_index + 1, cols.pop(cols.index("Loc")))
final_df = final_df[cols]

#sort
final_df["Date"] = pd.to_datetime(final_df["Date"])
final_df = final_df.sort_values(by="Date").reset_index(drop=True)

# Clean the W/L 
final_df["W/L"] = final_df["W/L"].str.replace(r"-wo", "", regex=True)

#transformations
final_df['Attendance'] = final_df['Attendance'].str.replace(',', '', regex=False).fillna('0').astype(int)
avg_attendance = final_df.loc[final_df['Attendance'] != 0, 'Attendance'].mean()
final_df.loc[final_df['Attendance'] == 0, 'Attendance'] = int(round(avg_attendance))

wl_map = {'W': 1, 'L': 0}
final_df['W/L'] = final_df['W/L'].map(wl_map)

final_df['At'] = final_df['At'].apply(lambda x: 1 if x == '@' else 0)

final_df['D/N'] = final_df['D/N'].map({'D': 1, 'N': 0}).fillna(0).astype(int)

final_df['Streak'] = final_df['Streak'].apply(lambda x: ''.join('0' if ch == '+' else '1' for ch in str(x)))

final_df['umpire_HP'] = final_df['umpire_HP'].fillna('N/A')
final_df.loc[final_df['umpire_HP'].astype(str).str.strip() == '', 'umpire_HP'] = 'N/A'

# Replace missing/empty values in batter name columns with 'N/A'
for i in range(1, 11):
    col = f"batter_{i}_Batting"
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna('N/A')
        final_df.loc[final_df[col].astype(str).str.strip() == '', col] = 'N/A'

# Replace missing/empty values in pitcher name columns with 'N/A'
for i in range(1, 2):
    col = f"pitcher_{i}_Pitching"
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna('N/A')
        final_df.loc[final_df[col].astype(str).str.strip() == '', col] = 'N/A'


final_df = final_df[cols]

#update names
position_codes = [
    'p',    # Pitcher
    'c',    # Catcher
    '1b',   # First base
    '2b',   # Second base
    '3b',   # Third base
    'ss',   # Shortstop
    'lf',   # Left field
    'cf',   # Center field
    'rf',   # Right field
    'dh',   # Designated hitter
    
    # Additional common abbreviations
    'ph',   # Pinch hitter
    'pr',   # Pinch runner
    'ut',   # Utility player
    'sf',   # Sacrifice fly? Sometimes used for position (rare)
    'fb',   # Foul ball? Sometimes misused abbreviation
    'gf',   # Ground foul? Rare - can omit if not used
    
    # More field roles/shifts variants
    'lm',   # Left midfield/Left (used sometimes)
    'rm',   # Right midfield/Right (used sometimes)
    'lb',   # Left base? (Rare, but sometimes used)
    'cfr',  # Center fielder Right?? Rare combo
    'cf-lf',# Center-left outfield combo
    '1b-rf',# Multi-position combo common in your data
    
    # Commonly observed combos (should be matched by allowing dashes)
    
    # Pitcher types/tags sometimes added
    'sp',   # Starting pitcher
    'rp',   # Relief pitcher
    'cl',   # Closer
    
    # Miscellaneous misc tags sometimes found
    'jr',   # Suffix for players (not a position but common)
    'sr',   # Same as above
    
    # You may also include numerics variations if used
    '4f',   # 4th fielder (rare)
    '5f',   # 5th fielder
    '6f',   # 6th fielder
    '7f',   # 7th fielder

    # etc.
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
    name = name.strip()
    
    # Remove anything after and including the first comma (e.g. "Jake Arrieta, W (1-0)" -> "Jake Arrieta")
    name = comma_and_after_pattern.sub("", name).strip()
    
    # Remove trailing parentheses with any content
    name = paren_pattern.sub("", name).strip()
    # Remove trailing commas (if any remain)
    name = comma_pattern.sub("", name).strip()

    # Iteratively remove trailing position codes
    while True:
        new_name = pos_pattern.sub("", name).strip()
        if new_name == name:
            break
        name = new_name
    return name

for i in range(1, 17):
    col = f'batter_{i}_Batting'
    if col in final_df.columns:
        final_df[col] = final_df[col].astype(str).apply(clean_player_name)
        final_df[f'batter_{i}_num'] = i

for i in range(1, 10):
    col = f'pitcher_{i}_Pitching'
    if col in final_df.columns:
        final_df[col] = final_df[col].astype(str).apply(clean_player_name)
        final_df[f'pitcher_{i}_num'] = i

def clean_column_percentages(df):
    for col in df.columns:
        # Process only string/object columns that might contain '%'
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            # If any values contain '%', clean the entire column
            if df[col].astype(str).str.contains('%').any():
                df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

final_df = clean_column_percentages(final_df)

########################################
final_df['At'] = final_df['At'].astype(int)
final_df = final_df.sort_values(by="At", ascending=True).reset_index(drop=True)

batter_cols = [f"batter_{i}_Batting" for i in range(1, 17) if f"batter_{i}_Batting" in final_df.columns]
pitcher_cols = [f"pitcher_{i}_Pitching" for i in range(1, 10) if f"pitcher_{i}_Pitching" in final_df.columns]

def count_valid_names(row, columns):
    count = 0
    for col in columns:
        val = str(row[col]).strip()
        if val and val != 'N/A':
            count += 1
    return count

final_df['num_Batters'] = final_df.apply(count_valid_names, columns=batter_cols, axis=1)
final_df['num_Pitchers'] = final_df.apply(count_valid_names, columns=pitcher_cols, axis=1)

def clean_name_col(col_series):
    return col_series.apply(lambda x: 'N/A' if (pd.isna(x) or str(x).strip().lower() in ['', 'nan']) else x)

for col in batter_cols:
    final_df[col] = clean_name_col(final_df[col])

for col in pitcher_cols:
    final_df[col] = clean_name_col(final_df[col])

cols = list(final_df.columns)
umpire_hp_index = cols.index('umpire_HP')

if 'num_Batters' in cols:
    cols.remove('num_Batters')
if 'num_Pitchers' in cols:
    cols.remove('num_Pitchers')

cols.insert(umpire_hp_index + 1, 'num_Pitchers')
cols.insert(umpire_hp_index + 1, 'num_Batters')

final_df = final_df[cols]

##0 out the data
def fill_missing_numeric_with_zero(df):
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].fillna(0)
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].fillna(0.0)
    return df

final_df = fill_missing_numeric_with_zero(final_df)
#
##cleans batterinvalid names and pitchers too

def remove_numeric_start_entries(df):
    batter_cols = [col for col in df.columns if col.startswith('batter_') and col.endswith('_Batting')]
    pitcher_cols = [col for col in df.columns if col.startswith('pitcher_') and col.endswith('_Pitching')]

    all_cols = batter_cols + pitcher_cols

    for col in all_cols:
        def filter_numeric_start(val):
            if pd.isna(val):
                return 'N/A'
            val_str = str(val).strip()
            if re.match(r'^\d+(\.\d+)?', val_str):
                return 'N/A'
            return val_str

        df[col] = df[col].apply(filter_numeric_start)
    
    return df

final_df = remove_numeric_start_entries(final_df)
### clean batter and pitcher names with comma and postions

def keep_before_comma(df):
    batter_cols = [col for col in df.columns if col.startswith('batter_') and col.endswith('_Batting')]
    pitcher_cols = [col for col in df.columns if col.startswith('pitcher_') and col.endswith('_Pitching')]

    all_cols = batter_cols + pitcher_cols 

    for col in all_cols:
        def process_val(val):
            if pd.isna(val):
                return val
            val_str = str(val)
            # Keep only part before the first comma
            return val_str.split(',', 1)[0].strip()

        df[col] = df[col].apply(process_val)
    
    return df
final_df = keep_before_comma(final_df)

###
final_away_df = final_df[final_df['At'] == 1].copy()
final_home_df = final_df[final_df['At'] == 0].copy()

# Assume final_home_df and final_away_df are your prepared DataFrames
home = final_home_df.copy()
away = final_away_df.copy()

home_copy = home.copy()
away_copy = away.copy()

# home_copy['game_Index'] = range(1, len(home_copy) + 1)
# cols = ['game_Index'] + [col for col in home_copy.columns if col != 'game_Index']
# home_copy = home_copy[cols]

# away_copy['game_Index'] = range(1, len(away_copy) + 1)
# cols = ['game_Index'] + [col for col in away_copy.columns if col != 'game_Index']
# away_copy = away_copy[cols]

##3 neccessary for future games.py
home_copy.to_csv('home.csv', index=False)
print("Home data saved to home.csv")
away_copy.to_csv('away.csv', index=False)
print("Away data saved to away.csv")


# Create merge keys
home['merge_key'] = home['Date'].astype(str) + '_' + home['Tm'] + '_' + home['Opp']
away['merge_key'] = away['Date'].astype(str) + '_' + away['Opp'] + '_' + away['Tm']

# Merge home and away on merge_key
merged = pd.merge(
    home, 
    away, 
    on='merge_key', 
    suffixes=('_home', '_away'), 
    how='inner'
)

# 1. General fields from home (no suffix)
general_fields_home = [
    'game_Index', 'Date', 'Tm', 'Opp', 'Loc',
    'Time', 'D/N', 'Attendance', 'Season', 'umpire_HP'
]
general_home_cols = [col + '_home' for col in general_fields_home]
general_rename = {col+'_home': col for col in general_fields_home}

# 2. Summary fields to be prefixed for home and away
summary_fields = [
    'W/L', 'R', 'RA', 'Rank', 'Win', 'Loss', 'cLI', 'Streak',
    'num_Batters', 'num_Pitchers'
]

home_summary_cols = [f"{col}_home" for col in summary_fields]
home_summary_rename = {col: f"home_{col.replace('_home','')}" for col in home_summary_cols}

away_summary_cols = [f"{col}_away" for col in summary_fields]
away_summary_rename = {col: f"away_{col.replace('_away','')}" for col in away_summary_cols}

# 3. Batter and pitcher cols (prefix for BOTH home and away)

# Exclude columns already included (general + summary + merge key)
excluded_home_cols = set(general_home_cols + home_summary_cols + ['merge_key'])
excluded_away_cols = set(away_summary_cols + ['merge_key'])

# Home batter/pitcher columns
home_batter_pitcher_cols = [
    col for col in merged.columns 
    if (col.startswith('batter_') or col.startswith('pitcher_')) and col.endswith('_home') 
    and col not in excluded_home_cols
]

# Away batter/pitcher columns
away_batter_pitcher_cols = [
    col for col in merged.columns 
    if (col.startswith('batter_') or col.startswith('pitcher_')) and col.endswith('_away') 
    and col not in excluded_away_cols
]

# Rename batter/pitcher columns adding prefixes explicitly, dropping suffixes
home_batter_pitcher_rename = {
    col: 'home_' + col[:-5] for col in home_batter_pitcher_cols  # remove '_home' suffix
}

away_batter_pitcher_rename = {
    col: 'away_' + col[:-5] for col in away_batter_pitcher_cols  # remove '_away' suffix
}

# Combine all columns for final DataFrame
final_columns = (
    general_home_cols 
    + home_summary_cols 
    + away_summary_cols 
    + home_batter_pitcher_cols 
    + away_batter_pitcher_cols
)

# Filter final_columns to only those existing
final_columns = [col for col in final_columns if col in merged.columns]

# Combine all renaming dicts
rename_map = {}
rename_map.update(general_rename)
rename_map.update(home_summary_rename)
rename_map.update(away_summary_rename)
rename_map.update(home_batter_pitcher_rename)
rename_map.update(away_batter_pitcher_rename)

# Build final DataFrame
final_df = merged[final_columns].rename(columns=rename_map)

final_df["Date"] = pd.to_datetime(final_df["Date"])
final_df = final_df.sort_values(by="Date").reset_index(drop=True)

final_df['game_Index'] = range(1, len(final_df) + 1)
cols = ['game_Index'] + [col for col in final_df.columns if col != 'game_Index']
final_df = final_df[cols]

##park factors

park_factors_df = pd.read_csv('./parkfactors/combined_parkfactors.csv')

park_factor_columns = [
    'Park Factor', 'wOBACon', 'xwOBACon', 'BACON', 'xBACON', 'HardHit',
    'R', 'OBP', 'H', '1B', '2B', '3B', 'HR', 'BB', 'SO'
]

# Merge park factors on Season and Loc
final_df = pd.merge(
    final_df,
    park_factors_df[['Season', 'Loc'] + park_factor_columns],
    how='left',
    on=['Season', 'Loc']
)

# Insert park factor columns after 'away_num_Pitchers'
insert_after = 'away_num_Pitchers'
col_list = list(final_df.columns)

if insert_after not in col_list:
    raise ValueError(f"Column '{insert_after}' not found in final_df")

insert_pos = col_list.index(insert_after) + 1

# Remove park factor columns from their current positions at the end
for col in park_factor_columns:
    if col in col_list:
        col_list.remove(col)

# Insert park factor columns at correct location
for i, col in enumerate(park_factor_columns):
    col_list.insert(insert_pos + i, col)

final_df.to_csv('merged_data.csv', index=False)
print("Successfully merged CSV files into 'merged_data.csv'")



#print
# final_df.to_csv("mix.csv", index=False)

