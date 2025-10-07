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
for i in range(1, 16):
    col = f"batter_{i}_Batting"
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna('N/A')
        final_df.loc[final_df[col].astype(str).str.strip() == '', col] = 'N/A'

# Replace missing/empty values in pitcher name columns with 'N/A'
for i in range(1, 10):
    col = f"pitcher_{i}_Pitching"
    if col in final_df.columns:
        final_df[col] = final_df[col].fillna('N/A')
        final_df.loc[final_df[col].astype(str).str.strip() == '', col] = 'N/A'

# Get current list of columns
cols = list(final_df.columns)
insert_pos = cols.index('batter_15_A') + 1

batter_16_cols = [col for col in cols if col.startswith('batter_16_')]

for col in batter_16_cols:
    cols.remove(col)

for i, col in enumerate(batter_16_cols):
    cols.insert(insert_pos + i, col)

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

# #clean%

# for i in range(1, 17):
#     old_col = f"batter_{i}_cWPA"
#     new_col = f"batter_{i}_cWPA(%)"
#     if old_col in final_df.columns:
#         final_df[old_col] = final_df[old_col].astype(str).str.replace('%', '', regex=False).str.strip()
#         final_df[old_col] = pd.to_numeric(final_df[old_col], errors='coerce')
#         final_df.rename(columns={old_col: new_col}, inplace=True)

# def clean_numeric_percent(val):
#     """
#     Convert values that can be a number or a number with '%' to a float.
#     If the value is a percentage (e.g. '-0.1%'), remove % and convert to float.
#     If value is a clean numeric string or float, convert to float.
#     Otherwise, return 0.
#     """
#     if pd.isna(val):
#         return 0.0
    
#     # Convert to string first and strip whitespace
#     val_str = str(val).strip()
    
#     # Check if value contains '%'
#     if val_str.endswith('%'):
#         # Remove percentage sign
#         val_str = val_str[:-1].strip()
#         # Attempt to convert to float
#         try:
#             return float(val_str)
#         except ValueError:
#             return 0.0
    
#     # For other values, try direct float conversion
#     # Also handle values like "-0.01", "0", "1.0", "2"
#     try:
#         # If numeric string or float, return float version
#         return float(val_str)
#     except ValueError:
#         # If not numeric, return 0
#         return 0.0

# # Example usage on your columns:
# cols_to_clean = [
#     'pitcher_1_R', 'pitcher_1_BF', 'pitcher_1_GB', 'pitcher_1_cWPA',
#     'pitcher_2_R', 'pitcher_2_BF', 'pitcher_2_GB', 'pitcher_2_cWPA',
#     'pitcher_3_R', 'pitcher_3_BF', 'pitcher_3_GB', 'pitcher_3_cWPA',
#     'pitcher_4_R', 'pitcher_4_BF', 'pitcher_4_cWPA',
#     'pitcher_5_R', 'pitcher_5_BF', 'pitcher_5_cWPA',
#     'pitcher_6_R', 'pitcher_6_BF', 'pitcher_6_cWPA',
#     'pitcher_7_R', 'pitcher_7_BF', 'pitcher_7_cWPA',
#     'pitcher_8_R', 'pitcher_8_cWPA',
#     'pitcher_9_cWPA'
# ]

# # To apply in your DataFrame (named final_df), something like:
# for col in cols_to_clean:
#     if col in final_df.columns:
#         final_df[col] = final_df[col].apply(clean_numeric_percent).astype(float)

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

# Function to consolidate individual player stats into team averages
def consolidate_team_stats(df):
    """Consolidate individual batter/pitcher stats into team averages"""
    consolidated = df.copy()
    
    # Get all batter columns (batter_1_Batting, batter_2_Batting, etc.)
    batter_cols = [col for col in df.columns if 'batter_' in col and '_Batting' in col]
    pitcher_cols = [col for col in df.columns if 'pitcher_' in col and '_Pitching' in col]
    
    # Get all stat columns for batters and pitchers
    batter_stat_cols = [col for col in df.columns if 'batter_' in col and '_Batting' not in col]
    pitcher_stat_cols = [col for col in df.columns if 'pitcher_' in col and '_Pitching' not in col]
    
    # Remove all individual player columns (names and stats)
    cols_to_remove = batter_cols + pitcher_cols + batter_stat_cols + pitcher_stat_cols
    consolidated = consolidated.drop(columns=[col for col in cols_to_remove if col in consolidated.columns])
    
    # Add team batting averages
    print("Consolidating team batting averages...")
    for idx, row in df.iterrows():
        # Calculate averages for all batter stats for this game
        team_batting_avgs = {}
        for stat_name in set([col.replace(f'batter_{i}_', '') for i in range(1, 17) for col in batter_stat_cols if col.startswith(f'batter_{i}_')]):
            values = []
            for i in range(1, 17):  # Up to 16 batters
                stat_col = f'batter_{i}_{stat_name}'
                if stat_col in df.columns and pd.notna(row[stat_col]):
                    value = pd.to_numeric(row[stat_col], errors='coerce')
                    if pd.notna(value):
                        values.append(value)
            
            if values:  # Only add if we have valid values
                avg_value = sum(values) / len(values)
                col_name = f'team_batting_{stat_name}_avg'
                if col_name not in consolidated.columns:
                    consolidated[col_name] = 0.0
                if consolidated[col_name].dtype == 'int64':
                    consolidated[col_name] = consolidated[col_name].astype(float)
                consolidated.at[idx, col_name] = avg_value
    
    print("Consolidating team pitching averages...")
    for idx, row in df.iterrows():
        # Calculate averages for all pitcher stats for this game
        team_pitching_avgs = {}
        for stat_name in set([col.replace(f'pitcher_{i}_', '') for i in range(1, 10) for col in pitcher_stat_cols if col.startswith(f'pitcher_{i}_')]):
            values = []
            for i in range(1, 10):  # Up to 9 pitchers
                stat_col = f'pitcher_{i}_{stat_name}'
                if stat_col in df.columns and pd.notna(row[stat_col]):
                    value = pd.to_numeric(row[stat_col], errors='coerce')
                    if pd.notna(value):
                        values.append(value)
            
            if values:  # Only add if we have valid values
                avg_value = sum(values) / len(values)
                col_name = f'team_pitching_{stat_name}_avg'
                if col_name not in consolidated.columns:
                    consolidated[col_name] = 0.0
                if consolidated[col_name].dtype == 'int64':
                    consolidated[col_name] = consolidated[col_name].astype(float)
                consolidated.at[idx, col_name] = avg_value
    
    return consolidated

# Function to calculate days rest for each team
def calculate_days_rest(df, team_column='Tm'):
    """Calculate days rest for each team in each game"""
    df_with_rest = df.copy()
    df_with_rest['Date'] = pd.to_datetime(df_with_rest['Date'])
    df_with_rest = df_with_rest.sort_values(['Tm', 'Date'])
    
    # Calculate days rest for each team
    df_with_rest['days_rest'] = 0
    
    for team in df_with_rest[team_column].unique():
        if pd.isna(team) or str(team).strip() == 'N/A':
            continue
            
        team_games = df_with_rest[df_with_rest[team_column] == team].copy()
        team_games = team_games.sort_values('Date')
        
        for i in range(1, len(team_games)):
            current_date = team_games.iloc[i]['Date']
            previous_date = team_games.iloc[i-1]['Date']
            days_rest = (current_date - previous_date).days - 1  # -1 because day of game doesn't count as rest
            df_with_rest.loc[team_games.iloc[i].name, 'days_rest'] = max(0, days_rest)
    
    return df_with_rest

# Function to calculate head-to-head record
def calculate_head_to_head(df, team_column='Tm', opp_column='Opp'):
    """Calculate head-to-head record for each team vs their opponent"""
    df_with_h2h = df.copy()
    df_with_h2h['Date'] = pd.to_datetime(df_with_h2h['Date'])
    df_with_h2h = df_with_h2h.sort_values('Date')
    
    # Initialize head-to-head columns
    df_with_h2h['h2h_wins'] = 0
    df_with_h2h['h2h_losses'] = 0
    df_with_h2h['h2h_record'] = 0.0  # Win percentage
    
    for idx, row in df_with_h2h.iterrows():
        team = row[team_column]
        opponent = row[opp_column]
        current_date = row['Date']
        
        if pd.isna(team) or pd.isna(opponent) or str(team).strip() == 'N/A' or str(opponent).strip() == 'N/A':
            continue
        
        # Get all previous games between these teams
        previous_games = df_with_h2h[
            (df_with_h2h['Date'] < current_date) & 
            ((df_with_h2h[team_column] == team) & (df_with_h2h[opp_column] == opponent) |
             (df_with_h2h[team_column] == opponent) & (df_with_h2h[opp_column] == team))
        ]
        
        if len(previous_games) > 0:
            # Count wins for current team
            wins = 0
            losses = 0
            
            for _, game in previous_games.iterrows():
                if game[team_column] == team:
                    # Current team was home team
                    if game.get('R', 0) > game.get('RA', 0):  # Home team won
                        wins += 1
                    else:
                        losses += 1
                else:
                    # Current team was away team
                    if game.get('R', 0) < game.get('RA', 0):  # Away team won
                        wins += 1
                    else:
                        losses += 1
            
            df_with_h2h.at[idx, 'h2h_wins'] = wins
            df_with_h2h.at[idx, 'h2h_losses'] = losses
            total_games = wins + losses
            if total_games > 0:
                df_with_h2h.at[idx, 'h2h_record'] = wins / total_games
    
    return df_with_h2h

# Function to calculate last 10 games average team stats for each game (rolling window)
def calculate_rolling_team_stats(df, team_column='Tm', home_away='home'):
    """Calculate last 10 games average team stats for each game using rolling window"""
    df_with_stats = df.copy()
    df_with_stats['Date'] = pd.to_datetime(df_with_stats['Date'])
    df_with_stats = df_with_stats.sort_values(['Tm', 'Date'])
    
    # Get all numeric columns (excluding team name, date, etc.)
    exclude_cols = ['Date', team_column, 'Opp', 'Rk', 'Gtm', 'Unnamed: 0']
    numeric_cols = []
    
    for col in df_with_stats.columns:
        if col not in exclude_cols:
            try:
                # Convert to numeric, coercing errors to NaN
                numeric_data = pd.to_numeric(df_with_stats[col], errors='coerce')
                # Only include if we have at least some numeric values
                if not numeric_data.isna().all():
                    numeric_cols.append(col)
            except:
                continue
    
    # Calculate rolling averages for each team
    for team in df_with_stats[team_column].unique():
        if pd.isna(team) or str(team).strip() == 'N/A':
            continue
            
        # Get all games for this team
        team_games = df_with_stats[df_with_stats[team_column] == team].copy()
        team_games = team_games.sort_values('Date')
        
        # For each game, calculate average of previous games (up to 10)
        for i, (idx, row) in enumerate(team_games.iterrows()):
            if i == 0:
                # First game - use current game stats (no history)
                for col in numeric_cols:
                    if col in team_games.columns:
                        try:
                            # Convert to numeric
                            value = pd.to_numeric(row[col], errors='coerce')
                            if pd.notna(value):
                                df_with_stats.at[idx, f'{col}_last10_avg'] = value
                        except:
                            # Skip columns that can't be converted to numeric
                            continue
            else:
                # Get previous games (up to 10)
                start_idx = max(0, i - 10)
                previous_games = team_games.iloc[start_idx:i]
                
                if len(previous_games) > 0:
                    # Calculate averages for numeric columns
                    for col in numeric_cols:
                        if col in previous_games.columns:
                            try:
                                # Convert to numeric and calculate mean
                                numeric_data = pd.to_numeric(previous_games[col], errors='coerce')
                                if not numeric_data.isna().all():
                                    avg_value = numeric_data.mean()
                                    df_with_stats.at[idx, f'{col}_last10_avg'] = avg_value
                            except:
                                # Skip columns that can't be converted to numeric
                                continue
                else:
                    # No previous games - use current game stats
                    for col in numeric_cols:
                        if col in team_games.columns:
                            try:
                                # Convert to numeric
                                value = pd.to_numeric(row[col], errors='coerce')
                                if pd.notna(value):
                                    df_with_stats.at[idx, f'{col}_last10_avg'] = value
                            except:
                                # Skip columns that can't be converted to numeric
                                continue
    
    return df_with_stats

# Add days rest and head-to-head features
print("Calculating days rest and head-to-head records...")
home_with_features = calculate_days_rest(home, 'Tm')
home_with_features = calculate_head_to_head(home_with_features, 'Tm', 'Opp')

# Consolidate individual player stats into team totals
print("Consolidating home team stats...")
home_consolidated = consolidate_team_stats(home_with_features)

# Process home team last 10 HOME games stats (rolling window)
print("Processing home team last 10 HOME games stats...")
home_last_10 = calculate_rolling_team_stats(home_consolidated, 'Tm', 'home')
print(f"Calculated rolling stats for home teams")

# Create separate DataFrame for away processing (don't affect home data)
print("Processing away team data separately...")
away_separate = away.copy()

# Add days rest and head-to-head features for away teams
print("Calculating days rest and head-to-head records for away teams...")
away_with_features = calculate_days_rest(away_separate, 'Tm')
away_with_features = calculate_head_to_head(away_with_features, 'Tm', 'Opp')

# Consolidate individual player stats into team totals
print("Consolidating away team stats...")
away_consolidated = consolidate_team_stats(away_with_features)

# Process away team last 10 AWAY games stats (rolling window)
print("Processing away team last 10 AWAY games stats...")
away_last_10 = calculate_rolling_team_stats(away_consolidated, 'Tm', 'away')
print(f"Calculated rolling stats for away teams")

print("Last 10 games stats calculation complete!")
print(f"Home team processed: {len(home_last_10)} games")
print(f"Away team processed: {len(away_last_10)} games")

# Create merge keys
home_last_10['merge_key'] = home_last_10['Date'].astype(str) + '_' + home_last_10['Tm'] + '_' + home_last_10['Opp'] + '_' + home_last_10['game_in_day'].astype(str)
away_last_10['merge_key'] = away_last_10['Date'].astype(str) + '_' + away_last_10['Opp'] + '_' + away_last_10['Tm'] + '_' + away_last_10['game_in_day'].astype(str)

# Merge home and away on merge_key
merged = pd.merge(
    home_last_10, 
    away_last_10, 
    on='merge_key', 
    suffixes=('_home', '_away'), 
    how='inner'
)

# Validate the merge
print(f"\nMerge validation:")
print(f"Total merged games: {len(merged)}")
print(f"Sample merge keys: {merged['merge_key'].head(3).tolist()}")

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

# Get only the last 10 rolling average columns (not team averages)
last10_feature_cols = [
    col for col in merged.columns 
    if (col.endswith('_last10_avg_home') or 
        col.endswith('_last10_avg_away') or
        col.startswith('days_rest') or 
        col.startswith('h2h_'))
]

# Combine all columns for final DataFrame
final_columns = (
    general_home_cols 
    + home_summary_cols 
    + away_summary_cols 
    + home_batter_pitcher_cols 
    + away_batter_pitcher_cols
    + last10_feature_cols  # Add only last 10 rolling averages
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

# Reorder dataframe columns
final_df = final_df[col_list]

weather_cols = [
    "datetime_utc", "temperature_2m", "relative_humidity_2m",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
    "wind_gusts_10m", "apparent_temperature", "Original_Loc", "Date"
]
weather_df = pd.read_csv("combined_weather_data.csv", usecols=weather_cols)

# Ensure Date format consistency in weather data
weather_df['Date'] = pd.to_datetime(weather_df['Date']).dt.strftime('%Y-%m-%d')
weather_df['Original_Loc'] = weather_df['Original_Loc'].astype(str)

weather_vars = [
    "temperature_2m", "relative_humidity_2m", 
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
    "wind_gusts_10m", "apparent_temperature"
]

# --- 2. Standardize final_df Matching Columns ---
final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%Y-%m-%d')
final_df['Loc'] = final_df['Loc'].astype(str)

# --- 3. Merge Weather Data On Date and Original_Loc ---
# Use final_df['Loc'] and merge on weather_df['Original_Loc']
merge_keys_left = ['Date', 'Loc']
merge_keys_right = ['Date', 'Original_Loc']

weather_merge_df = weather_df[['Date', 'Original_Loc'] + weather_vars]

final_df = pd.merge(
    final_df,
    weather_merge_df,
    how='left',
    left_on=merge_keys_left,
    right_on=merge_keys_right
)

# Drop the redundant 'Original_Loc' column after merge
final_df.drop(columns=['Original_Loc'], inplace=True)

# --- 4. Insert Weather Data Immediately After 'SO' Column (last one) ---
so_cols = [col for col in final_df.columns if col == 'SO']
if not so_cols:
    raise ValueError("No SO column found in final_df.")
so_col = so_cols[-1]  # Use the last occurrence
columns = list(final_df.columns)

# Remove weather_vars from their current positions before re-inserting
for wcol in weather_vars:
    if wcol in columns:
        columns.remove(wcol)

# Insert weather columns right after identified SO column
insert_idx = columns.index(so_col) + 1
for i, wcol in enumerate(weather_vars):
    columns.insert(insert_idx + i, wcol)

final_df = final_df[columns]
#
mlb_teams = [
    'ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET',
    'HOU', 'KCR', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK',
    'PHI', 'PIT', 'SDP', 'SEA', 'SFG', 'STL', 'TBR', 'TEX', 'TOR', 'WSN'
]

# Create dictionary for mapping team abbrev -> index
team_to_index = {team: idx+1 for idx, team in enumerate(mlb_teams)}

# Function to map team names to their indices, returns 0 if no match
def map_team_to_idx(team):
    if pd.isna(team):
        return 0
    team = str(team).strip().upper()
    return team_to_index.get(team, 0)

# Overwrite original columns with numeric indices
for col in ['Tm', 'Opp', 'Loc']:
    if col in final_df.columns:
        final_df[col] = final_df[col].apply(map_team_to_idx)

# # --- Load BERT ---
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# model.eval()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# def bert_encode_text(text):
#     if pd.isna(text) or str(text).strip() == '' or str(text).strip().upper() == 'N/A':
#         return np.zeros(model.config.hidden_size)
#     with torch.no_grad():
#         inputs = tokenizer(
#             str(text),
#             return_tensors='pt',
#             truncation=True,
#             max_length=32,
#             padding='max_length'
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
#         outputs = model(**inputs)
#         cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
#     return cls_embedding

# # List of columns to encode (must be present in final_df)
# umpire_cols = ['umpire_HP'] if 'umpire_HP' in final_df.columns else []
# win_loss_cols = [col for col in ['home_Win', 'home_Loss', 'away_Win', 'away_Loss'] if col in final_df.columns]
# text_cols_to_encode = umpire_cols + win_loss_cols

# print(f"Encoding {len(text_cols_to_encode)} text columns.")

# for col in text_cols_to_encode:
#     print(f"Encoding column: {col}")
#     emb_matrix = np.vstack(final_df[col].apply(bert_encode_text).tolist())  # (num_rows, 768)
#     print(f"Applying PCA for {col} (to 1D)...")
#     pca = PCA(n_components=1)
#     reduced_emb = pca.fit_transform(emb_matrix)  # (num_rows, 1)
#     # Replace the original column (keep same name, same position)
#     final_df[col] = reduced_emb.flatten()

# print("Replaced original text columns with 1-dimensional BERT embeddings, names unchanged.")


if 'Date' in final_df.columns:
    date_idx = final_df.columns.get_loc('Date')
    # Create the encoded date column
    date_enc = pd.to_datetime(final_df['Date'], errors='coerce').map(lambda x: x.toordinal() if not pd.isnull(x) else np.nan)
    # Insert it right after 'Date'
    final_df.insert(date_idx + 1, 'Date_enc', date_enc)
    print("Added 'Date_enc' as ordinal values next to the original 'Date'.")


# Overwrite 'Time' with sin/cos cyclical encoding, at the same position (replacing with two columns)
if 'Time' in final_df.columns:
    time_idx = final_df.columns.get_loc('Time')
    time_obj = pd.to_datetime(final_df['Time'], errors='coerce')
    if time_obj.isnull().any():
        try:
            time_obj = pd.to_datetime(final_df['Time'], format='%I:%M %p', errors='coerce')
        except Exception:
            pass
    minutes = time_obj.dt.hour * 60 + time_obj.dt.minute
    time_sin = np.sin(2 * np.pi * minutes / 1440)
    time_cos = np.cos(2 * np.pi * minutes / 1440)
    # Remove 'Time' column, then insert sin and cos at the same location
    final_df = final_df.drop(columns=['Time'])
    final_df.insert(time_idx, 'Time_sin', time_sin)
    final_df.insert(time_idx + 1, 'Time_cos', time_cos)
    print("Replaced 'Time' column with 'Time_sin' and 'Time_cos' at original position.")

# Save or use the final DataFrame
final_df.to_csv('c_merged_data.csv', index=False)
print("Success!!!!!!")

#print
# final_df.to_csv("mix.csv", index=False)