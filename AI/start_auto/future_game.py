import pandas as pd
import numpy as np
import re

# === CONFIG ===
historical_header_file = "home.csv"  # CSV with full columns in correct order
future_file = "mlb_schedule_processed.csv"

# 1️⃣ Load header from historical file
template_cols = list(pd.read_csv(historical_header_file, nrows=0).columns)

# 2️⃣ Load the future schedule
df = pd.read_csv(future_file)

# 3️⃣ Generate game_in_day flag
# create a key of Date+sorted teams to detect double-headers
df['match_key'] = df.apply(lambda r: f"{r['Date']}_{'_'.join(sorted([r['Tm'], r['Opp']]))}", axis=1)
df['game_in_day'] = df.groupby('match_key').cumcount() + 1
df.drop(columns=['match_key'], inplace=True)

# 4️⃣ Make home and away views
# home side
home_df = pd.DataFrame(columns=template_cols)  # Create empty DF with historical cols
home_df['Date'] = df['Date']
home_df['game_in_day'] = df['game_in_day']
home_df['Tm'] = df['Tm']
home_df['At'] = 0  # home team
home_df['Opp'] = df['Opp']
home_df['Loc'] = df['Loc']
home_df['Time'] = df['Time']
home_df['umpire_HP'] = df['umpire_HP']

# Fill only batter_#_Batting and pitcher_#_Pitching from input (home side)
for i in range(1, 18):  # up to batter_17 from your template
    col_name = f"batter_{i}_Batting"
    input_col = f"home_{col_name}"
    if input_col in df.columns and col_name in home_df.columns:
        home_df[col_name] = df[input_col]

for i in range(1, 12):  # up to pitcher_11 from your template
    col_name = f"pitcher_{i}_Pitching"
    input_col = f"home_{col_name}"
    if input_col in df.columns and col_name in home_df.columns:
        home_df[col_name] = df[input_col]

# away side
away_df = pd.DataFrame(columns=template_cols)
away_df['Date'] = df['Date']
away_df['game_in_day'] = df['game_in_day']
away_df['Tm'] = df['Opp']  # away team in future schedule Tm
away_df['At'] = 1
away_df['Opp'] = df['Tm']
away_df['Loc'] = df['Loc']
away_df['Time'] = df['Time']
away_df['umpire_HP'] = df['umpire_HP']

# Fill batter/pitcher for away
for i in range(1, 18):
    col_name = f"batter_{i}_Batting"
    input_col = f"away_{col_name}"
    if input_col in df.columns and col_name in away_df.columns:
        away_df[col_name] = df[input_col]

for i in range(1, 12):
    col_name = f"pitcher_{i}_Pitching"
    input_col = f"away_{col_name}"
    if input_col in df.columns and col_name in away_df.columns:
        away_df[col_name] = df[input_col]

# 5️⃣ Ensure correct column order & save
home_df = home_df[template_cols]
away_df = away_df[template_cols]

# Get number of rows
num_rows = len(df)  

home_combined = pd.read_csv("home.csv")
away_combined = pd.read_csv("away.csv")

home = pd.concat([home_combined, home_df])
away = pd.concat([away_combined, away_df])


def remove_unnamed_columns(df):
    """
    Remove columns from DataFrame that contain 'Unnamed' in their column name.
    Returns a new DataFrame without such columns.
    """
    cleaned_df = df.loc[:, ~df.columns.str.contains('^Unnamed', case=False)]
    return cleaned_df

home = remove_unnamed_columns(home)
away = remove_unnamed_columns(away)


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
    exclude_cols = [ team_column,
                    'game_Index', 'Date', 'game_in_day', 'Tm', 'At', 'Opp', 'Loc', 'W/L',
    'Rank', 'Win', 'Loss', 'Time', 'D/N', 'Attendance', 'cLI', 'Streak', 'Season',
    'Num_Batters', 'umpire_HP', 'num_Batters', 'num_Pitchers',
    'field_condition', 'weather_temp', 'weather_wind', 'weather_sky', 'weather_precip', 'days_rest', 'h2h_wins', 'h2h_losses', 'h2h_record',]
    # Add all batter_i_Batting and pitcher_i_Pitching columns dynamically like:
    batter_batting_cols = [col for col in df_with_stats.columns if re.match(r'^batter_\d+_Batting$', col)]
    pitcher_pitching_cols = [col for col in df_with_stats.columns if re.match(r'^pitcher_\d+_Pitching$', col)]

    exclude_cols.extend(batter_batting_cols)
    exclude_cols.extend(pitcher_pitching_cols)
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

# final_df.to_csv("future_game.csv", index=False)

# Fill missing Season with 2025
final_df['Season'] = final_df['Season'].fillna(2025)

# Ensure Attendance is numeric
final_df['Attendance'] = pd.to_numeric(final_df['Attendance'], errors='coerce')

# Calculate mean Attendance ignoring NaNs
mean_attendance = final_df['Attendance'].mean()

# Fill missing Attendance with the mean
final_df['Attendance'] = final_df['Attendance'].fillna(mean_attendance)
final_df = final_df.tail(num_rows)

# Loop through columns and fill based on dtype
for col in final_df.columns:
    if pd.api.types.is_integer_dtype(final_df[col]):
        final_df[col] = final_df[col].fillna(0)
    elif pd.api.types.is_float_dtype(final_df[col]):
        final_df[col] = final_df[col].fillna(0.0)
    else:  # object, string, category, etc.
        final_df[col] = final_df[col].fillna("N/A")

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
weather_df = pd.read_csv("future_weather_data.csv", usecols=weather_cols)

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


final_df.to_csv('final_future_processing_data.csv', index=False)
print("Success!!!!!!")