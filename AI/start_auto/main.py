import subprocess
import sys
import pandas as pd
from datetime import datetime
import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os
import re

def run_script(script_name, args=None):
    """Run another Python script, optionally with arguments."""
    cmd = [sys.executable, script_name]
    if args:
        cmd += args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] {script_name} failed!")
        sys.exit(result.returncode)
    print(f"[OK] Finished {script_name}")

def filter_times():
    input_csv = 'merged_data.csv'
    output_csv = 'filtered_times.csv'

    # Read the CSV file
    df = pd.read_csv(input_csv, low_memory=False)

    # Column names (update if needed)
    date_col = 'Date'
    time_col = 'Time'
    lod_col = 'Loc'

    def pm_to_24h(time_str):
        t = str(time_str).strip()
        match = re.match(r'^(\d{1,2})(?::(\d{0,2}))?$', t)
        if match:
            hour = int(match.group(1))
            minute = match.group(2)
            minute = int(minute) if minute and minute.isdigit() else 0
            hour += 12  # all are PM
            return f"{hour:02d}:{minute:02d}"
        return t  # unchanged if doesn't match

    # Build output DataFrame
    df_out = pd.DataFrame()
    df_out[date_col] = df[date_col]
    df_out[time_col] = df[time_col].astype(str).apply(pm_to_24h)
    df_out[lod_col] = df[lod_col]

    # Write to new CSV
    df_out.to_csv(output_csv, index=False)
    print("filtered.csv success")


def future_filter_times():
    input_csv = 'mlb_schedule_processed.csv'
    output_csv = 'future_filtered_times.csv'

    # Read the CSV file
    df = pd.read_csv(input_csv, low_memory=False)

    # Column names (update if needed)
    date_col = 'Date'
    time_col = 'Time'
    lod_col = 'Loc'

    def pm_to_24h(time_str):
        t = str(time_str).strip()
        match = re.match(r'^(\d{1,2})(?::(\d{0,2}))?$', t)
        if match:
            hour = int(match.group(1))
            minute = match.group(2)
            minute = int(minute) if minute and minute.isdigit() else 0
            hour += 12  # all are PM
            return f"{hour:02d}:{minute:02d}"
        return t  # unchanged if doesn't match

    # Build output DataFrame
    df_out = pd.DataFrame()
    df_out[date_col] = df[date_col]
    df_out[time_col] = df[time_col].astype(str).apply(pm_to_24h)
    df_out[lod_col] = df[lod_col]

    # Write to new CSV
    df_out.to_csv(output_csv, index=False)
    print("future_filtered.csv success")


def merge_weather_into_file(input_csv="merged_data.csv", weather_csv="combined_weather_data.csv"):
    """
    Merges weather data into merged_data.csv and overwrites it.
    """
    print(f"[INFO] Merging weather from {weather_csv} into {input_csv}...")

    # Load the merged data
    final_df = pd.read_csv(input_csv)

    # Keep current column order
    col_list = list(final_df.columns)
    final_df = final_df[col_list]

    # === WEATHER MERGE LOGIC ===
    weather_cols = [
        "datetime_utc", "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
        "wind_gusts_10m", "apparent_temperature", "Original_Loc", "Date"
    ]
    weather_df = pd.read_csv(weather_csv, usecols=weather_cols)

    # Ensure Date format consistency
    weather_df['Date'] = pd.to_datetime(weather_df['Date']).dt.strftime('%Y-%m-%d')
    weather_df['Original_Loc'] = weather_df['Original_Loc'].astype(str)

    weather_vars = [
        "temperature_2m", "relative_humidity_2m", 
        "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m",
        "wind_gusts_10m", "apparent_temperature"
    ]

    # Standardize final_df
    final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%Y-%m-%d')
    final_df['Loc'] = final_df['Loc'].astype(str)

    # Merge weather data
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

    # Drop redundant column
    final_df.drop(columns=['Original_Loc'], inplace=True)

    # Insert weather vars right after last 'SO' column
    so_cols = [col for col in final_df.columns if col == 'SO']
    if not so_cols:
        raise ValueError("No SO column found in merged_data.csv.")
    
    so_col = so_cols[-1]
    columns = list(final_df.columns)

    # Remove weather vars from current position
    for wcol in weather_vars:
        if wcol in columns:
            columns.remove(wcol)

    # Insert after SO column in same order
    insert_idx = columns.index(so_col) + 1
    for i, wcol in enumerate(weather_vars):
        columns.insert(insert_idx + i, wcol)

    final_df = final_df[columns]

    # Overwrite the merged_data.csv
    final_df.to_csv(input_csv, index=False)
    print(f"[OK] Weather merged and {input_csv} overwritten.")

def normalize_team_codes():
    """
    Replace values in columns Tm, Opp, and Loc of mlb_schedule_processed.csv
    using loc_translation. The columns keep their original names.
    """
    loc_translation = {
        "KC": "KCR",
        "SD": "SDP",
        "WSH": "WSN",
        "TB": "TBR",
        "SF": "SFG"
    }

    input_file = "mlb_schedule_processed.csv"
    try:
        df = pd.read_csv(input_file)
        # Replace values for each column IN PLACE (columns are not renamed)
        for col in ["Tm", "Opp", "Loc"]:
            if col in df.columns:
                df[col] = df[col].replace(loc_translation)
        df.to_csv(input_file, index=False)
        print(f"[OK] Normalized team codes in {input_file}")
    except FileNotFoundError:
        print(f"[ERROR] File {input_file} not found.")
    except Exception as e:
        print(f"[ERROR] Failed to normalize team codes: {e}")

def encode():
    final_df = pd.read_csv('final_future_processing_data.csv')  

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def bert_encode_text(text):
        if pd.isna(text) or str(text).strip() == '' or str(text).strip().upper() == 'N/A':
            return np.zeros(model.config.hidden_size)
        with torch.no_grad():
            inputs = tokenizer(
                str(text),
                return_tensors='pt',
                truncation=True,
                max_length=32,
                padding='max_length'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding

    # List of columns to encode (must be present in final_df)
    umpire_cols = ['umpire_HP'] if 'umpire_HP' in final_df.columns else []
    win_loss_cols = [col for col in ['home_Win', 'home_Loss', 'away_Win', 'away_Loss'] if col in final_df.columns]
    text_cols_to_encode = umpire_cols + win_loss_cols

    print(f"Encoding {len(text_cols_to_encode)} text columns.")

    for col in text_cols_to_encode:
        print(f"Encoding column: {col}")
        emb_matrix = np.vstack(final_df[col].apply(bert_encode_text).tolist())  # (num_rows, 768)
        print(f"Applying PCA for {col} (to 1D)...")
        pca = PCA(n_components=1)
        reduced_emb = pca.fit_transform(emb_matrix)  # (num_rows, 1)
        # Replace the original column (keep same name, same position)
        final_df[col] = reduced_emb.flatten()

    print("Replaced original text columns with 1-dimensional BERT embeddings, names unchanged.")
    final_df.to_csv('final_future_processing_data.csv', index=False)
    print("Success!!!!!!")


if __name__ == "__main__":

    # Step 0 - run update game logs to get the latest boxscore
    run_script("scrape.py")
    #should have replace OAK with ATH by now by running file oak_replacemnt.py

    #Step 0.5 - run update parkfactors
    run_script("parkfactors.py")

    run_script("oak_replacement.py")  # This will replace OAK with ATH in all relevant files

    # # Step 1 - raw_data_merge.py
    run_script("raw_data_merge.py")

    # Step 2 - filter_times function (not filtered_times.py)
    filter_times()

    # Step 3 - scrape weather data
    run_script("historical_weather_scrape.py")  # creates combined_weather_data.csv

    # NO NEED OF THIS STEP NOW
    # Step 4 - merge weather into merged_data.csv
    merge_weather_into_file("merged_data.csv", "combined_weather_data.csv")

    # Step 4.5 - fetch oddsline
    run_script("fetch_odds.py")

    # Step 5 - rolling_stats_merge.py
    run_script("final_merge.py")

    # Step 6 - Run mob_schedule_scraper.py for a specific date
    today_str = datetime.now().strftime("%Y-%m-%d")  # e.g. "2025-08-09"
    print(f"[INFO] Running mob_schedule_scraper.py for today's date: {today_str}")
    run_script("mlb_schedule_scraper.py", args=[today_str])

    # Step 7 - filter_times function on future schedule
    future_filter_times()

    # Step 8 - Run future_weather_scrape.py on future_filtered_times.csv
    run_script("future_weather_scrape.py")

    # #step 9 - rename team codes to match with our data
    normalize_team_codes()

    # #step 10 - prepare final future file for ai prediction
    run_script("future_game.py")
