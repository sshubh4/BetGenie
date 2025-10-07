import pandas as pd
import re

# Input/output files
input_csv = 'merged_data.csv'
output_csv = 'filtered_times.csv'

# Read the CSV file
df = pd.read_csv(input_csv)

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
