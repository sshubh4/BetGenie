import os
import pandas as pd

# Directory containing target CSV files
logs_dir = "teamgamelogs"

# Process all CSVs in logs_dir for 'Tm' and 'Opp' replacements
if not os.path.exists(logs_dir):
    print(f"❌ Directory not found: {logs_dir}")
    exit()

for fname in os.listdir(logs_dir):
    if fname.lower().endswith(".csv"):
        path = os.path.join(logs_dir, fname)
        try:
            df = pd.read_csv(path, low_memory=False)
            changed = False
            for col in ['Tm', 'Opp']:
                if col in df.columns:
                    before_count = (df[col] == "OAK").sum()
                    if before_count > 0:
                        df[col] = df[col].replace("OAK", "ATH")
                        changed = True
                        print(f"🔄 {fname}: Replaced {before_count} occurrences of 'OAK' in column '{col}'")

            if changed:
                df.to_csv(path, index=False)
                print(f"✅ Saved updated file: {fname}")

        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

# Process the combined_weather_data.csv for 'Loc' and 'original_loc' replacements
combined_path = "combined_weather_data.csv"
if os.path.exists(combined_path):
    try:
        df_combined = pd.read_csv(combined_path, low_memory=False)
        changed = False
        for col in ['Loc', 'Original_Loc']:
            if col in df_combined.columns:
                before_count = (df_combined[col] == "OAK").sum()
                if before_count > 0:
                    df_combined[col] = df_combined[col].replace("OAK", "ATH")
                    changed = True
                    print(f"🔄 {combined_path}: Replaced {before_count} occurrences of 'OAK' in column '{col}'")

        if changed:
            df_combined.to_csv(combined_path, index=False)
            print(f"✅ Saved updated file: {combined_path}")

    except Exception as e:
        print(f"⚠️ Error processing {combined_path}: {e}")
else:
    print(f"ℹ️ File not found: {combined_path}")

print("🏁 Replacement complete.")


parkfactors = "parkfactors/combined_parkfactors.csv"
if os.path.exists(parkfactors):
    combined_path = "parkfactors/combined_parkfactors.csv"
    if os.path.exists(combined_path):
        try:
            df_combined = pd.read_csv(combined_path, low_memory=False)
            changed = False
            for col in ['Loc', 'Original_Loc']:
                if col in df_combined.columns:
                    before_count = (df_combined[col] == "OAK").sum()
                    if before_count > 0:
                        df_combined[col] = df_combined[col].replace("OAK", "ATH")
                        changed = True
                        print(f"🔄 {combined_path}: Replaced {before_count} occurrences of 'OAK' in column '{col}'")

            if changed:
                df_combined.to_csv(combined_path, index=False)
                print(f"✅ Saved updated file: {combined_path}")

        except Exception as e:
            print(f"⚠️ Error processing {combined_path}: {e}")
    else:
        print(f"ℹ️ File not found: {combined_path}")
