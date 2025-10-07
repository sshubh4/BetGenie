import argparse
import glob
import os
import re
import sys

import pandas as pd


def add_game_in_day_to_csv(csv_path: str) -> bool:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"❌ Failed to read {csv_path}: {e}")
        return False

    if 'Date' not in df.columns:
        print(f"⏭️ Skipping {csv_path} (no 'Date' column)")
        return False

    # Vectorized extraction of (1) or (2) from the Date string
    # Default to 1 when not present
    extracted = df['Date'].astype(str).str.extract(r"\((1|2)\)")[0]
    game_in_day = (
        extracted
        .astype('Int64')
        .fillna(1)
        .astype(int)
    )

    # Insert or update the column, placing it right after 'Date'
    if 'game_in_day' in df.columns:
        df['game_in_day'] = game_in_day
    else:
        cols = list(df.columns)
        try:
            date_idx = cols.index('Date')
        except ValueError:
            date_idx = 0
        df.insert(date_idx + 1, 'game_in_day', game_in_day)

    try:
        df.to_csv(csv_path, index=False)
        print(f"✅ Updated {csv_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to write {csv_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Add game_in_day column to team game logs by parsing Date for doubleheaders.")
    parser.add_argument(
        "paths",
        nargs='*',
        default=[os.path.join(os.path.dirname(__file__), 'teamgamelogs', '*.csv')],
        help="One or more glob patterns to CSVs (default: AI/teamgamelogs/*.csv)",
    )
    args = parser.parse_args()

    patterns = args.paths if isinstance(args.paths, list) else [args.paths]
    matched_files = []
    for pattern in patterns:
        matched_files.extend(glob.glob(pattern, recursive=True))

    if not matched_files:
        print("ℹ️ No CSV files matched. Provide a path like 'BetGenie/AI/teamgamelogs/*.csv'.")
        sys.exit(0)

    updated = 0
    for csv_path in matched_files:
        if csv_path.lower().endswith('.csv'):
            if add_game_in_day_to_csv(csv_path):
                updated += 1

    print(f"\nDone. Updated {updated} file(s).")


if __name__ == "__main__":
    main()