import pandas as pd
import time
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os

# Mapping of standard and alternate MLB team location codes to latitude and longitude
loc_coords = {
    "ARI": (33.4455, -112.0667),
    "ATL": (33.8908, -84.4678),
    "BAL": (39.2840, -76.6216),
    "BOS": (42.3467, -71.0972),
    "CHW": (41.8300, -87.6339),  # Chicago White Sox alternate
    "CWS": (41.8300, -87.6339),  # Chicago White Sox standard
    "CHC": (41.9484, -87.6553),
    "CIN": (39.0976, -84.5071),
    "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942),
    "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),
    "KC":  (39.0516, -94.4805),  # Kansas City Royals standard
    "KCR": (39.0516, -94.4805),  # Kansas City Royals alternate
    "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),
    "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2779),
    "NYY": (40.8296, -73.9262),
    "NYM": (40.7571, -73.8458),
    "OAK": (37.7516, -122.2005),
    "PHI": (39.9057, -75.1665),
    "PIT": (40.4469, -80.0057),
    "SD":  (32.7073, -117.1566), # San Diego Padres standard
    "SDP": (32.7073, -117.1566), # San Diego Padres alternate
    "SEA": (47.5914, -122.3325),
    "SF":  (37.7786, -122.3893), # San Francisco Giants standard
    "SFG": (37.7786, -122.3893), # San Francisco Giants alternate
    "STL": (38.6226, -90.1928),
    "TB":  (27.7683, -82.6534),  # Tampa Bay Rays standard
    "TBR": (27.7683, -82.6534),  # Tampa Bay Rays alternate
    "TEX": (32.7513, -97.0820),
    "TOR": (43.6414, -79.3894),
    "WSH": (38.8728, -77.0074),  # Washington Nationals standard
    "WSN": (38.8728, -77.0074),  # Washington Nationals alternate
    "ATH": (38.5829, -121.5236)  # Your custom ATH location
}

# Translation for alternate keys in your CSV to loc_coords keys
loc_translation = {
    "CHW": "CHW",
    "KCR": "KC",
    "SDP": "SD",
    "WSN": "WSH",
    "TBR": "TB",
    "SFG": "SF",
    "ATH": "ATH"
}

# Load filtered CSV
filtered_csv_path = 'filtered_times.csv'
df = pd.read_csv(filtered_csv_path)

# Try to load combined CSV and get last processed (Date, Time)
combined_csv_path = 'combined_weather_data.csv'
if os.path.exists(combined_csv_path) and os.path.getsize(combined_csv_path) > 0:
    combined_df = pd.read_csv(combined_csv_path)
    # Get last row's Date and Time
    last_date = str(combined_df.iloc[-1]['Date'])
    last_time = str(combined_df.iloc[-1]['Time'])
    # Find the index in df with same Date & Time
    matches = df[(df['Date'] == last_date) & (df['Time'] == last_time)]
    if not matches.empty:
        start_index = matches.index[0] + 1  # Start after the matched row
    else:
        start_index = 0  # Not found, start from beginning
else:
    # If no combined file or empty, start at 0
    start_index = 0

print(f"Starting from index {start_index} in filtered_times.csv")


# Initialize Open-Meteo API client with caching and retries
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Input/output files
input_csv = 'filtered_times.csv'  # Your input CSV filename
output_csv = 'combined_weather_data.csv'  # Output CSV filename for combined hourly data

# Load input CSV file
df = pd.read_csv(input_csv)

all_hourly_dfs = []
calls_made = 0
start_hour_ts = time.time()  # Track hour window start for rate limiting

for idx, row in df.iloc[start_index:].iterrows():

    loc_raw = str(row['Loc']).upper()
    loc = loc_translation.get(loc_raw, loc_raw)

    if loc is None or loc not in loc_coords:
        print(f"Warning: Location '{loc_raw}' not recognized or missing coordinates. Skipping row {idx}.")
        continue

    # Rate limiting: pause every 600 calls for 60 seconds
    calls_made += 1
    if calls_made % 600 == 0:
        print(f"Reached {calls_made} calls, sleeping 60 seconds to respect per-minute limit...")
        time.sleep(60)

    # Rate limiting: pause after 5000 calls for remainder of hour
    if calls_made % 5000 == 0:
        elapsed = time.time() - start_hour_ts
        sleep_time = max(0, 3600 - elapsed)
        if sleep_time > 0:
            print(f"Reached {calls_made} calls, sleeping {int(sleep_time)} seconds to respect per-hour limit...")
            time.sleep(sleep_time)
        start_hour_ts = time.time()

    date = str(row['Date'])
    time_str = str(row['Time'])

    lat, lon = loc_coords[loc]

    # Use exact date and time for both start and end hour (single hour)
    start_hour = f"{date}T{time_str}"
    end_hour = start_hour

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_hour": start_hour,
        "end_hour": end_hour,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "wind_speed_10m",
            "wind_speed_100m",
            "wind_direction_10m",
            "wind_direction_100m",
            "wind_gusts_10m",
            "apparent_temperature"
        ],
    }

    try:
        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]

        hourly = response.Hourly()

        # Extract the single timestamp (in seconds since epoch) and convert to datetime UTC
        single_time = pd.to_datetime(hourly.Time(), unit="s", utc=True)

        hourly_data = {
            "datetime_utc": [single_time],
            "temperature_2m": [hourly.Variables(0).ValuesAsNumpy()[0]],
            "relative_humidity_2m": [hourly.Variables(1).ValuesAsNumpy()[0]],
            "precipitation": [hourly.Variables(2).ValuesAsNumpy()[0]],
            "rain": [hourly.Variables(3).ValuesAsNumpy()[0]],
            "wind_speed_10m": [hourly.Variables(4).ValuesAsNumpy()[0]],
            "wind_speed_100m": [hourly.Variables(5).ValuesAsNumpy()[0]],
            "wind_direction_10m": [hourly.Variables(6).ValuesAsNumpy()[0]],
            "wind_direction_100m": [hourly.Variables(7).ValuesAsNumpy()[0]],
            "wind_gusts_10m": [hourly.Variables(8).ValuesAsNumpy()[0]],
            "apparent_temperature": [hourly.Variables(9).ValuesAsNumpy()[0]],
            # Include metadata for clarity
            "Loc": loc,
            "Original_Loc": loc_raw,
            "Date": date,
            "Time": time_str
        }

        hourly_df = pd.DataFrame(hourly_data)
        all_hourly_dfs.append(hourly_df)

    except Exception as e:
        print(f"Error fetching data for row {idx}, location {loc_raw}: {e}")
        continue

# Combine all collected hourly data and save to CSV
import os

# Combine all collected hourly data
if all_hourly_dfs:
    combined_df = pd.concat(all_hourly_dfs, ignore_index=True)

    # === New: Append to existing file if it exists ===
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        existing_df = pd.read_csv(output_csv)
        # Concat and drop duplicate rows if needed
        final_df = pd.concat([existing_df, combined_df], ignore_index=True)
        # Optional: remove duplicates based on Date & Time
        final_df.drop_duplicates(subset=['Date', 'Time'], keep='last', inplace=True)
    else:
        final_df = combined_df

    final_df.to_csv(output_csv, index=False)
    print(f"Weather data appended and saved to '{output_csv}'.")
else:
    print("No weather data was collected.")
