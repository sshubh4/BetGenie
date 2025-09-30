import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import time
from datetime import datetime
import os

print("Script started")

# -------- Browser Setup --------
async def get_browser():
    """Initialize and return a Playwright browser instance"""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,  # Set to False for debugging
        args=[
            "--disable-gpu",
            "--ignore-certificate-errors",
            "--disable-blink-features=AutomationControlled",
            "--log-level=3"
        ]
    )
    print("[DEBUG] Playwright browser started successfully.")
    return playwright, browser

# -------- Load URL with Retry --------
async def load_url_with_retry(page, url, retries=3):
    """Load URL with retry logic for network issues"""
    for attempt in range(retries):
        try:
            print(f"[DEBUG] Attempting to load URL: {url}")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            print("[DEBUG] Page loaded successfully.")
            return True
        except Exception as e:
            print(f"⏱ Error loading {url}. Retrying ({attempt+1}/{retries})... Error: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(3)
            else:
                print(f"[ERROR] Failed to load URL after {retries} attempts.")
                return False
    return False

# -------- Wait for Table --------
async def wait_for_table(page, timeout=20000):
    """Wait for table to appear on the page"""
    print("[DEBUG] Waiting for table to appear...")
    try:
        await page.wait_for_selector("table", timeout=timeout)
        print("[DEBUG] Table found.")
        return True
    except Exception as e:
        print(f"[ERROR] Table did not appear within {timeout}ms: {e}")
        return False

# -------- Extract Table Data --------
async def extract_table_data(page):
    """Extract table headers and data from the page"""
    try:
        # Get all tables on the page
        tables = await page.query_selector_all("table")
        print(f"[DEBUG] Found {len(tables)} tables on the page")
        
        # Use the 16th table (index 15) as in the original script
        if len(tables) < 14:
            print(f"[WARNING] Expected at least 14 tables, found {len(tables)}")
            return [], []
            
        table = tables[13]
        
        # Extract headers
        header_elems = await table.query_selector_all("thead th")
        headers = []
        for h in header_elems:
            text = await h.text_content()
            if text and text.strip():
                headers.append(text.strip())
        
        # Remove "Rk." if it's the first header
        if headers and headers[0] == "Rk.":
            headers = headers[1:]
            
        print(f"[DEBUG] Headers: {headers}")
        
        # Extract data rows
        row_elems = await table.query_selector_all("tbody tr")
        rows = []
        
        for row in row_elems:
            # Find all span elements in the row
            span_elems = await row.query_selector_all("span")
            cell_texts = []
            for span in span_elems:
                text = await span.text_content()
                if text:
                    cell_texts.append(text.strip())
            
            if any(cell_texts):  # Make sure row has at least one non-empty cell
                rows.append(cell_texts)

        print(f"[DEBUG] Found {len(rows)} data rows.")
        if rows:
            return headers, rows
        else:
            return [], []
    except Exception as e:
        print(f"[ERROR] Exception while extracting table data: {e}")
        return [], []

# -------- Main Scrape Function --------
async def scrape(season):
    """Main scraping function for a specific season"""
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/statcast-park-factors?type=year&year={season}&batSide=&stat=index_wOBA&condition=All&rolling=1&parks=mlb"
    )

    print(f"[DEBUG] Scraping park factors from {season}")
    
    playwright, browser = await get_browser()
    
    try:
        # Create a new page
        page = await browser.new_page()
        
        # Set user agent
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        
        # Load the page
        if not await load_url_with_retry(page, url):
            print("❌ Failed to load page.")
            return
            
        # Wait for table to appear
        if not await wait_for_table(page, timeout=30000):
            print("❌ Table did not load. Exiting.")
            return
            
        # Wait a bit for any dynamic content
        await page.wait_for_timeout(5000)
        
        # Extract data
        headers, rows = await extract_table_data(page)
        
        if not headers or not rows:
            print("⚠️ No data extracted.")
        else:
            try:
                # Create output directory if it doesn't exist
                os.makedirs("./parkfactors", exist_ok=True)
                
                df = pd.DataFrame(rows, columns=headers)
                df.to_csv(f"./parkfactors/parkfactors_{season}.csv", index=False)
                print(f"✅ Saved parkfactors_{season}.csv")
            except Exception as e:
                print(f"[ERROR] Could not save CSV: {e}")
                
    except Exception as e:
        print(f"[ERROR] Error during scraping: {e}")
    finally:
        # Clean up
        await browser.close()
        await playwright.stop()

# -------- Main Execution --------
async def main():
    """Main function to run the scraper for all seasons"""
    seasons = [2021, 2022, 2023, 2024, 2025]
    
    for season in seasons:
        print(f"\n{'='*50}")
        print(f"Processing season: {season}")
        print(f"{'='*50}")
        await scrape(season)
        # Small delay between seasons
        await asyncio.sleep(2)

# Run the async main function
def combine_season_csvs(seasons, input_folder="./parkfactors", output_file="./parkfactors/combined_parkfactors.csv"):
    """Combine individual season CSV files into one CSV, adding 'Season' and 'Loc' columns"""

    # Mapping short team names (as appear in your dataset) to abbreviations
    short_name_to_loc = {
        "Tigers": "DET",
        "Reds": "CIN",
        "Brewers": "MIL",
        "Rockies": "COL",
        "Mariners": "SEA",
        "Phillies": "PHI",
        "Padres": "SDP",
        "Yankees": "NYY",
        "Marlins": "MIA",
        "Cubs": "CHC",
        "Angels": "LAA",
        "Royals": "KCR",
        "Athletics": "OAK",
        "Red Sox": "BOS",
        "Guardians": "CLE",
        "Rangers": "TEX",
        "Nationals": "WSN",
        "Cardinals": "STL",
        "Astros": "HOU",
        "Mets": "NYM",
        "Orioles": "BAL",
        "Twins": "MIN",
        "Pirates": "PIT",
        "White Sox": "CHW",
        "Blue Jays": "TOR",
        "Braves": "ATL",
        "Giants": "SFG",
        "Rays": "TBR",
        "Dodgers": "LAD",
        "D-backs": "ARI",
    }

    all_dfs = []
    for season in seasons:
        file_path = os.path.join(input_folder, f"parkfactors_{season}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['Season'] = season

                # Replace 'Team' with the actual column name that contains these short team names in your CSV
                if 'Team' in df.columns:
                    df['Loc'] = df['Team'].map(short_name_to_loc).fillna("Unknown")
                else:
                    df['Loc'] = "Unknown"

                all_dfs.append(df)
                print(f"[INFO] Loaded {file_path} with {len(df)} rows.")
            except Exception as e:
                print(f"[ERROR] Could not load {file_path}: {e}")
        else:
            print(f"[WARNING] File {file_path} does not exist and will be skipped.")
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"✅ Combined CSV saved to '{output_file}'. Total rows: {len(combined_df)}")
    else:
        print("[WARNING] No CSV files to combine.")


# Then call this function after main runs, for example:

if __name__ == "__main__":
    asyncio.run(main())
    seasons = [2021, 2022, 2023, 2024, 2025]
    combine_season_csvs(seasons)
