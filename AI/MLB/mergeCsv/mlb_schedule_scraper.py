from playwright.sync_api import sync_playwright
from datetime import datetime
import pandas as pd
import time
import sys

def extract_team_info(team_span):
    """Get team code and full name from the .Table__Team span anchor."""
    anchor = team_span.query_selector('a:last-of-type')
    href = anchor.get_attribute('href')
    # url pattern: /mlb/team/_/name/chw/chicago-white-sox
    parts = href.strip('/').split('/')
    code = parts[-2].upper()  # 'CHW' etc.
    name = anchor.inner_text().strip()
    return code, name

def get_home_plate_umpire(game_page):
    """
    Scrape Home Plate Umpire from the game detail page.
    Looks for the 'Umpires' list and extracts the Home Plate Umpire name.
    """
    try:
        # Locate the ul with umpire list (by text "Umpires:" and siblings)
        umpire_items = game_page.query_selector_all("ul.GameInfo__List li.GameInfo__List__Item")
        for item in umpire_items:
            text = item.inner_text().strip()
            if text.startswith("Home Plate Umpire"):
                # Format is "Home Plate Umpire - Phil Cuzzi"
                parts = text.split(" - ")
                if len(parts) == 2:
                    return parts[1].strip()
    except Exception as e:
        print(f"Error extracting umpire: {e}")
    return ""  # Return empty string if not found

def scrape_hitters_table(game_page):
    """
    Scrapes batter names from the currently visible hitters table on the active tab.
    """
    batters = []
    try:
        hitters_table = game_page.query_selector('table.Table')
        if hitters_table:
            rows = hitters_table.query_selector_all('tbody.Table__TBODY > tr')
            for row in rows:
                name_a = row.query_selector('a.Boxscore__Athlete_Name')
                if name_a:
                    name = name_a.inner_text().strip()
                    if name:
                        batters.append(name)
    except Exception as e:
        print(f"Error scraping hitters table: {e}")
    return batters

def get_batters_from_tab(game_page, batters_per_team=9):
    """
    Handles the tabbed lineups: scrapes away team hitters from first tab,
    clicks second tab and scrapes home team hitters.
    Returns two lists (away_batters, home_batters) each padded to batters_per_team.
    """
    away_batters = []
    home_batters = []

    tab_buttons = game_page.query_selector_all('nav.tabs__nav button')

    if not tab_buttons or len(tab_buttons) < 2:
        # If tabs missing or only one, scrape just one lineup (assign as away team)
        print("Tabs not found or only one tab found. Scraping single hitters table.")
        away_batters = scrape_hitters_table(game_page)
        home_batters = []
    else:
        # Click first tab (away team)
        tab_buttons[0].click()
        game_page.wait_for_timeout(1000)
        away_batters = scrape_hitters_table(game_page)

        # Click second tab (home team)
        tab_buttons[1].click()
        game_page.wait_for_timeout(1000)
        home_batters = scrape_hitters_table(game_page)

    # Pad lists to consistent length
    def pad(lst, size):
        return lst + [''] * (size - len(lst))

    away_batters = pad(away_batters, batters_per_team)[:batters_per_team]
    home_batters = pad(home_batters, batters_per_team)[:batters_per_team]

    return away_batters, home_batters


def scrape_schedule_rows_for_date(target_date_str):
    # Input e.g.: '2025-08-07'
    # ESPN's Table__Title always uses format: 'Thursday, August 7, 2025'
    input_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    espn_title = input_date.strftime('%A, %B') + f' {input_date.day}, {input_date.year}'
    print(f"Searching for MLB schedule table with title: {espn_title}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://www.espn.com/mlb/schedule")
        page.wait_for_selector('.Table__Title')

        # Find all section titles (one for each schedule date)
        all_titles = page.query_selector_all('.Table__Title')
        table_div = None
        for title_div in all_titles:
            if title_div.inner_text().strip() == espn_title:
                # Parent of title_div is the schedule table's div
                table_div = title_div.evaluate_handle('node => node.parentElement.parentElement.parentElement')
                break

        if not table_div:
            print("Could not find schedule for requested date!")
            browser.close()
            return

        tbody = table_div.query_selector('tbody.Table__TBODY')
        game_rows = tbody.query_selector_all('tr.Table__TR')

        data = []
        for row in game_rows:
            # Away team
            away_span = row.query_selector('td.events__col .Table__Team.away')
            away_code, away_name = extract_team_info(away_span)
            # Home team
            home_span = row.query_selector('td.colspan__col .Table__Team')
            home_code, home_name = extract_team_info(home_span)
            # Time and detail
            time_td = row.query_selector('td.date__col a.AnchorLink')
            if not time_td:
                # skip rows without a game time link
                print("Skipping row with no game time link")
                continue
            game_time = time_td.inner_text().strip()
            detail_link = 'https://www.espn.com' + time_td.get_attribute('href')
            # Probable pitchers
            pitchers = row.query_selector_all('td.probable__col p a')
            away_pitcher = pitchers[0].inner_text().strip() if len(pitchers) > 0 else ''
            home_pitcher = pitchers[1].inner_text().strip() if len(pitchers) > 1 else ''

            time_td = row.query_selector('td.date__col a.AnchorLink')
            if not time_td:
                continue
            game_time = time_td.inner_text().strip()
            detail_link = 'https://www.espn.com' + time_td.get_attribute('href')

            # Open detail page to get home plate umpire
            try:
                game_page = browser.new_page()
                game_page.goto(detail_link)
                game_page.wait_for_load_state('domcontentloaded')
                umpire_hp = get_home_plate_umpire(game_page)
                away_batters, home_batters = get_batters_from_tab(game_page)
                game_page.close()
            except Exception as e:
                print(f"Failed to open game detail page {detail_link}: {e}")
                umpire_hp = ""
                away_batters = []
                home_batters = []
            # Prepare the row dict with basic fields
            row_dict = {
                'date': espn_title,
                'away_code': away_code,
                'away_name': away_name,
                'home_code': home_code,
                'home_name': home_name,
                'game_time': game_time,
                'detail_url': detail_link,
                'away_pitcher_1_Pitching': away_pitcher,
                'home_pitcher_1_Pitching': home_pitcher,
                'umpire_HP': umpire_hp
            }

            # Add away batters with keys like 'away_batter_1_Batting', 'away_batter_2_Batting', ...
            for i, batter in enumerate(away_batters, 1):
                row_dict[f'away_batter_{i}_Batting'] = batter

            # Add home batters with keys like 'home_batter_1_Batting', 'home_batter_2_Batting', ...
            for i, batter in enumerate(home_batters, 1):
                row_dict[f'home_batter_{i}_Batting'] = batter

            # Now append this complete dictionary to the data list
            data.append(row_dict)



        browser.close()
        # Optional: return as DataFrame
        df = pd.DataFrame(data)
        print(df)
        # Or save to CSV
        df.to_csv(f'mlb_schedule.csv', index=False)
        print(f"Exported to mlb_schedule.csv")

def post_process_csv(filename_in, filename_out):
    # Wait 10 seconds to ensure the file is written
    print("Waiting 10 seconds before processing CSV...")
    time.sleep(10)

    df = pd.read_csv(filename_in)

    # Remove away_name and home_name columns if they exist
    df = df.drop(columns=[col for col in ['away_name', 'home_name', 'detail_url'] if col in df.columns])

    # Rename columns
    rename_map = {
        'away_code': 'Opp',
        'home_code': 'Tm',
        'date': 'Date',
        'game_time': 'Time',
    }
    df = df.rename(columns=rename_map)

    # Convert 'Date' to yyyy-mm-dd format
    def convert_date(d):
        try:
            dt = datetime.strptime(d, "%A, %B %d, %Y")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return d

    df['Date'] = df['Date'].apply(convert_date)

    # Remove 'PM' from Time and strip whitespace
    df['Time'] = df['Time'].str.replace(r'\s*PM$', '', regex=True).str.strip()

    # Add 'Loc' column from 'Tm' (home team code)
    df['Loc'] = df['Tm']

    # Reorder columns
    home_batters_cols = sorted([c for c in df.columns if c.startswith('home_batter_')])
    away_batters_cols = sorted([c for c in df.columns if c.startswith('away_batter_')])

    new_order = ['Date', 'Tm', 'Opp', 'Loc', 'Time', 'umpire_HP'] + home_batters_cols + away_batters_cols + ['home_pitcher_1_Pitching', 'away_pitcher_1_Pitching']
    df = df[new_order]

    # Save to new CSV
    df.to_csv(filename_out, index=False)
    print(f"Processed CSV saved as {filename_out}")

# Example usage (call this after your scraping CSV is saved):
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 mlb_schedule_scraper.py YYY-MM-DD in double quotes")
        sys.exit(1)

    target_date = sys.argv[1]  # First argument after script name

    # Run the scraper for the given date
    scrape_schedule_rows_for_date(target_date)

    time.sleep(10)  # Wait a bit to ensure the file is written
    input_file = 'mlb_schedule.csv'           # Replace with your generated CSV filename
    output_file = 'mlb_schedule_processed.csv'  # Desired output filename

    post_process_csv(input_file, output_file)