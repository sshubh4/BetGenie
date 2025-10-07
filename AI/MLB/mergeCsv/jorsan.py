import asyncio
import sys
import re
import html as html_module
from playwright.async_api import async_playwright
from collections import defaultdict
import pandas as pd
import numpy as np
import time
import os
import requests
from datetime import datetime
try:
    from lxml import html as lxml_html
    _LXML_OK = True
except Exception:
    _LXML_OK = False


MLB_TEAMS = {
    "ARI": "ArizonaDiamondbacks",
    "ATL": "AtlantaBraves",
    "BAL": "BaltimoreOrioles",
    "BOS": "BostonRedSox",
    "CHC": "ChicagoCubs",
    "CIN": "CincinnatiReds",
    "CLE": "ClevelandGuardians",
    "COL": "ColoradoRockies",
    "CHW": "ChicagoWhiteSox",
    "DET": "DetroitTigers",
    "HOU": "HoustonAstros",
    "KCR": "KansasCityRoyals",
    "LAD": "LosAngelesDodgers",
    "MIA": "MiamiMarlins",
    "MIL": "MilwaukeeBrewers",
    "MIN": "MinnesotaTwins",
    "NYM": "NewYorkMets",
    "NYY": "NewYorkYankees",
    "ATH": "Athletics",
    "PHI": "PhiladelphiaPhillies",
    "PIT": "PittsburghPirates",
    "SDP": "SanDiegoPadres",
    "SEA": "SeattleMariners",
    "SFG": "SanFranciscoGiants",
    "STL": "StLouisCardinals",
    "TBR": "TampaBayRays",
    "TEX": "TexasRangers",
    "TOR": "TorontoBlueJays",
    "WSN": "WashingtonNationals",
}


class PlaywrightMLBIncremental:
    def __init__(self, teams: list[str] = None, since_season: int | None = None):
        self.teams = teams or list(MLB_TEAMS.keys())
        self.since_season = since_season  # optional lower bound
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._rate_lock = asyncio.Lock()
        self._next_ts = 0.0

    async def _wait_rate(self, min_interval: float = 2.4):
        async with self._rate_lock:
            now = time.monotonic()
            if now < self._next_ts:
                await asyncio.sleep(self._next_ts - now)
            self._next_ts = time.monotonic() + min_interval

    async def setup(self):
        print("🚀 MLB Incremental Updater - append only (by date)")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
            ],
        )
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York",
        )
        self.page = await self.context.new_page()
        await self.page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4] });
            """
        )
        # Block heavy assets, allow xhr/fetch
        async def _route(route):
            try:
                if route.request.resource_type in {"image","media","font","stylesheet"}:
                    await route.abort()
                else:
                    await route.continue_()
            except Exception:
                try:
                    await route.continue_()
                except Exception:
                    pass
        await self.context.route("**/*", _route)

    async def cleanup(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    def get_team_name(self, team_code: str, year: int | None = None):
        if team_code == "CLE":
            return ["ClevelandIndians", "ClevelandGuardians"]
        return MLB_TEAMS.get(team_code, team_code)

    async def scrape_team_schedule(self, team_abbr: str, season: int) -> pd.DataFrame:
        url = f"https://www.baseball-reference.com/teams/{team_abbr}/{season}-schedule-scores.shtml"
        try:
            await self._wait_rate()
            resp = await self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
            if resp and resp.status == 429:
                await asyncio.sleep(300)
                await self._wait_rate()
                await self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(0.3)
            await self.page.wait_for_selector("#team_schedule", timeout=5000)
            result = await self.page.evaluate(
                """
                () => {
                  const table = document.getElementById('team_schedule');
                  if (!table) return { headers: [], rows: [] };
                  const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());
                  headers.splice(3, 0, 'Boxscore URL');
                  const rows = [];
                  const tbody = table.querySelector('tbody');
                  if (tbody) {
                    for (const row of tbody.querySelectorAll('tr')) {
                      const cells = row.querySelectorAll('th, td');
                      const rowData = [];
                      let boxscoreUrl = '';
                      for (const cell of cells) {
                        const link = cell.querySelector('a');
                        if (link && link.textContent.trim() === 'boxscore') {
                          boxscoreUrl = link.href;
                          rowData.push('boxscore');
                        } else {
                          rowData.push(cell.textContent.trim());
                        }
                      }
                      if (boxscoreUrl) rowData.splice(3, 0, boxscoreUrl);
                      if (rowData.length >= 5) rows.push(rowData);
                    }
                  }
                  return { headers, rows };
                }
                """
            )
        except Exception:
            if not _LXML_OK:
                return pd.DataFrame()
            try:
                resp = requests.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
                    "Referer": "https://www.baseball-reference.com/",
                }, timeout=30)
                if resp.status_code != 200:
                    return pd.DataFrame()
                root = lxml_html.fromstring(resp.content)
                table = root.get_element_by_id('team_schedule')
                headers = []
                rows = []
                if table is not None:
                    thead = table.xpath('.//thead')
                    if thead:
                        headers = [h.text_content().strip() for h in thead[0].xpath('.//th')]
                        if headers:
                            headers.insert(3, 'Boxscore URL')
                    tbody = table.xpath('.//tbody')
                    if tbody:
                        for tr in tbody[0].xpath('./tr'):
                            cells = tr.xpath('./th|./td')
                            if not cells:
                                continue
                            row = []
                            box_url = ''
                            for cell in cells:
                                a = cell.xpath('.//a')
                                if a and (a[0].text_content() or '').strip() == 'boxscore':
                                    href = a[0].get('href') or ''
                                    box_url = f"https://www.baseball-reference.com{href}" if href.startswith('/') else href
                                    row.append('boxscore')
                                else:
                                    row.append((cell.text_content() or '').strip())
                            if box_url:
                                row.insert(3, box_url)
                            if len(row) >= 5:
                                rows.append(row)
                result = { 'headers': headers, 'rows': rows }
            except Exception:
                return pd.DataFrame()

        df = pd.DataFrame(result["rows"], columns=result["headers"]) if result and result.get('rows') else pd.DataFrame()
        if df.empty:
            return df
        cols = list(df.columns)
        cols[2] = "Boxscore"
        df.columns = cols
        df = df[df.iloc[:, 2].astype(str).eq("boxscore")]
        date_raw = df["Date"].astype(str).str.replace(r"^\w+,\s*", "", regex=True)
        df["game_in_day"] = date_raw.str.extract(r"\((1|2)\)")[0].astype("Int64").fillna(1).astype(int)
        date_no_dh = date_raw.str.replace(r"\s*\((1|2)\)", "", regex=True).str.strip()
        # Ensure year present by appending season when missing
        df["Date_clean"] = np.where(
            date_no_dh.str.contains(r"\b\d{4}\b$"), date_no_dh, date_no_dh + f" {season}"
        )
        df["Date_parsed"] = pd.to_datetime(df["Date_clean"], errors="coerce")
        return df

    async def scrape_boxscore(self, boxscore_url: str, team_name: str, page=None):
        p = page or self.page
        await self._wait_rate()
        await p.goto(boxscore_url, wait_until="domcontentloaded", timeout=60000)
        await asyncio.sleep(0.3)
        # Cleveland handling
        actual_team_name = team_name
        if team_name in ["ClevelandIndians", "ClevelandGuardians"]:
            table_ids = await p.evaluate("""() => Array.from(document.querySelectorAll('table[id]')).map(t=>t.id)""")
            if any("ClevelandGuardians" in tid for tid in table_ids):
                actual_team_name = "ClevelandGuardians"
            elif any("ClevelandIndians" in tid for tid in table_ids):
                actual_team_name = "ClevelandIndians"
        # Wait briefly for batting table
        try:
            await p.wait_for_selector(f"table[id*='{actual_team_name}batting']", timeout=3000)
        except Exception:
            pass
        await asyncio.sleep(0.2)
        boxscore_data = await p.evaluate(
            f"""
            (teamName) => {{
                const data = {{ batter_headers: [], batter_stats: [], pitcher_headers: [], pitcher_stats: [], other_info: {{}} }};
                const battingTable = document.querySelector(`table[id*='${{teamName}}batting']`);
                if (battingTable) {{
                    const headers = Array.from(battingTable.querySelectorAll('thead th')).map(th => th.textContent.trim()).filter(Boolean);
                    data.batter_headers = headers.filter((h,i)=> i===0 || h !== headers[i-1]);
                    for (const row of battingTable.querySelectorAll('tbody tr')) {{
                        if (row.classList.contains('sum') || row.classList.contains('total')) continue;
                        const cells = row.querySelectorAll('td');
                        const nameCell = row.querySelector('th');
                        if (!nameCell || cells.length < (data.batter_headers.length - 1)) continue;
                        const arr = [nameCell.textContent.trim()];
                        if (arr[0].endsWith(' P')) {{
                            const ab = cells[0]?.textContent.trim();
                            if (!ab || ab==='0' || ab==='0.0') continue;
                        }}
                        for (let i=0; i<Math.min(cells.length, data.batter_headers.length-1); i++) arr.push(cells[i].textContent.trim());
                        data.batter_stats.push(arr);
                    }}
                }}
                const pitchingTable = document.querySelector(`table[id*='${{teamName}}pitching']`);
                if (pitchingTable) {{
                    const headers = Array.from(pitchingTable.querySelectorAll('thead th')).map(th => th.textContent.trim()).filter(Boolean);
                    data.pitcher_headers = headers.filter((h,i)=> i===0 || h !== headers[i-1]);
                    for (const row of pitchingTable.querySelectorAll('tbody tr')) {{
                        if (row.classList.contains('sum') || row.classList.contains('total')) continue;
                        const cells = row.querySelectorAll('td');
                        const nameCell = row.querySelector('th');
                        if (!nameCell || cells.length < (data.pitcher_headers.length - 1)) continue;
                        const arr = [nameCell.textContent.trim()];
                        for (let i=0; i<Math.min(cells.length, data.pitcher_headers.length-1); i++) arr.push(cells[i].textContent.trim());
                        data.pitcher_stats.push(arr);
                    }}
                }}
                return data;
            }}
            """,
            actual_team_name,
        )
        return boxscore_data

    def _align_headers_with_existing(self, existing_df: pd.DataFrame, append_df: pd.DataFrame):
        if existing_df is None or existing_df.empty:
            return existing_df, append_df
        aligned_columns = list(existing_df.columns) + [c for c in append_df.columns if c not in existing_df.columns]
        return existing_df.reindex(columns=aligned_columns), append_df.reindex(columns=aligned_columns)

    async def append_boxscores_per_team(self, all_entries: list[list], final_headers: list[str], team_code: str):
        other_info_keys = [
            'umpire_HP', 'umpire_1B', 'umpire_2B', 'umpire_3B', 'umpire_LF', 'umpire_RF',
            'time_of_game', 'attendance', 'field_condition', 'weather_temp', 'weather_wind',
            'weather_sky', 'weather_precip'
        ]
        team_path = os.path.join("teamgamelogs", f"{team_code}_boxscores.csv")
        os.makedirs("teamgamelogs", exist_ok=True)
        if os.path.exists(team_path):
            existing_df = pd.read_csv(team_path, low_memory=False)
        else:
            existing_df = pd.DataFrame()

        # Build set of existing keys to avoid duplicates
        existing_keys = set()
        if not existing_df.empty and 'Date' in existing_df.columns and 'Boxscore URL' in existing_df.columns:
            existing_keys = set(zip(existing_df['Date'].astype(str), existing_df['Boxscore URL'].astype(str)))

        max_batters = 0
        max_pitchers = 0
        batter_headers_ref: list[str] = []
        pitcher_headers_ref: list[str] = []
        enhanced_entries = []

        # Reuse one page for speed
        page = await self.context.new_page()
        for i, entry in enumerate(all_entries, 1):
            url = entry[3]
            key = (str(entry[1]), str(entry[3])) if len(entry) > 3 else None
            if key and key in existing_keys:
                continue
            game_start_ts = time.time()
            if url and isinstance(url, str) and url.startswith('http'):
                lookup = self.get_team_name(team_code, entry[-2] if len(entry) > 1 else None)
                team_name = lookup[0] if isinstance(lookup, list) else lookup
                data = await self.scrape_boxscore(url, team_name, page=page)
                batters_count = len(data['batter_stats'])
                pitchers_count = len(data['pitcher_stats'])
                if batters_count:
                    max_batters = max(max_batters, batters_count)
                    batter_headers_ref = data['batter_headers']
                if pitchers_count:
                    max_pitchers = max(max_pitchers, pitchers_count)
                    pitcher_headers_ref = data['pitcher_headers']
                row = entry.copy()
                row.extend([''] * len(other_info_keys))
                for batter in data['batter_stats']:
                    row.extend(batter)
                row.extend([''] * (len(batter_headers_ref) * (max_batters - batters_count)))
                for pitcher in data['pitcher_stats']:
                    row.extend(pitcher)
                row.extend([''] * (len(pitcher_headers_ref) * (max_pitchers - pitchers_count)))
                enhanced_entries.append(row)
        try:
            await page.close()
        except Exception:
            pass

        if not enhanced_entries:
            return

        final_enhanced_headers = final_headers.copy()
        final_enhanced_headers.extend(other_info_keys)
        for i in range(1, max_batters + 1):
            for h in batter_headers_ref:
                final_enhanced_headers.append(f"batter_{i}_{h}")
        for i in range(1, max_pitchers + 1):
            for h in pitcher_headers_ref:
                final_enhanced_headers.append(f"pitcher_{i}_{h}")

        append_df = pd.DataFrame(enhanced_entries, columns=final_enhanced_headers)
        # Clean accidental columns
        append_df = append_df.loc[:, ~append_df.columns.str.startswith('Unnamed:')]
        if not existing_df.empty:
            existing_df, append_df = self._align_headers_with_existing(existing_df, append_df)
            result_df = pd.concat([existing_df, append_df], ignore_index=True)
            if 'Date' in result_df.columns and 'Boxscore URL' in result_df.columns:
                result_df.drop_duplicates(subset=['Date', 'Boxscore URL'], keep='last', inplace=True)
        else:
            result_df = append_df
        result_df.to_csv(team_path, index=False)

    async def run(self):
        current_year = datetime.now().year
        for team_abbr in self.teams:
            print(f"\n== {team_abbr} incremental ==")
            team_path = os.path.join("teamgamelogs", f"{team_abbr}_boxscores.csv")
            if os.path.exists(team_path):
                try:
                    existing_team_df = pd.read_csv(team_path, low_memory=False)
                except Exception:
                    existing_team_df = pd.DataFrame()
            else:
                existing_team_df = pd.DataFrame()

            # Determine latest date/season from existing file
            latest_date = None
            min_season = self.since_season or (current_year - 2)
            if not existing_team_df.empty and 'Date' in existing_team_df.columns:
                tdate = existing_team_df['Date'].astype(str).str.replace(r"^\w+,\s*", "", regex=True)
                if 'Season' in existing_team_df.columns:
                    team_date_parsed = pd.to_datetime(tdate + ' ' + existing_team_df['Season'].astype(str), errors='coerce')
                else:
                    team_date_parsed = pd.to_datetime(tdate, errors='coerce')
                if team_date_parsed.notna().any():
                    latest_date = team_date_parsed.max()
                    if 'Season' in existing_team_df.columns:
                        try:
                            min_season = int(existing_team_df['Season'].max())
                        except Exception:
                            pass

            seasons_to_check = list(range(min_season, current_year + 1))
            per_season_dfs = []
            for season in seasons_to_check:
                print(f"  📅 schedule {team_abbr} {season}...")
                df = await self.scrape_team_schedule(team_abbr, season)
                if df is None or df.empty:
                    print("    • no rows")
                    continue
                df['Season'] = season
                per_season_dfs.append(df)
            if not per_season_dfs:
                print("  • nothing to append")
                continue
            combined = pd.concat(per_season_dfs, ignore_index=True)
            # Do not consider future games; clamp to today
            today = pd.Timestamp.today().normalize()
            if 'Date_parsed' in combined.columns:
                combined = combined[combined['Date_parsed'].notna() & (combined['Date_parsed'] <= today)]
            if latest_date is not None and 'Date_parsed' in combined.columns:
                mask_new = combined['Date_parsed'] > latest_date
                new_df = combined[mask_new].copy()
            else:
                new_df = combined.copy()
            if new_df.empty:
                print("  • up to date")
                continue
            helper_cols = ["Date_clean", "Date_parsed"]
            final_df = new_df.drop(columns=helper_cols, errors="ignore").copy()
            if 'Season' not in final_df.columns:
                final_df['Season'] = final_df.get('Season', datetime.now().year)
            final_headers = list(final_df.columns)
            entries = final_df.values.tolist()
            print(f"  ▶ new games: {len(entries)}")
            await self.append_boxscores_per_team(entries, final_headers, team_abbr)


async def main():
    updater = PlaywrightMLBIncremental()
    try:
        await updater.setup()
        await updater.run()
    finally:
        await updater.cleanup()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    asyncio.run(main())


