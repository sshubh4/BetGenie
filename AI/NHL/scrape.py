import asyncio
from playwright.async_api import async_playwright
import pandas as pd
from bs4 import BeautifulSoup

class NHLLeagueScraper:
    def __init__(self):
        self.base_url = "https://www.hockey-reference.com/leagues/"
        self.seasons = range(2015, 2026)
        self.headers = [
            "Season", "Date", "Time", "Visitor", "Visitor_G", "Home", "Home_G", "Attendance",
            "Arena", "Duration", "Boxscore_URL",
            "Team1_Name", "Team1_Skaters", "Team1_Goalies", "Team1_Adv_All", "Team2_Name",
            "Team2_Skaters", "Team2_Goalies", "Team2_Adv_All"
        ]

    async def get_league_schedule(self, page, season):
        url = f"{self.base_url}NHL_{season}_games.html"
        await page.goto(url, timeout=60000)
        
        tables = await page.query_selector_all("table")
        all_rows = []
        len_games = 0
        for tbl in tables:
            header_text = await tbl.inner_text()
            if "Visitor" in header_text and "Home" in header_text:
                rows = await tbl.query_selector_all("tbody > tr")
                print(f"Season {season}: schedule rows found = {len(rows)}", flush=True)
                for row in rows:
                    tds = await row.query_selector_all("td")
                    if not tds or len(tds) < 7:
                        continue                    
                    date_cell = await row.query_selector("th[data-stat='date_game']")
                    date_txt = await date_cell.inner_text() if date_cell else ""
                    link = await date_cell.query_selector("a")
                    box_url = await link.get_attribute("href") if link else ""
                    box_url = "https://www.hockey-reference.com" + box_url if box_url else ""
                    time_txt = await tds[0].inner_text()
                    visitor = await tds[1].inner_text()
                    visitor_g = await tds[2].inner_text()
                    home = await tds[3].inner_text()
                    home_g = await tds[4].inner_text()
                    att = await tds[6].inner_text()
                    all_rows.append([
                        season, date_txt, time_txt, visitor, visitor_g, home, home_g, att, box_url
                    ])
                    len_games += 1
                    print(f"Season {season}: games done = {len_games}", flush=True)
        return all_rows

    async def get_boxscore_details(self, page, box_url):
        await page.goto(box_url, timeout=90000)
        soup = BeautifulSoup(await page.content(), "lxml")

        # Meta info: sometimes not .scorebox_meta, so search by label
        meta = soup.find("div", class_="scorebox_meta")
        attendance, arena, duration = "", "", ""
        if meta:
            text = meta.get_text(" ")
            for div in meta.find_all('div'):
                t = div.get_text(" ")
                if "Arena" in t:
                    arena = t.split("Arena")[-1].strip(' :')
                if "Game Duration" in t:
                    duration = t.split("Game Duration")[-1].strip(' :')

        # Team tables extraction: loop in page order
        h2s = soup.find_all("h2")
        team_tables = []
        adv_tables = []

        # Find both team's entire blocks by moving from h2 onwards
        for h2 in h2s:
            name = h2.get_text(strip=True)
            # Next <table> for skaters
            skaters_table_html = ""
            goalies_table_html = ""
            advanced_table_html = ""
            tbl = h2.find_next_sibling("div", class_="table_container")
            if not tbl:
                tbl = h2.find_next_sibling("table")
            while tbl:
                cap = tbl.find("caption")
                if cap and "Advanced" in cap.get_text():
                    advanced_table_html = str(tbl)
                    break
                if skaters_table_html == "" and (cap is None or "Goalies" not in cap.get_text()):
                    skaters_table_html = str(tbl)
                elif goalies_table_html == "" and cap and "Goalies" in cap.get_text():
                    goalies_table_html = str(tbl)
                tbl = tbl.find_next_sibling()
            team_tables.append((name, skaters_table_html, goalies_table_html, advanced_table_html))

        # If structure is team1 then team2, always order as [home, visitor] per main table
        if len(team_tables) < 2:
            team_tables += [("", "", "", "")] * (2 - len(team_tables))

        # Compose output: unpack names/tables for clarity
        t1, t2 = team_tables[0], team_tables[1]

        return {
            "attendance": attendance,
            "arena": arena,
            "duration": duration,
            "team1_name": t1[0], "team1_skaters": t1[1], "team1_goalies": t1[2], "team1_adv": t1[3],
            "team2_name": t2[0], "team2_skaters": t2[1], "team2_goalies": t2[2], "team2_adv": t2[3]
        }

    async def run(self, output_csv="nhl_full_season_details.csv"):
        records = []
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            for season in self.seasons:
                print(f"Scraping season {season}", flush=True)
                schedule = await self.get_league_schedule(page, season)
                print(f"Season {season}: schedule games found = {len(schedule)}", flush=True)
                processed = 0
                attempted = 0
                for row in schedule:
                    box_url = row[10]
                    if not box_url:
                        continue
                    attempted += 1
                    details = await self.get_boxscore_details(page, box_url)
                    row[7] = details["attendance"]
                    row[8] = details["arena"]
                    row[9] = details["duration"]
                    row.extend([
                        details["team1_name"], details["team1_skaters"], details["team1_goalies"], details["team1_adv"],
                        details["team2_name"], details["team2_skaters"], details["team2_goalies"], details["team2_adv"]
                    ])
                    records.append(row)
                    processed += 1
                    print(f"Done: {row[0]} {row[1]} {row[3]} vs {row[5]}", flush=True)
                print(
                    f"Season {season}: processed {processed} games (with boxscores) out of {attempted} attempted; schedule total {len(schedule)}",
                    flush=True,
                )
            await browser.close()
        df = pd.DataFrame(records, columns=self.headers)
        df.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}", flush=True)

if __name__ == "__main__":
    asyncio.run(NHLLeagueScraper().run())       