import asyncio
from playwright.async_api import async_playwright
from collections import defaultdict
import pandas as pd
import time
import re
import os
from datetime import datetime, timedelta


class PlaywrightMLBScraper:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
    async def setup(self):
        """Initialize Playwright browser"""
        print("🚀 MLB SCRAPER - Playwright Edition")
        print("✨ Features: 3x faster, better anti-detection, auto-wait")
        
        self.playwright = await async_playwright().start()
        
        # Launch browser with anti-detection settings
        self.browser = await self.playwright.chromium.launch(
            headless=False,  # Set to True for production
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor"
            ]
        )
        
        # Create context with realistic settings
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/New_York"
        )
        
        # Create page
        self.page = await self.context.new_page()
        
        # Add JavaScript to hide automation indicators
        await self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            window.chrome = {
                runtime: {},
            };
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
        """)
        
    async def scrape_team_schedule(self, team_abbr, season, max_retries=3):
        """Scrape team schedule for a season with retry logic"""
        url = f"https://www.baseball-reference.com/teams/{team_abbr}/{season}-schedule-scores.shtml"
        print(f"📊 Scraping {team_abbr} {season} schedule...")
        
        for attempt in range(max_retries):
            try:
                print(f"🌐 Loading {url} (attempt {attempt + 1}/{max_retries})")
                
                # Navigate to page with longer timeout and less strict wait condition
                await self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
                
                # Add a delay to let page fully load
                await asyncio.sleep(3)
                
                # Wait for table to load with longer timeout
                await self.page.wait_for_selector("#team_schedule", timeout=30000)
                
                # Add another small delay
                await asyncio.sleep(1)
                
                # Extract table data using JavaScript
                table_data = await self.page.evaluate("""
                    () => {
                        const table = document.getElementById('team_schedule');
                        if (!table) return [];
                        
                        const rows = [];
                        const tbody = table.querySelector('tbody');
                        if (!tbody) return [];
                        
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
                            
                            if (boxscoreUrl) {
                                rowData.splice(3, 0, boxscoreUrl);
                            }
                            
                            if (rowData.length >= 5) {
                                rows.push(rowData);
                            }
                        }
                        return rows;
                    }
                """)
                
                print(f"✅ Scraped {len(table_data)} games from {team_abbr} {season}")
                return table_data
                
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)[:100]}...")
                
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Increasing delay: 5s, 10s, 15s
                    print(f"🔄 Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                    
                    # Try to refresh the page context
                    try:
                        await self.page.reload(wait_until="domcontentloaded", timeout=30000)
                        await asyncio.sleep(2)
                    except:
                        pass
        
        print(f"❌ Failed to scrape {team_abbr} {season} after {max_retries} attempts")
        return []

    async def scrape_boxscore(self, boxscore_url, team_name, max_retries=2):
        """Scrape individual boxscore data with retry logic"""
        for attempt in range(max_retries):
            try:
                # Navigate with longer timeout and less strict wait condition
                await self.page.goto(boxscore_url, wait_until="domcontentloaded", timeout=60000)
                
                # Add delay to let page load
                await asyncio.sleep(2)
                
                # Debug: Let's see what tables actually exist
                print(f"🔍 DEBUG: Looking for team '{team_name}' tables...")
                
                table_ids = await self.page.evaluate("""
                    () => {
                        const tables = document.querySelectorAll('table[id]');
                        return Array.from(tables).map(t => t.id);
                    }
                """)
                print(f"🔍 DEBUG: Found table IDs: {table_ids}")
                
                # Try multiple possible batting table selectors
                batting_selectors = [
                    f"table[id*='{team_name}batting']",
                    f"table[id='{team_name}batting']", 
                    f"table[id*='{team_name.lower()}batting']"
                ]
                
                batting_table_found = False
                for selector in batting_selectors:
                    try:
                        await self.page.wait_for_selector(selector, timeout=5000)
                        print(f"✅ Found batting table with selector: {selector}")
                        batting_table_found = True
                        break
                    except:
                        print(f"❌ No table found with selector: {selector}")
                        continue
                
                if not batting_table_found:
                    print(f"❌ No batting table found for team: {team_name}")
                    raise Exception(f"No batting table found for {team_name}")
                
                # Add small delay before extraction
                await asyncio.sleep(1)
                
                # Extract all data at once with JavaScript - much faster!
                boxscore_data = await self.page.evaluate(f"""
                    (teamName) => {{
                        const data = {{
                            batter_headers: [],
                            batter_stats: [],
                            pitcher_headers: [],
                            pitcher_stats: [],
                            other_info: {{
                                umpire_HP: '', umpire_1B: '', umpire_2B: '', umpire_3B: '', 
                                umpire_LF: '', umpire_RF: '', time_of_game: '', attendance: '',
                                field_condition: '', weather_temp: '', weather_wind: '', 
                                weather_sky: '', weather_precip: ''
                            }},
                            debug_info: []
                        }};
                        
                        data.debug_info.push(`🔍 Looking for team '${{teamName}}' tables...`);
                        
                        // Get batting stats - only for OUR team
                        const battingTable = document.querySelector(`table[id*='${{teamName}}batting']`);
                        if (battingTable) {{
                            data.debug_info.push(`✅ Found batting table: ${{battingTable.id}}`);
                            const headers = Array.from(battingTable.querySelectorAll('thead th')).map(th => th.textContent.trim()).filter(h => h);
                            data.batter_headers = headers.filter((h, i) => i === 0 || h !== headers[i-1]);
                            
                            const rows = battingTable.querySelectorAll('tbody tr');
                            for (const row of rows) {{
                                if (row.classList.contains('sum') || row.classList.contains('total')) continue;
                                
                                const cells = row.querySelectorAll('td');
                                const nameCell = row.querySelector('th');
                                
                                if (cells.length >= data.batter_headers.length - 1 && nameCell) {{
                                    const batterName = nameCell.textContent.trim();
                                    if (batterName.endsWith(' P')) {{
                                        const ab = cells[0]?.textContent.trim();
                                        if (!ab || ab === '0' || ab === '0.0') continue;
                                    }}
                                    
                                    const batterData = [batterName];
                                    for (let i = 0; i < Math.min(cells.length, data.batter_headers.length - 1); i++) {{
                                        batterData.push(cells[i].textContent.trim());
                                    }}
                                    data.batter_stats.push(batterData);
                                }}
                            }}
                        }} else {{
                            data.debug_info.push(`❌ No batting table found for team: ${{teamName}}`);
                        }}
                        
                        // Get pitching stats - only for OUR team
                        const pitchingTable = document.querySelector(`table[id*='${{teamName}}pitching']`);
                        if (pitchingTable) {{
                            data.debug_info.push(`✅ Found pitching table: ${{pitchingTable.id}}`);
                            const headers = Array.from(pitchingTable.querySelectorAll('thead th')).map(th => th.textContent.trim()).filter(h => h);
                            data.pitcher_headers = headers.filter((h, i) => i === 0 || h !== headers[i-1]);
                            
                            const rows = pitchingTable.querySelectorAll('tbody tr');
                            for (const row of rows) {{
                                if (row.classList.contains('sum') || row.classList.contains('total')) continue;
                                
                                const cells = row.querySelectorAll('td');
                                const nameCell = row.querySelector('th');
                                
                                if (cells.length >= data.pitcher_headers.length - 1 && nameCell) {{
                                    const pitcherName = nameCell.textContent.trim();
                                    const pitcherData = [pitcherName];
                                    for (let i = 0; i < Math.min(cells.length, data.pitcher_headers.length - 1); i++) {{
                                        pitcherData.push(cells[i].textContent.trim());
                                    }}
                                    data.pitcher_stats.push(pitcherData);
                                }}
                            }}
                        }} else {{
                            data.debug_info.push(`❌ No pitching table found for team: ${{teamName}}`);
                        }}
                        
                        // Get other info (umpires, weather, etc.)
                        data.debug_info.push('🔍 Looking for game info sections...');
                        
                        // Try multiple possible selectors for game info
                        const possibleSelectors = [
                            'div.section_content',
                            'div[class*="section"]',
                            'div[id*="other_info"]', 
                            'div[class*="game_info"]',
                            'div.scorebox',
                            'div[class*="scorebox"]'
                        ];
                        
                        let gameInfoText = '';
                        for (const selector of possibleSelectors) {{
                            const elements = document.querySelectorAll(selector);
                            data.debug_info.push(`Found ${{elements.length}} elements with selector: ${{selector}}`);
                            
                            for (const element of elements) {{
                                const text = element.textContent;
                                if (text.includes('Umpires') || text.includes('Time of Game') || 
                                    text.includes('Attendance') || text.includes('Weather')) {{
                                    gameInfoText += text + ' ';
                                    data.debug_info.push(`Found relevant game info: ${{text.substring(0, 200)}}...`);
                                }}
                            }}
                        }}
                        
                        // Also check for specific game info in common areas
                        const scoresArea = document.querySelector('.scores');
                        if (scoresArea) {{
                            gameInfoText += scoresArea.textContent + ' ';
                            data.debug_info.push(`Found scores area text: ${{scoresArea.textContent.substring(0, 200)}}...`);
                        }}
                        
                        // Try to find game info in any div that contains relevant keywords
                        const allDivs = document.querySelectorAll('div');
                        for (const div of allDivs) {{
                            const text = div.textContent;
                            if ((text.includes('Umpires:') || text.includes('Time of Game:')) && text.length < 2000) {{
                                gameInfoText += text + ' ';
                                data.debug_info.push(`Found game info in generic div: ${{text.substring(0, 300)}}...`);
                                break; // Just take the first one we find
                            }}
                        }}
                        
                        if (gameInfoText) {{
                            data.debug_info.push(`Processing game info text: ${{gameInfoText.substring(0, 500)}}...`);
                            
                            // Extract umpire info - try multiple patterns
                            const umpirePatterns = [
                                /Umpires:([^.]+)\\./g,
                                /Umpires:([^\\n]+)/g,
                                /HP[:\\-\\s]+([^,\\n]+)/g,
                                /1B[:\\-\\s]+([^,\\n]+)/g,
                                /2B[:\\-\\s]+([^,\\n]+)/g,
                                /3B[:\\-\\s]+([^,\\n]+)/g,
                                /LF[:\\-\\s]+([^,\\n]+)/g,
                                /RF[:\\-\\s]+([^,\\n]+)/g
                            ];
                            
                            // Try to extract full umpire section first
                            let umpiresFound = false;
                            const umpiresSectionMatch = gameInfoText.match(/Umpires:([^.]+)\\./i);
                            if (umpiresSectionMatch) {{
                                data.debug_info.push(`Found umpires section: ${{umpiresSectionMatch[1]}}`);
                                const umpires = umpiresSectionMatch[1];
                                
                                // Parse individual umpires - more flexible parsing
                                const umpireParts = umpires.split(/[,;]/);
                                for (const part of umpireParts) {{
                                    const trimmed = part.trim();
                                    if (trimmed.includes('HP')) {{
                                        const name = trimmed.replace(/.*HP[:\\-\\s]*/, '').trim();
                                        data.other_info.umpire_HP = name;
                                    }}
                                    if (trimmed.includes('1B')) {{
                                        const name = trimmed.replace(/.*1B[:\\-\\s]*/, '').trim();
                                        data.other_info.umpire_1B = name;
                                    }}
                                    if (trimmed.includes('2B')) {{
                                        const name = trimmed.replace(/.*2B[:\\-\\s]*/, '').trim();
                                        data.other_info.umpire_2B = name;
                                    }}
                                    if (trimmed.includes('3B')) {{
                                        const name = trimmed.replace(/.*3B[:\\-\\s]*/, '').trim();
                                        data.other_info.umpire_3B = name;
                                    }}
                                    if (trimmed.includes('LF')) {{
                                        const name = trimmed.replace(/.*LF[:\\-\\s]*/, '').trim();
                                        data.other_info.umpire_LF = name;
                                    }}
                                    if (trimmed.includes('RF')) {{
                                        const name = trimmed.replace(/.*RF[:\\-\\s]*/, '').trim();
                                        data.other_info.umpire_RF = name;
                                    }}
                                }}
                                umpiresFound = true;
                            }}
                            
                            // Extract time, attendance, weather - more flexible patterns
                            const timePatterns = [
                                /Time of Game:\\s*([^\\n.]+)/i,
                                /Game Time:\\s*([^\\n.]+)/i,
                                /Duration:\\s*([^\\n.]+)/i
                            ];
                            
                            for (const pattern of timePatterns) {{
                                const match = gameInfoText.match(pattern);
                                if (match) {{
                                    data.other_info.time_of_game = match[1].trim();
                                    data.debug_info.push(`Found time: ${{match[1].trim()}}`);
                                    break;
                                }}
                            }}
                            
                            const attendancePatterns = [
                                /Attendance:\\s*([^\\n.]+)/i,
                                /Crowd:\\s*([^\\n.]+)/i
                            ];
                            
                            for (const pattern of attendancePatterns) {{
                                const match = gameInfoText.match(pattern);
                                if (match) {{
                                    data.other_info.attendance = match[1].trim();
                                    data.debug_info.push(`Found attendance: ${{match[1].trim()}}`);
                                    break;
                                }}
                            }}
                            
                            const fieldMatch = gameInfoText.match(/Field Condition:\\s*([^\\n.]+)/i);
                            if (fieldMatch) {{
                                data.other_info.field_condition = fieldMatch[1].trim();
                                data.debug_info.push(`Found field condition: ${{fieldMatch[1].trim()}}`);
                            }}
                            
                            // Weather extraction - multiple patterns
                            const weatherPatterns = [
                                /Start Time Weather:\\s*([^\\n.]+)/i,
                                /Weather:\\s*([^\\n.]+)/i,
                                /Game Weather:\\s*([^\\n.]+)/i
                            ];
                            
                            for (const pattern of weatherPatterns) {{
                                const weatherMatch = gameInfoText.match(pattern);
                                if (weatherMatch) {{
                                    const weather = weatherMatch[1];
                                    data.debug_info.push(`Found weather info: ${{weather}}`);
                                    
                                    // Extract temperature
                                    const tempMatch = weather.match(/(\\d+)°\\s*F/i);
                                    if (tempMatch) data.other_info.weather_temp = tempMatch[0];
                                    
                                    // Extract wind
                                    const windMatch = weather.match(/Wind\\s+([^,\\n]+)/i);
                                    if (windMatch) data.other_info.weather_wind = windMatch[1].trim();
                                    
                                    // Extract sky condition
                                    const skyPatterns = [
                                        /,\\s*([A-Za-z\\s]+),/,
                                        /(Clear|Cloudy|Overcast|Partly Cloudy|Sunny)/i
                                    ];
                                    for (const skyPattern of skyPatterns) {{
                                        const skyMatch = weather.match(skyPattern);
                                        if (skyMatch) {{
                                            data.other_info.weather_sky = skyMatch[1].trim();
                                            break;
                                        }}
                                    }}
                                    
                                    // Extract precipitation
                                    const precipMatch = weather.match(/(No Precipitation|Rain|Snow|Drizzle)/i);
                                    if (precipMatch) data.other_info.weather_precip = precipMatch[1].trim();
                                    
                                    break;
                                }}
                            }}
                            
                            data.debug_info.push(`Extracted other info: ${{JSON.stringify(data.other_info)}}`);
                        }} else {{
                            data.debug_info.push('❌ No game info text found');
                        }}
                        
                        data.debug_info.push(`Final data stats: batters=${{data.batter_stats.length}}, pitchers=${{data.pitcher_stats.length}}`);
                        
                        return data;
                    }}
                """, team_name)
                
                # Print debug information only if there are issues
                debug_info = boxscore_data.get('debug_info', [])
                has_data = len(boxscore_data['batter_stats']) > 0 or len(boxscore_data['pitcher_stats']) > 0
                
                if not has_data:
                    print(f"      ⚠️ No data extracted - Debug info:")
                    for debug_msg in debug_info[-3:]:  # Only show last 3 debug messages
                        print(f"         {debug_msg}")
                
                return boxscore_data
                
            except Exception as e:
                print(f"⚠️ Boxscore attempt {attempt + 1} failed: {str(e)[:100]}...")
                if attempt < max_retries - 1:
                    await asyncio.sleep(3)
        
        print(f"❌ Failed to scrape boxscore after {max_retries} attempts")
        return {
            "batter_headers": [], "batter_stats": [], "pitcher_headers": [], "pitcher_stats": [],
            "other_info": {k: '' for k in ['umpire_HP', 'umpire_1B', 'umpire_2B', 'umpire_3B', 'umpire_LF', 'umpire_RF',
                                           'time_of_game', 'attendance', 'field_condition', 'weather_temp', 
                                           'weather_wind', 'weather_sky', 'weather_precip']}
        }

    def get_team_name(self, team_code):
        """Convert team code to team name for CSS selectors"""
        team_map = {
            "ATH": "Athletics",
            "ARI": "ArizonaDiamondbacks", "ATL": "AtlantaBraves","BAL": "BaltimoreOrioles", 
            "BOS": "BostonRedSox", "CHC": "ChicagoCubs", "CHW": "ChicagoWhiteSox", 
            "CIN": "CincinnatiReds", "CLE": "ClevelandGuardians", "COL": "ColoradoRockies", 
            "DET": "DetroitTigers", "HOU": "HoustonAstros", 
            "KCR": "KansasCityRoyals",
            "SDP" : "SanDiegoPadres",
            "TBR" : "TampaBayRays",
            "SFG" : "SanFranciscoGiants",
            "LAA": "LosAngelesAngels", "LAD": "LosAngelesDodgers", "MIA": "MiamiMarlins",
            "MIL": "MilwaukeeBrewers", "MIN": "MinnesotaTwins", "NYM": "NewYorkMets",
            "NYY": "New YorkYankees", "PHI": "PhiladelphiaPhillies",
            "PIT": "PittsburghPirates", 
            "SEA": "SeattleMariners", "STL": "StLouisCardinals", 
            "TEX": "TexasRangers", "TOR": "TorontoBlueJays", "WSN": "WashingtonNationals"
        }
        return team_map.get(team_code, team_code).replace(" ", "")

    async def scrape_maintable(self):
        """Main scraping function"""
        final_headers = [
            'Gm#', 'Date', 'Boxscore', 'Boxscore URL', 'Tm', 'At', 'Opp', 'W/L', 'R', 'RA', 'Inn',
            'W-L', 'Rank', 'GB', 'Win', 'Loss', 'Save', 'Time', 'D/N', 'Attendance',
            'cLI', 'Streak', 'Orig. Scheduled', 'Season', 'Num_Batters'
        ]
        
        # Check if simple logs already exist
        if os.path.exists("simple_team_logs.csv"):
            print("📋 Found existing simple_team_logs.csv - loading data...")
            df = pd.read_csv("simple_team_logs.csv")
            all_entries = df.values.tolist()
            print(f"✅ Loaded {len(all_entries)} existing games")
            await self.scrape_boxscores(all_entries, final_headers)
            return

        print("🚀 Starting team-by-team scraping...")
        
        mlb_team_map = {
            "ARI": "ArizonaDiamondbacks", "ATL": "AtlantaBraves","BAL": "BaltimoreOrioles", 
            "BOS": "BostonRedSox", "CHC": "ChicagoCubs", "CHW": "ChicagoWhiteSox", 
            "CIN": "CincinnatiReds", "CLE": "ClevelandGuardians", "COL": "ColoradoRockies", 
            "DET": "DetroitTigers", "HOU": "HoustonAstros", "KCR": "KansasCityRoyals",
            "LAA": "LosAngelesAngels", "LAD": "LosAngelesDodgers", "MIA": "MiamiMarlins",
            "MIL": "MilwaukeeBrewers", "MIN": "MinnesotaTwins", "NYM": "NewYorkMets",
            "NYY": "New YorkYankees", "ATH": "OaklandAthletics", "PHI": "PhiladelphiaPhillies",
            "PIT": "PittsburghPirates", "SDP": "SanDiegoPadres", "SFG": "SanFranciscoGiants",
            "SEA": "SeattleMariners", "STL": "StLouisCardinals", "TBR": "TampaBayRays",
            "TEX": "TexasRangers", "TOR": "TorontoBlueJays", "WSN": "WashingtonNationals"
        }
        seasons = [2025]

        all_entries = []
        for team_abbr in mlb_team_map:
            team_entries = []

            # Find last saved 2025 date for this team, if CSV exists
            last_date = None
            csv_path = f"./teamgamelogs/{team_abbr}_boxscores.csv"
            if os.path.exists(csv_path):
                try:
                    df_team = pd.read_csv(csv_path)
                    df2025 = df_team[df_team['Season'] == 2025]
                    if not df2025.empty:
                        last_str = df2025['Date'].iloc[-1]
                        last_date = datetime.strptime(last_str + " 2025", "%A, %b %d %Y")
                except:
                    pass

            for season in seasons:
                games = await self.scrape_team_schedule(team_abbr, season)

                for game in games:
                    raw_date_str = game[1]  # e.g. "Saturday, Aug 9 (1)"

                    # Remove suffix like (1), (2) for parsing/comparison purposes
                    base_date_str = re.sub(r"\s*\(\d+\)$", "", raw_date_str).strip()

                    try:
                        g_date = datetime.strptime(base_date_str + f" {season}", "%A, %b %d %Y")
                    except Exception as e:
                        # Skip if we can't parse it as a date
                        continue

                    # ✅ Compare only clean date without suffix
                    if last_date and g_date <= last_date:
                        continue

                    # Preserve original date with suffix in dataset
                    full_row = game + [season, 0]
                    full_row[1] = raw_date_str  # Ensure CSV keeps "(1)" or "(2)"

                    if len(full_row) == len(final_headers):
                        team_entries.append(full_row)


                await asyncio.sleep(2)

            all_entries.extend(team_entries)
            print(f"📦 Finished {team_abbr}: {len(team_entries)} new games after saved date.")
            await asyncio.sleep(5)

        if all_entries:
            # Save simple logs
            df = pd.DataFrame(all_entries, columns=final_headers)
            df.to_csv("simple_team_logs.csv", index=False)
            print(f"✅ Saved {len(all_entries)} games to simple_team_logs.csv")
            
            # Now scrape boxscores
            await self.scrape_boxscores(all_entries, final_headers)

    async def scrape_boxscores(self, all_entries, final_headers):
        """Scrape boxscores for all games"""
        # Check if enhanced logs already exist
        if os.path.exists("enhanced_team_logs.csv"):
            print("📋 Enhanced boxscore data already exists - skipping boxscore scraping")
            return

        other_info_keys = [
            'umpire_HP', 'umpire_1B', 'umpire_2B', 'umpire_3B', 'umpire_LF', 'umpire_RF',
            'time_of_game', 'attendance', 'field_condition', 'weather_temp', 'weather_wind',
            'weather_sky', 'weather_precip'
        ]

        print("🏟️ Starting boxscore scraping...")

        # Organize entries by team
        team_entries = defaultdict(list)
        for entry in all_entries:
            entry_team = entry[4]
            team_entries[entry_team].append(entry)

        os.makedirs("boxstats", exist_ok=True)

        for team_code, entries in team_entries.items():
            print(f"\n🔄 {team_code} | {len(entries)} games")

            max_batters = 0
            max_pitchers = 0
            batter_headers_ref = []
            pitcher_headers_ref = []
            enhanced_entries = []
            
            team_name = self.get_team_name(team_code)

            for i, entry in enumerate(entries, 1):
                start_time = time.time()
                url = entry[3]

                other_info = {k: '' for k in other_info_keys}
                boxscore_loaded = False
                
                # Try to scrape boxscore
                if url and url.startswith('http'):
                    try:
                        boxscore_data = await self.scrape_boxscore(url, team_name)
                        
                        if len(boxscore_data['batter_stats']) > max_batters:
                            max_batters = len(boxscore_data['batter_stats'])
                            batter_headers_ref = boxscore_data['batter_headers']
                        if len(boxscore_data['pitcher_stats']) > max_pitchers:
                            max_pitchers = len(boxscore_data['pitcher_stats'])
                            pitcher_headers_ref = boxscore_data['pitcher_headers']

                        row = entry.copy()
                        row.extend([boxscore_data['other_info'].get(k, '') for k in other_info_keys])

                        for batter in boxscore_data['batter_stats']:
                            row.extend(batter)
                        row.extend([''] * (len(batter_headers_ref) * (max_batters - len(boxscore_data['batter_stats']))))

                        for pitcher in boxscore_data['pitcher_stats']:
                            row.extend(pitcher)
                        row.extend([''] * (len(pitcher_headers_ref) * (max_pitchers - len(boxscore_data['pitcher_stats']))))

                        enhanced_entries.append(row)
                        boxscore_loaded = True
                        
                        # Success output
                        elapsed = time.time() - start_time
                        batters_count = len(boxscore_data['batter_stats'])
                        pitchers_count = len(boxscore_data['pitcher_stats'])
                        print(f"   ✅ Game {i:>3}/{len(entries)} | {batters_count}B, {pitchers_count}P | {elapsed:.1f}s | {team_code}")
                        
                    except Exception as e:
                        elapsed = time.time() - start_time
                        print(f"   ❌ Game {i:>3}/{len(entries)} | Failed: {str(e)[:50]}... | {elapsed:.1f}s | {team_code}")

                if not boxscore_loaded:
                    # Add empty row for failed boxscore
                    row = entry.copy()
                    row.extend([''] * len(other_info_keys))
                    row.extend([''] * (len(batter_headers_ref) * max_batters))
                    row.extend([''] * (len(pitcher_headers_ref) * max_pitchers))
                    enhanced_entries.append(row)

                # Small delay between boxscores - increased for reliability
                await asyncio.sleep(1)

            # Build final headers and save
            final_enhanced_headers = final_headers.copy()
            final_enhanced_headers.extend(other_info_keys)
            for i in range(1, max_batters + 1):
                for h in batter_headers_ref:
                    final_enhanced_headers.append(f"batter_{i}_{h}")
            for i in range(1, max_pitchers + 1):
                for h in pitcher_headers_ref:
                    final_enhanced_headers.append(f"pitcher_{i}_{h}")

            df = pd.DataFrame(enhanced_entries, columns=final_enhanced_headers)
            output_path = f"./teamgamelogs/{team_code}_boxscores.csv"

            # If file exists, load old, concat with new; else just use new
            if os.path.exists(output_path):
                df_old = pd.read_csv(output_path)
                df_new = pd.concat([df_old, df], ignore_index=True)

            df_new.to_csv(output_path, index=False)

            print(f"📦 {team_code} Complete | {len(df)} new games | Saved/Appended: {output_path}")
            print()


        # Save combined enhanced logs
        print("📊 Building final combined dataset...")
        all_enhanced = []
        
        for team_code in team_entries.keys():
            team_file = f"./teamgamelogs/{team_code}_boxscores.csv"
            if os.path.exists(team_file):
                team_df = pd.read_csv(team_file)
                all_enhanced.extend(team_df.values.tolist())

        if all_enhanced:
            # Use headers from first team file
            first_team = list(team_entries.keys())[0]
            first_file = f"./teamgamelogs/{first_team}_boxscores.csv"
            if os.path.exists(first_file):
                sample_df = pd.read_csv(first_file)
                combined_headers = sample_df.columns.tolist()
                
                combined_df = pd.DataFrame(all_enhanced, columns=combined_headers)
                combined_df.to_csv("enhanced_team_logs.csv", index=False)
                print(f"✅ Saved combined enhanced dataset: {len(combined_df)} total games")

    async def cleanup(self):
        """Clean up resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


async def main():
    """Main execution function"""
    scraper = PlaywrightMLBScraper()
    
    try:
        await scraper.setup()
        await scraper.scrape_maintable()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await scraper.cleanup()


if __name__ == "__main__":
    # ====== STEP 1: Delete the two main CSV files ======
    for file in ["simple_team_logs.csv", "enhanced_team_logs.csv"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ Deleted old file: {file}")

    # ====== STEP 2: Remove 'game_in_day' column from every CSV in teamgamelogs ======
    logs_dir = "teamgamelogs"
    if os.path.exists(logs_dir):
        for fname in os.listdir(logs_dir):
            if fname.endswith(".csv"):
                path = os.path.join(logs_dir, fname)
                try:
                    df = pd.read_csv(path, low_memory=False)
                    if 'game_in_day' in df.columns:
                        df.drop(columns=['game_in_day'], inplace=True)
                        df.to_csv(path, index=False)
                        print(f"✂️ Removed 'game_in_day' column from: {fname}")
                except Exception as e:
                    print(f"⚠️ Could not process {fname}: {e}")

    # ====== STEP 3: Status print (as in your original code) ======
    if os.path.exists("enhanced_team_logs.csv"):
        print("📋 Status: Enhanced boxscore data already exists")
    elif os.path.exists("simple_team_logs.csv"):
        print("📋 Status: Simple team logs exist, will scrape boxscores")
    else:
        print("📋 Status: Starting fresh scrape - will get team schedules then boxscores")

    # ====== STEP 4: Run scraper ======
    asyncio.run(main())
