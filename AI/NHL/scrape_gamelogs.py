import asyncio
import argparse
import csv
import sys
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from bs4 import BeautifulSoup
import time
import random

HR_BASE = "https://www.hockey-reference.com/leagues"
TEAM_BASE = "https://www.hockey-reference.com/teams"

DEFAULT_TEAMS = [
    "ANA",
    "ARI",
    "BOS",
    "BUF",
    "CGY",
    "CAR",
    "CHI",
    "COL",
    "CBJ",
    "DAL",
    "DET",
    "EDM",
    "FLA",
    "LAK",
    "MIN",
    "MTL",
    "NSH",
    "NJD",
    "NYI",
    "NYR",
    "OTT",
    "PHI",
    "PIT",
    "SJS",
    "SEA",
    "STL",
    "TBL",
    "TOR",
    "VAN",
    "VGK",
    "WSH",
    "WPG",
]


def build_season_string(start_year: int) -> str:
    return f"{start_year}{start_year + 1}"


def compute_default_end_start_year(today: Optional[date] = None) -> int:
    d = today or date.today()
    return d.year if d.month >= 8 else d.year - 1


class AsyncRateLimiter:
    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self._last: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = (self._last + self.min_interval_sec) - now
            if wait > 0:
                await asyncio.sleep(wait + random.uniform(0.05, 0.25))
            self._last = time.monotonic()


async def fetch_text_with_retries(ctx, url: str, retries: int, timeout_ms: int, rate_limiter: Optional[AsyncRateLimiter] = None, verbose: bool = False) -> str:
    delay = 0.75
    last_err: Optional[BaseException] = None
    for _ in range(max(1, retries)):
        try:
            if rate_limiter is not None:
                await rate_limiter.acquire()
            resp = await ctx.get(url, timeout=timeout_ms)
            if resp.ok:
                return await resp.text()
            if resp.status in (400, 404):
                if verbose:
                    print(f"HTTP {resp.status} on {url} — skipping", flush=True)
                return ""
            if resp.status == 429:
                retry_after_hdr = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
                retry_after = None
                if retry_after_hdr:
                    try:
                        retry_after = float(retry_after_hdr)
                    except Exception:
                        retry_after = None
                sleep_s = retry_after if retry_after and retry_after > 0 else 65.0
                if verbose:
                    print(f"HTTP 429 on {url} — cooling down for {sleep_s:.0f}s", flush=True)
                await asyncio.sleep(sleep_s)
            last_err = RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        except PlaywrightTimeoutError as e:
            last_err = e
        except Exception as e:
            last_err = e
        await asyncio.sleep(delay)
        delay = min(delay * 2.0, 8.0)
    if last_err:
        raise last_err
    return ""


def parse_hr_preseason_table(html: str, season_end_year: int) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="games")
    if not table:
        return []
    tbody = table.find("tbody")
    if not tbody:
        return []
    rows_out: List[Dict[str, Any]] = []
    for tr in tbody.find_all("tr"):
        cls = tr.get("class", [])
        if "thead" in cls:
            continue
        # Use data-stat keys where available
        def cell(stat: str) -> Optional[str]:
            td = tr.find("td", attrs={"data-stat": stat})
            if td is None:
                return None
            txt = td.get_text(strip=True)
            return txt if txt != "" else None
        # date is on th with data-stat="date_game"
        th_date = tr.find(["th"], attrs={"data-stat": "date_game"})
        date_game = th_date.get_text(strip=True) if th_date else None
        visitor = cell("visitor_team_name")
        visitor_g = cell("visitor_goals")
        home = cell("home_team_name")
        home_g = cell("home_goals")
        ot = cell("overtime")
        attendance = cell("attendance")
        location = cell("game_location")
        notes = cell("notes")
        if not (date_game and visitor and home):
            continue
        rows_out.append(
            {
                "season": season_end_year,
                "gameType": "Preseason",
                "date": date_game,
                "visitor": visitor,
                "visitor_goals": visitor_g,
                "home": home,
                "home_goals": home_g,
                "ot": ot,
                "attendance": attendance,
                "location": location,
                "notes": notes,
            }
        )
    return rows_out


def parse_team_preseason_table(html: str, team: str, season_end_year: int) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="games")
    if not table:
        return []
    tbody = table.find("tbody")
    if not tbody:
        return []
    out: List[Dict[str, Any]] = []
    for tr in tbody.find_all("tr"):
        if "thead" in tr.get("class", []):
            continue
        def cell(stat: str) -> Optional[str]:
            td = tr.find("td", attrs={"data-stat": stat})
            if td is None:
                return None
            txt = td.get_text(strip=True)
            return txt if txt != "" else None
        th_date = tr.find(["th"], attrs={"data-stat": "date_game"})
        date_game = th_date.get_text(strip=True) if th_date else None
        game_type = (cell("game_type") or "").strip().lower()
        if "preseason" not in game_type:
            continue
        loc = cell("game_location") or ""
        opp = cell("opp_name") or cell("opp_name_abbr")
        goals_for = cell("goals") or cell("team_goal_t")
        goals_against = cell("goals_against") or cell("opp_goal_t")
        ot = cell("overtime")
        attendance = cell("attendance")
        notes = cell("notes")
        if not (date_game and opp):
            continue
        # Map to league-style home/visitor fields
        is_away = loc == "@"
        if is_away:
            visitor, home = (team, opp)
            visitor_goals, home_goals = (goals_for, goals_against)
        else:
            visitor, home = (opp, team)
            visitor_goals, home_goals = (goals_against, goals_for)
        out.append(
            {
                "season": season_end_year,
                "gameType": "Preseason",
                "date": date_game,
                "visitor": visitor,
                "visitor_goals": visitor_goals,
                "home": home,
                "home_goals": home_goals,
                "ot": ot,
                "attendance": attendance,
                "location": loc,
                "notes": notes,
            }
        )
    return out


async def fetch_preseason_for_season(ctx, end_year: int, retries: int, timeout_ms: int, rate_limiter: Optional[AsyncRateLimiter], verbose: bool) -> List[Dict[str, Any]]:
    url = f"{HR_BASE}/NHL_{end_year}_games-preseason.html"
    if verbose:
        print(f"Season {end_year}: requesting {url}", flush=True)
    html = await fetch_text_with_retries(ctx, url, retries=retries, timeout_ms=timeout_ms, rate_limiter=rate_limiter, verbose=verbose)
    if not html:
        return []
    return parse_hr_preseason_table(html, end_year)


async def fetch_team_season(ctx, team: str, end_year: int, retries: int, timeout_ms: int, rate_limiter: Optional[AsyncRateLimiter], verbose: bool) -> List[Dict[str, Any]]:
    url = f"{TEAM_BASE}/{team}/{end_year}_games.html"
    if verbose:
        print(f"Team {team} Season {end_year}: requesting {url}", flush=True)
    html = await fetch_text_with_retries(ctx, url, retries=retries, timeout_ms=timeout_ms, rate_limiter=rate_limiter, verbose=verbose)
    if not html:
        return []
    return parse_team_preseason_table(html, team, end_year)


async def gather_with_semaphore(tasks: List[asyncio.Task], limit: int) -> List[Any]:
    semaphore = asyncio.Semaphore(limit)

    async def with_sem(coro):
        async with semaphore:
            return await coro

    wrapped = [asyncio.create_task(with_sem(t)) for t in tasks]
    return await asyncio.gather(*wrapped)


def dedupe_by_key(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        k = r.get(key)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = [
        "season",
        "gameType",
        "date",
        "visitor",
        "visitor_goals",
        "home",
        "home_goals",
        "ot",
        "attendance",
        "location",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


async def run(start_season: int, end_season: int, out_path: str, concurrency: int, retries: int, timeout_ms: int, min_interval_ms: int, user_agent: Optional[str], verbose: bool, mode: str, teams: List[str], progress: bool) -> None:
    async with async_playwright() as p:
        headers = {
            "User-Agent": user_agent
            or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.hockey-reference.com/",
        }
        req = await p.request.new_context(extra_http_headers=headers)
        rate_limiter = AsyncRateLimiter(min_interval_sec=max(0.0, float(min_interval_ms) / 1000.0))
        years = list(range(start_season, end_season + 1))
        if progress:
            print(f"Starting NHL preseason scrape | mode={mode} | seasons={years} | concurrency={max(1, concurrency)}", flush=True)
        elif verbose:
            print(f"Seasons: {years}", flush=True)
        tasks: List[asyncio.Task] = []
        index_keys: List[Tuple[str, int]] = []
        if mode == "team":
            if progress:
                print(f"Teams: {teams}", flush=True)
            for y in years:
                for t in teams:
                    tasks.append(asyncio.create_task(fetch_team_season(req, t, y, retries=retries, timeout_ms=timeout_ms, rate_limiter=rate_limiter, verbose=verbose)))
                    index_keys.append((t, y))
        else:
            for y in years:
                tasks.append(asyncio.create_task(fetch_preseason_for_season(req, y, retries=retries, timeout_ms=timeout_ms, rate_limiter=rate_limiter, verbose=verbose)))
                index_keys.append(("LEAGUE", y))
        results_nested = await gather_with_semaphore(tasks, limit=max(1, concurrency))
        all_rows: List[Dict[str, Any]] = []
        for idx, part in enumerate(results_nested):
            key = index_keys[idx]
            if progress or verbose:
                if key[0] == "LEAGUE":
                    print(f"Season {key[1]}: parsed {len(part)} rows", flush=True)
                else:
                    print(f"Team {key[0]} Season {key[1]}: parsed {len(part)} preseason rows", flush=True)
            all_rows.extend(part)
        # League page has no unique ID; dedupe by composite key
        seen_keys = set()
        deduped: List[Dict[str, Any]] = []
        for r in all_rows:
            key = (r.get("season"), r.get("date"), r.get("visitor"), r.get("home"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(r)
        deduped.sort(key=lambda r: (r.get("season"), r.get("date"), r.get("visitor"), r.get("home")))
        write_csv(out_path, deduped)
        if progress or verbose:
            print(f"Wrote {len(deduped)} rows to {out_path}", flush=True)
        await req.dispose()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    default_end = compute_default_end_start_year()
    parser = argparse.ArgumentParser(prog="scrape_gamelogs", add_help=True)
    parser.add_argument("--start-season", type=int, default=2015)
    parser.add_argument("--end-season", type=int, default=default_end)
    parser.add_argument("--out", type=str, default=f"nhl_preseason_gamelogs_2015_{default_end}.csv")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--min-interval-ms", type=int, default=3500)
    parser.add_argument("--user-agent", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress prints")
    parser.add_argument("--mode", choices=["league", "team"], default="team")
    parser.add_argument("--teams", type=str, default="ALL", help="Comma-separated team abbreviations or ALL")
    return parser.parse_args(argv)


def validate_range(start_season: int, end_season: int) -> None:
    if end_season < start_season:
        raise ValueError("end-season must be >= start-season")
    if start_season < 1917 or end_season > 2100:
        raise ValueError("season bounds out of range")


def parse_teams_arg(arg: str) -> List[str]:
    if not arg or arg.upper() == "ALL":
        return DEFAULT_TEAMS
    teams = [t.strip().upper() for t in arg.split(",") if t.strip()]
    return teams if teams else DEFAULT_TEAMS


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        validate_range(args.start_season, args.end_season)
        teams = parse_teams_arg(args.teams)
        progress = not args.no_progress
        asyncio.run(
            run(
                start_season=args.start_season,
                end_season=args.end_season,
                out_path=args.out,
                concurrency=args.concurrency,
                retries=args.retries,
                timeout_ms=args.timeout_ms,
                min_interval_ms=args.min_interval_ms,
                user_agent=args.user_agent,
                verbose=args.verbose,
                mode=args.mode,
                teams=teams,
                progress=progress,
            )
        )
        return 0
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


