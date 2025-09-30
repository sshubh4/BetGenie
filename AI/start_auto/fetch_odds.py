#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_odds_season_auto.py
(HISTORICAL snapshots, fast + safe)

- Hits The Odds API v4 historical endpoint across multiple snapshots per day.
- Collects MLB totals (Over/Under) for the exact ET game date (handles DHs cleanly).
- Supports restricting to a CSV (one date per line) or a hardcoded list of dates.
- Uses robust retrying session; appends output (no global dedupe unless you add it).
"""

import csv
import os
import datetime as dt
import time
from typing import List, Dict, Optional, Iterable, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds"
MARKETS = "totals"
REGIONS = "us,us2"
BOOKMAKERS: Optional[str] = ""

SNAPSHOT_TIMES_UTC = [
    "00:00:00","03:00:00","06:00:00","09:00:00","02:00:00","12:00:00","15:00:00","21:00:00",
    "12:00:00","16:00:00","17:00:00","18:00:00","19:00:00","20:00:00","22:00:00","23:59:00",
    
]
NEXT_DAY_PEEKS_UTC = []

# --- Date selection controls ---
DATE_CSV: str = "missing_dates"    # "" to ignore and use ranges
SELECT_DATES: List[str] = []           # explicit list; leave [] to ignore
YEARS_TO_FETCH: List[int] = [2021, 2022, 2023, 2024, 2025]  # which seasons to include from the table below
INCLUDE_2024_SEOUL_SERIES = False      # True to start 2024 at 2024-03-20 instead of 2024-03-28
CAP_RANGES_TO_TODAY = True             # never query future dates

OUTPUT = "mlb_odds.csv"
API_KEY = "c091e2b1fa5b1ec04094ed7cc74bcad4"

TZ_ET = ZoneInfo("America/New_York")

# ── HTTP session ───────────────────────────────────────────────────────────────

def make_session(total_retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    retry = Retry(
        total=total_retries,
        read=total_retries,
        connect=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# ── Helpers ────────────────────────────────────────────────────────────────────

def iso_at_utc(date: dt.date, time_str: str) -> str:
    hh, mm, ss = map(int, time_str.split(":"))
    dtobj = dt.datetime(date.year, date.month, date.day, hh, mm, ss, tzinfo=dt.timezone.utc)
    return dtobj.isoformat().replace("+00:00", "Z")

def et_date_of_commence(commence_iso: str) -> Optional[dt.date]:
    try:
        dt_utc = dt.datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
        return dt_utc.astimezone(TZ_ET).date()
    except Exception:
        return None

def season_date_ranges() -> List[Tuple[dt.date, dt.date]]:
    """
    Exact MLB regular-season windows (no preseason) for 2021–2025.
    2024 note: set INCLUDE_2024_SEOUL_SERIES=True to start on Mar 20 (LG vs LAD in Seoul).
    2025 is capped to today if CAP_RANGES_TO_TODAY is True.
    """
    start_2024 = dt.date(2024, 3, 20) if INCLUDE_2024_SEOUL_SERIES else dt.date(2024, 3, 28)
    seasons: Dict[int, Tuple[dt.date, dt.date]] = {
        #2021: (dt.date(2021, 4, 1),  dt.date(2021, 10, 3)),
        #2022: (dt.date(2022, 4, 7),  dt.date(2022, 10, 5)),
        #2023: (dt.date(2023, 3, 30), dt.date(2023, 10, 1)),
        #2024: (start_2024,           dt.date(2024, 9, 29)),
        #2025: (dt.date(2025, 3, 27), dt.date(2025, 9, 28)),
    }
    today = dt.datetime.now(dt.timezone.utc).date()
    out: List[Tuple[dt.date, dt.date]] = []
    for y in YEARS_TO_FETCH:
        if y not in seasons:
            continue
        s, e = seasons[y]
        if CAP_RANGES_TO_TODAY:
            if s > today:
                continue
            e = min(e, today)
        if s <= e:
            out.append((s, e))
    return out

def iter_days(ranges: List[Tuple[dt.date, dt.date]]) -> Iterable[dt.date]:
    for s, e in ranges:
        d = s
        one = dt.timedelta(days=1)
        while d <= e:
            yield d
            d += one

def load_dates_from_csv(path: str) -> List[dt.date]:
    out: List[dt.date] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            cell = row[0].strip()
            if not cell or cell.lower() in {"date","game_date","dates"}:
                continue
            try:
                out.append(dt.date.fromisoformat(cell))
            except ValueError:
                continue
    return sorted({d for d in out})

def normalize_hardcoded_dates(dates: List[str]) -> List[dt.date]:
    out: List[dt.date] = []
    for s in dates:
        try:
            out.append(dt.date.fromisoformat((s or "").strip()))
        except Exception:
            continue
    return sorted({d for d in out})

# ── Core fetchers ──────────────────────────────────────────────────────────────

def fetch_snapshot(session: requests.Session, api_key: str, iso_ts: str, *, regions=REGIONS, markets=MARKETS, bookmakers=BOOKMAKERS) -> List[Dict]:
    params = {
        "apiKey": api_key,
        "markets": markets,
        "oddsFormat": "decimal",
        "date": iso_ts,
    }
    if bookmakers and bookmakers.strip():
        params["bookmakers"] = bookmakers
    else:
        params["regions"] = regions

    try:
        resp = session.get(BASE_URL, params=params, timeout=30)
    except Exception:
        return []
    if resp.status_code != 200:
        return []
    try:
        js = resp.json()
    except Exception:
        return []
    if isinstance(js, list):
        return js
    if isinstance(js, dict):
        return js.get("data", [])
    return []

def fetch_day_totals_multi_snapshots(session, api_key, day: dt.date, *, snapshot_times_utc, next_day_peeks_utc, target_date_et=None) -> List[Dict]:
    if target_date_et is None:
        target_date_et = day
    latest: Dict[Tuple[str,str,str],Dict] = {}

    def process_events(events: List[Dict], ts_label: str):
        for ev in events:
            ev_id = ev.get("id")
            ct = ev.get("commence_time")
            if not ev_id or not ct:
                continue
            if et_date_of_commence(ct) != target_date_et:
                continue
            home, away = ev.get("home_team"), ev.get("away_team")
            for bm in ev.get("bookmakers") or []:
                book_key = bm.get("key")
                for mk in bm.get("markets") or []:
                    if mk.get("key") != "totals":
                        continue
                    for outcome in mk.get("outcomes") or []:
                        name = (outcome.get("name") or "").title()
                        key = (ev_id, book_key or "", name)
                        latest[key] = {
                            "snapshot": ts_label,
                            "event_id": ev_id,
                            "commence_time": ct,
                            "home_team": home,
                            "away_team": away,
                            "bookmaker": book_key,
                            "name": name,
                            "point": outcome.get("point"),
                            "price": outcome.get("price"),
                        }

    for t in snapshot_times_utc:
        iso_ts = iso_at_utc(day, t)
        process_events(fetch_snapshot(session, api_key, iso_ts), ts_label=iso_ts)

    next_day = day + dt.timedelta(days=1)
    for t in next_day_peeks_utc:
        iso_ts = iso_at_utc(next_day, t)
        process_events(fetch_snapshot(session, api_key, iso_ts), ts_label=iso_ts)

    return list(latest.values())

# ── Main ───────────────────────────────────────────────────────────────────────

def build_date_list() -> List[dt.date]:
    # 1) CSV takes precedence if provided
    csv_dates = load_dates_from_csv(DATE_CSV) if DATE_CSV else []
    if csv_dates:
        return csv_dates
    # 2) Hardcoded explicit list
    hardcoded = normalize_hardcoded_dates(SELECT_DATES)
    if hardcoded:
        return hardcoded
    # 3) Otherwise, use season ranges
    return list(iter_days(season_date_ranges()))

def main(api_key: str, output_csv: str, pause: float = 0.15):
    dates = build_date_list()
    if not dates:
        print("No valid dates to query.")
        return
    print(f"Loaded {len(dates)} date(s): {dates[0]} → {dates[-1]}")
    session = make_session()

    wrote_header_once = os.path.exists(output_csv)  # if file exists, don't write header again

    for day in dates:
        rows = fetch_day_totals_multi_snapshots(
            session, api_key, day,
            snapshot_times_utc=SNAPSHOT_TIMES_UTC,
            next_day_peeks_utc=NEXT_DAY_PEEKS_UTC,
            target_date_et=day,
        )
        print(f"{day} → {len(rows)} rows")
        if rows:
            day_df = pd.DataFrame(rows).drop_duplicates(
                subset=["event_id","bookmaker","name","point","price"], keep="last"
            )
            day_df["commence_time"] = pd.to_datetime(day_df["commence_time"], utc=True, errors="coerce")
            day_df["game_date_et"] = day_df["commence_time"].dt.tz_convert("America/New_York").dt.date

            # append-only write (no read-back, no cross-run dedupe)
            day_df.to_csv(
                output_csv,
                mode="a",
                header=not wrote_header_once,  # write header only the first time
                index=False,
            )
            wrote_header_once = True

        if pause and pause > 0:
            time.sleep(max(0.0, pause - 0.25) + 0.5 * (time.time() % 0.5))

    print(f"✅ Finished appending to {output_csv}")

# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main(API_KEY, OUTPUT, pause=0.05)
