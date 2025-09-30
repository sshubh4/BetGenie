#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_odds_season_auto.py
(HISTORICAL snapshots, fast + safe)

- Hits The Odds API v4 historical endpoint across multiple snapshots per day.
- Collects MLB totals (Over/Under) for the exact ET game date (handles DHs cleanly).
- Supports restricting to a CSV/TXT (one date per line or a 'date' column) or a hardcoded list of dates.
- Uses robust retrying session; appends output (no global dedupe unless you add it).
"""

import csv
import glob
import os
import re
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
BOOKMAKERS: Optional[str] = ""  # e.g. "draftkings,caesars"; leave "" to use REGIONS (cheaper)

# sensible, deduped spread of UTC snapshots
SNAPSHOT_TIMES_UTC = [
    "00:00:00","02:00:00","03:00:00","06:00:00","09:00:00",
    "12:00:00","15:00:00","16:00:00","17:00:00","18:00:00",
    "19:00:00","20:00:00","21:00:00","22:00:00","23:59:00",
]
NEXT_DAY_PEEKS_UTC: List[str] = []  # e.g. ["00:30:00","01:30:00"]

# --- Date selection controls ---
DATE_FILE: str = "missing_dates.csv"  # can omit extension; CSV or TXT supported. "" to ignore and use ranges.
SELECT_DATES: List[str] = []          # explicit list of ISO dates; leave [] to ignore
YEARS_TO_FETCH: List[int] = []        # [] means don't use season ranges unless DATE_FILE/SELECT_DATES are empty
INCLUDE_2024_SEOUL_SERIES = False     # True to start 2024 at 2024-03-20 instead of 2024-03-28
CAP_RANGES_TO_TODAY = True            # never query future dates
LIMIT_DATES: Optional[int] = None     # e.g. 25 to only do first 25 dates from the file

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

def _find_existing_path(path_like: str) -> Optional[str]:
    """
    Return an existing file path trying common variants:
    - as given
    - with .csv / .txt
    - relative to this script directory
    - glob 'path*' (pick most recent)
    """
    if not path_like:
        return None

    candidates = [path_like, f"{path_like}.csv", f"{path_like}.txt"]
    here = os.path.dirname(os.path.abspath(__file__))
    for c in candidates:
        if os.path.exists(c):
            return c
        c2 = os.path.join(here, c)
        if os.path.exists(c2):
            return c2

    # try globbing
    for pat in (path_like + "*", os.path.join(here, path_like + "*")):
        matches = glob.glob(pat)
        if matches:
            # pick latest modified
            return max(matches, key=os.path.getmtime)
    return None

_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

def _parse_dates_from_dataframe(df: pd.DataFrame) -> List[dt.date]:
    cols = [c for c in df.columns if isinstance(c, str)]
    # Prefer explicit columns
    for name in ["date", "game_date", "Date", "DATE"]:
        if name in df.columns:
            s = pd.to_datetime(df[name].astype(str).str.strip(), errors="coerce").dt.date
            return [d for d in s.dropna().tolist()]  # type: ignore
    # Else: scan all string cells for YYYY-MM-DD
    found: List[dt.date] = []
    for c in cols:
        ser = df[c].astype(str)
        for cell in ser:
            m = _DATE_RE.search(cell)
            if m:
                try:
                    found.append(dt.date.fromisoformat(m.group(0)))
                except ValueError:
                    pass
    return found

def load_dates_from_file(path_like: str) -> List[dt.date]:
    path = _find_existing_path(path_like)
    if not path:
        return []
    ext = os.path.splitext(path)[1].lower()

    dates: List[dt.date] = []
    try:
        if ext in {".csv", ".tsv", ".txt", ""}:
            # Try pandas first (handles CSV/TSV nicely)
            try:
                sep = "\t" if ext == ".tsv" else None
                df = pd.read_csv(path, dtype=str, sep=sep, engine="python")
                dates = _parse_dates_from_dataframe(df)
            except Exception:
                # Fallback: one date per line text file
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # allow "date,..." or plain "YYYY-MM-DD"
                        token = line.split(",")[0].strip()
                        if token.lower() in {"date","game_date","dates"}:
                            continue
                        m = _DATE_RE.search(token)
                        if m:
                            try:
                                dates.append(dt.date.fromisoformat(m.group(0)))
                            except ValueError:
                                pass
        else:
            # Unknown extension; try pandas anyway
            df = pd.read_csv(path, dtype=str, engine="python")
            dates = _parse_dates_from_dataframe(df)
    except FileNotFoundError:
        return []

    # dedupe + sort
    dates = sorted({d for d in dates})
    return dates

def normalize_hardcoded_dates(dates: List[str]) -> List[dt.date]:
    out: List[dt.date] = []
    for s in dates:
        try:
            out.append(dt.date.fromisoformat((s or "").strip()))
        except Exception:
            continue
    return sorted({d for d in out})

def season_date_ranges() -> List[Tuple[dt.date, dt.date]]:
    """
    Example hook for season windows (left empty by default to avoid long runs).
    Toggle YEARS_TO_FETCH above if you want to use these.
    """
    start_2024 = dt.date(2024, 3, 20) if INCLUDE_2024_SEOUL_SERIES else dt.date(2024, 3, 28)
    seasons: Dict[int, Tuple[dt.date, dt.date]] = { }
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

def fetch_day_totals_multi_snapshots(
    session: requests.Session,
    api_key: str,
    day: dt.date,
    *,
    snapshot_times_utc: List[str],
    next_day_peeks_utc: List[str],
    target_date_et: Optional[dt.date] = None
) -> List[Dict]:
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

# ── Date selection ─────────────────────────────────────────────────────────────

def build_date_list() -> List[dt.date]:
    # 1) File takes precedence if provided
    file_dates = load_dates_from_file(DATE_FILE) if DATE_FILE else []
    if file_dates:
        if LIMIT_DATES:
            file_dates = file_dates[:LIMIT_DATES]
        print(f"Loaded {len(file_dates)} date(s) from file: {DATE_FILE}")
        return file_dates

    # 2) Hardcoded explicit list
    hardcoded = normalize_hardcoded_dates(SELECT_DATES)
    if hardcoded:
        if LIMIT_DATES:
            hardcoded = hardcoded[:LIMIT_DATES]
        print(f"Loaded {len(hardcoded)} date(s) from SELECT_DATES")
        return hardcoded

    # 3) Otherwise, use season ranges (only if explicitly requested)
    ranges = season_date_ranges() if YEARS_TO_FETCH else []
    dates = list(iter_days(ranges)) if ranges else []
    if LIMIT_DATES and dates:
        dates = dates[:LIMIT_DATES]
    return dates

# ── Main ───────────────────────────────────────────────────────────────────────

def main(api_key: str, output_csv: str, pause: float = 0.15):
    dates = build_date_list()
    if not dates:
        print("No valid dates to query. (Set DATE_FILE, SELECT_DATES, or YEARS_TO_FETCH)")
        return
    print(f"Running for {len(dates)} date(s): {dates[0]} → {dates[-1]}")
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
