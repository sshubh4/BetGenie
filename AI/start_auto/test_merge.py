#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe multiple historical snapshots on The Odds API for MLB game totals (Over/Under).

- Hits the v4 historical odds endpoint multiple times across the date in UTC.
- Prints any O/U lines it finds (points + prices) per bookmaker.
- Designed to catch doubleheader legs that show up at different snapshot times.

Edit HOME, AWAY, DATE to target a specific matchup/day.
"""

import requests

# ========== EDIT THESE ==========
API_KEY = "c091e2b1fa5b1ec04094ed7cc74bcad4"  # YOUR Odds API key (you asked to hardcode)
SPORT   = "baseball_mlb"

DATE = "2021-06-19"  # YYYY-MM-DD (local calendar day you care about)
HOME = "Washington Nationals"
AWAY = "New York Mets"

# Try many UTC timestamps so we don't miss when totals get posted
TIMES_UTC = [
    "00:00:00","03:00:00","06:00:00","09:00:00",
    "12:00:00","15:00:00","16:00:00","17:00:00",
    "18:00:00","19:00:00","20:00:00","21:00:00",
    "22:00:00","23:59:00",
    # also peek slightly into the next day UTC (late ET games spill over)
    # comment out if not needed:
    # next-day probes:
]
NEXT_DAY_PEEKS = ["00:30:00","01:00:00","02:00:00"]  # used with DATE+1
INCLUDE_NEXT_DAY_PEEKS = True
# ===============================

def fetch_snapshot(iso_ts: str):
    """Return list of events for a given historical snapshot (closest <= iso_ts)."""
    url = f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us,us2",
        "markets": "totals",
        "oddsFormat": "decimal",
        "date": iso_ts,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    if isinstance(js, list):
        return js
    elif isinstance(js, dict):
        # Unexpected structure (e.g., error payload). Print for debugging, return empty list.
        return js.get("data", []) if isinstance(js.get("data"), list) else []
    else:
        print(f"[DEBUG] Unexpected JSON type at {iso_ts}: {type(js)}")
        return []

def print_totals(events, home_name, away_name, iso_ts):
    """Scan events for the given home/away and print any totals found."""
    any_found = False
    for ev in events:
        if not isinstance(ev, dict):
            continue
        home = ev.get("home_team")
        away = ev.get("away_team")
        # Match regardless of order (DH legs are separate events)
        if {home, away} == {home_name, away_name}:
            commence = ev.get("commence_time")
            for bm in ev.get("bookmakers", []) or []:
                title = bm.get("title")
                for mk in bm.get("markets", []) or []:
                    if mk.get("key") != "totals":
                        continue
                    outs = mk.get("outcomes", []) or []
                    over = next((o for o in outs if (o.get("name") or "").lower() == "over"), None)
                    under = next((o for o in outs if (o.get("name") or "").lower() == "under"), None)
                    if over and under:
                        any_found = True
                        print(f"[{iso_ts}] {away} @ {home}  ({commence})  | {title} "
                              f"| O/U {over.get('point')}  Over={over.get('price')}  Under={under.get('price')}")
    if not any_found:
        print(f"[{iso_ts}] no totals yet for {AWAY} @ {HOME}")

def next_day(date_str: str) -> str:
    from datetime import datetime, timedelta
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")

def main():
    # Probe the target calendar day
    for t in TIMES_UTC:
        iso = f"{DATE}T{t}Z"
        try:
            events = fetch_snapshot(iso)
        except Exception as e:

            continue
        print_totals(events, HOME, AWAY, iso)

    # Optionally peek into early next-day UTC (helps with late ET starts / Game 2 of DH)
    if INCLUDE_NEXT_DAY_PEEKS:
        nd = next_day(DATE)
        for t in NEXT_DAY_PEEKS:
            iso = f"{nd}T{t}Z"
            try:
                events = fetch_snapshot(iso)
            except Exception as e:
                continue
            print_totals(events, HOME, AWAY, iso)

if __name__ == "__main__":
    main()
