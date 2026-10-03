"""Download the SONATE-2 pass recordings and SatNOGS ground-truth frames.

Usage:
    python fetch_data.py                         # our three passes
    python fetch_data.py 15104225 15052729       # specific SatNOGS observation IDs

Pass audio needs no account. Ground-truth frames come from the SatNOGS DB API and need
a free account's API token (db.satnogs.org -> profile settings):
    SATNOGS_API_TOKEN=<token> python fetch_data.py
"""
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

DATA = "data/satnogs"
NORAD = 59112   # SONATE-2
PASSES = [15039241, 15052729, 15104225]


def get_json(url, headers=None):
    """GET a JSON page, waiting out rate limits. Returns (data, next_page_url)."""
    while True:
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=60)
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link") or "")
            return json.load(r), (m.group(1) if m else None)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            wait = int(e.headers.get("Retry-After") or 60) + 2
            print(f"  rate limited, waiting {wait}s", flush=True)
            time.sleep(wait)


def fetch_pass(obs_id):
    d = f"{DATA}/sonate2_{obs_id}"
    os.makedirs(d, exist_ok=True)
    o, _ = get_json(f"https://network.satnogs.org/api/observations/{obs_id}/?format=json")
    if o["norad_cat_id"] != NORAD:
        raise ValueError(f"Observation {obs_id} is NORAD {o['norad_cat_id']}, not SONATE-2 ({NORAD})")
    json.dump(o, open(f"{d}/obs.json", "w"), indent=1)
    if not os.path.exists(f"{d}/audio.ogg"):
        urllib.request.urlretrieve(o["payload"], f"{d}/audio.ogg")
    print(f"pass {obs_id}: {o['start']} {o['station_name']} -> {d}/audio.ogg")
    return o


def fetch_frames(observations, token):
    headers = {"Authorization": "Token " + token, "Accept": "application/json"}
    rows = []
    for o in observations:
        url = (f"https://db.satnogs.org/api/telemetry/?format=json&satellite={NORAD}"
               f"&start={o['start']}&end={o['end']}")
        n = 0
        while url:
            page, url = get_json(url, headers)
            if isinstance(page, dict):
                page = page.get("results", page)
            rows += page
            n += len(page)
        print(f"frames for pass {o['id']}: {n}")
    out = f"{DATA}/sonate2_frames.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out} ({len(rows)} rows)")


def main():
    ids = [int(a) for a in sys.argv[1:]] or PASSES
    observations = [fetch_pass(i) for i in ids]
    token = os.environ.get("SATNOGS_API_TOKEN")
    if token:
        fetch_frames(observations, token)
    else:
        print("SATNOGS_API_TOKEN not set: skipping ground-truth frames (the decoder runs without them).")


if __name__ == "__main__":
    main()
