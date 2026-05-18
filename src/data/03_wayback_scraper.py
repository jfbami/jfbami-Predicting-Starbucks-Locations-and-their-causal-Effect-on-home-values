"""Derive Starbucks opening-date upper bounds from the Wayback Machine.

A store's page on starbucks.com cannot be archived before the store exists,
so the earliest snapshot of
    starbucks.com/store-locator/store/<store_id>
is an upper bound on the opening date. The Wayback CDX API is keyless.

COVERAGE CAVEAT: Starbucks launched this /store-locator/store/<id> URL scheme
around 2021-2022. Stores that opened earlier will therefore all show an
earliest snapshot of ~2022 -- a true but uninformative bound. This method is
sharp for 2022+ openings (the recent BeltLine-corridor expansion the project
cares about most); for older "long-standing" stores, fall back to county
business-license / building-permit data.

Run after 02_assign_tracts.py. By default only company-operated (CO) stores
are processed -- licensed kiosks inside groceries/hotels are not the
neighborhood amenity the project models.

Output: data/raw/starbucks_openings.csv
"""

import csv
import time
import requests

CDX_API = "http://web.archive.org/cdx/search/cdx"
IN_CSV = "data/raw/starbucks_clipped.csv"
OUT_CSV = "data/raw/starbucks_openings.csv"
ONLY_COMPANY_OPERATED = True


def earliest_snapshot(store_id, retries=3):
    """Return (YYYY-MM, n_snapshots) for the earliest archive of a store page.

    Returns (None, 0) if never archived.
    """
    params = {
        "url": f"starbucks.com/store-locator/store/{store_id}*",
        "output": "json",
        "fl": "timestamp",
        "collapse": "timestamp:6",  # one row per year-month
    }
    for attempt in range(retries):
        try:
            resp = requests.get(CDX_API, params=params, timeout=60)
        except requests.Timeout:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 503:  # Wayback rate-limit; back off
            time.sleep(5 * (attempt + 1))
            continue
        resp.raise_for_status()
        rows = resp.json()
        if len(rows) <= 1:  # row 0 is the header
            return None, 0
        timestamps = sorted(r[0] for r in rows[1:])
        e = timestamps[0]  # YYYYMMDDhhmmss
        return f"{e[:4]}-{e[4:6]}", len(timestamps)
    raise requests.RequestException(f"503 after {retries} retries for {store_id}")


def main():
    with open(IN_CSV, newline="", encoding="utf-8") as f:
        stores = list(csv.DictReader(f))
    if ONLY_COMPANY_OPERATED:
        stores = [s for s in stores if s["ownership"] == "CO"]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["id", "store_number", "lat", "lon",
             "opening_year_month_upper_bound", "n_snapshots", "source"]
        )
        for i, s in enumerate(stores, 1):
            try:
                ym, n = earliest_snapshot(s["id"])
            except requests.RequestException as e:
                print(f"  [{i}/{len(stores)}] {s['id']}: error {e}")
                continue
            writer.writerow([s["id"], s["store_number"], s["lat"], s["lon"],
                             ym, n, "wayback_cdx"])
            time.sleep(1)  # be polite to the CDX API

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
