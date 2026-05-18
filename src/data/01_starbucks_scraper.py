"""Scrape current Starbucks locations in metro Atlanta.

Endpoint (confirmed via DevTools):
    https://www.starbucks.com/apiproxy/v1/locations?place=...&lat=...&lng=...

The API returns only the 50 stores nearest the given lat/lng -- limit/offset/
page params are ignored. To cover the whole metro we query a grid of points
and dedupe by store id.

Output: data/raw/starbucks_current.csv
"""

import csv
import time
import requests

API_URL = "https://www.starbucks.com/apiproxy/v1/locations"
HEADERS = {
    "accept": "application/json",
    "referer": "https://www.starbucks.com/store-locator",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/148.0.0.0 Safari/537.36",
    "x-requested-with": "XMLHttpRequest",
}

# Metro Atlanta bounding box. Step ~0.08 deg; each query pulls the 50 nearest
# stores, so overlapping cells guarantee full coverage.
LAT_MIN, LAT_MAX = 33.45, 34.15
LNG_MIN, LNG_MAX = -84.70, -84.00
STEP = 0.08

OUT_CSV = "data/raw/starbucks_current.csv"


def frange(start, stop, step):
    x = start
    while x <= stop:
        yield round(x, 4)
        x += step


def fetch_near(lat, lng):
    params = {"place": "Atlanta, GA, USA", "lat": lat, "lng": lng}
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()  # list of {distance, store: {...}}


def main():
    stores = {}  # id -> row, deduped across grid cells
    points = [(la, ln) for la in frange(LAT_MIN, LAT_MAX, STEP)
              for ln in frange(LNG_MIN, LNG_MAX, STEP)]
    for i, (lat, lng) in enumerate(points, 1):
        try:
            results = fetch_near(lat, lng)
        except requests.RequestException as e:
            print(f"  [{i}/{len(points)}] {lat},{lng}: error {e}")
            continue
        for item in results:
            s = item["store"]
            stores[s["id"]] = {
                "id": s["id"],
                "store_number": s["storeNumber"],
                "name": s["name"],
                "lat": s["coordinates"]["latitude"],
                "lon": s["coordinates"]["longitude"],
                "ownership": s["ownershipTypeCode"],  # CO=company, LS=licensed
                "city": s["address"]["city"],
                "postal_code": s["address"]["postalCode"],
                "address": s["address"]["singleLine"],
            }
        time.sleep(1)  # be polite

    rows = list(stores.values())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    n_co = sum(r["ownership"] == "CO" for r in rows)
    print(f"\nWrote {len(rows)} stores to {OUT_CSV}")
    print(f"  company-operated (CO): {n_co}   licensed (LS): {len(rows) - n_co}")


if __name__ == "__main__":
    main()
