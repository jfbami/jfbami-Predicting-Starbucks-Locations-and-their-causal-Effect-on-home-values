"""Assign each Starbucks store a census tract, then clip to the study area.

Uses the Census geocoder coordinates->geography endpoint (keyless) to look up
the tract for each store's lat/lon. Keeps only stores in the two study-area
counties named in the project plan:

    Georgia state FIPS 13  ->  Fulton County 121,  DeKalb County 089

Input:  data/raw/starbucks_current.csv
Output: data/raw/starbucks_clipped.csv  (in-study stores, with tract_geoid)
"""

import csv
import time
import requests

GEOCODER = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
IN_CSV = "data/raw/starbucks_current.csv"
OUT_CSV = "data/raw/starbucks_clipped.csv"

STATE_FIPS = "13"                  # Georgia
STUDY_COUNTIES = {"121", "089"}    # Fulton, DeKalb


def lookup_tract(lat, lon, retries=3):
    """Return (geoid, county_fips) for a point, or (None, None) if unresolved."""
    params = {
        "x": lon, "y": lat,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(GEOCODER, params=params, timeout=30)
            resp.raise_for_status()
            tracts = resp.json()["result"]["geographies"].get("Census Tracts", [])
            if not tracts:
                return None, None
            ct = tracts[0]
            return ct["GEOID"], ct["COUNTY"]
        except (requests.RequestException, KeyError, ValueError):
            time.sleep(2 * (attempt + 1))
    return None, None


def main():
    with open(IN_CSV, newline="", encoding="utf-8") as f:
        stores = list(csv.DictReader(f))

    kept, dropped, unresolved = [], 0, 0
    for s in stores:
        geoid, county = lookup_tract(s["lat"], s["lon"])
        if geoid is None:
            unresolved += 1
        elif county in STUDY_COUNTIES and geoid.startswith(STATE_FIPS):
            s["tract_geoid"] = geoid
            kept.append(s)
        else:
            dropped += 1
        time.sleep(0.3)

    fieldnames = list(stores[0].keys()) + ["tract_geoid"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    n_co = sum(r["ownership"] == "CO" for r in kept)
    print(f"\nWrote {len(kept)} in-study stores to {OUT_CSV}")
    print(f"  kept: {len(kept)}  |  dropped (out of study area): {dropped}  "
          f"|  unresolved: {unresolved}")
    print(f"  of kept, company-operated (CO): {n_co}")


if __name__ == "__main__":
    main()
