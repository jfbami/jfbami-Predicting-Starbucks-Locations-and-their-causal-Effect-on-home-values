"""Pull built-environment POIs for metro Atlanta from OpenStreetMap.

Uses the Overpass API (keyless; just needs a User-Agent). Collects every POI in
the chosen categories across a metro bounding box into one flat file. Downstream
scripts compute "count within X miles of a location" from this -- both the ZIP
panel covariates and the site-selection model's catchment features.

Note: Overpass returns the CURRENT map only (no history). These are present-day
counts; treat them as a static covariate snapshot.

Output: data/raw/osm_pois.csv
"""

import csv
import time
import requests

OVERPASS = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "starlight-research/1.0 (academic project)"}
OUT_CSV = "data/raw/osm_pois.csv"

# Generous metro-Atlanta bounding box: south,west,north,east
BBOX = "33.40,-84.90,34.30,-83.85"

# category -> list of OSM tag filters that count as that category
CATEGORIES = {
    "cafe": ['["amenity"="cafe"]', '["shop"="coffee"]'],
    "restaurant": ['["amenity"="restaurant"]', '["amenity"="fast_food"]'],
    "gym": ['["leisure"="fitness_centre"]'],
    "bank": ['["amenity"="bank"]'],
    "grocery": ['["shop"="supermarket"]', '["shop"="grocery"]'],
    "school": ['["amenity"="school"]'],
    "park": ['["leisure"="park"]'],
}


def fetch_category(filters):
    """Run one Overpass query for a category; return its elements."""
    parts = "".join(f'nwr{f}({BBOX});' for f in filters)
    query = f"[out:json][timeout:180];({parts});out center;"
    resp = requests.post(OVERPASS, data={"data": query},
                         headers=HEADERS, timeout=240)
    resp.raise_for_status()
    return resp.json()["elements"]


def coords(el):
    """Return (lat, lon) for a node, or the center for a way/relation."""
    if el["type"] == "node":
        return el.get("lat"), el.get("lon")
    c = el.get("center", {})
    return c.get("lat"), c.get("lon")


def main():
    rows = []
    for category, filters in CATEGORIES.items():
        elements = fetch_category(filters)
        for el in elements:
            lat, lon = coords(el)
            if lat is None:
                continue
            rows.append({
                "category": category,
                "osm_type": el["type"],
                "osm_id": el["id"],
                "name": (el.get("tags", {}) or {}).get("name", ""),
                "lat": lat,
                "lon": lon,
            })
        time.sleep(3)  # be polite to the public Overpass server

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} POIs to {OUT_CSV}")


if __name__ == "__main__":
    main()
