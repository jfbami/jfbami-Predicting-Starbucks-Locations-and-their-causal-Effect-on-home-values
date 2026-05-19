"""Phase 2, step 1-2: assemble the site-selection training set.

positives = company-operated Starbucks locations.
negatives = other retail POIs (restaurants, banks, gyms, grocery) -- real,
            commercially-viable spots with no Starbucks, sampled ~7x the
            positives.

Both classes are restricted to the bounding box where the Starbucks scrape
was comprehensive, so a "negative" genuinely means we checked and found no
Starbucks there. Negatives co-located with a Starbucks are dropped.

This produces labeled locations only; features come in the next step.

Output: data/processed/site_candidates.csv
"""

import numpy as np
import pandas as pd

CURRENT = "data/raw/starbucks_current.csv"
OSM = "data/raw/osm_pois.csv"
OUT = "data/processed/site_candidates.csv"

# bounding box of the comprehensive Starbucks scrape (the 01_starbucks grid)
LAT_MIN, LAT_MAX = 33.45, 34.15
LON_MIN, LON_MAX = -84.70, -84.00

NEG_CATEGORIES = ["restaurant", "bank", "gym", "grocery"]
EXCLUDE_RADIUS_M = 150   # drop negative candidates this close to any Starbucks
NEG_PER_POS = 7          # negatives : positives ratio
SEED = 42


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in metres (NumPy broadcasting)."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def in_box(df):
    return df[df["lat"].between(LAT_MIN, LAT_MAX) &
              df["lon"].between(LON_MIN, LON_MAX)]


def main():
    sb = pd.read_csv(CURRENT)
    sb_lat, sb_lon = sb["lat"].to_numpy(), sb["lon"].to_numpy()  # all stores

    # --- positives: company-operated stores inside the comprehensive box ---
    pos = in_box(sb[sb["ownership"] == "CO"])
    positives = pd.DataFrame({
        "site_id": "sbux_" + pos["id"].astype(str),
        "lat": pos["lat"], "lon": pos["lon"],
        "label": 1, "source": "starbucks", "name": pos["name"],
    })

    # --- negative pool: other retail POIs in the same box ---
    osm = pd.read_csv(OSM)
    pool = in_box(osm[osm["category"].isin(NEG_CATEGORIES)]).copy()
    # distance from each candidate to the nearest Starbucks; drop co-located
    dist = haversine_m(pool["lat"].to_numpy()[:, None],
                       pool["lon"].to_numpy()[:, None],
                       sb_lat[None, :], sb_lon[None, :]).min(axis=1)
    pool = pool[dist > EXCLUDE_RADIUS_M]

    n_neg = min(len(pool), NEG_PER_POS * len(positives))
    neg = pool.sample(n=n_neg, random_state=SEED)
    negatives = pd.DataFrame({
        "site_id": "osm_" + neg["osm_id"].astype(str),
        "lat": neg["lat"], "lon": neg["lon"],
        "label": 0, "source": neg["category"], "name": neg["name"],
    })

    sites = pd.concat([positives, negatives], ignore_index=True)
    sites.to_csv(OUT, index=False)

    print(f"Wrote {len(sites)} sites to {OUT}")
    print(f"  positives (Starbucks): {len(positives)}")
    print(f"  negatives (retail):    {len(negatives)}  "
          f"(from a pool of {len(pool)} eligible)")
    print("  negatives by type:")
    for cat, n in negatives["source"].value_counts().items():
        print(f"    {cat:12} {n}")


if __name__ == "__main__":
    main()
