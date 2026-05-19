"""Phase 2, step 2: Build features for site candidates.

For each site in site_candidates.csv:
1. Reverse geocode lat/lon to ZIP code using uszipcode.
2. Join ZIP-level demographics and economic features from the latest year of panel_modeling.parquet.
3. Compute spatial features:
   - count of cafes within 400m (competitor density)
   - count of grocery/bank/gym within 400m (co-tenancy)

Output: data/processed/site_features.csv
"""

import numpy as np
import pandas as pd
from uszipcode import SearchEngine
import warnings
warnings.filterwarnings("ignore")

CANDIDATES = "data/processed/site_candidates.csv"
PANEL = "data/processed/panel_modeling.parquet"
OSM = "data/raw/osm_pois.csv"
OUT = "data/processed/site_features.csv"

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def main():
    print("Loading candidates...")
    sites = pd.read_csv(CANDIDATES)
    
    print("Reverse geocoding to ZIP codes...")
    search = SearchEngine()
    zips = []
    for row in sites.itertuples():
        res = search.by_coordinates(row.lat, row.lon, radius=10, returns=1)
        if res:
            zips.append(res[0].zipcode)
        else:
            zips.append(np.nan)
    sites["zip"] = zips
    
    # Fill any missing zips with a fallback spatial nearest if necessary, but uszipcode is usually good.
    missing_zips = sites["zip"].isna().sum()
    if missing_zips > 0:
        print(f"Warning: {missing_zips} sites could not be mapped to a ZIP code.")
        sites = sites.dropna(subset=["zip"])
        
    print("Loading panel features...")
    panel = pd.read_parquet(PANEL)
    max_year = panel["year"].max()
    print(f"Using panel data from year: {max_year}")
    
    # Get latest cross-section of features by zip
    panel_latest = panel[panel["year"] == max_year].drop_duplicates(subset=["zip"])
    
    features = [
        "zip", "median_income", "population", "median_age", "pct_college", 
        "pct_white", "pct_black", "pct_renter", "median_gross_rent", 
        "median_home_value", "total_jobs"
    ]
    panel_latest = panel_latest[features]
    
    # Merge
    sites = sites.merge(panel_latest, on="zip", how="left")
    
    # Impute missing demographics with medians
    for col in features[1:]:
        sites[col] = sites[col].fillna(sites[col].median())
        
    print("Computing spatial OSM features...")
    osm = pd.read_csv(OSM)
    cafes = osm[osm["category"] == "cafe"]
    retail = osm[osm["category"].isin(["grocery", "bank", "gym", "restaurant"])]
    
    s_lat = sites["lat"].to_numpy()[:, None]
    s_lon = sites["lon"].to_numpy()[:, None]
    
    # Cafe density (within 400m)
    c_lat = cafes["lat"].to_numpy()[None, :]
    c_lon = cafes["lon"].to_numpy()[None, :]
    dist_to_cafes = haversine_m(s_lat, s_lon, c_lat, c_lon)
    sites["cafes_within_400m"] = (dist_to_cafes <= 400).sum(axis=1)
    
    # Retail co-tenancy (within 400m)
    r_lat = retail["lat"].to_numpy()[None, :]
    r_lon = retail["lon"].to_numpy()[None, :]
    dist_to_retail = haversine_m(s_lat, s_lon, r_lat, r_lon)
    sites["retail_within_400m"] = (dist_to_retail <= 400).sum(axis=1)
    
    # Distance to nearest retail (just as another feature)
    if dist_to_retail.shape[1] > 0:
        sites["dist_to_nearest_retail_m"] = dist_to_retail.min(axis=1)
    else:
        sites["dist_to_nearest_retail_m"] = 10000

    print(f"Saving to {OUT}...")
    sites.to_csv(OUT, index=False)
    print(f"Saved {len(sites)} sites with {len(sites.columns)} features.")

if __name__ == "__main__":
    main()
