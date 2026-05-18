"""Pull ACS 5-year demographics at ZCTA (~ZIP) level for metro Atlanta.

The Census API now requires a key (free, instant, no billing -- it used to be
keyless under 500 requests/day, but that changed). Get one at
    https://api.census.gov/data/key_signup.html
then create a file named .env in the project root containing:
    CENSUS_API_KEY=your_key_here
(.env is gitignored.)

Pulls one row per (zcta, year). Years are 5-year estimates; covariates move
slowly, so an annual series is fine to join onto the monthly panel by year.

Output: data/raw/acs_zcta.csv
"""

import csv
import os
import requests

OUT_CSV = "data/raw/acs_zcta.csv"
ZHVI_CSV = "data/raw/zhvi_zip.csv"
YEARS = range(2013, 2024)  # ACS5 2013-2023

# Census variable code -> output column name
VARS = {
    "B19013_001E": "median_income",
    "B01003_001E": "population",
    "B01002_001E": "median_age",
    "B15003_001E": "pop_25plus",
    "B15003_022E": "edu_bachelor",
    "B15003_023E": "edu_master",
    "B15003_024E": "edu_professional",
    "B15003_025E": "edu_doctorate",
    "B02001_001E": "race_total",
    "B02001_002E": "race_white",
    "B02001_003E": "race_black",
    "B25003_001E": "tenure_total",
    "B25003_003E": "tenure_renter",
    "B25064_001E": "median_gross_rent",
    "B25077_001E": "median_home_value",
}


def load_key():
    """Read CENSUS_API_KEY from the environment or a .env file."""
    key = os.environ.get("CENSUS_API_KEY")
    if key:
        return key.strip()
    if os.path.exists(".env"):
        for line in open(".env", encoding="utf-8"):
            if line.strip().startswith("CENSUS_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        "No Census API key found. Get a free key at "
        "https://api.census.gov/data/key_signup.html and put "
        "CENSUS_API_KEY=... in a .env file in the project root.")


def metro_zips():
    """The ZIP set to keep -- Atlanta-metro ZIPs from the ZHVI file."""
    zips = set()
    with open(ZHVI_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if "Atlanta" in (r.get("Metro") or ""):
                zips.add(str(r["RegionName"]).zfill(5))
    return zips


def fetch_year(year, key):
    """Return list of dict rows for all ZCTAs in `year`, or [] if unavailable."""
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME," + ",".join(VARS),
        "for": "zip code tabulation area:*",
        "key": key,
    }
    resp = requests.get(url, params=params, timeout=120)
    if resp.status_code != 200:
        print(f"  {year}: unavailable (HTTP {resp.status_code}) -- skipped")
        return []
    table = resp.json()
    header = table[0]
    return [dict(zip(header, row)) for row in table[1:]]


def derive(rec):
    """Add interpretable shares; return None for unusable values."""
    def num(v):
        try:
            x = float(v)
            return x if x >= 0 else None  # Census uses negatives as null flags
        except (TypeError, ValueError):
            return None

    out = {VARS[k]: num(rec.get(k)) for k in VARS}
    edu = sum(out[k] or 0 for k in
              ["edu_bachelor", "edu_master", "edu_professional", "edu_doctorate"])
    if out["pop_25plus"]:
        out["pct_college"] = round(100 * edu / out["pop_25plus"], 2)
    if out["race_total"]:
        out["pct_white"] = round(100 * (out["race_white"] or 0) / out["race_total"], 2)
        out["pct_black"] = round(100 * (out["race_black"] or 0) / out["race_total"], 2)
    if out["tenure_total"]:
        out["pct_renter"] = round(100 * (out["tenure_renter"] or 0) / out["tenure_total"], 2)
    return out


def main():
    key = load_key()
    keep = metro_zips()

    fieldnames = (["zcta", "year", "median_income", "population", "median_age",
                   "pct_college", "pct_white", "pct_black", "pct_renter",
                   "median_gross_rent", "median_home_value"])
    rows = []
    for year in YEARS:
        records = fetch_year(year, key)
        for rec in records:
            z = str(rec.get("zip code tabulation area", "")).zfill(5)
            if z not in keep:
                continue
            d = derive(rec)
            rows.append({"zcta": z, "year": year,
                         **{k: d.get(k) for k in fieldnames[2:]}})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
