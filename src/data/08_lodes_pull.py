"""Pull daytime worker population per ZIP from LEHD LODES (keyless).

LODES WAC (Workplace Area Characteristics) counts jobs by the census block of
the *workplace* -- a good proxy for daytime population / foot traffic, which
the site-selection model needs (Starbucks favors high-daytime-population spots).

WAC is block-level; the LODES geographic crosswalk maps each block to a ZCTA
(~ZIP), so we sum jobs up to ZIP. All files are public .csv.gz downloads.

Output: data/raw/lodes_zcta.csv  (one row per zcta x year)
"""

import csv
import gzip
import io
import requests

BASE = "https://lehd.ces.census.gov/data/lodes/LODES8/ga"
XWALK_URL = f"{BASE}/ga_xwalk.csv.gz"
WAC_URL = BASE + "/wac/ga_wac_S000_JT00_{year}.csv.gz"
YEARS = range(2010, 2023)  # LODES8 WAC availability
ZHVI_CSV = "data/raw/zhvi_zip.csv"
OUT_CSV = "data/raw/lodes_zcta.csv"


def metro_zips():
    zips = set()
    with open(ZHVI_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if "Atlanta" in (r.get("Metro") or ""):
                zips.add(str(r["RegionName"]).zfill(5))
    return zips


def fetch_csv_gz(url):
    """Download a .csv.gz and return a csv.DictReader over it."""
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    text = gzip.decompress(resp.content).decode("utf-8")
    return csv.DictReader(io.StringIO(text))


def main():
    keep = metro_zips()
    block_to_zcta = {}
    for row in fetch_csv_gz(XWALK_URL):
        z = str(row["zcta"]).zfill(5)
        if z in keep:
            block_to_zcta[row["tabblk2020"]] = z

    rows = []
    for year in YEARS:
        try:
            reader = fetch_csv_gz(WAC_URL.format(year=year))
        except requests.HTTPError:
            print(f"  {year}: not available -- skipped")
            continue
        jobs = {}  # zcta -> total jobs
        for r in reader:
            z = block_to_zcta.get(r["w_geocode"])
            if z:
                jobs[z] = jobs.get(z, 0) + int(r["C000"])
        for z, n in jobs.items():
            rows.append({"zcta": z, "year": year, "total_jobs": n})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["zcta", "year", "total_jobs"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
