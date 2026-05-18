"""Join ACS + LODES covariates onto the panel -> final modeling table.

Covariates are annual; the panel is monthly. We join by ZIP and year. Where a
panel year falls outside a source's range, we clamp to the nearest available
year (covariates move slowly, so 2000-2012 uses the 2013 ACS, etc.).

OSM is NOT joined here. OSM POIs are points, not ZIP-keyed -- they feed the
site-selection model directly as point-radius features. Aggregating them to
per-ZIP counts needs ZCTA polygons and is deferred until a model needs it.

Input:  data/processed/panel.parquet  +  data/raw/acs_zcta.csv, lodes_zcta.csv
Output: data/processed/panel_modeling.parquet
"""

import pandas as pd

PANEL = "data/processed/panel.parquet"
ACS = "data/raw/acs_zcta.csv"
LODES = "data/raw/lodes_zcta.csv"
OUT = "data/processed/panel_modeling.parquet"


def main():
    panel = pd.read_parquet(PANEL)
    panel["year"] = panel["year_month"].str.slice(0, 4).astype(int)

    acs = pd.read_csv(ACS, dtype={"zcta": str})
    acs["zcta"] = acs["zcta"].str.zfill(5)
    lodes = pd.read_csv(LODES, dtype={"zcta": str})
    lodes["zcta"] = lodes["zcta"].str.zfill(5)

    # clamp panel year into each source's available range (nearest-year join)
    panel["acs_year"] = panel["year"].clip(acs["year"].min(), acs["year"].max())
    panel["lodes_year"] = panel["year"].clip(lodes["year"].min(), lodes["year"].max())

    panel = panel.merge(
        acs.rename(columns={"zcta": "zip", "year": "acs_year"}),
        on=["zip", "acs_year"], how="left")
    panel = panel.merge(
        lodes.rename(columns={"zcta": "zip", "year": "lodes_year"}),
        on=["zip", "lodes_year"], how="left")
    panel = panel.drop(columns=["acs_year", "lodes_year"])

    panel.to_parquet(OUT, index=False)

    cov = ["median_income", "pct_college", "pct_renter", "total_jobs"]
    print(f"Wrote {len(panel):,} rows x {len(panel.columns)} cols to {OUT}")
    print("Covariate coverage (non-null share of panel rows):")
    for c in cov:
        print(f"  {c:16} {panel[c].notna().mean():.1%}")
    print("\nColumns:", list(panel.columns))


if __name__ == "__main__":
    main()
