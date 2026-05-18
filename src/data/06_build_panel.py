"""Build the core ZIP x month panel: ZHVI prices + Starbucks treatment timing.

One row per (zip, year_month) for every Atlanta-metro ZIP. This is the spine of
the analysis; ACS / LODES / OSM covariates get joined on in later steps.

treated = 1 for a ZIP-month at or after the first company-operated Starbucks
opened in that ZIP. ZIPs with no Starbucks at all are clean controls; ZIPs that
have a Starbucks we did not study (licensed, or outside Fulton/DeKalb) are
marked separately so they can be excluded from the control group.

Inputs:
  data/raw/zhvi_zip.csv                     Zillow ZHVI, wide (ZIP x month)
  data/interim/opening_dates_worksheet.csv  per-store opening dates
  data/raw/starbucks_clipped.csv            per-store ZIP (postal_code)
  data/raw/starbucks_current.csv            all 271 metro stores (control hygiene)

Output:
  data/processed/panel.parquet
"""

import os
import re
import pandas as pd

ZHVI_CSV = "data/raw/zhvi_zip.csv"
WORKSHEET = "data/interim/opening_dates_worksheet.csv"
CLIPPED = "data/raw/starbucks_clipped.csv"
CURRENT = "data/raw/starbucks_current.csv"
OUT = "data/processed/panel.parquet"

# Treated ZIPs are intown Fulton/DeKalb. Controls must be comparable urban
# units -- the 5-county Atlanta core -- not rural exurbs 40-60 mi out, which
# appreciate fast off a low base and break the parallel-trends assumption.
CORE_COUNTIES = {"Fulton County", "DeKalb County", "Cobb County",
                 "Gwinnett County", "Clayton County"}


def ym_ord(ym):
    """'YYYY-MM' -> integer month ordinal (for month arithmetic)."""
    return int(ym[:4]) * 12 + (int(ym[5:7]) - 1)


def parse_opening(s):
    """worksheet opening_date -> 'YYYY-MM' or None.

    Handles exact 'YYYY-MM' and bound '<= YYYY-MM' alike (the bound month is
    a fine treatment date for old always-treated stores).
    """
    m = re.search(r"(\d{4})-(\d{2})", str(s or ""))
    return f"{m.group(1)}-{m.group(2)}" if m else None


def main():
    # --- ZHVI: wide -> long, Atlanta metro only ---
    zhvi = pd.read_csv(ZHVI_CSV)
    zhvi = zhvi[zhvi["Metro"].fillna("").str.contains("Atlanta")].copy()
    date_cols = [c for c in zhvi.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", c)]
    long = zhvi.melt(id_vars=["RegionName", "CountyName"], value_vars=date_cols,
                     var_name="date", value_name="zhvi")
    long["zip"] = long["RegionName"].astype(str).str.zfill(5)
    long["year_month"] = long["date"].str.slice(0, 7)
    long = long.rename(columns={"CountyName": "county"})[
        ["zip", "year_month", "zhvi", "county"]]

    # --- treatment timing per ZIP ---
    ws = pd.read_csv(WORKSHEET, dtype=str)
    clip = pd.read_csv(CLIPPED, dtype=str)[["id", "postal_code"]]
    ws = ws.merge(clip, on="id", how="left")
    ws["open_ym"] = ws["opening_date"].map(parse_opening)
    ws["zip"] = ws["postal_code"].str.zfill(5)
    first_open = (ws.dropna(subset=["open_ym"]).groupby("zip")["open_ym"].min()
                  .rename("first_open_month").reset_index())
    treated_zips = set(ws["zip"].dropna())

    # --- metro ZIPs with ANY Starbucks (so controls can exclude them) ---
    cur = pd.read_csv(CURRENT, dtype=str)
    starbucks_zips = set(cur["postal_code"].astype(str).str.zfill(5))

    # --- assemble panel ---
    panel = long.merge(first_open, on="zip", how="left")
    panel["ever_treated"] = panel["zip"].isin(treated_zips)
    panel["ord"] = panel["year_month"].map(ym_ord)
    panel["first_open_ord"] = panel["first_open_month"].map(
        lambda x: ym_ord(x) if isinstance(x, str) else None)
    panel["months_since_treatment"] = panel["ord"] - panel["first_open_ord"]
    panel["treated"] = (panel["ever_treated"] &
                        (panel["months_since_treatment"] >= 0)).astype(int)

    def label(r):
        if r["ever_treated"]:
            return "treated"
        if r["zip"] in starbucks_zips:
            return "has_starbucks_unstudied"
        return "control" if r["county"] in CORE_COUNTIES else "out_of_core"
    panel["group"] = panel.apply(label, axis=1)

    panel = panel[["zip", "year_month", "zhvi", "treated",
                   "months_since_treatment", "ever_treated",
                   "first_open_month", "group", "county"]]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    panel.to_parquet(OUT, index=False)

    n_zip = panel["zip"].nunique()
    g = panel.drop_duplicates("zip")["group"].value_counts()
    print(f"Wrote {len(panel):,} rows ({n_zip} ZIPs x "
          f"{panel['year_month'].nunique()} months) to {OUT}")
    print(f"  treated ZIPs:  {g.get('treated', 0)}")
    print(f"  control ZIPs (core county, no Starbucks):  {g.get('control', 0)}")
    print(f"  excluded -- has unstudied Starbucks:  {g.get('has_starbucks_unstudied', 0)}")
    print(f"  excluded -- rural exurb (out of core):  {g.get('out_of_core', 0)}")
    print(f"  ZHVI missing values: {panel['zhvi'].isna().sum():,} "
          f"({panel['zhvi'].isna().mean():.1%})")


if __name__ == "__main__":
    main()
