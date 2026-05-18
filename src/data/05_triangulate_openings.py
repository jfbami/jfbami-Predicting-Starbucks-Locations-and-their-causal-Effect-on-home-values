"""Triangulate Starbucks opening dates into the treatment-timing worksheet.

Sources, in priority order:
  1. starbuckseverywhere.net (sbe_opening_dates.csv) -- a visitor-maintained
     catalog with exact opening dates, joined by Starbucks's internal store
     name. Primary source: when it has a date, that date is used.
  2. starbuckseverywhere "original visit" date -- for old stores the catalog
     lists "OPENED: ???"; the visit date is then an upper bound (these are
     1990s/early-2000s stores, i.e. always-treated in any modern panel).
  3. Wayback bound (starbucks_openings.csv) -- cross-check / last-resort bound.

(Atlanta building permits were tried and dropped: the dataset is City-of-
Atlanta-only and matched just 1 of 56 stores.)

Output: data/interim/opening_dates_worksheet.csv
"""

import csv
import os
import re
from collections import Counter

CLIPPED_CSV = "data/raw/starbucks_clipped.csv"
SBE_CSV = "data/raw/sbe_opening_dates.csv"
WAYBACK_CSV = "data/raw/starbucks_openings.csv"
OUT_CSV = "data/interim/opening_dates_worksheet.csv"

TIGHT_CUTOFF = "2022-01"   # Wayback bounds at/after this are trustworthy


def norm(name):
    """Normalize a store name for joining across sources."""
    s = name.lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def synthesize(sbe_opened, sbe_visit, wb_bound):
    """Return (opening_date, confidence, needs_manual_check, notes)."""
    wb_quality = "none" if not wb_bound else (
        "tight" if wb_bound >= TIGHT_CUTOFF else "loose")

    if sbe_opened:
        notes = "starbuckseverywhere.net catalogued opening date"
        # sanity check: an archived store page cannot predate the opening
        if wb_quality == "tight" and wb_bound < sbe_opened:
            return sbe_opened, "medium", "yes", (
                notes + f" (NOTE: Wayback bound {wb_bound} is earlier -- "
                "verify name match)")
        return sbe_opened, "high", "no", notes

    # No catalogued open date. A physical visit date is a hard upper bound
    # and dominates any Wayback bound (which may just be a crawl artifact).
    if sbe_visit:
        return f"<= {sbe_visit}", "medium", "no", (
            f"no catalogued open date; physically visited {sbe_visit}, so "
            "opened on or before that -- long-standing, classify as always-treated")

    # No open date and no visit date -> rely on Wayback.
    if wb_quality == "tight":
        return wb_bound, "medium", "no", (
            f"not yet catalogued by starbuckseverywhere; Wayback first-archive "
            f"{wb_bound} indicates a recent opening")
    if wb_quality == "loose":
        return f"<= {wb_bound}", "low", "yes", (
            f"no catalogued open date; only a loose Wayback bound ({wb_bound}) "
            "-- verify whether mid-panel or always-treated")
    return "", "low", "yes", "no date from any source -- manual check needed"


def main():
    with open(CLIPPED_CSV, newline="", encoding="utf-8") as f:
        stores = [r for r in csv.DictReader(f) if r["ownership"] == "CO"]

    sbe = {}
    with open(SBE_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sbe[norm(r["name"])] = (r["opened_year_month"],
                                    r["original_visit_year_month"])

    wayback = {}
    with open(WAYBACK_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            wayback[r["id"]] = r["opening_year_month_upper_bound"]

    rows = []
    for s in stores:
        sbe_opened, sbe_visit = sbe.get(norm(s["name"]), ("", ""))
        wb_bound = wayback.get(s["id"], "")
        opening, conf, manual, notes = synthesize(sbe_opened, sbe_visit, wb_bound)
        rows.append({
            "id": s["id"],
            "store_number": s["store_number"],
            "name": s["name"],
            "tract_geoid": s["tract_geoid"],
            "address": s["address"],
            "sbe_opened": sbe_opened,
            "sbe_original_visit": sbe_visit,
            "wayback_bound": wb_bound,
            "opening_date": opening,
            "confidence": conf,
            "needs_manual_check": manual,
            "notes": notes,
        })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    conf = Counter(r["confidence"] for r in rows)
    manual = sum(r["needs_manual_check"] == "yes" for r in rows)
    exact = sum(bool(r["sbe_opened"]) for r in rows)
    print(f"Wrote {len(rows)} stores to {OUT_CSV}")
    print(f"  exact opening date (starbuckseverywhere): {exact}")
    print(f"  confidence: high={conf['high']} medium={conf['medium']} low={conf['low']}")
    print(f"  still need a manual check: {manual}")


if __name__ == "__main__":
    main()
