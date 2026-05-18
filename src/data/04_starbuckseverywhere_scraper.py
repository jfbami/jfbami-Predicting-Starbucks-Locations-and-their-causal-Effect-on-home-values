"""Scrape Starbucks opening dates from starbuckseverywhere.net.

This site is maintained by a visitor who catalogs every Starbucks with its
internal store name and exact opening date -- the same internal names the
store-locator API returns, so stores join cleanly by name.

Covers metro Atlanta via two pages (city + suburbs). Some 1990s-era stores
show "OPENED: ???" (visited before the cataloger tracked dates).

Output: data/raw/sbe_opening_dates.csv
"""

import csv
import re
import requests

PAGES = [
    "https://www.starbuckseverywhere.net/Atlanta.htm",
    "https://www.starbuckseverywhere.net/AtlantaSuburbs.htm",
]
OUT_CSV = "data/raw/sbe_opening_dates.csv"

# "#19908: Northside & 11th - INTERLOCK, Atlanta, Georgia"
HEADER_RE = re.compile(r"#(\d+):\s*(.+?),\s*Georgia", re.IGNORECASE)
OPENED_RE = re.compile(r"OPENED:\s*([^,<\n]+)", re.IGNORECASE)
# "ORIGINAL VISIT: 11/6/1999" -- an upper bound for stores with OPENED: ???
VISIT_RE = re.compile(r"ORIGINAL VISIT:\s*([^,<\n]+)", re.IGNORECASE)
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def strip_tags(html):
    text = re.sub(r"<[^>]+>", "\n", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return text


def normalize_opened(raw):
    """Return YYYY-MM, or '' if the date is unknown/unparseable."""
    raw = raw.strip()
    if not raw or raw.startswith("?"):
        return ""
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", raw)  # M/D/YYYY
    if m:
        return f"{m.group(3)}-{int(m.group(1)):02d}"
    m = re.match(r"([A-Za-z]{3})[a-z]*\s+(\d{4})", raw)  # "Sep 2016"
    if m and m.group(1).lower() in MONTHS:
        return f"{m.group(2)}-{MONTHS[m.group(1).lower()]:02d}"
    m = re.match(r"(\d{4})", raw)  # bare year
    if m:
        return f"{m.group(1)}-01"
    return ""


def parse_page(html):
    """Yield (sbe_number, store_name, city, opened_ym, original_visit_ym)."""
    text = strip_tags(html)
    headers = list(HEADER_RE.finditer(text))
    for i, h in enumerate(headers):
        parts = [p.strip() for p in h.group(2).split(",")]
        name, city = ", ".join(parts[:-1]), parts[-1] if len(parts) > 1 else ""
        # the OPENED line for this store sits before the next header
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        opened = OPENED_RE.search(text, h.end(), end)
        visit = VISIT_RE.search(text, h.end(), end)
        yield (h.group(1), name, city,
               normalize_opened(opened.group(1)) if opened else "",
               normalize_opened(visit.group(1)) if visit else "")


def main():
    rows = []
    for url in PAGES:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        page_rows = list(parse_page(resp.text))
        rows.extend(page_rows)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sbe_number", "name", "city",
                         "opened_year_month", "original_visit_year_month"])
        writer.writerows(rows)

    dated = sum(bool(r[3]) for r in rows)
    print(f"\nWrote {len(rows)} stores to {OUT_CSV}  ({dated} with a known date)")


if __name__ == "__main__":
    main()
