"""Week-1 EDA -- the project plan's end-of-week-1 decision point.

This does NOT estimate a causal effect. It checks whether Atlanta is a viable
testbed: do ZIPs that got a Starbucks appreciate more than ZIPs that did not,
and what does the pre-opening price trend look like (a first peek at the
selection story the project is built to decompose)?

Input:  data/processed/panel.parquet
Output: reports/figures/parallel_trends.png
        reports/figures/event_study_descriptive.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANEL = "data/processed/panel.parquet"
FIGDIR = "reports/figures"


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    panel = pd.read_parquet(PANEL)
    groups = panel.drop_duplicates("zip").set_index("zip")["group"]
    wide = panel.pivot_table(index="zip", columns="year_month",
                             values="zhvi", aggfunc="first").sort_index(axis=1)

    def zips(g):
        return groups[groups == g].index.intersection(wide.index)

    # --- 1. raw appreciation gap, a window every ZIP shares ---
    base, end = "2012-01", "2025-01"
    print(f"Raw appreciation, {base} -> {end}:")
    for g in ["treated", "control"]:
        sub = wide.loc[zips(g), [base, end]].dropna()
        growth = (sub[end] / sub[base] - 1) * 100
        print(f"  {g:8}  n={len(growth):3}  median appreciation = {growth.median():.0f}%")

    # --- 2. parallel-trends plot: mean ZHVI indexed to 2000-01 = 100 ---
    fig, ax = plt.subplots(figsize=(11, 6))
    months = list(wide.columns)
    for g, color in [("treated", "#00704A"), ("control", "#888888")]:
        idx = wide.loc[zips(g)].div(wide.loc[zips(g), "2000-01"], axis=0) * 100
        ax.plot(range(len(months)), idx.mean(axis=0), label=f"{g} ZIPs (n={len(zips(g))})",
                color=color, linewidth=2)
    ticks = [i for i, m in enumerate(months) if m.endswith("-01") and int(m[:4]) % 3 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([months[i][:4] for i in ticks])
    ax.set_ylabel("Mean ZHVI, indexed (2000-01 = 100)")
    ax.set_title("Parallel trends: home-value index, Starbucks vs no-Starbucks ZIPs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/parallel_trends.png", dpi=130)
    plt.close(fig)

    # --- 3. descriptive event study: treated ZIPs, price vs months-since-open ---
    tr = panel[panel["ever_treated"] & panel["months_since_treatment"].notna()].copy()
    tr["log_zhvi"] = np.log(tr["zhvi"])
    base_ev = (tr[tr["months_since_treatment"] == -12]
               .set_index("zip")["log_zhvi"].rename("base_log"))
    tr = tr.join(base_ev, on="zip").dropna(subset=["base_log", "log_zhvi"])
    tr["delta"] = tr["log_zhvi"] - tr["base_log"]
    win = tr[tr["months_since_treatment"].between(-48, 48)]
    curve = win.groupby("months_since_treatment")["delta"].mean()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(curve.index, curve.values * 100, color="#00704A", linewidth=2)
    ax.axvline(0, color="red", linestyle="--", label="Starbucks opens")
    ax.set_xlabel("Months since the first Starbucks opened in the ZIP")
    ax.set_ylabel("Mean ZHVI change vs. 12 months pre-open (%)")
    ax.set_title("Descriptive event study (treated ZIPs only -- not net of secular trend)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/event_study_descriptive.png", dpi=130)
    plt.close(fig)

    # --- treatment-timing spread ---
    fo = panel.drop_duplicates("zip")["first_open_month"].dropna()
    yrs = fo.str.slice(0, 4).astype(int)
    print(f"\nTreated ZIPs with a dated first opening: {len(fo)}")
    print(f"  opening years span {yrs.min()}-{yrs.max()}; "
          f"{(yrs >= 2010).sum()} opened 2010 or later")
    pre = curve.loc[-48:-1].iloc[-1] - curve.loc[-48:-1].iloc[0] if len(curve) else 0
    print(f"\nFigures written to {FIGDIR}/")
    print(f"Pre-opening 4-yr price drift in treated ZIPs: {pre*100:+.0f}% "
          "(if large, selection is visible even before the store opens)")


if __name__ == "__main__":
    main()
