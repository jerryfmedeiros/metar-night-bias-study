#!/usr/bin/env python3
"""snow_insitu_validation.py — is the ERA5-Land snow screen a fair stand-in for airport records?

Two questions behind the snow-filter justification (metar-night-bias, sec. snow filter):

  1. Could we have screened on the airports' own snow-on-ground reports instead of ERA5-Land?
     Not across the network: over 2020-2025 the in-situ record is badly incomplete at most
     sites and absent at some (station automation dropped the manual snow-depth observation).
     Reported here as the fraction of days, and of snow-season days (Nov-Apr), carrying a value.

  2. Where the in-situ record does exist, does ERA5-Land agree with it? On the shared days we
     cross-tabulate snow-free (<= 1 cm) vs snow (> 1 cm). The disagreement is almost entirely
     ERA5 excluding days the airport pad reports bare (conservative); ERA5 admitting a snow day
     as bare -- the leak that a coarse grid might invite -- is the "leak" column, and it is small.

Reads data/snow_insitu/<slug>.csv (fetch_snow_insitu.py) and the ERA5-Land daily depths in
data/snow_cache/. Writes results/snow_insitu_validation.csv.
"""
import argparse
import csv
import datetime as dt
import json
import os

from sites import SITES

SNOW_CM = 1.0                      # snow-free threshold, matching the paper's 1 cm
COLD_MONTHS = {11, 12, 1, 2, 3, 4}
N_DAYS = {2020: 366, 2021: 365, 2022: 365, 2023: 365, 2024: 366, 2025: 365}


def load_insitu(slug):
    """Date -> snow-on-ground cm for reported days; None if the site has no ECCC station."""
    p = f"data/snow_insitu/{slug}.csv"
    if not os.path.exists(p):
        return None
    return {r["date"]: float(r["snow_on_grnd_cm"]) for r in csv.DictReader(open(p))}


def load_era5_cm(slug):
    j = json.load(open(f"data/snow_cache/snow_{slug}.json"))
    return {t: v * 100 for t, v in zip(j["daily"]["time"], j["daily"]["snow_depth_max"])
            if v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/snow_insitu_validation.csv")
    ap.add_argument("--years", type=int, nargs=2, default=(2020, 2025))
    args = ap.parse_args()
    d0, d1 = dt.date(args.years[0], 1, 1), dt.date(args.years[1], 12, 31)
    span = [d0 + dt.timedelta(i) for i in range((d1 - d0).days + 1)]
    total_days = len(span)
    cold_denom = sum(1 for d in span if d.month in COLD_MONTHS)  # calendar snow-season days

    rows = []
    print(f"{'site':10} {'valid% all':11} {'valid% Nov-Apr':15} {'agree%':8} {'leak%':7} {'overexcl%':9}")
    for slug in SITES:
        ins = load_insitu(slug)
        if ins is None:                # no ECCC airport station for this site
            continue
        era = load_era5_cm(slug)
        n_valid = len(ins)
        n_cold = sum(1 for d in ins if int(d[5:7]) in COLD_MONTHS)
        agree = leak = overexcl = matched = 0
        for d, sv in ins.items():
            if d not in era:
                continue
            matched += 1
            era_free, ins_free = era[d] <= SNOW_CM, sv <= SNOW_CM
            if era_free == ins_free:
                agree += 1
            elif era_free and not ins_free:
                leak += 1            # ERA5 says bare, airport says snow (the risky miss)
            else:
                overexcl += 1        # ERA5 says snow, airport pad bare (conservative)
        vp = round(100 * n_valid / total_days, 1)
        vpc = round(100 * n_cold / cold_denom, 1)
        if matched >= 30:              # enough shared days to cross-validate
            ag = round(100 * agree / matched, 1)
            lk = round(100 * leak / matched, 1)
            ox = round(100 * overexcl / matched, 1)
        else:
            ag = lk = ox = ""          # in-situ too sparse to validate (e.g. Vancouver)
        rows.append((slug, vp, vpc, matched, ag, lk, ox))
        astr = f"{ag:5.1f}%" if ag != "" else "  --  "
        print(f"{slug:10} {vp:5.1f}%      {vpc:5.1f}%          {astr}  "
              f"{f'{lk:.1f}%' if lk != '' else '--':7} {f'{ox:.1f}%' if ox != '' else '--':9}")

    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "insitu_valid_pct_all", "insitu_valid_pct_snow_season",
                    "n_matched_days", "agree_pct", "leak_pct_era5free_insitu_snow",
                    "overexclude_pct_era5snow_insitu_bare"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
