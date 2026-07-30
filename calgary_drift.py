#!/usr/bin/env python3
"""calgary_drift.py — per-year decomposition of the Calgary difference-of-differences.

The six-year Calgary DoD (CYYC human minus CYBW auto, night minus day) does not hold
still. This reports it year by year and shows what moves. The widening splits roughly
evenly between the two stations; the biggest single piece is CYBW's daytime clear-sky
fraction rising off an unusually cloudy 2020. CYYC carries 0% night AUTO in every year,
so it is not an artifact of the human record. Backs the drift paragraph in the paper
(automated-partner section).

Estimator is the study's: a night is the local-noon-to-noon unit
(observable_nights.night_key), dark = solar altitude <= -12 deg, day = >= 6 deg,
usable = okta <= 2, each unit weighted equally (mean of per-unit means). Reads only
the shipped METAR cache; writes results/calgary_drift.csv.
"""
import argparse
import csv
import glob
import math
from collections import defaultdict
from statistics import mean
import datetime as dt

import ephem

from fetch_metar import parse_metar_cloud
from observable_nights import night_key, LOCAL_TZ
from sites import SITES

SITE = SITES["calgary"]


def sun_alt(ts_utc):
    o = ephem.Observer()
    o.lat, o.lon, o.elevation = str(SITE.lat), str(SITE.lon), SITE.elevation_m
    o.date = ts_utc.strftime("%Y/%m/%d %H:%M:%S")
    s = ephem.Sun()
    s.compute(o)
    return math.degrees(s.alt)


def load(station):
    """Return night/day per-unit usable lists and per-year AUTO counts."""
    night = defaultdict(lambda: defaultdict(list))   # year -> night_date -> [usable]
    day = defaultdict(lambda: defaultdict(list))     # year -> date -> [usable]
    auto = defaultdict(lambda: [0, 0])               # year -> [n_auto_dark, n_dark]
    for path in sorted(glob.glob(f"data/metar_cache/{station}_????-01-01_????-12-31.csv")):
        for r in csv.DictReader(open(path)):
            raw = (r.get("metar") or "").strip()
            v = (r.get("valid") or "").strip()
            if not raw or not v:
                continue
            ts = dt.datetime.strptime(v, "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc)
            a = sun_alt(ts)
            usable = parse_metar_cloud(raw)[0] <= 2
            if a <= -12.0:
                nk = night_key(ts)
                night[nk.year][nk].append(usable)
                auto[nk.year][1] += 1
                auto[nk.year][0] += ("AUTO" in raw.split())
            elif a >= 6.0:
                d = ts.astimezone(LOCAL_TZ).date()
                day[d.year][d].append(usable)
    return night, day, auto


def yr_pct(bucket, year, min_obs=1):
    units = [mean(u) for u in bucket[year].values() if len(u) >= min_obs]
    return (100 * mean(units) if units else float("nan")), len(units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/calgary_drift.csv")
    ap.add_argument("--years", type=int, nargs=2, default=(2020, 2025))
    args = ap.parse_args()

    cyn, cyd, cya = load("CYYC")
    bwn, bwd, _ = load("CYBW")
    years = range(args.years[0], args.years[1] + 1)

    rows = []
    print(f"{'yr':4} {'CYYCn':6} {'CYYCd':6} {'CYYCΔ':6} {'nAUTO%':7} "
          f"{'CYBWn':6} {'CYBWd':6} {'CYBWΔ':6} {'DoD':6}")
    vals = {}
    for y in years:
        cn, ncu = yr_pct(cyn, y, 2)     # >= 2 obs per unit, nights and days alike,
        cd, ncd = yr_pct(cyd, y, 2)     # matching human_deltas/dynamic_tables MIN_OBS
        bn, nbu = yr_pct(bwn, y, 2)
        bd, nbd = yr_pct(bwd, y, 2)
        na = 100 * cya[y][0] / cya[y][1] if cya[y][1] else float("nan")
        dod = (cn - cd) - (bn - bd)
        vals[y] = (cn, cd, bn, bd, dod)
        rows.append((y, round(cn, 1), round(cd, 1), round(cn - cd, 1), round(na, 2),
                     round(bn, 1), round(bd, 1), round(bn - bd, 1), round(dod, 1),
                     ncu, ncd, nbu, nbd))
        print(f"{y:4} {cn:6.1f} {cd:6.1f} {cn-cd:+6.1f} {na:6.2f}% "
              f"{bn:6.1f} {bd:6.1f} {bn-bd:+6.1f} {dod:+6.1f}")

    a, b = vals[years[0]], vals[years[-1]]
    contrib = {"CYYC_night": b[0]-a[0], "CYYC_day": -(b[1]-a[1]),
               "CYBW_night": -(b[2]-a[2]), "CYBW_day": b[3]-a[3]}
    print(f"\nDoD {years[0]}={a[4]:+.1f} -> {years[-1]}={b[4]:+.1f} "
          f"(total {b[4]-a[4]:+.1f} pp). Contributions to the rise:")
    for k, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:11} {v:+.1f}")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "cyyc_night_pct", "cyyc_day_pct", "cyyc_delta_pp",
                    "cyyc_night_auto_pct", "cybw_night_pct", "cybw_day_pct",
                    "cybw_delta_pp", "dod_pp",
                    "n_cyyc_night_units", "n_cyyc_day_units",
                    "n_cybw_night_units", "n_cybw_day_units"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
