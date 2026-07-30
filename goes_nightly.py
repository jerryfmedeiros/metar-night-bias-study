"""
goes_nightly.py — Night-unit GOES deltas from a logged draw file.

Consumes goes_draws_<site>_<family>.csv (from climatology_goes.py --log-draws) and
computes the GOES night-minus-day clear-fraction delta with the SAME uncertainty
treatment as the METAR record: draws are aggregated into night units (local-noon to
local-noon, site timezone) and day units (local calendar date), each unit's clear
fraction is the mean of its draws, and a block bootstrap (B=2000, seeded) over units
gives percentile CIs. This replaces the draw-level Wilson/normal CIs, which assumed
independent draws and understated uncertainty where several draws share one night.

NOTE ON WINDOWS: the "calendar_JunNov" window here is a crude month-based stand-in for bare
ground, kept only as the diagnostic contrast described in the paper. It is NOT the snow-free
filter behind the published tables -- that is the daily ERA5-Land snow-depth filter in
dynamic_tables.py, which is what the main results table reports. The two give different numbers; use
dynamic_tables.py for anything quoted as "snow-free".

Reports full-year and calendar_JunNov windows for three footprints:
  nearest — nearest-pixel BCM class
  box3    — 3x3 box mean BCM <= 0.5 counts clear
  box5    — 5x5 box mean BCM <= 0.5 counts clear

Usage:
  python3 goes_nightly.py [--glob 'data/goes_draws/goes_draws_*.csv'] [--boot 2000] [--seed 42]
                                 [--csv results/goes_nightly.csv]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob as globmod
import sys
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

try:
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

CALENDAR_BARE_MONTHS = set(range(6, 12))   # crude month proxy, not the published snow filter
BOX_CLEAR_THRESH = 0.5
FOOTPRINTS = ("nearest", "box3", "box5")
WINDOWS = (("full", None), ("calendar_JunNov", CALENDAR_BARE_MONTHS),
           ("winter", {12, 1, 2}), ("spring", {3, 4, 5}),
           ("summer", {6, 7, 8}), ("autumn", {9, 10, 11}))


def unit_key(ts: dt.datetime, regime: str, tz: ZoneInfo) -> dt.date:
    loc = ts.astimezone(tz)
    return (loc - dt.timedelta(hours=12)).date() if regime == "NIGHT" else loc.date()


def boot_delta(nf, df, B, seed):
    rng = np.random.default_rng(seed)
    n, d = np.asarray(nf, float), np.asarray(df, float)
    bn = n[rng.integers(0, len(n), size=(B, len(n)))].mean(axis=1)
    bd = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    dist = bn - bd
    return ((n.mean() - d.mean()) * 100,
            np.percentile(dist, 2.5) * 100, np.percentile(dist, 97.5) * 100)


def process(path, B, seed):
    import os
    parts = os.path.basename(path).replace(".csv", "").split("_")  # goes, draws, <slug...>, <family>
    family = parts[-1]
    slug = "_".join(parts[2:-1])
    site = SITES[slug]
    tz = ZoneInfo(site.iana_tz)

    # units[station][footprint][regime][unit_date] -> [0/1 clear indicators]
    units = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    known = set(SITES[slug].coords) if slug in SITES else None
    with open(path) as f:
        for row in csv.DictReader(f):
            st = row["station"]
            if known is not None and st not in known:
                continue   # e.g. CYKZ: logged during sampling, later dropped from the registry
            ts = dt.datetime.fromisoformat(row["ts_utc"])
            uk = unit_key(ts, row["regime"], tz)
            fp_clear = {"nearest": int(row["nearest_clear"])}
            for fp, col in (("box3", "box3_mean_bcm"), ("box5", "box5_mean_bcm")):
                if row[col] != "":
                    fp_clear[fp] = int(float(row[col]) <= BOX_CLEAR_THRESH)
            for fp, clear in fp_clear.items():
                units[st][fp][row["regime"]][uk].append(clear)

    rows = []
    for st, by_fp in sorted(units.items()):
        for fp in FOOTPRINTS:
            if fp not in by_fp:
                continue
            for window, keep in WINDOWS:
                sel = {}
                for regime in ("NIGHT", "DAY"):
                    sel[regime] = [np.mean(v) for uk, v in sorted(by_fp[fp][regime].items())
                                   if keep is None or uk.month in keep]
                if not sel["NIGHT"] or not sel["DAY"]:
                    continue
                dpt, dlo, dhi = boot_delta(sel["NIGHT"], sel["DAY"], B, seed + 100)
                rows.append(dict(
                    site=slug, family=family, station=st, footprint=fp, window=window,
                    n_nights=len(sel["NIGHT"]), n_days=len(sel["DAY"]),
                    n_draws_night=sum(len(v) for v in by_fp[fp]["NIGHT"].values()),
                    n_draws_day=sum(len(v) for v in by_fp[fp]["DAY"].values()),
                    night_pct=round(float(np.mean(sel["NIGHT"])) * 100, 1),
                    day_pct=round(float(np.mean(sel["DAY"])) * 100, 1),
                    delta_pp=round(dpt, 1), ci_lo=round(dlo, 1), ci_hi=round(dhi, 1),
                ))
                print(f"  {slug}/{family} {st} {fp:7s} {window:8s}: "
                      f"night {rows[-1]['night_pct']:5.1f}% day {rows[-1]['day_pct']:5.1f}%  "
                      f"delta {dpt:+5.1f} pp [{dlo:+5.1f},{dhi:+5.1f}]  "
                      f"({rows[-1]['n_nights']} nights/{rows[-1]['n_days']} days from "
                      f"{rows[-1]['n_draws_night']}+{rows[-1]['n_draws_day']} draws)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="data/goes_draws/goes_draws_*.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="results/goes_nightly.csv")
    args = ap.parse_args()

    all_rows = []
    for path in sorted(globmod.glob(args.glob)):
        print(f"\n=== {path} ===")
        all_rows.extend(process(path, args.boot, args.seed))
    if not all_rows:
        sys.exit(f"No files matched {args.glob}")
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
