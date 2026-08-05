"""
dynamic_tables.py — snow-free comparison tables under the dynamic (snow-depth) filter.

Computes, for each site and each snow-depth threshold, with night/day units and the
study's block bootstrap (B=2000, seed 42):

  - human night-day usable delta (okta<=2) at the human station and, where a true
    automated partner exists, at the partner + the difference-of-differences
  - GOES night-day clear delta from the logged draws (nearest, 3x3, 5x5 footprints)
  - the human-GOES gap, paired night/day-unit block bootstrap on matched nights (nearest pixel)

A unit (local-noon-to-noon night, or local calendar day) is snow-free if its key date
is not flagged in data/snow_cache/snow_<site>.json (daily max ERA5-Land depth > threshold).

Usage:
  python3 dynamic_tables.py [--thresholds ...] [--csv results/dynamic_tables.csv] [--era ...]

The GOES draws span two cloud-mask algorithms (baseline threshold mask before 2021-11-29
1900 UTC, naive Bayesian Enterprise Cloud Mask after). --era selects one; the default,
enterprise, is the paper's primary GOES reference. --era all pools both and is reported
alongside it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

try:
    from fetch_metar import fetch_metar_csv
    from metar_climatology import calculate_sun_alt
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

YEARS = (2020, 2025)
MIN_OBS = 2
B, SEED = 2000, 42
# Genuine automated ceilometer partners (100% night AUTO, report sky condition): the METAR
# difference-of-differences is computed for all four.
AUTO_OK = {"CYBW", "CYHU", "CYTZ", "CZVL"}
# Partners whose GOES pixel was also logged in the per-draw files, so the satellite
# cross-check at the partner location can be built. CYTZ/CZVL pixels were backfilled at
# the published draw timestamps (partner_pixel_backfill.py) and merged into the east logs.
AUTO_GOES = {"CYBW", "CYHU", "CYTZ", "CZVL"}


def snow_depths(slug):
    d = json.loads((Path("data/snow_cache") / f"snow_{slug}.json").read_text())["daily"]
    return {dt.date.fromisoformat(t): (v if v is not None else 0.0)
            for t, v in zip(d["time"], d["snow_depth_max"])}


def boot_delta(nf, df, seed):
    rng = np.random.default_rng(seed)
    n, d = np.asarray(nf, float), np.asarray(df, float)
    bn = n[rng.integers(0, len(n), size=(B, len(n)))].mean(axis=1)
    bd = d[rng.integers(0, len(d), size=(B, len(d)))].mean(axis=1)
    return ((n.mean() - d.mean()) * 100, (bn - bd))


def metar_units(site, station):
    tz = ZoneInfo(site.iana_tz)
    nights, days = defaultdict(list), defaultdict(list)
    for year in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(station, dt.date(year, 1, 1), dt.date(year, 12, 31)):
            alt = calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m)
            u = o.coverage_okta <= 2
            if alt <= -12.0:
                nights[(o.timestamp.astimezone(tz) - dt.timedelta(hours=12)).date()].append(u)
            elif alt >= 6.0:
                days[o.timestamp.astimezone(tz).date()].append(u)
    f = lambda dct: {d: float(np.mean(v)) for d, v in dct.items() if len(v) >= MIN_OBS}
    return f(nights), f(days)


# The ABI cloud mask product switched algorithms mid-record: the baseline threshold mask
# (Heidinger and Straka 2012) ran until 2021-11-29 1900 UTC, the naive Bayesian Enterprise
# Cloud Mask (Heidinger and Botambekov 2020) from then on. BCM keeps its meaning across the
# switch, so the draws are directly comparable, but the two algorithms do not report the
# same night-day contrast. ERA_CUT splits the record for the robustness check.
ERA_CUT = dt.datetime(2021, 11, 29, 19, 0, tzinfo=dt.timezone.utc)
ERAS = ("all", "baseline", "enterprise")
SEASONS = ("WINTER", "SPRING", "SUMMER", "AUTUMN")
SEASON_MONTHS = {"WINTER": {12, 1, 2}, "SPRING": {3, 4, 5},
                 "SUMMER": {6, 7, 8}, "AUTUMN": {9, 10, 11}}
# GOES-West platform handover, GOES-17 -> GOES-18.
HANDOVER_1718 = dt.datetime(2023, 1, 4, tzinfo=dt.timezone.utc)


def in_era(ts_utc, era):
    if era == "all":
        return True
    return (ts_utc < ERA_CUT) if era == "baseline" else (ts_utc >= ERA_CUT)


def goes_units(site, family, station=None, era="all", season=None, since=None, until=None):
    """Night/day GOES clear fractions at one station's pixel (default: the human station).

    Passing the automated partner's ICAO samples the same draws at the partner pixel, which
    is what shows that the satellite sees the same diurnal sky change at both airports.

    era restricts the draws to one cloud-mask algorithm; "all" is the pooled record and
    reproduces the pooled tables exactly. season, if given, keeps only that meteorological
    season's draws (the season each draw was logged under).
    """
    station = station or site.human
    tz = ZoneInfo(site.iana_tz)
    cols = {"nearest": "nearest_clear", "box3": "box3_mean_bcm", "box5": "box5_mean_bcm"}
    out = {fp: {"NIGHT": defaultdict(list), "DAY": defaultdict(list)} for fp in cols}
    with open(f"data/goes_draws/goes_draws_{site.slug}_{family}.csv") as fh:
        for row in csv.DictReader(fh):
            if row["station"] != station:
                continue
            if season is not None and row["season"] != season:
                continue
            ts_utc = dt.datetime.fromisoformat(row["ts_utc"])
            if not in_era(ts_utc, era):
                continue
            if (since and ts_utc < since) or (until and ts_utc >= until):
                continue
            ts = ts_utc.astimezone(tz)
            key = (ts - dt.timedelta(hours=12)).date() if row["regime"] == "NIGHT" else ts.date()
            out["nearest"][row["regime"]][key].append(int(row["nearest_clear"]))
            for fp in ("box3", "box5"):
                if row[cols[fp]] != "":
                    out[fp][row["regime"]][key].append(int(float(row[cols[fp]]) <= 0.5))
    return {fp: {rg: {d: float(np.mean(v)) for d, v in dd.items()}
                 for rg, dd in per.items()} for fp, per in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.005, 0.01, 0.02, 0.05])
    ap.add_argument("--csv", default="results/dynamic_tables.csv")
    ap.add_argument("--era", choices=ERAS, default="enterprise",
                    help="which cloud-mask algorithm's draws to use (baseline: before "
                         "2021-11-29 1900 UTC; enterprise: on/after; all: pooled). Default "
                         "enterprise, the paper's primary GOES reference -- one algorithm "
                         "throughout, and the currently operational one.")
    args = ap.parse_args()

    rows, dod_dists = [], {}
    for slug, site in SITES.items():
        depth = snow_depths(slug)
        family = "west" if slug == "vancouver" else "east"
        metar = {site.human: metar_units(site, site.human)}
        if site.auto in AUTO_OK:
            metar[site.auto] = metar_units(site, site.auto)
        goes = goes_units(site, family, era=args.era)
        # Cross-check family (the paper's exploratory East/West comparison): GOES-East at
        # Vancouver, GOES-West at Calgary and Edmonton. Logged so the cross-check numbers
        # quoted in the text come out of the derived table like every other figure.
        xfam = {"vancouver": "east", "calgary": "west", "edmonton": "west"}.get(slug)
        goes_x = goes_units(site, xfam, era=args.era) if xfam else None
        # Same draws at the automated partner's pixel: shows the satellite sees the same
        # diurnal change at both airports, so the station difference is method, not sky.
        goes_partner = (goes_units(site, family, site.auto, era=args.era)
                        if site.auto in AUTO_GOES else None)
        # Season-stratified GOES, primary family and cross-check family. Only the satellite
        # side is split here; the human seasonal deltas come from human_deltas.py, which is
        # METAR-only and so unaffected by which cloud-mask algorithm is in force.
        goes_seas = {s: goes_units(site, family, era=args.era, season=s) for s in SEASONS}
        goes_x_seas = ({s: goes_units(site, xfam, era=args.era, season=s) for s in SEASONS}
                       if xfam else None)

        for thr in args.thresholds:
            snowfree = lambda d: depth.get(d, 0.0) <= thr
            res = {}
            for station, (nu, du) in metar.items():
                nf = [f for d, f in nu.items() if snowfree(d)]
                df = [f for d, f in du.items() if snowfree(d)]
                pt, dist = boot_delta(nf, df, SEED + 100)
                lo, hi = np.percentile(dist, 2.5) * 100, np.percentile(dist, 97.5) * 100
                res[station] = (pt, lo, hi, dist, len(nf), len(df), nf, df)
                # Night/day levels as well as the delta: the main results table quotes all three, so all
                # three should come out of the pipeline rather than off a screen.
                rows.append(dict(site=slug, record=f"metar_{station}", threshold_cm=thr * 100,
                                 delta_pp=round(pt, 1), ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                                 n_night=len(nf), n_day=len(df),
                                 night_pct=round(float(np.mean(nf)) * 100, 1),
                                 day_pct=round(float(np.mean(df)) * 100, 1)))
                # Seasonal human deltas on the same snow-free ground, so the seasonal
                # human-vs-satellite comparison (the Calgary seasonal-structure sentence)
                # is matched-window on both sides. Season is the unit key date's month.
                if station == site.human:
                    for s, months in SEASON_MONTHS.items():
                        nf_s = [f for d, f in nu.items() if snowfree(d) and d.month in months]
                        df_s = [f for d, f in du.items() if snowfree(d) and d.month in months]
                        if len(nf_s) < 2 or len(df_s) < 2:
                            continue
                        pt_s, dist_s = boot_delta(nf_s, df_s, SEED + 100)
                        lo_s = np.percentile(dist_s, 2.5) * 100
                        hi_s = np.percentile(dist_s, 97.5) * 100
                        rows.append(dict(site=slug, record=f"metar_{station}_{s}",
                                         threshold_cm=thr * 100, delta_pp=round(pt_s, 1),
                                         ci_lo=round(lo_s, 1), ci_hi=round(hi_s, 1),
                                         n_night=len(nf_s), n_day=len(df_s),
                                         night_pct=round(float(np.mean(nf_s)) * 100, 1),
                                         day_pct=round(float(np.mean(df_s)) * 100, 1)))
            for fp in ("nearest", "box3", "box5"):
                nf = [f for d, f in goes[fp]["NIGHT"].items() if snowfree(d)]
                df = [f for d, f in goes[fp]["DAY"].items() if snowfree(d)]
                pt, dist = boot_delta(nf, df, SEED + 100)
                lo, hi = np.percentile(dist, 2.5) * 100, np.percentile(dist, 97.5) * 100
                res[fp] = (pt, lo, hi, dist, len(nf), len(df))
                rows.append(dict(site=slug, record=f"goes_{fp}", threshold_cm=thr * 100,
                                 delta_pp=round(pt, 1), ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                                 n_night=len(nf), n_day=len(df)))
            if goes_x is not None:
                nf = [f for d, f in goes_x["nearest"]["NIGHT"].items() if snowfree(d)]
                df = [f for d, f in goes_x["nearest"]["DAY"].items() if snowfree(d)]
                pt, dist = boot_delta(nf, df, SEED + 100)
                lo, hi = np.percentile(dist, 2.5) * 100, np.percentile(dist, 97.5) * 100
                rows.append(dict(site=slug, record=f"goes_nearest_{xfam}",
                                 threshold_cm=thr * 100, delta_pp=round(pt, 1),
                                 ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                                 n_night=len(nf), n_day=len(df)))
            # GOES-West split at the GOES-17/18 handover. The 17 loop-heat-pipe anomaly
            # degraded nighttime imagery, so the paper checks whether the West readings at the
            # three cross-check cities depend on it. Emitted here so those numbers come out of
            # the pipeline rather than a one-off script.
            if xfam == "west" or family == "west":
                fam_w = "west"
                for tag, lo, hi in (("GOES17", None, HANDOVER_1718),
                                    ("GOES18", HANDOVER_1718, None)):
                    u = goes_units(site, fam_w, era=args.era, since=lo, until=hi)
                    nf = [f for d, f in u["nearest"]["NIGHT"].items() if snowfree(d)]
                    df = [f for d, f in u["nearest"]["DAY"].items() if snowfree(d)]
                    if len(nf) < 10 or len(df) < 10:
                        continue
                    pt, dist = boot_delta(nf, df, SEED + 100)
                    lo_, hi_ = np.percentile(dist, 2.5) * 100, np.percentile(dist, 97.5) * 100
                    rows.append(dict(site=slug, record=f"goes_nearest_west_{tag}",
                                     threshold_cm=thr * 100, delta_pp=round(pt, 1),
                                     ci_lo=round(lo_, 1), ci_hi=round(hi_, 1),
                                     n_night=len(nf), n_day=len(df)))
            for s in SEASONS:
                for tag, src in (("", goes_seas), (f"_{xfam}", goes_x_seas)):
                    if src is None:
                        continue
                    nf = [f for d, f in src[s]["nearest"]["NIGHT"].items() if snowfree(d)]
                    df = [f for d, f in src[s]["nearest"]["DAY"].items() if snowfree(d)]
                    if len(nf) < 2 or len(df) < 2:
                        continue
                    pt, dist = boot_delta(nf, df, SEED + 100)
                    lo, hi = np.percentile(dist, 2.5) * 100, np.percentile(dist, 97.5) * 100
                    rows.append(dict(site=slug, record=f"goes_nearest{tag}_{s}",
                                     threshold_cm=thr * 100, delta_pp=round(pt, 1),
                                     ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                                     n_night=len(nf), n_day=len(df)))
            if goes_partner is not None:
                nf = [f for d, f in goes_partner["nearest"]["NIGHT"].items() if snowfree(d)]
                df = [f for d, f in goes_partner["nearest"]["DAY"].items() if snowfree(d)]
                pt, dist = boot_delta(nf, df, SEED + 100)
                lo, hi = np.percentile(dist, 2.5) * 100, np.percentile(dist, 97.5) * 100
                rows.append(dict(site=slug, record=f"goes_nearest_{site.auto}",
                                 threshold_cm=thr * 100, delta_pp=round(pt, 1),
                                 ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                                 n_night=len(nf), n_day=len(df)))
            # Paired human-minus-GOES: on units carrying both a human record and a GOES draw,
            # take the per-unit human-minus-GOES gap and night/day-unit block bootstrap it -- the
            # same paired treatment as the VIIRS test. This captures the shared-weather covariance
            # the old quadrature CI discarded, and cancels the fixed dome-vs-pixel offset within
            # each unit, so the interval is tighter and the estimand matched night for night.
            hnu, hdu = metar[site.human]
            gnu, gdu = goes["nearest"]["NIGHT"], goes["nearest"]["DAY"]
            cN = sorted(d for d in hnu if d in gnu and snowfree(d))
            cD = sorted(d for d in hdu if d in gdu and snowfree(d))
            gapN = np.array([hnu[d] - gnu[d] for d in cN])
            gapD = np.array([hdu[d] - gdu[d] for d in cD])
            hg = (gapN.mean() - gapD.mean()) * 100
            rng = np.random.default_rng(SEED + 100)
            bhg = (gapN[rng.integers(0, len(gapN), (B, len(gapN)))].mean(1)
                   - gapD[rng.integers(0, len(gapD), (B, len(gapD)))].mean(1)) * 100
            rows.append(dict(site=slug, record="human_minus_goes", threshold_cm=thr * 100,
                             delta_pp=round(hg, 1), ci_lo=round(float(np.percentile(bhg, 2.5)), 1),
                             ci_hi=round(float(np.percentile(bhg, 97.5)), 1),
                             n_night=len(cN), n_day=len(cD)))
            # The same paired statistic against the cross-check family, so the claim
            # that no human-excess verdict changes when the other disk is substituted
            # is checkable from this table rather than by column arithmetic.
            if goes_x is not None:
                gnu, gdu = goes_x["nearest"]["NIGHT"], goes_x["nearest"]["DAY"]
                cN = sorted(d for d in hnu if d in gnu and snowfree(d))
                cD = sorted(d for d in hdu if d in gdu and snowfree(d))
                gapN = np.array([hnu[d] - gnu[d] for d in cN])
                gapD = np.array([hdu[d] - gdu[d] for d in cD])
                hgx = (gapN.mean() - gapD.mean()) * 100
                rng = np.random.default_rng(SEED + 100)
                bhg = (gapN[rng.integers(0, len(gapN), (B, len(gapN)))].mean(1)
                       - gapD[rng.integers(0, len(gapD), (B, len(gapD)))].mean(1)) * 100
                rows.append(dict(site=slug, record=f"human_minus_goes_{xfam}",
                                 threshold_cm=thr * 100, delta_pp=round(hgx, 1),
                                 ci_lo=round(float(np.percentile(bhg, 2.5)), 1),
                                 ci_hi=round(float(np.percentile(bhg, 97.5)), 1),
                                 n_night=len(cN), n_day=len(cD)))
            if site.auto in AUTO_OK:
                # Redraw both stations from ONE rng stream so they are resampled
                # independently, as observable_nights.py and human_deltas.py do.
                # Differencing the two per-station distributions instead would reuse
                # SEED+100 for both and couple their draws, narrowing the interval.
                rng = np.random.default_rng(SEED + 999)
                nh, dh = res[site.human][6], res[site.human][7]
                na, da = res[site.auto][6], res[site.auto][7]
                bm = lambda a: np.asarray(a, float)[
                    rng.integers(0, len(a), size=(B, len(a)))].mean(axis=1)
                dodd = (bm(nh) - bm(dh)) - (bm(na) - bm(da))
                pt = res[site.human][0] - res[site.auto][0]
                lo, hi = np.percentile(dodd, 2.5) * 100, np.percentile(dodd, 97.5) * 100
                rows.append(dict(site=slug, record="auto_dod", threshold_cm=thr * 100,
                                 delta_pp=round(pt, 1), ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                                 n_night="", n_day=""))
            print(f"[{slug} thr={thr*100:.1f}cm] human {res[site.human][0]:+.1f} "
                  f"goes {res['nearest'][0]:+.1f} H-G {hg:+.1f}", flush=True)

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["site", "record", "threshold_cm", "delta_pp",
                                          "ci_lo", "ci_hi", "n_night", "n_day",
                                          "night_pct", "day_pct"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
