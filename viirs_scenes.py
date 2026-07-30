"""
viirs_scenes.py — per-scene VIIRS matched analysis from viirs_matches_<site>.csv.

Consumes the per-scene logs written by climatology_viirs.py --log-matches and reports,
per site and per snow window (calendar vs dynamic ERA5-Land snow depth):

  - the matched night gap (human clear minus VIIRS clear, (CN-NC)/N) with a
    continuity-corrected McNemar z on the discordant pairs
  - the day-referenced matched difference-of-deltas, with two intervals: the
    scene-level one (treats scenes as independent) and a night/day-unit block
    bootstrap that absorbs intra-night clustering of overpasses
  - the moonlit vs dark split of the night gap, split on a modelled moonlight illuminance
    (Krisciunas & Schaefer phase brightness x airmass extinction) recomputed per scene
  - the realized METAR match offsets (median / mean / p90)

Duplicate catalog entries (same timestamp + platform) are dropped before analysis.
Snow flags come from data/snow_cache/snow_<site>.json (daily ERA5-Land snow depth > 1 cm).

Usage:
  python3 viirs_scenes.py [--glob 'data/viirs_matches/viirs_matches_*.csv'] [--boot 2000] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob as globmod
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import ephem
import numpy as np

try:
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

SNOW_THRESH_M = 0.01   # overridden by --snow-thresh
DJF_EXCLUDED = {"toronto", "ottawa", "montreal", "winnipeg", "calgary", "edmonton"}

K_EXT = 0.28              # V-band clear-sky extinction, mag per airmass
MOON_ILLUM_THRESH = 0.1   # relative ground illuminance (full Moon at zenith = 1) to count "moonlit"


def _airmass(h_deg):
    """Kasten-Young (1989) airmass at apparent altitude h_deg.

    Kasten-Young is bounded at the horizon: it converges to ~38 at h=0, where the naive
    secant 1/sin(h) diverges, so there is no singularity to trip over. The h<=0 guard only
    covers the below-horizon case, and moon_illuminance returns 0 for alt<=0 before this is
    ever called, so a Moon at or under the horizon yields ~0 illuminance rather than a blow-up.
    """
    if h_deg <= 0:
        return 40.0
    return 1.0 / (math.sin(math.radians(h_deg)) + 0.50572 * (h_deg + 6.07995) ** -1.6364)


def moon_illuminance(ts_utc, lat, lon, elev):
    """Relative moonlight reaching the ground (full Moon at zenith = 1; 0 if the Moon is down).

    Krisciunas & Schaefer (1991) phase-angle brightness attenuated by atmospheric extinction at
    the Moon's airmass. Replaces the bare above-horizon, quarter-phase flag stored in the logs,
    which counts low or thin-crescent Moons that extinction renders effectively dark.
    """
    o = ephem.Observer()
    o.lat, o.lon, o.elevation = str(lat), str(lon), elev
    o.date = ephem.Date(ts_utc)
    m = ephem.Moon(o)
    alt = math.degrees(m.alt)
    if alt <= 0:
        return 0.0
    frac = max(0.0, min(1.0, m.phase / 100.0))
    alpha = math.degrees(math.acos(max(-1.0, min(1.0, 2 * frac - 1))))   # phase angle, 0 = full
    brightness = 10 ** (-0.4 * (0.026 * alpha + 4e-9 * alpha ** 4))       # = 1 at full Moon
    transmission = 10 ** (-0.4 * K_EXT * (_airmass(alt) - 1.0))           # extra extinction vs zenith
    return brightness * transmission


def snow_days(slug, thresh=None):
    p = Path("data/snow_cache") / f"snow_{slug}.json"
    if not p.exists():
        return None
    thresh = SNOW_THRESH_M if thresh is None else thresh
    d = json.loads(p.read_text())["daily"]
    return {dt.date.fromisoformat(t) for t, v in zip(d["time"], d["snow_depth_max"])
            if v is not None and v > thresh}


def mcnemar_z(cn, nc):
    return (abs(cn - nc) - 1) / math.sqrt(cn + nc) if cn + nc else float("nan")


def gap_stats(scenes):
    """scenes: list of (h, v). Returns (gap pp, CN, NC, N)."""
    cn = sum(1 for h, v in scenes if h and not v)
    nc = sum(1 for h, v in scenes if not h and v)
    n = len(scenes)
    return ((cn - nc) / n * 100 if n else float("nan")), cn, nc, n


def unit_gaps(scenes_by_unit):
    return [float(np.mean([h - v for h, v in sc])) for sc in scenes_by_unit.values()]


def boot_mean(vals, B, rng):
    a = np.asarray(vals, float)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    return a[idx].mean(axis=1)


RESULTS = []
MOON_RESULTS = []
MOON_MIN_SCENES = 200   # below this a stratum is too thin to read (e.g. Edmonton under snow)

def analyze(path, B, seed, thresh=None):
    slug = Path(path).stem.replace("viirs_matches_", "")
    site = SITES[slug]
    tz = ZoneInfo(site.iana_tz)
    snow = snow_days(slug, thresh)

    seen, rows = set(), []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r["ts_utc"], r["platform"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)

    # matched scenes with the box decision (the paper's primary footprint). The moonlit/dark
    # split uses a modelled moonlight illuminance recomputed here from each scene's timestamp
    # (moon_illuminance), not the binary flag stored in the log.
    lat, lon = site.coords[site.human]
    elev = site.elevation_m
    scenes = []   # (regime, local unit date, month, moon illuminance, h, v_box, v_pixel, offset_s)
    for r in rows:
        if r["human_usable"] == "" or r["box_clear"] == "" or r["pixel_clear"] == "":
            continue
        ts_utc = dt.datetime.fromisoformat(r["ts_utc"])
        ts = ts_utc.astimezone(tz)
        unit = (ts - dt.timedelta(hours=12)).date() if r["regime"] == "NIGHT" else ts.date()
        illum = (moon_illuminance(ts_utc.replace(tzinfo=None), lat, lon, elev)
                 if r["regime"] == "NIGHT" else 0.0)
        scenes.append((r["regime"], unit, unit.month, illum,
                       int(r["human_usable"]), int(r["box_clear"]), int(r["pixel_clear"]),
                       int(r["metar_offset_s"])))

    offs = np.asarray([s[7] for s in scenes]) / 60.0
    print(f"\n=== {slug}: {len(rows)} scenes ({len(scenes)} matched, dupes dropped) ===")
    print(f"  METAR match offset: median {np.median(offs):.0f} min, "
          f"mean {offs.mean():.0f} min, p90 {np.percentile(offs, 90):.0f} min")

    windows = {"calendar": (lambda m, u: m not in (12, 1, 2)) if slug in DJF_EXCLUDED
                           else (lambda m, u: True)}
    if snow is not None:
        windows["dynamic"] = lambda m, u: u not in snow

    for wname, keep in windows.items():
        sel = [s for s in scenes if keep(s[2], s[1])]
        by = {"NIGHT": [(h, v) for rg, u, m, il, h, v, p, o in sel if rg == "NIGHT"],
              "DAY":   [(h, v) for rg, u, m, il, h, v, p, o in sel if rg == "DAY"]}
        by_px = {"NIGHT": [(h, p) for rg, u, m, il, h, v, p, o in sel if rg == "NIGHT"],
                 "DAY":   [(h, p) for rg, u, m, il, h, v, p, o in sel if rg == "DAY"]}
        units = {"NIGHT": defaultdict(list), "DAY": defaultdict(list)}
        units_px = {"NIGHT": defaultdict(list), "DAY": defaultdict(list)}
        for rg, u, m, il, h, v, p, o in sel:
            units[rg][u].append((h, v))
            units_px[rg][u].append((h, p))
        if not by["NIGHT"] or not by["DAY"]:
            print(f"  {wname:9s}: insufficient data")
            continue
        gn, cn_n, nc_n, n_n = gap_stats(by["NIGHT"])
        gd, cn_d, nc_d, n_d = gap_stats(by["DAY"])
        z = mcnemar_z(cn_n, nc_n)
        dod = gn - gd
        # scene-level CI (independence assumption; reported for comparison only —
        # climatology_viirs.matched_report uses an unpaired variance, this one is paired)
        var_n = (cn_n + nc_n - (cn_n - nc_n) ** 2 / n_n) / n_n ** 2
        var_d = (cn_d + nc_d - (cn_d - nc_d) ** 2 / n_d) / n_d ** 2
        ci = 1.96 * math.sqrt(var_n + var_d) * 100
        # night/day-unit block bootstrap (cluster-honest)
        rng = np.random.default_rng(seed + 100)
        bn = boot_mean(unit_gaps(units["NIGHT"]), B, rng) * 100
        bd = boot_mean(unit_gaps(units["DAY"]), B, rng) * 100
        dodu = np.mean(unit_gaps(units["NIGHT"])) * 100 - np.mean(unit_gaps(units["DAY"])) * 100
        lo, hi = np.percentile(bn - bd, 2.5), np.percentile(bn - bd, 97.5)
        # cluster-honest night gap: 2-4 overpasses share a night's cloud field (and often
        # the same matched METAR), so the McNemar z above overstates the night gap's
        # precision. Bootstrap the mean per-night-unit gap instead; this is the interval
        # the significance statements should rest on.
        ngu = unit_gaps(units["NIGHT"])
        ngu_pt = float(np.mean(ngu)) * 100
        bng = boot_mean(ngu, B, np.random.default_rng(seed + 200)) * 100
        ngu_lo, ngu_hi = np.percentile(bng, 2.5), np.percentile(bng, 97.5)
        # single-pixel variant (750 m nearest pixel instead of the 7x7 box), same units
        pgn, pcn_n, pnc_n, pn_n = gap_stats(by_px["NIGHT"])
        pgd, _, _, _ = gap_stats(by_px["DAY"])
        pz = mcnemar_z(pcn_n, pnc_n)
        rng_px = np.random.default_rng(seed + 300)
        pbn = boot_mean(unit_gaps(units_px["NIGHT"]), B, rng_px) * 100
        pbd = boot_mean(unit_gaps(units_px["DAY"]), B, rng_px) * 100
        pdodu = (np.mean(unit_gaps(units_px["NIGHT"])) * 100
                 - np.mean(unit_gaps(units_px["DAY"])) * 100)
        plo, phi = np.percentile(pbn - pbd, 2.5), np.percentile(pbn - pbd, 97.5)
        RESULTS.append(dict(site=slug, window=wname,
                            snow_thresh_cm=round((thresh if thresh is not None else SNOW_THRESH_M) * 100, 1),
                            night_gap_pp=round(gn, 1),
                            mcnemar_z=round(z, 2), n_night=n_n, day_gap_pp=round(gd, 1),
                            n_day=n_d, dod_scene=round(dod, 1),
                            dod_scene_lo=round(dod - ci, 1), dod_scene_hi=round(dod + ci, 1),
                            # Two decimals: the snow-threshold sweep turns on whether the
                            # Winnipeg difference-of-deltas bound clears zero, and 1 dp
                            # rounds that margin away.
                            dod_unit=round(dodu, 1), dod_unit_lo=round(lo, 2),
                            dod_unit_hi=round(hi, 2), n_night_units=len(units["NIGHT"]),
                            n_day_units=len(units["DAY"]),
                            night_gap_unit_pp=round(ngu_pt, 1),
                            night_gap_unit_lo=round(ngu_lo, 2),
                            night_gap_unit_hi=round(ngu_hi, 2),
                            pixel_night_gap_pp=round(pgn, 1), pixel_mcnemar_z=round(pz, 2),
                            pixel_day_gap_pp=round(pgd, 1),
                            pixel_dod_unit=round(pdodu, 1),
                            pixel_dod_unit_lo=round(plo, 2),
                            pixel_dod_unit_hi=round(phi, 2)))
        print(f"  {wname:9s}: night gap {gn:+5.1f} pp (z={z:.1f}, N={n_n}), day {gd:+5.1f} "
              f"(N={n_d})")
        print(f"             night gap unit-bootstrap {ngu_pt:+5.1f} [{ngu_lo:+5.1f},{ngu_hi:+5.1f}]")
        print(f"             DoD scene-level {dod:+5.1f} [{dod-ci:+5.1f},{dod+ci:+5.1f}]  |  "
              f"unit-bootstrap {dodu:+5.1f} [{lo:+5.1f},{hi:+5.1f}] "
              f"({len(units['NIGHT'])} nights/{len(units['DAY'])} days)")
        print(f"             single-pixel: night gap {pgn:+5.1f} (z={pz:.1f}), day {pgd:+5.1f}, "
              f"DoD unit {pdodu:+5.1f} [{plo:+5.1f},{phi:+5.1f}]")
        # moon split within the window. Persisted, not just printed: these are the numbers
        # behind the moonlit/dark paragraph, so they belong in a table like everything else.
        for tag, keep_moon in (("moonlit", lambda e: e >= MOON_ILLUM_THRESH),
                               ("dark",    lambda e: e < MOON_ILLUM_THRESH)):
            ms = [(h, v) for rg, u, m, il, h, v, p, o in sel if rg == "NIGHT" and keep_moon(il)]
            if len(ms) > MOON_MIN_SCENES:
                g, cn, nc, n = gap_stats(ms)
                mz = mcnemar_z(cn, nc)
                MOON_RESULTS.append(dict(
                    site=slug, window=wname,
                    snow_thresh_cm=round((thresh if thresh is not None else SNOW_THRESH_M) * 100, 1),
                    stratum=tag, night_gap_pp=round(g, 1), mcnemar_z=round(mz, 2), n_scenes=n))
                print(f"             night gap, {tag:7s}: {g:+5.1f} pp (z={mz:.1f}, N={n})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="data/viirs_matches/viirs_matches_*.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snow-thresh", type=float, nargs="+", default=[None],
                    help="snow-day depth threshold(s) in metres (default module constant)")
    ap.add_argument("--csv-out", default="results/viirs_scenes.csv")
    ap.add_argument("--moon-csv-out", default="results/viirs_moon_split.csv")
    args = ap.parse_args()
    paths = sorted(globmod.glob(args.glob))
    if not paths:
        sys.exit(f"No files matched {args.glob}")
    for thresh in args.snow_thresh:
        for p in paths:
            analyze(p, args.boot, args.seed, thresh)
    if RESULTS:
        import csv as csvmod
        with open(args.csv_out, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(RESULTS[0].keys()))
            w.writeheader()
            w.writerows(RESULTS)
        print(f"\nWrote {len(RESULTS)} rows to {args.csv_out}")
    if MOON_RESULTS:
        import csv as csvmod
        with open(args.moon_csv_out, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(MOON_RESULTS[0].keys()))
            w.writeheader()
            w.writerows(MOON_RESULTS)
        print(f"Wrote {len(MOON_RESULTS)} rows to {args.moon_csv_out}")


if __name__ == "__main__":
    main()
