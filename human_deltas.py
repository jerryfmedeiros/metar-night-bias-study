"""
human_deltas.py — Human night-day usable deltas per site and window.

Computes the night-minus-day clear-or-few delta for every study station, on
full-year, calendar Jun-Nov (a crude bare-ground proxy, diagnostic only), and
seasonal windows, with night/day units and a block bootstrap. The full-year and
seasonal values back the paper's text; the SNOW-FREE columns of the main
results table come from dynamic_tables.py (daily ERA5-Land screen), NOT from
the Jun-Nov window here. The script also produces the full-year
difference-of-differences against the four automated partner stations.

Replicates observable_nights.py's conventions exactly:
  - night = sun <= -12 deg (the regime actually used throughout the study;
    --dark-deg -18 reruns the astronomical-dark sensitivity check),
    unit = local-noon-to-noon night; day = sun >= +6 deg, unit = local calendar date
  - a unit needs >= 2 obs to contribute a fraction
  - usable = okta <= 2 (SKC+FEW); SKC-only (okta = 0) computed alongside
  - night block bootstrap, B = 2000; full-year run uses observable_nights.py's exact
    seed offsets (42 night / 43 day / 142 delta / 1041 DoD) so the main results table reproduces.
    The difference-of-differences draws its two stations from ONE rng stream seeded
    42+999, exactly as observable_nights.py does, so the two records are resampled
    independently. Deriving the DoD from the per-station delta distributions instead
    would reuse the same seed for both stations, coupling their draws and narrowing
    the interval below the independent-resampling CI the paper reports.
  - snow-free window: unit month in Jun-Nov (night keyed by its local evening date),
    matching the GOES SUMMER+AUTUMN (meteorological) filter

Usage (from repo root; reads metar_cache/ only, no network needed):
  python3 human_deltas.py [--boot 2000] [--seed 42] [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

try:
    from fetch_metar import fetch_metar_csv
    from metar_climatology import calculate_sun_alt
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root: {e}")

MIN_OBS = 2
# Crude month-based proxy for bare ground, NOT the published snow-free filter (which is
# the daily ERA5-Land snow-depth screen in dynamic_tables.py). Kept for the diagnostic
# contrast only; anything quoted as "snow-free" in the paper comes from dynamic_tables.py.
CALENDAR_BARE_MONTHS = set(range(6, 12))     # Jun-Nov = GOES SUMMER+AUTUMN
YEARS = (2020, 2025)


def collect_units(station, site, dark_deg=-12.0, hourly_only=False):
    """Per-unit usable fractions: (night_units, day_units) as lists of
    (month, frac_okta2, frac_skc, n_obs). dark_deg sets the night threshold;
    -12 (nautical) is the study default, -18 reproduces the astronomical-dark
    sensitivity check quoted in the paper. hourly_only keeps top-of-hour
    (minute :00) reports only: the IEM archive strips the SPECI token, so
    off-hour timestamps are the event-triggered specials, which are much
    cloudier than routine reports and issued at different rates by human and
    automated stations; --hourly-only reruns the SPECI-exclusion sensitivity
    check quoted in the paper."""
    tz = ZoneInfo(site.iana_tz)
    nights = defaultdict(list)   # night date -> [(usable2, skc)]
    days = defaultdict(list)     # local date  -> [(usable2, skc)]
    for year in range(YEARS[0], YEARS[1] + 1):
        for o in fetch_metar_csv(station, dt.date(year, 1, 1), dt.date(year, 12, 31)):
            if hourly_only and o.timestamp.minute != 0:
                continue
            alt = calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m)
            rec = (o.coverage_okta <= 2, o.coverage_okta == 0)
            if alt <= dark_deg:
                nk = (o.timestamp.astimezone(tz) - dt.timedelta(hours=12)).date()
                nights[nk].append(rec)
            elif alt >= 6.0:
                days[o.timestamp.astimezone(tz).date()].append(rec)

    def units(dct):
        out = []
        for d, recs in sorted(dct.items()):
            if len(recs) >= MIN_OBS:
                a = np.asarray(recs, dtype=float)
                out.append((d.month, a[:, 0].mean(), a[:, 1].mean(), len(recs)))
        return out

    return units(nights), units(days)


def boot_mean(fracs, B, rng):
    arr = np.asarray(fracs, float)
    idx = rng.integers(0, len(arr), size=(B, len(arr)))
    return arr[idx].mean(axis=1)


def delta_ci(nf, df, B, seed):
    """(delta pp, lo, hi) with observable_nights-style independent resampling."""
    rng = np.random.default_rng(seed)
    d = boot_mean(nf, B, rng) - boot_mean(df, B, rng)
    point = (np.mean(nf) - np.mean(df)) * 100
    return point, np.percentile(d, 2.5) * 100, np.percentile(d, 97.5) * 100, d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--csv", default="results/human_deltas.csv")
    ap.add_argument("--dark-deg", type=float, default=-12.0,
                    help="solar-altitude night threshold; -18 reproduces the "
                         "astronomical-dark sensitivity check in the paper")
    ap.add_argument("--hourly-only", action="store_true",
                    help="keep top-of-hour reports only (drop event-triggered "
                         "SPECIs); reruns the SPECI-exclusion sensitivity check")
    args = ap.parse_args()
    B = args.boot

    rows = []
    dod_stash = {}   # (slug, window, metric) -> delta bootstrap distribution + point

    for slug, site in SITES.items():
        stations = [site.human] + ([site.auto] if site.auto in ("CYBW", "CYHU", "CYTZ", "CZVL") else [])
        for station in stations:
            print(f"\n=== {slug} / {station} ===", flush=True)
            nunits, dunits = collect_units(station, site, dark_deg=args.dark_deg,
                                           hourly_only=args.hourly_only)
            for window, keep in (("full", None), ("calendar_JunNov", CALENDAR_BARE_MONTHS),
                                 ("winter", {12, 1, 2}), ("spring", {3, 4, 5}),
                                 ("summer", {6, 7, 8}), ("autumn", {9, 10, 11})):
                nsel = [u for u in nunits if keep is None or u[0] in keep]
                dsel = [u for u in dunits if keep is None or u[0] in keep]
                if not nsel or not dsel:
                    continue
                for metric, col in (("okta2", 1), ("skc", 2)):
                    nf = [u[col] for u in nsel]
                    df = [u[col] for u in dsel]
                    dpt, dlo, dhi, ddist = delta_ci(nf, df, B, args.seed + 100)
                    rows.append(dict(
                        site=slug, station=station, window=window, metric=metric,
                        n_nights=len(nf), n_days=len(df),
                        night_pct=round(np.mean(nf) * 100, 1),
                        day_pct=round(np.mean(df) * 100, 1),
                        delta_pp=round(dpt, 1),
                        ci_lo=round(dlo, 1), ci_hi=round(dhi, 1),
                    ))
                    # Stash the raw unit fractions, not the delta distribution: the DoD
                    # must redraw both stations from a single shared rng stream.
                    dod_stash[(slug, station, window, metric)] = (nf, df, dpt)
                    print(f"  {window:9s} {metric:5s}: night {np.mean(nf)*100:5.1f}%  "
                          f"day {np.mean(df)*100:5.1f}%  delta {dpt:+5.1f} pp "
                          f"[{dlo:+5.1f},{dhi:+5.1f}]  (N={len(nf)} nights/{len(df)} days)",
                          flush=True)

    # DoD for the two sites with a true automated partner, both windows.
    print("\n=== Difference of differences (human - auto) ===")
    for slug, auto in (("calgary", "CYBW"), ("montreal", "CYHU"),
                       ("toronto", "CYTZ"), ("edmonton", "CZVL")):
        human = SITES[slug].human
        for window in ("full", "calendar_JunNov"):
            for metric in ("okta2", "skc"):
                kh, ka = (slug, human, window, metric), (slug, auto, window, metric)
                if kh not in dod_stash or ka not in dod_stash:
                    continue
                (nh, dhu, ph), (na, dau, pa) = dod_stash[kh], dod_stash[ka]
                # One rng stream, four sequential draws — mirrors observable_nights.py
                # so the two stations are resampled independently of each other.
                rng = np.random.default_rng(args.seed + 999)
                d_human = boot_mean(nh, B, rng) - boot_mean(dhu, B, rng)
                d_auto = boot_mean(na, B, rng) - boot_mean(dau, B, rng)
                dod = d_human - d_auto
                point = ph - pa
                lo, hi = np.percentile(dod, 2.5) * 100, np.percentile(dod, 97.5) * 100
                rows.append(dict(site=slug, station=f"{human}-{auto}", window=window,
                                 metric=metric, n_nights="", n_days="",
                                 night_pct="", day_pct="",
                                 delta_pp=round(point, 1),
                                 ci_lo=round(lo, 1), ci_hi=round(hi, 1)))
                print(f"  {slug:9s} {window:9s} {metric:5s}: DoD {point:+5.1f} pp "
                      f"[{lo:+5.1f},{hi:+5.1f}]", flush=True)

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
