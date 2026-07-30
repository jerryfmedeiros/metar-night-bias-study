"""fetch_all_sites.py — multi-site driver for the cloud-bias study.

Turns the single-site Calgary pipeline into a cross-Canada one. For each site in
the registry (sites.py) it:

  1. AUTO SCREEN  — fetches (cached) METAR for the site's station(s) and reports
     night-regime %AUTO. This is the *selection filter*: the whole study hinges on
     a station being genuinely human-augmented AT NIGHT (CYYC is 0% AUTO at night;
     many airports silently revert to AUTO after the tower closes and must drop).
  2. OBSERVABLE   — runs observable_nights.py --site <slug>, which fetches METAR +
     ERA5 (transparency/seeing) + NAPS PM2.5 (smoke) and produces the day/night
     usable bias, the human-auto difference-of-differences, and the observable-night
     budget.
  3. CLIMATOLOGY  — runs metar_climatology.py --site <slug> for sky-state, cloud
     character, and the augmentation/ceilometer breakdowns.
  4. GOES         — runs climatology_goes.py --site <slug>, sampling the correct
     satellite family (East 16->19, West 17->18; Vancouver samples BOTH).

Each analysis runs as a SUBPROCESS so its site-geometry globals stay isolated, and
its full output is captured under results/<slug>/.

Usage:
  # quick selection screen only (no heavy fetch/analysis):
  python fetch_all_sites.py --sites all --screen-only
  # one site end-to-end with a small GOES sample (smoke test):
  python fetch_all_sites.py --sites winnipeg --goes-samples 200
  # full run for several sites:
  python fetch_all_sites.py --sites calgary vancouver toronto --goes-samples 5000
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from sites import resolve_sites
from fetch_metar import fetch_metar_csv
from metar_climatology import calculate_sun_alt, get_solar_regime

PROJECT_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "results"


def auto_screen(site, years: tuple[int, int]) -> list[dict]:
    """Fetch (cached) METAR and report night-regime %AUTO per station.

    A human station qualifies only if it is human-augmented at night (low %AUTO);
    an automated partner should be ~100% AUTO. Returns one summary dict per station."""
    y0, y1 = years
    print(f"\n=== [{site.slug}] METAR AUTO screen {y0}-{y1} ===")
    print(f"{'Station':<7} | {'role':<6} | {'night N':>8} | {'night %AUTO':>11} | "
          f"{'night usable%':>13} | verdict")
    print("-" * 78)
    rows: list[dict] = []
    for station in site.stations:
        role = "human" if station == site.human else "auto"
        tot = defaultdict(int); auto = defaultdict(int); usable = defaultdict(int)
        for year in range(y0, y1 + 1):
            try:
                obs = fetch_metar_csv(station, dt.date(year, 1, 1), dt.date(year, 12, 31))
            except Exception as e:
                print(f"  {station} {year}: fetch error {e}")
                continue
            for o in obs:
                reg = get_solar_regime(
                    calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m))
                tot[reg] += 1
                if o.is_auto:
                    auto[reg] += 1
                if o.coverage_okta <= 2:
                    usable[reg] += 1
        n_night = tot["NIGHT"]
        pa = auto["NIGHT"] / n_night * 100 if n_night else float("nan")
        pu = usable["NIGHT"] / n_night * 100 if n_night else float("nan")
        if n_night < 1000:
            # the screen the paper states: too few night reports to classify
            verdict = f"UNDER {1000} night reports → DROP"
            qualifies = False
        elif role == "human":
            verdict = ("OK human-augmented" if pa < 5 else
                       "MIXED — inspect" if pa < 50 else "AUTO at night → DROP")
            qualifies = None if math.isnan(pa) else bool(pa < 5)
        else:
            verdict = "OK automated" if pa > 95 else "not fully AUTO — inspect"
            qualifies = None if math.isnan(pa) else bool(pa > 95)
        print(f"{station:<7} | {role:<6} | {n_night:8d} | {pa:10.1f}% | {pu:12.1f}% | {verdict}")
        rows.append({
            "station": station, "role": role, "night_n": n_night,
            "night_pct_auto": None if math.isnan(pa) else round(pa, 1),
            "night_usable_pct": None if math.isnan(pu) else round(pu, 1),
            "qualifies": qualifies, "verdict": verdict,
        })

    # Persist this site's screen so the attrition is durable & machine-readable.
    outdir = RESULTS_DIR / site.slug
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "auto_screen.json").write_text(json.dumps(
        {"site": site.slug, "human": site.human, "auto": site.auto,
         "years": [y0, y1], "stations": rows}, indent=2))
    return rows


SCREEN_CSV_COLS = ["site", "station", "role", "years", "night_n",
                   "night_pct_auto", "night_usable_pct", "qualifies", "verdict"]


def merge_screen_csv(all_rows: list[dict]) -> Path:
    """Merge screen rows into results/screen_summary.csv, keyed by (site, station).

    Re-running a subset of sites updates only those rows, so the combined table
    accumulates across separate runs (mirrors the GOES checkpoint philosophy)."""
    path = RESULTS_DIR / "screen_summary.csv"
    existing: dict[tuple, dict] = {}
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                existing[(r["site"], r["station"])] = r
    for r in all_rows:
        existing[(r["site"], r["station"])] = r
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCREEN_CSV_COLS)
        w.writeheader()
        for key in sorted(existing):
            w.writerow({c: existing[key].get(c, "") for c in SCREEN_CSV_COLS})
    return path


def run_step(cmd: list[str], logpath: Path) -> int:
    """Run an analysis subprocess, tee-ing combined output to logpath."""
    print(f"  $ {' '.join(cmd)}")
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with open(logpath, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"    -> {logpath.relative_to(PROJECT_ROOT)} [{status}]")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", required=True,
                    help="site slugs from sites.py, or 'all'")
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2025])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--goes-samples", type=int, default=200,
                    help="GOES samples per regime per family (small for smoke tests)")
    ap.add_argument("--screen-only", action="store_true",
                    help="run just the METAR AUTO screen (the selection filter)")
    ap.add_argument("--skip-era5", action="store_true", help="pass --no-era5 to observable_nights")
    ap.add_argument("--skip-pm25", action="store_true", help="pass --no-pm25 to observable_nights")
    ap.add_argument("--skip-goes", action="store_true", help="skip GOES sampling")
    ap.add_argument("--skip-viirs", action="store_true", help="skip VIIRS sampling")
    ap.add_argument("--viirs-samples", type=int, default=1500,
                    help="VIIRS samples per regime (needs an Earthdata token; see fetch_viirs.py)")
    args = ap.parse_args()

    sites = resolve_sites(args.sites)
    y0, y1 = args.years
    yr = [str(y0), str(y1)]
    print(f"Multi-site driver — {len(sites)} site(s): {', '.join(s.slug for s in sites)}")

    screen_summary: list[tuple[str, dict]] = []
    all_rows: list[dict] = []
    for site in sites:
        outdir = RESULTS_DIR / site.slug
        rows = auto_screen(site, (y0, y1))
        for r in rows:
            all_rows.append({"site": site.slug, "years": f"{y0}-{y1}", **r})
            if r["role"] == "human":
                screen_summary.append((site.slug, r))
        if args.screen_only:
            continue

        py = sys.executable
        # 2) Observable budget + bias (fetches METAR + ERA5 + PM2.5).
        obs_cmd = [py, "observable_nights.py", "--site", site.slug, "--years", *yr,
                   "--seed", str(args.seed)]
        if args.skip_era5:
            obs_cmd.append("--no-era5")
        if args.skip_pm25:
            obs_cmd.append("--no-pm25")
        run_step(obs_cmd, outdir / "observable.txt")

        # 3) Sky-state / cloud character / augmentation detail.
        run_step([py, "metar_climatology.py", "--site", site.slug, "--years", *yr],
                 outdir / "climatology.txt")

        # 4) GOES — correct satellite family (both for western sites); --discard for disk.
        if not args.skip_goes:
            run_step([py, "climatology_goes.py", "--site", site.slug, "--years", *yr,
                      "--samples", str(args.goes_samples), "--seed", str(args.seed),
                      "--discard"],
                     outdir / "goes.txt")

        # 5) VIIRS — second satellite (polar, DNB at night); needs an Earthdata token.
        if not args.skip_viirs:
            run_step([py, "climatology_viirs.py", "--site", site.slug, "--years", *yr,
                      "--samples", str(args.viirs_samples), "--seed", str(args.seed),
                      "--discard"],
                     outdir / "viirs.txt")

    # Persist the merged screen table (durable, machine-readable attrition record).
    csv_path = merge_screen_csv(all_rows)

    # Cross-site selection summary: the headline filter for choosing study sites.
    print("\n" + "=" * 78)
    print("[SELECTION SUMMARY] human stations — night-augmentation screen")
    print(f"{'Site':<10} | {'Station':<7} | {'night N':>8} | {'night %AUTO':>11} | verdict")
    print("-" * 70)
    for slug, r in screen_summary:
        pa = r["night_pct_auto"]
        pa_str = f"{pa:10.1f}%" if pa is not None else f"{'n/a':>11}"
        print(f"{slug:<10} | {r['station']:<7} | {r['night_n']:8d} | {pa_str} | {r['verdict']}")
    print(f"\nScreen table: {csv_path.relative_to(PROJECT_ROOT)}  "
          f"(+ results/<slug>/auto_screen.json per site)")
    print("Analysis under results/<slug>/ (observable.txt, climatology.txt, goes.txt)")


if __name__ == "__main__":
    main()
