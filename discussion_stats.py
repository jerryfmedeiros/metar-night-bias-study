"""
discussion_stats.py — the Discussion/Limitations numbers that back single
sentences in the paper, none of which previously had a generating script.

Computes, offline, from the shipped caches and derived tables:

  corr        pairwise cross-site correlation of nightly clear-or-few fractions
              (shared nights, full record) + the effective number of independent
              weather samples N_eff = (sum lambda)^2 / sum lambda^2 over the
              correlation matrix's eigenvalues
  spearman    Spearman rank correlation of site latitude vs the snow-free H-G
              excess (read from results/dynamic_tables.csv), t-approximation p
  siteyears   per-year human night-day delta for every site (the "all 48
              site-years positive" claim) and the per-pair yearly DoD with an
              OLS slope (pp/yr) for the four automated-partner pairs
  diurnal     clear-or-few fraction by local clock hour per station; the
              pre-dawn (02-05 h) mean minus the afternoon (12-18 h) minimum for
              the paired stations, and each automated partner's max deviation
              from its own diurnal mean
  layers      share of reports carrying any cloud layer based above 12,000 ft
              (the ceilometer ceiling), per station
  viirsdelta  the VIIRS mask's own night-minus-day clear delta per site
              (unit-weighted, snow-free, 7x7 box) — the satellite-side diurnal
              signal the geostationary-family discussion compares against
  erashare    era balance of the GOES draws: per site, the largest gap between
              the baseline and Enterprise eras in any season-and-regime cell's
              share of the draws, over all draws and over snow-free draws —
              the "not by sampling accident" sentence in the cloud-mask
              inhomogeneity subsection

Usage (from the repo root, reads data/ only):
  python3 discussion_stats.py [--csv results/discussion_stats.csv]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
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
ROWS: list[dict] = []


def emit(section, key, value):
    ROWS.append(dict(section=section, key=key, value=value))


def get_obs(station, _cache={}):
    if station not in _cache:
        obs = []
        for y in range(YEARS[0], YEARS[1] + 1):
            obs += fetch_metar_csv(station, dt.date(y, 1, 1), dt.date(y, 12, 31))
        _cache[station] = obs
    return _cache[station]


def units(station, site):
    """night/day unit -> clear-or-few fraction, study conventions."""
    tz = ZoneInfo(site.iana_tz)
    nights, days = defaultdict(list), defaultdict(list)
    for o in get_obs(station):
        alt = calculate_sun_alt(o.timestamp, site.lat, site.lon, site.elevation_m)
        rec = o.coverage_okta <= 2
        if alt <= -12.0:
            nights[(o.timestamp.astimezone(tz) - dt.timedelta(hours=12)).date()].append(rec)
        elif alt >= 6.0:
            days[o.timestamp.astimezone(tz).date()].append(rec)
    keep = lambda d: {k: float(np.mean(v)) for k, v in d.items() if len(v) >= MIN_OBS}
    return keep(nights), keep(days)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/discussion_stats.csv")
    args = ap.parse_args()

    hum = {s.human: (slug, s) for slug, s in SITES.items()}
    pairs = {slug: s for slug, s in SITES.items()
             if s.auto in ("CYBW", "CYHU", "CYTZ", "CZVL")}

    print("== building night/day units (12 stations, ~2 min) ==", flush=True)
    NU, DU = {}, {}
    for slug, site in SITES.items():
        for st in site.stations:
            NU[st], DU[st] = units(st, site)

    # ---- corr: pairwise nightly correlations + effective N -----------------
    print("\n== cross-site nightly correlations ==")
    stations = [s.human for s in SITES.values()]
    n = len(stations)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = NU[stations[i]], NU[stations[j]]
            shared = sorted(set(a) & set(b))
            r = float(np.corrcoef([a[d] for d in shared], [b[d] for d in shared])[0, 1])
            mat[i, j] = mat[j, i] = r
            emit("corr", f"{stations[i]}-{stations[j]}", round(r, 2))
            if r > 0.40:
                print(f"  {stations[i]}-{stations[j]}: r={r:.2f}")
    lam = np.linalg.eigvalsh(mat)
    neff = float(lam.sum() ** 2 / (lam ** 2).sum())
    emit("corr", "n_eff_eigen", round(neff, 1))
    off = mat[np.triu_indices(n, 1)]
    print(f"  pairs > 0.40: {(off > 0.40).sum()}; max {off.max():.2f}; "
          f"N_eff (eigenvalue) = {neff:.1f}")

    # ---- spearman: latitude vs snow-free H-G -------------------------------
    hg = {}
    with open("results/dynamic_tables.csv") as fh:
        for r in csv.DictReader(fh):
            if r["record"] == "human_minus_goes" and r["threshold_cm"] == "1.0":
                hg[r["site"]] = float(r["delta_pp"])
    slugs = [slug for slug in SITES if slug in hg]
    lats = [SITES[s].lat for s in slugs]
    vals = [hg[s] for s in slugs]
    rank = lambda v: np.argsort(np.argsort(v)) + 1
    rho = float(np.corrcoef(rank(lats), rank(vals))[0, 1])
    t = rho * math.sqrt((len(slugs) - 2) / (1 - rho ** 2))
    # two-sided p via Student-t survival (series approximation adequate at df=6)
    from statistics import NormalDist
    df = len(slugs) - 2
    x = df / (df + t * t)
    # regularized incomplete beta via continued fraction (numerical, no scipy)
    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-9, 1e-30
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, max(1.0 - qab * x / qap, FPMIN)
        d = 1.0 / d
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = max(1.0 + aa * d, FPMIN); c = max(1.0 + aa / c, FPMIN)
            d = 1.0 / d; h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = max(1.0 + aa * d, FPMIN); c = max(1.0 + aa / c, FPMIN)
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < EPS:
                break
        return h
    def ibeta(a, b, x):
        if x in (0.0, 1.0):
            return x
        ln = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
              + a * math.log(x) + b * math.log(1 - x))
        if x < (a + 1) / (a + b + 2):
            return math.exp(ln) * betacf(a, b, x) / a
        return 1.0 - math.exp(math.lgamma(a + b) - math.lgamma(b) - math.lgamma(a)
                              + b * math.log(1 - x) + a * math.log(x)) * betacf(b, a, 1 - x) / b
    p = ibeta(df / 2.0, 0.5, x)
    emit("spearman", "rho_lat_vs_HG", round(rho, 2))
    emit("spearman", "p_two_sided", round(p, 2))
    print(f"\n== Spearman latitude vs H-G: rho={rho:+.2f}, p={p:.2f} (t-approx, df={df})")

    # ---- siteyears: 48 site-year deltas + pair DoD slopes -------------------
    print("\n== per-year human deltas ==")
    n_nonpos = 0
    for slug, site in SITES.items():
        st = site.human
        for y in range(YEARS[0], YEARS[1] + 1):
            nf = [v for d, v in NU[st].items() if d.year == y]
            df_ = [v for d, v in DU[st].items() if d.year == y]
            dl = 100 * (np.mean(nf) - np.mean(df_))
            emit("siteyears", f"{st}_{y}", round(dl, 1))
            n_nonpos += dl <= 0
    emit("siteyears", "n_nonpositive_of_48", n_nonpos)
    print(f"  non-positive site-years: {n_nonpos}/48")
    for slug, site in pairs.items():
        ds = []
        for y in range(YEARS[0], YEARS[1] + 1):
            def yd(st):
                return 100 * (np.mean([v for d, v in NU[st].items() if d.year == y])
                              - np.mean([v for d, v in DU[st].items() if d.year == y]))
            ds.append(yd(site.human) - yd(site.auto))
        slope = float(np.polyfit(range(len(ds)), ds, 1)[0])
        emit("siteyears", f"dod_slope_{slug}", round(slope, 2))
        print(f"  {slug:10s} DoD by year " + " ".join(f"{x:+.1f}" for x in ds)
              + f"  slope {slope:+.2f} pp/yr")

    # ---- diurnal ------------------------------------------------------------
    print("\n== diurnal (local clock hour, study parser) ==")
    for slug, site in pairs.items():
        tz = ZoneInfo(site.iana_tz)
        for st in (site.human, site.auto):
            by = defaultdict(list)
            for o in get_obs(st):
                by[o.timestamp.astimezone(tz).hour].append(o.coverage_okta <= 2)
            cur = {h: 100 * float(np.mean(v)) for h, v in by.items()}
            predawn = float(np.mean([cur[h] for h in (2, 3, 4, 5)]))
            aft_min = min(cur[h] for h in range(12, 19))
            vals = np.array([cur[h] for h in range(24)])
            emit("diurnal", f"{st}_predawn_minus_aftmin_pp", round(predawn - aft_min, 1))
            emit("diurnal", f"{st}_max_dev_from_mean_pp",
                 round(max(vals.max() - vals.mean(), vals.mean() - vals.min()), 1))
            print(f"  {st}: predawn(02-05) {predawn:.1f}, afternoon min {aft_min:.1f}, "
                  f"excess {predawn-aft_min:+.1f}; max dev from mean "
                  f"{max(vals.max()-vals.mean(), vals.mean()-vals.min()):.1f}")

    # ---- layers above 12,000 ft ---------------------------------------------
    print("\n== reports with a layer based above 12,000 ft ==")
    for slug, site in SITES.items():
        for st in site.stations:
            obs = get_obs(st)
            share = 100 * sum(1 for o in obs if any(b > 12000 for _, b in o.layers)) / len(obs)
            emit("layers", f"{st}_pct_above_12kft", round(share, 1))
            print(f"  {st}: {share:.1f}%")

    # ---- VIIRS own night-day delta (snow-free, unit-weighted, box) ----------
    print("\n== VIIRS own night-day clear delta (snow-free, unit-weighted) ==")
    for slug, site in SITES.items():
        p_snow = Path("data/snow_cache") / f"snow_{slug}.json"
        d = json.loads(p_snow.read_text())["daily"]
        snow = {t for t, v in zip(d["time"], d["snow_depth_max"])
                if v is not None and v > 0.01}
        tz = ZoneInfo(site.iana_tz)
        uniq = {}
        with open(f"data/viirs_matches/viirs_matches_{slug}.csv") as fh:
            for r in csv.DictReader(fh):
                uniq.setdefault((r["ts_utc"], r["platform"]), r)
        per = {"NIGHT": defaultdict(list), "DAY": defaultdict(list)}
        for r in uniq.values():
            if r["box_clear"] not in ("0", "1"):
                continue
            ts = dt.datetime.fromisoformat(r["ts_utc"]).astimezone(tz)
            u = (ts - dt.timedelta(hours=12)).date() if r["regime"] == "NIGHT" else ts.date()
            if str(u) in snow:
                continue
            per[r["regime"]][u].append(int(r["box_clear"]))
        delta = 100 * (np.mean([np.mean(v) for v in per["NIGHT"].values()])
                       - np.mean([np.mean(v) for v in per["DAY"].values()]))
        emit("viirsdelta", f"{slug}_night_minus_day_pp", round(delta, 1))
        print(f"  {slug:10s} {delta:+.1f} pp")

    # ---- erashare: era balance of the GOES draws -----------------------------
    # Draws are deduped by timestamp at the human station's pixel of the site's
    # primary family; the era cut is the cloud-mask algorithm change
    # (2021-11-29 1900 UTC, as in dynamic_tables.py); snow-free keys the draw to
    # its night/day unit date and drops units over > 1 cm ERA5-Land snow.
    print("\n== era season-regime share balance (baseline vs enterprise) ==")
    era_cut = dt.datetime(2021, 11, 29, 19, 0, tzinfo=dt.timezone.utc)
    worst = {"all": 0.0, "snowfree": 0.0}
    for slug, site in SITES.items():
        family = "west" if slug == "vancouver" else "east"
        d = json.loads((Path("data/snow_cache") / f"snow_{slug}.json").read_text())["daily"]
        snow = {t for t, v in zip(d["time"], d["snow_depth_max"])
                if v is not None and v > 0.01}
        tz = ZoneInfo(site.iana_tz)
        uniq = {}
        with open(f"data/goes_draws/goes_draws_{slug}_{family}.csv") as fh:
            for r in csv.DictReader(fh):
                if r["station"] == site.human:
                    uniq.setdefault(r["ts_utc"], r)
        for tag in ("all", "snowfree"):
            counts = {0: defaultdict(int), 1: defaultdict(int)}
            for r in uniq.values():
                ts_utc = dt.datetime.fromisoformat(r["ts_utc"])
                if tag == "snowfree":
                    ts = ts_utc.astimezone(tz)
                    u = ((ts - dt.timedelta(hours=12)).date()
                         if r["regime"] == "NIGHT" else ts.date())
                    if str(u) in snow:
                        continue
                counts[int(ts_utc >= era_cut)][(r["season"], r["regime"])] += 1
            tot = {e: sum(c.values()) for e, c in counts.items()}
            cells = set(counts[0]) | set(counts[1])
            gap = 100 * max(abs(counts[0][c] / tot[0] - counts[1][c] / tot[1])
                            for c in cells)
            emit("erashare", f"{slug}_max_cell_gap_pp_{tag}", round(gap, 1))
            worst[tag] = max(worst[tag], gap)
        print(f"  {slug:10s} all {worst['all']:.1f} (running max), "
              f"snow-free {worst['snowfree']:.1f}")
    emit("erashare", "max_over_sites_all", round(worst["all"], 1))
    emit("erashare", "max_over_sites_snowfree", round(worst["snowfree"], 1))
    print(f"  max over sites: all {worst['all']:.1f} pp, "
          f"snow-free {worst['snowfree']:.1f} pp")

    Path(args.csv).parent.mkdir(exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "key", "value"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nwrote {args.csv} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
