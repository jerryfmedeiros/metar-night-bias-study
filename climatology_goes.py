"""
climatology_goes.py — Validate METAR results using satellite (GOES ABI) data.

Estimates day/night clear-sky frequency from the GOES-East ABI Clear-Sky Mask
(product ACMC → variable BCM) by random sampling, without downloading the whole
archive.

Two methodological fixes vs. the original spot-check:

1. Date-aware satellite selection. GOES-19 became the operational GOES-East on
   2025-04-04; GOES-16 stops producing operational CONUS ACMC around then. The
   original script pinned the GOES-16 bucket and silently skipped every date with
   no S3 listing, so a "random sample across 2025" actually only drew from
   Jan–early Apr 2025 (winter/early spring) and never observed summer. We now pick
   the bucket per-timestamp so sampling can span the full study period, and we
   report per-season exposure so the truncation is visible if it recurs.

2. Reproducibility & uncertainty. Sampling is seeded (--seed) and every reported
   proportion carries a Wilson 95% confidence interval.

NOTE: the ABI Clear-Sky Mask is itself day/night-dependent (reflectance tests by
day, IR-only at night) and Calgary sits at a steep slant angle from GOES-East
(~75degW), so this is an independent *imperfect* estimate, not bias-free ground
truth. Interpret accordingly.

Reports annual AND per-season day/night clear-sky with Wilson 95% CIs and the
night-day difference (with CI) — so the satellite can be tested season-by-season
against the human record (e.g. does it show ANY nighttime clearing in summer?).

Usage:
  python climatology_goes.py --years 2020 2025 --samples 100 --seed 42
  # big overnight run, disk-light (download -> sample -> delete each file):
  python climatology_goes.py --years 2020 2025 --samples 3000 --discard
"""
import argparse
import datetime as dt
import json
import math
import sys
import random
from collections import defaultdict
from pathlib import Path

# Reuse functions from fetch_goes.py and metar_climatology.py
try:
    from fetch_goes import find_nearest_file, download_cached, sample_product, sample_product_box
    from metar_climatology import calculate_sun_alt, get_solar_regime, get_season
except ImportError:
    sys.exit("Error: Could not import fetch_goes.py or metar_climatology.py.")

import fetch_goes  # mutated per-timestamp to switch satellites

# GOES family cutovers: (operational-handover UTC date, bucket before, bucket after).
#   East: GOES-16 -> GOES-19 on 2025-04-04 (already used by the published Calgary run).
#   West: GOES-17 -> GOES-18 on 2023-01-04 (GOES-18 declared operational GOES-West).
# Family is a per-site choice (geometry): West for Pacific/BC, East elsewhere; Vancouver
# samples both as a cross-check (East is steeply oblique there). Known minor edge cases:
# GOES-18's Aug-Nov 2022 interleave and GOES-17 ABI cooling degradation (see paper limits).
GOES_FAMILIES = {
    "east": (dt.datetime(2025, 4, 4, tzinfo=dt.timezone.utc), "noaa-goes16", "noaa-goes19"),
    "west": (dt.datetime(2023, 1, 4, tzinfo=dt.timezone.utc), "noaa-goes17", "noaa-goes18"),
}


def select_goes_bucket(ts: dt.datetime, family: str = "east") -> str:
    """Point fetch_goes at whichever satellite was operational for `family` at ts."""
    cutover, before, after = GOES_FAMILIES[family]
    bucket = after if ts >= cutover else before
    fetch_goes.GOES_BUCKET = bucket
    fetch_goes.S3_HTTPS_BASE = f"https://{bucket}.s3.amazonaws.com"
    return bucket


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; returns (lo, hi) in %."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def delta_ci(cd: int, nd: int, cn: int, nn: int, z: float = 1.96):
    """95% CI for the (night - day) clear-sky proportion difference, in pp."""
    pd, pn = cd / nd, cn / nn
    se = math.sqrt(pd * (1 - pd) / nd + pn * (1 - pn) / nn)
    d = (pn - pd) * 100
    return d, d - z * se * 100, d + z * se * 100


def _new_stats():
    return defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))


def save_checkpoint(path, years, target, stats, samples_found, attempts):
    """Atomically persist accumulated tallies so a killed run can resume."""
    plain = {st: {rg: {se: dict(d) for se, d in rgd.items()}
                  for rg, rgd in std.items()} for st, std in stats.items()}
    payload = {"years": list(years), "target": target, "stats": plain,
               "samples_found": samples_found, "attempts": attempts}
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def load_checkpoint(path, years, target):
    """Return (stats, samples_found, attempts) from a matching checkpoint, else fresh."""
    p = Path(path)
    if not p.exists():
        return _new_stats(), {"DAY": 0, "NIGHT": 0}, 0
    data = json.loads(p.read_text())
    if data.get("years") != list(years) or data.get("target") != target:
        print(f"  Checkpoint {path} is for different years/target — ignoring it.")
        return _new_stats(), {"DAY": 0, "NIGHT": 0}, 0
    stats = _new_stats()
    for st, std in data["stats"].items():
        for rg, rgd in std.items():
            for se, d in rgd.items():
                stats[st][rg][se].update({k: int(v) for k, v in d.items()})
    sf = {"DAY": int(data["samples_found"]["DAY"]), "NIGHT": int(data["samples_found"]["NIGHT"])}
    print(f"  Resuming from {path}: DAY={sf['DAY']}, NIGHT={sf['NIGHT']}")
    return stats, sf, int(data.get("attempts", 0))


SEASONS = ["WINTER", "SPRING", "SUMMER", "AUTUMN"]


def checkpoint_path(slug: str, family: str, prefix: str = "goes_progress") -> str:
    """Per-(site, family) checkpoint file. Calgary/east keeps the legacy filename so the
    published 5000+5000-sample run is reused verbatim (reproducibility)."""
    if slug == "calgary" and family == "east":
        return f"{prefix}.json"
    return f"{prefix}_{slug}_{family}.json"


def _agg(st, station, regime, season=None):
    """(clear, total) for a stats structure / station / regime, optionally one season."""
    cl = to = 0
    for s in ([season] if season else SEASONS):
        d = st[station][regime][s]
        cl += d["CLEAR"]; to += d["TOTAL"]
    return cl, to


def report(st, tag, stations):
    # '*' marks a day-night delta whose 95% CI excludes zero (significant).
    for station in stations:
        print(f"\n[GOES CLEAR-SKY{tag}] {station} — clear% [95% CI] (N); '*' = delta CI excludes 0")
        print(f"{'Bucket':<7} | {'Day clear':>21} | {'Night clear':>21} | {'Night-Day (pp)':>20}")
        print("-" * 78)
        for label, season in [("ANNUAL", None)] + [(s.title(), s) for s in SEASONS]:
            cd, nd = _agg(st, station, "DAY", season)
            cn, nn = _agg(st, station, "NIGHT", season)
            if nd == 0 or nn == 0:
                continue
            ld, hd = wilson_ci(cd, nd)
            ln, hn = wilson_ci(cn, nn)
            dd, dlo, dhi = delta_ci(cd, nd, cn, nn)
            sig = "*" if not (dlo <= 0 <= dhi) else " "
            print(f"{label:<7} | {cd/nd*100:5.1f} [{ld:4.1f},{hd:4.1f}] {nd:5d} | "
                  f"{cn/nn*100:5.1f} [{ln:4.1f},{hn:4.1f}] {nn:5d} | "
                  f"{dd:+5.1f} [{dlo:+5.1f},{dhi:+5.1f}]{sig}")


def sample_family(site, family, years, target, seed, *, max_attempts=2_000_000,
                  discard=False, checkpoint=None, checkpoint_every=100,
                  box=False, box_thresh=0.5, months=None, log_path=None):
    """Random-sample the GOES clear-sky mask for one site and one satellite family.

    Samples every station pixel in `site.coords` per draw (so they share one sample
    set), switching satellites per-timestamp via the family cutover. Resumable via a
    per-(site, family) checkpoint. With `log_path`, every accepted draw is appended to
    a CSV (timestamp, station, nearest-pixel class, 3x3/5x5 box means) so uncertainty
    can be computed on night units downstream instead of assuming independent draws.
    Returns (stats, stats_box3, stats_box5, samples_found, attempts, buckets_used)."""
    y_start, y_end = years
    checkpoint = checkpoint or checkpoint_path(site.slug, family)
    listing_cache: dict = {}
    stats, samples_found, attempts = load_checkpoint(checkpoint, (y_start, y_end), target)
    stats_box3, stats_box5 = _new_stats(), _new_stats()   # only populated if box; not checkpointed
    buckets_used: dict = defaultdict(int)
    # Re-seed off progress so a resumed run draws a fresh (non-repeating) sequence.
    random.seed(seed + samples_found["DAY"] + samples_found["NIGHT"])
    span_days = (dt.date(y_end, 12, 31) - dt.date(y_start, 1, 1)).days
    since_ckpt = 0

    while (samples_found["DAY"] < target or samples_found["NIGHT"] < target) \
            and attempts < max_attempts:
        attempts += 1
        ts = dt.datetime(y_start, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
            days=random.randint(0, span_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        if months and ts.month not in months:
            continue
        sun_alt = calculate_sun_alt(ts, site.lat, site.lon, site.elevation_m)
        regime = get_solar_regime(sun_alt)
        if regime not in samples_found or samples_found[regime] >= target:
            continue

        bucket = select_goes_bucket(ts, family)
        result = find_nearest_file(getattr(site, "goes_product", "ACMC"), ts, listing_cache)
        if not result:
            continue  # no operational scan for this satellite/time — skip
        key, file_dt = result

        try:
            local = download_cached(key)
        except Exception:
            continue

        # Require a valid pixel at every station so all stations share one sample set
        values, box3, box5, box_raw = {}, {}, {}, {}
        ok = True
        for station, (lat, lon) in site.coords.items():
            val = sample_product(local, "BCM", lat, lon)
            if val is None:
                ok = False
                break
            values[station] = "CLEAR" if val == 0 else "CLOUDY"
            if box:
                b3 = sample_product_box(local, "BCM", lat, lon, half=1)
                b5 = sample_product_box(local, "BCM", lat, lon, half=2)
                box_raw[station] = (b3, b5)
                box3[station] = "CLEAR" if (b3 is not None and b3 <= box_thresh) else "CLOUDY"
                box5[station] = "CLEAR" if (b5 is not None and b5 <= box_thresh) else "CLOUDY"
        if discard:
            try:
                local.unlink()
            except OSError:
                pass
        if not ok:
            continue

        season = get_season(ts.month)
        samples_found[regime] += 1
        since_ckpt += 1
        buckets_used[bucket] += 1
        for station, label in values.items():
            d = stats[station][regime][season]
            d[label] += 1
            d["TOTAL"] += 1
            if box:
                for stx, lab in ((stats_box3, box3[station]), (stats_box5, box5[station])):
                    db = stx[station][regime][season]
                    db[lab] += 1
                    db["TOTAL"] += 1
        if log_path:
            lp = Path(log_path)
            write_header = not lp.exists()
            with lp.open("a") as lf:
                if write_header:
                    lf.write("ts_utc,family,bucket,regime,season,station,"
                             "nearest_clear,box3_mean_bcm,box5_mean_bcm\n")
                for station, label in values.items():
                    b3, b5 = box_raw.get(station, (None, None))
                    fmt = lambda v: "" if v is None else f"{v:.4f}"
                    lf.write(f"{ts.isoformat()},{family},{bucket},{regime},{season},"
                             f"{station},{int(label == 'CLEAR')},{fmt(b3)},{fmt(b5)}\n")

        if since_ckpt >= checkpoint_every:
            save_checkpoint(checkpoint, (y_start, y_end), target, stats, samples_found, attempts)
            since_ckpt = 0
        if sum(samples_found.values()) % 50 == 0:
            print(f"  [{site.slug}/{family}] DAY={samples_found['DAY']}/{target}, "
                  f"NIGHT={samples_found['NIGHT']}/{target} (attempt {attempts})")

    save_checkpoint(checkpoint, (y_start, y_end), target, stats, samples_found, attempts)
    return stats, stats_box3, stats_box5, samples_found, attempts, buckets_used


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs=2, type=int, default=[2020, 2025],
                        help="Start and end year inclusive")
    parser.add_argument("--samples", type=int, default=100,
                        help="Target samples per regime (DAY and NIGHT); e.g. 3000 for an overnight run")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible sampling")
    parser.add_argument("--max-attempts", type=int, default=2_000_000,
                        help="Safety cap on random draws (target stops earlier)")
    parser.add_argument("--discard", action="store_true",
                        help="Delete each GOES file after sampling (keeps disk tiny for big runs)")
    parser.add_argument("--checkpoint", default=None,
                        help="Explicit resume file (default: per-site/family auto name)")
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Write the checkpoint every N new samples")
    parser.add_argument("--box", action="store_true",
                        help="Also sample 3x3 and 5x5 pixel boxes (parallax/all-sky sensitivity test)")
    parser.add_argument("--box-thresh", type=float, default=0.5,
                        help="Box mean BCM at/below this counts as CLEAR")
    parser.add_argument("--months", type=int, nargs="+", default=None,
                        help="Restrict sampling to these calendar months (e.g. 6 7 8 9 10 11 for snow-free)")
    parser.add_argument("--log-draws", action="store_true",
                        help="Log every accepted draw to goes_draws_<site>_<family>.csv (implies "
                             "--box) and use a separate 'goes_progress_logged*' checkpoint so the "
                             "legacy aggregate checkpoints are untouched")
    parser.add_argument("--site", default="calgary", help="site slug from sites.py")
    parser.add_argument("--families", nargs="+", default=None, choices=list(GOES_FAMILIES),
                        help="override the site's GOES families (default: site.goes)")
    args = parser.parse_args()

    from sites import get_site
    site = get_site(args.site)
    families = args.families if args.families else list(site.goes)
    y_start, y_end = args.years

    print(f"GOES Clear-Sky Mask validation {y_start}-{y_end} — {site.name}")
    print(f"Families: {', '.join(families)}; up to {args.samples} samples/regime (seed={args.seed})")
    print("-" * 80)

    family_stats = {}
    for family in families:
        cutover, before, after = GOES_FAMILIES[family]
        print(f"\n#### GOES-{family.upper()} ({before} before {cutover.date()}, {after} on/after) ####")
        # An explicit --checkpoint only applies when a single family is requested.
        ckpt = args.checkpoint if (args.checkpoint and len(families) == 1) else None
        log_path = None
        if args.log_draws:
            args.box = True
            log_path = f"goes_draws_{site.slug}_{family}.csv"
            if ckpt is None:
                ckpt = checkpoint_path(site.slug, family, prefix="goes_progress_logged")
            print(f"  Logging draws to {log_path} (checkpoint {ckpt})")
        stats, box3, box5, sf, attempts, buckets = sample_family(
            site, family, (y_start, y_end), args.samples, args.seed,
            max_attempts=args.max_attempts, discard=args.discard, checkpoint=ckpt,
            checkpoint_every=args.checkpoint_every, box=args.box, box_thresh=args.box_thresh,
            months=set(args.months) if args.months else None, log_path=log_path)
        family_stats[family] = stats

        print(f"\nCollected after {attempts} attempts: DAY={sf['DAY']}, NIGHT={sf['NIGHT']}")
        if sf["DAY"] < args.samples or sf["NIGHT"] < args.samples:
            print("  WARNING: target not reached — likely limited available scans.")
        print(f"  buckets used: {dict(buckets)}")

        ref = site.human  # reference station for the exposure summary
        print("\n[SEASONAL EXPOSURE] samples per regime/season")
        print(f"{'Regime':<8} | " + " | ".join(f"{s:>7}" for s in SEASONS))
        print("-" * 48)
        for regime in ["DAY", "NIGHT"]:
            print(f"{regime:<8} | " + " | ".join(f"{_agg(stats, ref, regime, s)[1]:7d}" for s in SEASONS))

        report(stats, f" (nearest pixel, {family})", site.coords)
        if args.box:
            report(box3, f" (3x3 box {family}, clear<=BCM {args.box_thresh})", site.coords)
            report(box5, f" (5x5 box {family}, clear<=BCM {args.box_thresh})", site.coords)

    # East-vs-West cross-check (Vancouver etc.): the human station's annual clear% per family.
    if len(families) > 1:
        ref = site.human
        print(f"\n[EAST vs WEST] {site.name} — {ref} annual clear% by family")
        print(f"{'Family':<7} | {'Day clear':>11} | {'Night clear':>11} | {'Night-Day pp':>12}")
        print("-" * 50)
        for family in families:
            st = family_stats[family]
            cd, nd = _agg(st, ref, "DAY")
            cn, nn = _agg(st, ref, "NIGHT")
            if nd == 0 or nn == 0:
                continue
            print(f"{family:<7} | {cd/nd*100:10.1f}% | {cn/nn*100:10.1f}% | {(cn/nn-cd/nd)*100:+11.1f}")


if __name__ == "__main__":
    main()
