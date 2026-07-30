"""climatology_viirs.py — VIIRS MVCM cloud mask vs the human record, per site.

The polar-orbiting counterpart to climatology_goes.py, but with one twist on the
comparison. VIIRS only sees a site at its ~13:30 and ~01:30 overpass times, so a raw
"day vs night" VIIRS delta is really an afternoon-vs-post-midnight snapshot — it captures
the diurnal convective cycle, not the all-hours average GOES/METAR use, and is therefore
NOT apples-to-apples with them. We keep that aggregate (clearly caveated) but the real
test is the **per-moment collocation (gold standard)**: for every VIIRS sample we pull
the nearest human METAR (+/-30 min) and compare the *same scene*. The night contingency
(human=clear while VIIRS=cloud, vs the reverse) and its McNemar test are the most direct
possible evidence of the human over-reporting nighttime clarity, free of sampling-time
confounds. Night samples are also split moonlit (DNB-capable) vs dark (IR-fallback).

Reuses the GOES stats/report/checkpoint helpers; METAR is read from the local cache
(free), so collocation adds no downloads beyond the VIIRS granules themselves.

Usage:
  python climatology_viirs.py --site calgary --samples 1500 --seed 42 --discard
"""
import argparse
import bisect
import datetime as dt
import math
import random
import sys
from pathlib import Path
from collections import defaultdict

import ephem

try:
    from fetch_viirs import (cmr_granules, download_cached, sample_granule,
                             sample_granule_box, earthdata_token, SHORT_NAME)
    from fetch_metar import fetch_metar_csv
    from climatology_goes import (wilson_ci, _new_stats, save_checkpoint,
                                  load_checkpoint, report, SEASONS, _agg)
    from metar_climatology import calculate_sun_alt, get_solar_regime, get_season
except ImportError as e:
    sys.exit(f"Error importing project modules: {e}")

USABLE_OKTA = 2                # human "clear/usable" = SKC+FEW, matching the rest of the study
MATCH_WINDOW_S = 30 * 60       # collocate a VIIRS sample with a METAR within +/-30 min
MOONLIT_MIN_ALT_DEG = 0.0
MOONLIT_MIN_ILLUM_PCT = 25.0


def moon_is_up(ts, lat, lon, elev) -> bool:
    """Moon above horizon AND >=25% illuminated (i.e. the DNB has light to work with)."""
    o = ephem.Observer()
    o.lat, o.lon, o.elevation = str(lat), str(lon), elev
    o.date = ephem.Date(ts)
    m = ephem.Moon(o)
    return math.degrees(m.alt) > MOONLIT_MIN_ALT_DEG and m.phase >= MOONLIT_MIN_ILLUM_PCT


def load_human_metar(station, y0, y1):
    """Sorted (epoch_seconds[], usable_bool[]) for the human station, from the cache."""
    pairs = []
    for year in range(y0, y1 + 1):
        try:
            for o in fetch_metar_csv(station, dt.date(year, 1, 1), dt.date(year, 12, 31)):
                pairs.append((o.timestamp.timestamp(), o.coverage_okta <= USABLE_OKTA))
        except Exception as e:
            print(f"  METAR {station} {year}: {e}")
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


def nearest_usable(epochs, usable, t, window_s=MATCH_WINDOW_S):
    """(human usable?, offset seconds) for the METAR nearest to t; None if none close."""
    if not epochs:
        return None
    te = t.timestamp()
    i = bisect.bisect_left(epochs, te)
    best, bestd = None, None
    for j in (i - 1, i):
        if 0 <= j < len(epochs):
            d = abs(epochs[j] - te)
            if bestd is None or d < bestd:
                bestd, best = d, j
    if best is None or bestd > window_s:
        return None
    return usable[best], bestd


def _mcnemar_z(b, c):
    """Continuity-corrected McNemar z for discordant pairs b, c (None if too few)."""
    if b + c == 0:
        return None
    return (abs(b - c) - 1) / math.sqrt(b + c)


def matched_report(stats, MATCH, title, dod_label):
    """Print a same-scene human-vs-VIIRS comparison for one footprint (pixel or box)."""
    def mc(h, v, regime):
        return _agg(stats, MATCH[(h, v)], regime)[1]   # TOTAL in that 2x2 cell

    print(f"\n[VIIRS MATCHED — {title}]")
    print(f"{'regime':<7} | {'N':>5} | {'human clear%':>12} | {'VIIRS clear%':>12} | {'human-VIIRS pp':>14}")
    print("-" * 64)
    rates = {}
    for regime in ("DAY", "NIGHT"):
        cc, cx, xc, xx = (mc("c", "c", regime), mc("c", "x", regime),
                          mc("x", "c", regime), mc("x", "x", regime))
        n = cc + cx + xc + xx
        if n == 0:
            print(f"{regime:<7} | {0:>5} | {'(no matches)':>12} |")
            continue
        h_rate, v_rate = (cc + cx) / n * 100, (cc + xc) / n * 100
        rates[regime] = (h_rate, v_rate, n, cx, xc)
        print(f"{regime:<7} | {n:5d} | {h_rate:11.1f}% | {v_rate:11.1f}% | {h_rate-v_rate:+14.1f}")

    if "NIGHT" in rates:
        _h, _v, _n, cx_n, xc_n = rates["NIGHT"]
        z = _mcnemar_z(cx_n, xc_n)
        print(f"  night contingency: human-clear/VIIRS-cloud={cx_n}  vs  "
              f"human-cloud/VIIRS-clear={xc_n}")
        if z is not None:
            p = math.erfc(z / math.sqrt(2))
            verdict = ("human over-reports clarity" if cx_n > xc_n else "no over-reporting")
            print(f"  McNemar z={z:.2f}, p={p:.4g} ({verdict})")
    if "NIGHT" in rates and "DAY" in rates:
        h_n, v_n, n_n = rates["NIGHT"][:3]
        h_d, v_d, n_d = rates["DAY"][:3]
        dod = (h_n - h_d) - (v_n - v_d)
        var = sum((r / 100) * (1 - r / 100) / n
                  for r, n in ((h_n, n_n), (h_d, n_d), (v_n, n_n), (v_d, n_d)))
        ci = 1.96 * math.sqrt(var) * 100
        print(f"  {dod_label}: {dod:+.1f} pp  95% CI [{dod-ci:+.1f}, {dod+ci:+.1f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2025])
    ap.add_argument("--site", default="calgary", help="site slug from sites.py")
    ap.add_argument("--samples", type=int, default=1500, help="target samples per regime")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--platforms", nargs="+", default=None, choices=list(SHORT_NAME),
                    help="override the site's VIIRS platforms")
    ap.add_argument("--max-attempts", type=int, default=200_000)
    ap.add_argument("--discard", action="store_true",
                    help="delete each granule after sampling")
    ap.add_argument("--box-half", type=int, default=3,
                    help="half-width of the whole-sky box in pixels ((2h+1)^2; 3 -> 7x7 ~5km)")
    ap.add_argument("--box-thresh", type=float, default=0.25,
                    help="box cloudy-fraction at/below this counts as CLEAR (~okta<=2)")
    ap.add_argument("--log-matches", action="store_true",
                    help="log every accepted overpass to viirs_matches_<site>.csv and use a "
                         "separate viirs_progress_logged_<site>.json checkpoint")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    args = ap.parse_args()

    from sites import get_site
    site = get_site(args.site)
    platforms = args.platforms or list(getattr(site, "viirs_platforms", tuple(SHORT_NAME)))
    # Sample at the human station itself (site.lat/lon is a site reference point that
    # can differ from the station; at Calgary it is 8 km away).
    lat, lon = site.coords[site.human]
    elev = site.elevation_m
    # Pixel sampling uses the human station's own coordinates (matches climatology_goes.py's
    # site.coords convention), not the site reference point used for sun/moon geometry above.
    # These coincide everywhere except Calgary, whose site.lat/lon is the original pilot
    # study's home coordinate, ~7.5 km from CYYC.
    sample_lat, sample_lon = site.coords[site.human]
    y0, y1 = args.years
    target = args.samples
    if args.log_matches:
        checkpoint = args.checkpoint or f"viirs_progress_logged_{site.slug}.json"
        log_path = Path(f"viirs_matches_{site.slug}.csv")
        print(f"Logging matches to {log_path} (checkpoint {checkpoint})")
    else:
        checkpoint = args.checkpoint or f"viirs_progress_{site.slug}.json"
        log_path = None
    token = earthdata_token()

    print(f"VIIRS MVCM vs human {y0}-{y1} — {site.name} (human station {site.human})")
    print(f"Platforms: {', '.join(platforms)}; up to {target} samples/regime (seed={args.seed})")
    print("-" * 80)

    print("Loading human METAR for collocation ...")
    h_epochs, h_usable = load_human_metar(site.human, y0, y1)
    print(f"  {len(h_epochs)} {site.human} reports loaded")

    # GOES-shaped stats tree. Stations: MAIN (aggregate), MOONLIT/DARK (night DNB split),
    # and four matched 2x2 pseudo-stations m_HV (H/V in {c=clear, x=cloud}) holding the
    # per-moment human-vs-VIIRS contingency. All persist via the shared checkpoint.
    MAIN, MOONLIT, DARK = site.slug, f"{site.slug}/moonlit", f"{site.slug}/dark"
    MAIN_BOX = f"{site.slug}/box"
    MATCH = {(h, v): f"{site.slug}/m_{h}{v}" for h in "cx" for v in "cx"}     # single pixel
    MATCHB = {(h, v): f"{site.slug}/mb_{h}{v}" for h in "cx" for v in "cx"}   # whole-sky box
    stats, samples_found, attempts = load_checkpoint(checkpoint, (y0, y1), target)
    cmr_cache: dict = {}
    used: set = set()
    random.seed(args.seed + samples_found["DAY"] + samples_found["NIGHT"])
    span_days = (dt.date(y1, 12, 31) - dt.date(y0, 1, 1)).days
    since_ckpt = n_dl = n_matched = 0

    while (samples_found["DAY"] < target or samples_found["NIGHT"] < target) \
            and attempts < args.max_attempts:
        attempts += 1
        day = dt.datetime(y0, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
            days=random.randint(0, span_days))
        for platform in platforms:
            grans = cmr_granules(platform, sample_lat, sample_lon, day, day + dt.timedelta(days=1), cmr_cache)
            for url, t in grans:
                if url in used:
                    continue
                regime = get_solar_regime(calculate_sun_alt(t, lat, lon, elev))
                if regime not in samples_found or samples_found[regime] >= target:
                    continue
                try:
                    local = download_cached(url, token)
                    n_dl += 1
                except Exception as ex:
                    print(f"  download failed: {ex}")
                    continue
                res = sample_granule(local, sample_lat, sample_lon)
                resb = sample_granule_box(local, sample_lat, sample_lon, args.box_half, args.box_thresh)
                if args.discard:
                    try:
                        local.unlink()
                    except OSError:
                        pass
                if res is None:
                    continue
                label, _dist = res
                box_label = resb[0] if resb else None
                used.add(url)
                season = get_season(t.month)
                samples_found[regime] += 1
                since_ckpt += 1
                moonlit = moon_is_up(t, lat, lon, elev) if regime == "NIGHT" else None
                for st in [MAIN] + ([MOONLIT if moonlit else DARK]
                                    if regime == "NIGHT" else []):
                    d = stats[st][regime][season]
                    d[label] += 1
                    d["TOTAL"] += 1
                if box_label:
                    db = stats[MAIN_BOX][regime][season]
                    db[box_label] += 1
                    db["TOTAL"] += 1
                # Gold-standard collocation: nearest human METAR to this exact scene.
                match = nearest_usable(h_epochs, h_usable, t)
                if match is not None:
                    hu, hdt = match
                    H = "c" if hu else "x"
                    stats[MATCH[(H, "c" if label == "CLEAR" else "x")]][regime][season]["TOTAL"] += 1
                    if box_label:
                        stats[MATCHB[(H, "c" if box_label == "CLEAR" else "x")]][regime][season]["TOTAL"] += 1
                    n_matched += 1
                if log_path:
                    write_header = not log_path.exists()
                    with log_path.open("a") as lf:
                        if write_header:
                            lf.write("ts_utc,platform,regime,season,moonlit,"
                                     "pixel_clear,box_clear,human_usable,metar_offset_s,granule\n")
                        hu_s = "" if match is None else int(match[0])
                        dt_s = "" if match is None else int(match[1])
                        ml = "" if moonlit is None else int(moonlit)
                        lf.write(f"{t.isoformat()},{platform},{regime},{season},{ml},"
                                 f"{int(label == 'CLEAR')},"
                                 f"{'' if box_label is None else int(box_label == 'CLEAR')},"
                                 f"{hu_s},{dt_s},{url.rsplit('/', 1)[-1]}\n")
                if since_ckpt >= args.checkpoint_every:
                    save_checkpoint(checkpoint, (y0, y1), target, stats, samples_found, attempts)
                    since_ckpt = 0
                if sum(samples_found.values()) % 50 == 0:
                    print(f"  [{site.slug}] DAY={samples_found['DAY']}/{target}, "
                          f"NIGHT={samples_found['NIGHT']}/{target} "
                          f"(days {attempts}, dl {n_dl}, matched {n_matched})")
            if samples_found["DAY"] >= target and samples_found["NIGHT"] >= target:
                break

    save_checkpoint(checkpoint, (y0, y1), target, stats, samples_found, attempts)
    print(f"\nCollected: DAY={samples_found['DAY']}, NIGHT={samples_found['NIGHT']}, "
          f"matched to METAR={n_matched} ({n_dl} downloads, {attempts} day-draws)")

    # --- Aggregate VIIRS day/night (CAVEAT: overpass-time snapshot, not all-hours) ---
    print("\n  NOTE: the aggregate below samples only ~13:30/01:30 overpasses, so it")
    print("  reflects the diurnal cycle, NOT an all-hours day/night average. The matched")
    print("  comparison further down is the apples-to-apples artifact test.")
    report(stats, " (VIIRS aggregate, overpass-time)", [MAIN])

    # --- Night DNB split (same overpass times; isolates the moonlight benefit) ---
    cd, nd = _agg(stats, MAIN, "DAY")
    day_clear = cd / nd * 100 if nd else float("nan")
    print(f"\n[VIIRS NIGHT DNB SPLIT] {site.name} — night clear% by Moon illumination")
    print(f"{'regime':<16} | {'clear%':>18} (N) | {'vs day (pp)':>11}")
    print("-" * 56)
    print(f"{'DAY (ref)':<16} | {day_clear:5.1f} {'':12} {nd:5d} |")
    for tag, stn in (("night moonlit", MOONLIT), ("night dark", DARK), ("night all", MAIN)):
        cn, nn = _agg(stats, stn, "NIGHT")
        if nn == 0:
            print(f"{tag:<16} | {'(none)':>18}     0 |")
            continue
        lo, hi = wilson_ci(cn, nn)
        print(f"{tag:<16} | {cn/nn*100:5.1f} [{lo:4.1f},{hi:4.1f}] {nn:5d} | {cn/nn*100-day_clear:+11.1f}")

    # --- GOLD STANDARD: per-moment human-vs-VIIRS collocation, two footprints ---
    # Single pixel (point) vs a whole-sky box (cloud fraction) that better matches the
    # human's dome-wide okta. If the night artifact is real but washed out by the point
    # footprint, the box version should sharpen it (whole-sky VIIRS catches more cloud).
    print(f"\nSame scenes: METAR within +/-{MATCH_WINDOW_S//60} min of each VIIRS overpass.")
    matched_report(stats, MATCH, f"single pixel — {site.name}", "Matched diff-of-deltas")
    matched_report(stats, MATCHB,
                   f"whole-sky box {2*args.box_half+1}x{2*args.box_half+1}, "
                   f"clear<= {args.box_thresh:.0%} cloud — {site.name}",
                   "Matched diff-of-deltas (box)")


if __name__ == "__main__":
    main()
