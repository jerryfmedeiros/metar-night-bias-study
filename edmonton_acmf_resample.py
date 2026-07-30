"""
edmonton_acmf_resample.py — re-read every published Edmonton GOES draw from the
full-disk clear-sky mask (ABI-L2-ACMF) at correctly-navigated pixels.

Why: Edmonton (CYEG 53.31N, CZVL 53.67N) lies OUTSIDE the ABI CONUS sector in
both GOES families — the CONUS north edge crosses ~52.9N at Edmonton's
longitude — and the original sampler's nearest-pixel lookup clamped to the
sector edge, so every published Edmonton "GOES" value was measured 105-215 km
from the station (fetch_goes now raises OutOfSector instead of clamping). This
script re-reads the SAME published draw timestamps from the full-disk ACMF
product (same BCM field, same algorithm and 2021-11-29 Enterprise boundary,
10-min cadence instead of 5), so the night/day units, era split, and seasonal
shares of the published design are preserved exactly; only the pixel is
corrected.

Two stages:

  # 1. sample (slow, network, fully resumable; ~10,000 ACMF files per family
  #    at ~20-26 MB each, cache discarded per file unless --keep-cache)
  python3 edmonton_acmf_resample.py --family east
  python3 edmonton_acmf_resample.py --family west

  # 2. assemble (instant, offline): rebuild the draw log row-for-row from the
  #    sample cache, preserving the published file's row order and multiplicity
  python3 edmonton_acmf_resample.py --family east --assemble
  python3 edmonton_acmf_resample.py --family west --assemble

Outputs:
  data/goes_draws/acmf_resample_edmonton_<family>.csv   sample cache (stage 1)
  data/goes_draws/goes_draws_edmonton_<family>_acmf.csv corrected draw log (stage 2)

The corrected log keeps the standard 9-column schema plus a `pix_km` audit
column: the great-circle distance from the station to the centre of the pixel
actually read (should be <= ~3 km at this latitude; the run aborts if any
sampled pixel exceeds MAX_PIX_KM). Swap the corrected logs in for the
originals (archive, don't delete, the published ones) before regenerating
dynamic_tables.py / goes_nightly.py.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import netCDF4

import fetch_goes as fg
from sites import SITES

SITE = SITES["edmonton"]
PRODUCT = "ACMF"                      # full disk; see sites.py goes_product note
STATIONS = {"east": ["CYEG", "CZVL"],  # matches the stations in the published logs
            "west": ["CYEG"]}
MAX_PIX_KM = 5.0                      # sanity guard; ~2 km grid, oblique view


def _invert_goes_xy(xs: float, ys: float, pv: dict) -> tuple[float, float]:
    """Fixed-grid scan angles -> geodetic lat/lon (ABI PUG inverse navigation)."""
    req = pv["semi_major_axis"]
    rpol = pv["semi_minor_axis"]
    H = pv["perspective_point_height"] + req
    lam0 = math.radians(pv["longitude_of_projection_origin"])
    a = math.sin(xs) ** 2 + math.cos(xs) ** 2 * (
        math.cos(ys) ** 2 + (req ** 2 / rpol ** 2) * math.sin(ys) ** 2)
    b = -2.0 * H * math.cos(xs) * math.cos(ys)
    c = H * H - req * req
    rs = (-b - math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    sx = rs * math.cos(xs) * math.cos(ys)
    sy = -rs * math.sin(xs)
    sz = rs * math.cos(xs) * math.sin(ys)
    lat = math.degrees(math.atan((req ** 2 / rpol ** 2) * sz
                                 / math.sqrt((H - sx) ** 2 + sy ** 2)))
    lon = math.degrees(lam0 - math.atan(sy / (H - sx)))
    return lat, lon


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def sample_stations(path: Path, stations: list[str]):
    """Read BCM at each station's nearest full-disk pixel + 3x3/5x5 box means.

    Returns {station: (nearest_clear, box3_str, box5_str, pix_km)} or None if
    any station's centre pixel is fill/masked (mirroring the original rule that
    a draw counts only when every station pixel is valid)."""
    out = {}
    with netCDF4.Dataset(str(path)) as nc:
        gip = nc.variables["goes_imager_projection"]
        pv = {a: getattr(gip, a) for a in
              ("perspective_point_height", "semi_major_axis", "semi_minor_axis",
               "longitude_of_projection_origin")}
        xs = np.asarray(nc.variables["x"][:], dtype=float)
        ys = np.asarray(nc.variables["y"][:], dtype=float)
        var = nc.variables["BCM"]
        ny, nx = var.shape[-2:]
        for st in stations:
            lat, lon = SITE.coords[st]
            sat_x, sat_y = fg.latlon_to_goes_xy(lat, lon, pv)
            j, i = fg._grid_indices(xs, ys, sat_x, sat_y, lat, lon)
            plat, plon = _invert_goes_xy(xs[j], ys[i], pv)
            pix_km = _haversine_km(lat, lon, plat, plon)
            if pix_km > MAX_PIX_KM:
                raise RuntimeError(
                    f"{st}: sampled pixel {pix_km:.1f} km from station — navigation wrong")
            y0, y1 = max(0, i - 2), min(ny, i + 3)
            x0, x1 = max(0, j - 2), min(nx, j + 3)
            win = np.ma.masked_invalid(var[y0:y1, x0:x1]).astype(float)
            win = np.ma.masked_outside(win, 0, 1)     # BCM fill 255 -> masked
            centre = win[i - y0, j - x0]
            if win.count() == 0 or np.ma.is_masked(centre):
                return None
            b3 = win[max(0, i - 1 - y0):i + 2 - y0, max(0, j - 1 - x0):j + 2 - x0]
            nearest_clear = int(float(centre) == 0.0)
            box3 = f"{float(b3.mean()):.4f}" if b3.count() else ""
            box5 = f"{float(win.mean()):.4f}" if win.count() else ""
            out[st] = (nearest_clear, box3, box5, f"{pix_km:.2f}")
    return out


def load_source_rows(family: str) -> list[dict]:
    src = Path(f"data/goes_draws/goes_draws_edmonton_{family}.csv")
    with open(src) as fh:
        return list(csv.DictReader(fh))


def stage_sample(family: str, limit: int | None, keep_cache: bool) -> None:
    rows = load_source_rows(family)
    stations = STATIONS[family]
    # unique draw timestamps in first-appearance order, with their draw metadata
    draws, seen = [], set()
    for r in rows:
        if r["ts_utc"] in seen:
            continue
        seen.add(r["ts_utc"])
        draws.append((r["ts_utc"], r["bucket"]))

    out = Path(f"data/goes_draws/acmf_resample_edmonton_{family}.csv")
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {r["ts_utc"] for r in csv.DictReader(fh)}
    is_new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if is_new:
        w.writerow(["ts_utc", "station", "nearest_clear", "box3_mean_bcm",
                    "box5_mean_bcm", "pix_km"])

    todo = [d for d in draws if d[0] not in done]
    if limit:
        todo = todo[:limit]
    print(f"edmonton/{family}: {len(draws)} unique draws, {len(done)} sampled, "
          f"{len(todo)} to do ({PRODUCT}, stations {stations})", flush=True)

    listing_cache: dict = {}
    n_done = n_fail = 0
    for ts_s, bucket in todo:
        ts = dt.datetime.fromisoformat(ts_s)
        fg.GOES_BUCKET = bucket
        fg.S3_HTTPS_BASE = f"https://{bucket}.s3.amazonaws.com"
        try:
            res = fg.find_nearest_file(PRODUCT, ts, listing_cache)
            if res is None:
                n_fail += 1
                continue
            key, _ = res
            local = fg.download_cached(key)
            try:
                samp = sample_stations(local, stations)
            finally:
                if not keep_cache:
                    local.unlink(missing_ok=True)
            if samp is None:
                n_fail += 1
                continue
            for st in stations:
                w.writerow([ts_s, st, *samp[st]])
            n_done += 1
            if n_done % 25 == 0:
                fh.flush()
            if n_done % 100 == 0:
                print(f"  {n_done}/{len(todo)} sampled ({n_fail} failed)", flush=True)
        except KeyboardInterrupt:
            print("interrupted; resume by re-running", flush=True)
            break
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  fail {ts_s}: {type(e).__name__} {e}", flush=True)
    fh.close()
    print(f"sample stage done: {n_done} new, {n_fail} failed -> {out}", flush=True)


def stage_assemble(family: str) -> None:
    rows = load_source_rows(family)
    cache_path = Path(f"data/goes_draws/acmf_resample_edmonton_{family}.csv")
    sampled: dict[tuple[str, str], dict] = {}
    with open(cache_path) as fh:
        for r in csv.DictReader(fh):
            sampled[(r["ts_utc"], r["station"])] = r

    out = Path(f"data/goes_draws/goes_draws_edmonton_{family}_acmf.csv")
    n_out, drops = 0, Counter()
    flips = Counter()
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ts_utc", "family", "bucket", "regime", "season", "station",
                    "nearest_clear", "box3_mean_bcm", "box5_mean_bcm", "pix_km"])
        for r in rows:
            s = sampled.get((r["ts_utc"], r["station"]))
            if s is None:
                drops[r["station"]] += 1
                continue
            w.writerow([r["ts_utc"], r["family"], r["bucket"], r["regime"],
                        r["season"], r["station"], s["nearest_clear"],
                        s["box3_mean_bcm"], s["box5_mean_bcm"], s["pix_km"]])
            flips[(r["station"], r["regime"],
                   r["nearest_clear"], s["nearest_clear"])] += 1
            n_out += 1
    print(f"assembled {n_out}/{len(rows)} rows -> {out}"
          + (f"  (dropped: {dict(drops)})" if drops else ""))
    # old (clamped, 105-215 km off) vs new (correct pixel) clear fractions
    for st in STATIONS[family]:
        for regime in ("DAY", "NIGHT"):
            tot = sum(v for k, v in flips.items() if k[0] == st and k[1] == regime)
            if not tot:
                continue
            old = sum(v for k, v in flips.items()
                      if k[0] == st and k[1] == regime and k[2] == "1") / tot
            new = sum(v for k, v in flips.items()
                      if k[0] == st and k[1] == regime and k[3] == "1") / tot
            agree = sum(v for k, v in flips.items()
                        if k[0] == st and k[1] == regime and k[2] == k[3]) / tot
            print(f"  {st} {regime}: clear% old {100*old:.1f} -> new {100*new:.1f} "
                  f"(per-draw agreement {100*agree:.1f}%)")


def stage_topup(family: str, log_path: Path, seed: int = 42) -> None:
    """Draw fresh random timestamps until CYEG holds 5,000 rows per regime.

    A draw whose full-disk scan is missing is skipped and sampling continues,
    exactly as the primary sampler behaves at every site; the re-seeding
    convention matches climatology_goes.py (seed + draws already held, so a
    continued run draws a fresh sequence; the logs are the record)."""
    import random
    from metar_climatology import calculate_sun_alt, get_solar_regime, get_season

    stations = STATIONS[family]
    lat, lon = SITE.coords["CYEG"]
    counts = {"DAY": 0, "NIGHT": 0}
    with open(log_path) as fh:
        for r in csv.DictReader(fh):
            if r["station"] == "CYEG":
                counts[r["regime"]] += 1
    need = {k: 5000 - v for k, v in counts.items()}
    print(f"[top-up {family}] have {counts}, need {need}", flush=True)
    if all(v <= 0 for v in need.values()):
        print(f"[top-up {family}] nothing to do", flush=True)
        return

    random.seed(seed + counts["DAY"] + counts["NIGHT"])
    y0, y1 = 2020, 2025
    span = (dt.date(y1, 12, 31) - dt.date(y0, 1, 1)).days
    listing_cache: dict = {}
    fh = open(log_path, "a", newline="")
    w = csv.writer(fh)
    n_fail = 0
    while any(v > 0 for v in need.values()):
        ts = dt.datetime(y0, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
            days=random.randint(0, span), hours=random.randint(0, 23),
            minutes=random.randint(0, 59))
        regime = get_solar_regime(calculate_sun_alt(ts, lat, lon, SITE.elevation_m))
        if regime not in need or need[regime] <= 0:
            continue
        if family == "east":
            bucket = "noaa-goes16" if ts.date() < dt.date(2025, 4, 4) else "noaa-goes19"
        else:
            bucket = "noaa-goes17" if ts.date() < dt.date(2023, 1, 4) else "noaa-goes18"
        fg.GOES_BUCKET = bucket
        fg.S3_HTTPS_BASE = f"https://{bucket}.s3.amazonaws.com"
        try:
            res = fg.find_nearest_file(PRODUCT, ts, listing_cache)
            if res is None:
                n_fail += 1
                continue
            key, _ = res
            local = fg.download_cached(key)
            try:
                samp = sample_stations(local, stations)
            finally:
                local.unlink(missing_ok=True)
            if samp is None:
                n_fail += 1
                continue
            ts_s = ts.isoformat()
            season = get_season(ts.month)
            for st in stations:
                w.writerow([ts_s, family, bucket, regime, season, st, *samp[st]])
            fh.flush()
            need[regime] -= 1
            print(f"  [top-up {family}] {ts_s} {regime} ok; remaining {need}", flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  [top-up {family}] fail {ts}: {type(e).__name__} {e}", flush=True)
    fh.close()
    print(f"[top-up {family}] done ({n_fail} skipped/failed draws)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=("east", "west"))
    ap.add_argument("--assemble", action="store_true",
                    help="stage 2: build the corrected draw log from the sample cache")
    ap.add_argument("--top-up", metavar="LOG",
                    help="stage 3: append fresh random draws to LOG until CYEG "
                         "holds 5,000 rows per regime")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: stop after N draws")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep downloaded ACMF files (default: delete after reading)")
    args = ap.parse_args()
    if args.top_up:
        stage_topup(args.family, Path(args.top_up))
    elif args.assemble:
        stage_assemble(args.family)
    else:
        stage_sample(args.family, args.limit, args.keep_cache)


if __name__ == "__main__":
    main()
