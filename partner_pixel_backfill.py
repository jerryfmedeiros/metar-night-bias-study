"""
partner_pixel_backfill.py — sample a new partner station's GOES pixel at the exact
timestamps of an existing per-draw log.

The published draw logs for Toronto and Edmonton predate the CYTZ / CZVL partners, so
those pixels were never logged and the siting cross-check (does the satellite see the
same diurnal sky change at both airports of a pair?) could not be run there. This
script re-reads the published log's (ts_utc, bucket) pairs, downloads each ACMC scan
again, and samples ONLY the new partner pixel, writing a parallel log with identical
row structure. Nothing in the published logs or derived tables changes; the output is
purely additive.

Usage:
  python3 partner_pixel_backfill.py --site toronto  --station CYTZ [--limit N]
  python3 partner_pixel_backfill.py --site edmonton --station CZVL [--limit N]

Output: data/goes_draws/goes_draws_<site>_east_<station>.csv
Resumable: rows already in the output are skipped on restart.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np
import netCDF4

import fetch_goes as fg
from sites import SITES


def sample_pixel(path: Path, lat: float, lon: float):
    """(nearest_clear, box3_mean_bcm, box5_mean_bcm) at the pixel nearest (lat, lon)."""
    with netCDF4.Dataset(str(path)) as nc:
        gip = nc.variables["goes_imager_projection"]
        pv = {a: getattr(gip, a) for a in
              ("perspective_point_height", "semi_major_axis", "semi_minor_axis",
               "longitude_of_projection_origin")}
        x, y = fg.latlon_to_goes_xy(lat, lon, pv)
        xs = np.asarray(nc.variables["x"][:])
        ys = np.asarray(nc.variables["y"][:])
        j, i = fg._grid_indices(xs, ys, x, y, lat, lon)  # raises OutOfSector, never clamps
        bcm = np.asarray(nc.variables["BCM"][:], dtype="float32")
    v = bcm[i, j]
    if not (0 <= v <= 1):
        return None
    def box(h):
        b = bcm[max(0, i - h):i + h + 1, max(0, j - h):j + h + 1]
        b = b[(b >= 0) & (b <= 1)]
        return f"{b.mean():.4f}" if b.size else ""
    return int(v == 0), box(1), box(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", required=True)
    ap.add_argument("--station", required=True, help="new partner ICAO to sample")
    ap.add_argument("--family", default="east")
    ap.add_argument("--limit", type=int, default=None, help="stop after N draws (smoke test)")
    args = ap.parse_args()

    site = SITES[args.site]
    lat, lon = site.coords[args.station]
    src = Path(f"data/goes_draws/goes_draws_{args.site}_{args.family}.csv")
    out = Path(f"data/goes_draws/goes_draws_{args.site}_{args.family}_{args.station}.csv")

    # unique draws from the published log (ts, bucket, regime, season)
    draws, seen = [], set()
    with open(src) as fh:
        for r in csv.DictReader(fh):
            k = r["ts_utc"]
            if k in seen:
                continue
            seen.add(k)
            draws.append((r["ts_utc"], r["bucket"], r["regime"], r["season"]))

    done = set()
    if out.exists():
        with open(out) as fh:
            done = {r["ts_utc"] for r in csv.DictReader(fh)}
    new = out.exists() is False
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["ts_utc", "family", "bucket", "regime", "season", "station",
                    "nearest_clear", "box3_mean_bcm", "box5_mean_bcm"])

    listing_cache: dict = {}
    n_done = n_skip = n_fail = 0
    todo = [d for d in draws if d[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.site}/{args.station}: {len(draws)} draws in log, "
          f"{len(done)} already sampled, {len(todo)} to do", flush=True)

    for k, (ts_s, bucket, regime, season) in enumerate(todo, 1):
        ts = dt.datetime.fromisoformat(ts_s)
        fg.GOES_BUCKET = bucket
        fg.S3_HTTPS_BASE = f"https://{bucket}.s3.amazonaws.com"
        try:
            res = fg.find_nearest_file("ACMC", ts, listing_cache)
            if res is None:
                n_fail += 1
                continue
            key, _ = res
            local = fg.download_cached(key)
            samp = sample_pixel(local, lat, lon)
            local.unlink(missing_ok=True)          # discard, like the original run
            if samp is None:
                n_fail += 1
                continue
            nearest, b3, b5 = samp
            w.writerow([ts_s, args.family, bucket, regime, season, args.station,
                        nearest, b3, b5])
            n_done += 1
            if n_done % 100 == 0:
                fh.flush()
                print(f"  {n_done}/{len(todo)} sampled ({n_fail} failed)", flush=True)
        except KeyboardInterrupt:
            break
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  fail {ts_s}: {type(e).__name__} {e}", flush=True)
    fh.close()
    print(f"done: {n_done} sampled, {n_fail} failed -> {out}", flush=True)


if __name__ == "__main__":
    main()
