"""
fetch_snow_cds.py — daily ERA5-Land snow depth per site, straight from the CDS.

Writes snow_cache/snow_<site>.json in the same schema the snow filters read
({"daily": {"time": [...], "snow_depth_max": [...]}}), so it is a drop-in
replacement for the Open-Meteo-sourced cache. Uses the CDS derived daily
statistics for ERA5-Land: one request per site-year, daily maximum, shifted to
the site's standard-time offset so a "day" is a local calendar day.

Requests are small (a single grid box) and resume cheaply: existing per-year
scratch files are kept, and a site's JSON is only written when all years are in.

Usage:
  python3 fetch_snow_cds.py --site calgary --years 2021 2021   # one-year test
  python3 fetch_snow_cds.py --site all                          # full fetch
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

try:
    import cdsapi
    import netCDF4
    import numpy as np
    from sites import SITES
except ImportError as e:
    sys.exit(f"Run from the repo root with cdsapi+netCDF4 installed: {e}")

SCRATCH = Path("snow_cds_scratch")
OUT = Path("snow_cache")


def utc_offset_str(hours: int) -> str:
    return f"utc{hours:+03d}:00"


def fetch_span(client, site, y0: int, y1: int, pad: float = 0.06) -> list[Path]:
    """Daily-max snow depth for one site over [y0, y1]. Tries the whole span in one
    CDS request; on a cost-limit rejection, splits the span in half and recurses
    (single years are known to pass). Returns the NetCDF paths covering the span.

    `pad` sets the request box half-width. The default box holds one ERA5-Land
    cell at most sites; a wider box is used on retry when every cell in the small
    box is water-masked (Vancouver: the airport sits on a delta island whose cell
    is ocean in ERA5-Land)."""
    SCRATCH.mkdir(exist_ok=True)
    tag = "" if pad == 0.06 else f"_pad{pad:g}"
    p = SCRATCH / f"{site.slug}_{y0}_{y1}{tag}.nc"
    if p.exists():
        return [p]
    try:
        client.retrieve(
            "derived-era5-land-daily-statistics",
            {
                "variable": ["snow_depth"],
                "year": [str(y) for y in range(y0, y1 + 1)],
                "month": [f"{m:02d}" for m in range(1, 13)],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "daily_statistic": "daily_maximum",
                "time_zone": utc_offset_str(site.naps_std_offset_h),
                "frequency": "1_hourly",
                "area": [site.lat + pad, site.lon - pad,
                         site.lat - pad, site.lon + pad],
            },
            str(p),
        )
        return [p]
    except Exception as e:
        msg = str(e)
        if "temporarily limited" in msg or "queued requests" in msg:
            import time
            print(f"[{site.slug}] {y0}-{y1}: CDS throttled, retrying in 15 min", flush=True)
            time.sleep(900)
            return fetch_span(client, site, y0, y1)
        if "cost limits" not in msg and "too large" not in msg:
            raise
        if y0 == y1:
            raise
        mid = (y0 + y1) // 2
        print(f"[{site.slug}] {y0}-{y1} over cost limit, splitting", flush=True)
        return (fetch_span(client, site, y0, mid, pad)
                + fetch_span(client, site, mid + 1, y1, pad))


MAX_CELL_KM = 25.0     # refuse a fallback cell farther than this from the station
MIN_VALID_FRAC = 0.95  # a usable cell must have data on nearly every day


def _dist_km(lat1, lon1, lat2, lon2):
    import math
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def read_year(path: Path, site):
    """[(date_iso, max_depth_m)] from one site-span file.

    Uses the valid (land) ERA5-Land cell NEAREST the station. At most sites the
    request box holds a single land cell, so this is identical to the old
    box-mean; where the station's own cell is water-masked (Vancouver) the
    nearest land cell within MAX_CELL_KM stands in, and the choice is printed.
    Returns None if no cell in the file has data on >= MIN_VALID_FRAC of days
    (caller retries with a wider box rather than writing a dead cache)."""
    with netCDF4.Dataset(str(path)) as nc:
        tvar = nc.variables["valid_time"]
        times = netCDF4.num2date(tvar[:], tvar.units)
        var = nc.variables["sde" if "sde" in nc.variables else "sd"]
        arr = np.ma.filled(var[:], np.nan)          # (time, lat, lon)
        lats = np.atleast_1d(nc.variables["latitude"][:]).astype(float)
        lons = np.atleast_1d(nc.variables["longitude"][:]).astype(float)
    arr = arr.reshape(arr.shape[0], len(lats), len(lons))
    best = None   # (dist_km, ilat, ilon)
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            frac = np.isfinite(arr[:, i, j]).mean()
            if frac < MIN_VALID_FRAC:
                continue
            d = _dist_km(site.lat, site.lon, la, lo)
            if d <= MAX_CELL_KM and (best is None or d < best[0]):
                best = (d, i, j)
    if best is None:
        return None
    d, i, j = best
    if d > 8.0:   # farther than one grid cell: this is a stand-in, say so
        print(f"[{site.slug}] station cell water-masked; using nearest land cell "
              f"({lats[i]:.1f},{lons[j]:.1f}), {d:.1f} km away", flush=True)
    series = arr[:, i, j]
    return [(t.strftime("%Y-%m-%d"), None if np.isnan(v) else float(v))
            for t, v in zip(times, series)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="all")
    ap.add_argument("--years", nargs=2, type=int, default=[2020, 2025])
    args = ap.parse_args()
    slugs = list(SITES) if args.site == "all" else [args.site]
    y0, y1 = args.years

    client = cdsapi.Client(quiet=True)
    OUT.mkdir(exist_ok=True)
    for slug in slugs:
        site = SITES[slug]
        print(f"[{slug}] requesting {y0}-{y1} ...", flush=True)
        rows = []
        for pad in (0.06, 0.30):    # small box first; widen only if all-water
            rows = []
            failed = False
            for path in fetch_span(client, site, y0, y1, pad):
                got = read_year(path, site)
                if got is None:
                    failed = True
                    break
                rows.extend(got)
            if not failed:
                break
            print(f"[{slug}] no valid land cell in the {pad:g} deg box, widening",
                  flush=True)
        n_valid = sum(1 for r in rows if r[1] is not None)
        if not rows or n_valid < len(rows) * MIN_VALID_FRAC:
            sys.exit(f"[{slug}] REFUSING to write a dead cache: "
                     f"{n_valid}/{len(rows)} days valid. Fix the request box.")
        rows.sort()
        out = OUT / f"snow_{slug}.json"
        out.write_text(json.dumps({"daily": {
            "time": [r[0] for r in rows],
            "snow_depth_max": [r[1] for r in rows],
        }}))
        print(f"[{slug}] wrote {out} ({len(rows)} days, {n_valid} valid)", flush=True)


if __name__ == "__main__":
    main()
