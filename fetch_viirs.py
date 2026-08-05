"""fetch_viirs.py — VIIRS MVCM cloud mask (NASA CLDMSK_L2_VIIRS) access.

A second satellite reference with the OPPOSITE weaknesses to GOES: VIIRS is polar-
orbiting (near-nadir over Canada, vs GOES's oblique slant) and its cloud mask uses
the Day/Night Band, so at night it detects cloud by reflected moonlight rather than
GOES's IR-only test. Confirming the human nighttime-clearing artifact with both
satellites closes the "day/night algorithm change" loophole GOES alone leaves open.

Access (NASA Earthdata):
  * Granule discovery — CMR (cmr.earthdata.nasa.gov) point+temporal search. ANONYMOUS.
    CMR returns exactly the granule(s) over a site at a time, so we don't reproduce
    GOES's fixed-grid navigation OR any swath-windowing — CMR solves coverage for us.
  * Download — LAADS archive (ladsweb.modaps.eosdis.nasa.gov), needs a (free) Earthdata
    bearer token in EARTHDATA_TOKEN or ~/.edl_token. One-time signup:
    https://urs.earthdata.nasa.gov ; long-lived LAADS "App Key" recommended.

Product: CLDMSK_L2_VIIRS_{SNPP,NOAA20,NOAA21}, v2.0 (ArchiveSet 5200), netCDF4, 6-min
swath granules (2012-03-01 onward). Cloud mask = group geophysical_data /
Integer_Cloud_Mask (0 confident cloudy, 1 probably cloudy, 2 probably clear,
3 confident clear); geolocation = group geolocation_data / latitude, longitude.

Run a bring-up to confirm structure against a real file:
  python fetch_viirs.py --describe --platform NOAA20 --date 2024-01-15 \
                        --lat 51.1139 --lon -114.0203
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
# VIIRS_CACHE_DIR lets parallel per-site runs use separate cache dirs so concurrent
# downloads never race on the same .part file. Defaults to one shared viirs_cache/.
CACHE_DIR = Path(os.environ.get("VIIRS_CACHE_DIR") or (PROJECT_ROOT / "viirs_cache"))

CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Platform key -> CLDMSK short_name. NOAA21 came online ~2023; pre-2023 searches
# just return nothing, which the sampler treats as "no coverage" and moves on.
SHORT_NAME = {
    "SNPP":   "CLDMSK_L2_VIIRS_SNPP",
    "NOAA20": "CLDMSK_L2_VIIRS_NOAA20",
    "NOAA21": "CLDMSK_L2_VIIRS_NOAA21",
}

# MVCM Integer_Cloud_Mask confidence codes. "Clear" = probably + confident clear,
# the standard clear-sky grouping and the closest analogue to GOES's binary clear.
CLEAR_CODES = (2, 3)

# Tolerant variable/group lookup (file labels can vary slightly by collection).
_CMASK_VARS = ["Integer_Cloud_Mask"]
_LAT_VARS = ["latitude", "Latitude"]
_LON_VARS = ["longitude", "Longitude"]
_GEO_GROUP = "geolocation_data"
_PHYS_GROUP = "geophysical_data"

# A site is "covered" only if the nearest swath pixel is within this distance
# (M-band pixels are ~750 m at nadir, larger at swath edge); else the granule
# clipped past the site and we reject it.
COVERAGE_MAX_KM = 2.0

# Minimum unmasked pixels for a box to count. A box is kept if at least this many
# of its (2*half+1)^2 pixels carry a mask value; boxes with none are dropped as
# all-fill. One is the loosest such rule: a box on the swath edge survives on a
# single valid pixel, the limiting case of thinning the box down to a point.
MIN_VALID_BOX_PIXELS = 1


# --------------------------------------------------------------------------- #
# Earthdata token + HTTP
# --------------------------------------------------------------------------- #
def earthdata_token() -> str:
    """Bearer token from EARTHDATA_TOKEN env or ~/.edl_token. Errors if absent."""
    tok = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if tok:
        return tok
    path = Path.home() / ".edl_token"
    if path.exists():
        tok = path.read_text().strip()
        if tok:
            return tok
    sys.exit(
        "No Earthdata token. Set EARTHDATA_TOKEN or write ~/.edl_token "
        "(free signup at https://urs.earthdata.nasa.gov; LAADS App Key recommended)."
    )


def curl_json(url: str, timeout: int = 90) -> dict:
    r = subprocess.run(["curl", "-sS", "--fail", "--retry", "3", "--retry-delay", "2",
                        "--max-time", str(timeout), url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"CMR query failed: {url}\n{r.stderr.strip()}")
    return json.loads(r.stdout)


def curl_download(url: str, dest: Path, token: str, timeout: int = 180) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    r = subprocess.run(
        ["curl", "-sS", "--fail", "-L", "--retry", "4", "--retry-delay", "2",
         "--max-time", str(timeout), "-H", f"Authorization: Bearer {token}",
         "-o", str(tmp), url])
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"LAADS download failed (token valid?): {url}")
    tmp.replace(dest)


# --------------------------------------------------------------------------- #
# CMR granule discovery (anonymous)
# --------------------------------------------------------------------------- #
def _data_url(entry: dict) -> str | None:
    """Pick the downloadable .nc link from a CMR granule entry."""
    for link in entry.get("links", []):
        href = link.get("href", "")
        rel = link.get("rel", "")
        if href.endswith(".nc") and ("/data#" in rel or "ladsweb" in href):
            return href
    # fallback: any .nc href
    for link in entry.get("links", []):
        if link.get("href", "").endswith(".nc"):
            return link["href"]
    return None


def cmr_granules(platform: str, lat: float, lon: float,
                 start: dt.datetime, end: dt.datetime,
                 cache: dict | None = None,
                 page_size: int = 50) -> list[tuple[str, dt.datetime]]:
    """Granules of `platform`'s CLDMSK covering (lat,lon) in [start,end].

    Returns [(download_url, time_start_utc), ...] sorted by time. CMR's `point`
    spatial filter does the swath-coverage test for us. Results cached per call key.
    page_size must exceed the granule count of the span (CMR caps it at 2000):
    the daily spans of the sampler fit in the default 50; a month-long span
    (viirs_restore_granules.py) needs ~200 and passes 2000."""
    short = SHORT_NAME[platform]
    key = (short, round(lat, 3), round(lon, 3), start.isoformat(), end.isoformat())
    if cache is not None and key in cache:
        return cache[key]
    url = (f"{CMR_GRANULES}?short_name={short}"
           f"&point={lon:.4f},{lat:.4f}"
           f"&temporal={start.strftime('%Y-%m-%dT%H:%M:%SZ')},{end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
           f"&page_size={page_size}&sort_key=start_date")
    out: list[tuple[str, dt.datetime]] = []
    try:
        data = curl_json(url)
        for e in data.get("feed", {}).get("entry", []):
            durl = _data_url(e)
            ts = e.get("time_start", "")
            if not durl or not ts:
                continue
            t = dt.datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
            out.append((durl, t))
    except Exception as ex:
        print(f"  CMR {platform} {start.date()}: {ex}")
    if cache is not None:
        cache[key] = out
    return out


def download_cached(url: str, token: str) -> Path:
    local = CACHE_DIR / url.rsplit("/", 1)[-1]
    if local.exists() and local.stat().st_size > 1024:
        return local
    curl_download(url, local, token)
    return local


# --------------------------------------------------------------------------- #
# NetCDF sampling (swath: nearest pixel in 2-D lat/lon)
# --------------------------------------------------------------------------- #
def _find(group, names):
    for n in names:
        if n in group.variables:
            return n
    return None


def _haversine_km_arr(lat0, lon0, lat, lon):
    R = 6371.0
    p0 = math.radians(lat0)
    p = np.radians(lat)
    dp = np.radians(lat - lat0)
    dl = np.radians(lon - lon0)
    a = np.sin(dp / 2) ** 2 + math.cos(p0) * np.cos(p) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _groups(nc):
    """Return (geo_group, phys_group), tolerating a flat (no-group) file layout."""
    geo = nc.groups.get(_GEO_GROUP, nc)
    phys = nc.groups.get(_PHYS_GROUP, nc)
    return geo, phys


def sample_granule(nc_path: Path, lat: float, lon: float,
                   clear_codes=CLEAR_CODES) -> tuple[str, float] | None:
    """Sample the cloud mask at the swath pixel nearest (lat,lon).

    Returns ("CLEAR"|"CLOUDY", dist_km), or None if the site isn't actually under
    this granule's swath (nearest pixel farther than COVERAGE_MAX_KM) or vars missing."""
    with netCDF4.Dataset(str(nc_path)) as nc:
        geo, phys = _groups(nc)
        latn, lonn = _find(geo, _LAT_VARS), _find(geo, _LON_VARS)
        cmn = _find(phys, _CMASK_VARS)
        if not (latn and lonn and cmn):
            return None
        glat = np.asarray(geo.variables[latn][:], dtype="float64")
        glon = np.asarray(geo.variables[lonn][:], dtype="float64")
        d = _haversine_km_arr(lat, lon, glat, glon)
        iy, ix = np.unravel_index(int(np.argmin(d)), d.shape)
        dist = float(d[iy, ix])
        if dist > COVERAGE_MAX_KM:
            return None
        val = phys.variables[cmn][iy, ix]
        if np.ma.is_masked(val):
            return None
        return ("CLEAR" if int(val) in clear_codes else "CLOUDY", dist)


def sample_granule_box(nc_path: Path, lat: float, lon: float, half: int = 3,
                       cloud_frac_thresh: float = 0.25) -> tuple[str, float] | None:
    """Whole-sky proxy: cloud FRACTION over a (2*half+1)^2 pixel box around the site.

    A single 750 m pixel can be clear between scattered clouds while the dome is not;
    averaging a box better matches the human's whole-sky okta. "CLEAR" if the cloudy
    fraction (Integer_Cloud_Mask in {0,1}) is at/below cloud_frac_thresh (default 0.25
    ~ the SKC+FEW "usable" cut of okta<=2). Returns ("CLEAR"|"CLOUDY", dist_km) or None
    if the site isn't under the swath / the box is all fill."""
    with netCDF4.Dataset(str(nc_path)) as nc:
        geo, phys = _groups(nc)
        latn, lonn = _find(geo, _LAT_VARS), _find(geo, _LON_VARS)
        cmn = _find(phys, _CMASK_VARS)
        if not (latn and lonn and cmn):
            return None
        glat = np.asarray(geo.variables[latn][:], dtype="float64")
        glon = np.asarray(geo.variables[lonn][:], dtype="float64")
        d = _haversine_km_arr(lat, lon, glat, glon)
        iy, ix = np.unravel_index(int(np.argmin(d)), d.shape)
        dist = float(d[iy, ix])
        if dist > COVERAGE_MAX_KM:
            return None
        ny, nx = phys.variables[cmn].shape[-2:]
        y0, y1 = max(0, iy - half), min(ny, iy + half + 1)
        x0, x1 = max(0, ix - half), min(nx, ix + half + 1)
        win = np.ma.asarray(phys.variables[cmn][y0:y1, x0:x1])
        valid = int(win.count())
        if valid < MIN_VALID_BOX_PIXELS:
            return None
        cloudy = int(np.ma.sum((win == 0) | (win == 1)))   # confident/probably cloudy
        frac = cloudy / valid
        return ("CLEAR" if frac <= cloud_frac_thresh else "CLOUDY", dist)


def describe_granule(nc_path: Path, lat: float, lon: float) -> None:
    """Bring-up helper: print group/variable layout and a sample at (lat,lon)."""
    with netCDF4.Dataset(str(nc_path)) as nc:
        print(f"  file: {nc_path.name}  ({nc_path.stat().st_size/1e6:.1f} MB)")
        print(f"  root groups: {list(nc.groups)}")
        for gname, g in (list(nc.groups.items()) or [("(root)", nc)]):
            print(f"  [{gname}] vars: {list(g.variables)[:20]}")
    res = sample_granule(nc_path, lat, lon)
    print(f"  sample at ({lat},{lon}): {res}")


# --------------------------------------------------------------------------- #
# CLI (bring-up)
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--describe", action="store_true",
                    help="download one covering granule and dump its structure + a sample")
    ap.add_argument("--platform", default="NOAA20", choices=list(SHORT_NAME))
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (UTC day to search)")
    ap.add_argument("--lat", type=float, default=51.1139)     # CYYC
    ap.add_argument("--lon", type=float, default=-114.0203)
    args = ap.parse_args()

    day = dt.datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    grans = cmr_granules(args.platform, args.lat, args.lon, day,
                         day + dt.timedelta(days=1))
    print(f"CMR: {len(grans)} {args.platform} granule(s) over "
          f"({args.lat},{args.lon}) on {args.date}")
    for url, t in grans:
        print(f"  {t.isoformat()}  {url.rsplit('/',1)[-1]}")

    if args.describe and grans:
        token = earthdata_token()
        url, t = grans[0]
        print(f"\nDownloading {url.rsplit('/',1)[-1]} ...")
        describe_granule(download_cached(url, token), args.lat, args.lon)


if __name__ == "__main__":
    main()
