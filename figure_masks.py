"""
figure_masks.py — Figure 2: pixel-level cloud masks over Calgary
from GOES ABI (BCM) and VIIRS MVCM (Integer_Cloud_Mask) for the same scene.

Produces figures/figure_masks.pdf/.png

Usage:
  python3 figure_masks.py                   # default scene
  python3 figure_masks.py --date 2024-10-15 --hour 19
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import netCDF4
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── project modules ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from fetch_goes import (
    find_nearest_file, download_cached as goes_download,
    latlon_to_goes_xy,
)
import fetch_goes as _fg
from fetch_viirs import (
    cmr_granules, download_cached as viirs_download, earthdata_token,
    _groups, _find, _LAT_VARS, _LON_VARS, _CMASK_VARS,
)
from fetch_metar import fetch_metar_csv

# ── constants ──────────────────────────────────────────────────────────────────
SITE_LAT, SITE_LON = 51.11, -114.02   # CYYC Calgary Int'l
SITE_NAME = "CYYC"

HALF        = 45      # ±45 GOES pixels around station (~180–270 km box)
VIIRS_PLAT  = "NOAA20"   # default; per-row override via --viirs-plat/--snow-viirs-plat
VIIRS_WIN_H = 8       # search ±8 h around the GOES scene for a VIIRS overpass

# VIIRS sampling box used in the study (sample_granule_box, half=3 → 7×7 pixels @ 750 m)
VIIRS_BOX_HALF_KM = 3 * 0.75  # 2.25 km → half-side of the 7×7 box
VIIRS_BOX_DLAT = VIIRS_BOX_HALF_KM / 111.0
VIIRS_BOX_DLON = VIIRS_BOX_HALF_KM / (111.0 * np.cos(np.radians(SITE_LAT)))

OUT_DIR = PROJECT_ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Grayscale: dark = cloudy, light = clear (physical intuition, print-safe)
_CMAP_GOES  = mcolors.ListedColormap(["#d8d8d8", "#404040"])   # 0=clear (light), 1=cloudy (dark)
_CMAP_VIIRS = mcolors.ListedColormap([
    "#2a2a2a",   # 0 = confident cloudy
    "#787878",   # 1 = probably cloudy
    "#c0c0c0",   # 2 = probably clear
    "#f0f0f0",   # 3 = confident clear
])


# ── GOES helpers ───────────────────────────────────────────────────────────────
def _goes_proj_vars(nc):
    gip = nc.variables["goes_imager_projection"]
    return {a: getattr(gip, a) for a in [
        "perspective_point_height", "semi_major_axis", "semi_minor_axis",
        "longitude_of_projection_origin",
    ]}


def fetch_goes_tile(ts: dt.datetime, listing_cache: dict):
    """Download GOES ACMC scene nearest ts; return (bcm_tile, lons_2d, lats_2d, file_ts)."""
    result = find_nearest_file("ACMC", ts, listing_cache)
    if result is None:
        sys.exit(f"No GOES ACMC file found near {ts.isoformat()}")
    key, file_ts = result
    local = goes_download(key)

    with netCDF4.Dataset(str(local)) as nc:
        pv  = _goes_proj_vars(nc)
        x_all = np.asarray(nc.variables["x"][:])
        y_all = np.asarray(nc.variables["y"][:])
        bcm_all = np.asarray(nc.variables["BCM"][:], dtype="float32")

        # find station pixel
        sat_x, sat_y = latlon_to_goes_xy(SITE_LAT, SITE_LON, pv)
        cx = int(np.argmin(np.abs(x_all - sat_x)))
        cy = int(np.argmin(np.abs(y_all - sat_y)))

        ix0, ix1 = max(0, cx - HALF), min(x_all.size,  cx + HALF + 1)
        iy0, iy1 = max(0, cy - HALF), min(y_all.size, cy + HALF + 1)

        x_tile = x_all[ix0:ix1]
        y_tile = y_all[iy0:iy1]
        bcm    = bcm_all[iy0:iy1, ix0:ix1]

    # mask fill / invalid
    bcm = np.where((bcm >= 0) & (bcm <= 1), bcm, np.nan)

    # convert x/y scan angles → lat/lon via GOES-R ABI PUG inverse formula (§5.1.2.8.2)
    H   = pv["perspective_point_height"] + pv["semi_major_axis"]
    req = pv["semi_major_axis"]
    rpol = pv["semi_minor_axis"]
    lon_0 = np.radians(pv["longitude_of_projection_origin"])

    X, Y = np.meshgrid(x_tile, y_tile)          # scan angles in radians (2-D)

    a = (np.sin(X)**2
         + np.cos(X)**2 * (np.cos(Y)**2 + (req / rpol)**2 * np.sin(Y)**2))
    b = -2.0 * H * np.cos(X) * np.cos(Y)
    c = H**2 - req**2

    disc = b**2 - 4.0 * a * c
    rs = np.where(disc >= 0, (-b - np.sqrt(np.where(disc >= 0, disc, 0.0))) / (2.0 * a), np.nan)

    sx = rs * np.cos(X) * np.cos(Y)
    sy = -rs * np.sin(X)
    sz =  rs * np.cos(X) * np.sin(Y)

    lats = np.degrees(np.arctan((req / rpol)**2 * sz / np.sqrt((H - sx)**2 + sy**2)))
    lons = np.degrees(lon_0 - np.arctan(sy / (H - sx)))

    return bcm, lons, lats, file_ts


# ── VIIRS helpers ──────────────────────────────────────────────────────────────
def fetch_viirs_tile(target: dt.datetime, token: str, platform: str = VIIRS_PLAT):
    """Fetch VIIRS granule covering Calgary within ±VIIRS_WIN_H of target.

    Returns (mask_2d, lat_2d, lon_2d, granule_ts) using raw swath arrays
    cropped to a bounding box around the site, or exits if nothing found.
    """
    t0 = target - dt.timedelta(hours=VIIRS_WIN_H)
    t1 = target + dt.timedelta(hours=VIIRS_WIN_H)
    granules = cmr_granules(platform, SITE_LAT, SITE_LON, t0, t1)
    if not granules:
        sys.exit(f"No VIIRS {platform} granule found covering Calgary in [{t0}, {t1}].\n"
                 "Try --date / --hour with a different scene.")

    # pick granule closest to target
    url, gts = min(granules, key=lambda g: abs((g[1] - target).total_seconds()))
    print(f"  VIIRS granule: {url.rsplit('/',1)[-1]}  ({gts.strftime('%Y-%m-%d %H:%M')} UTC)")
    local = viirs_download(url, token)

    deg = 2.5  # degrees around station to crop
    with netCDF4.Dataset(str(local)) as nc:
        geo, phys = _groups(nc)
        latn = _find(geo, _LAT_VARS)
        lonn = _find(geo, _LON_VARS)
        cmn  = _find(phys, _CMASK_VARS)
        if not (latn and lonn and cmn):
            sys.exit("VIIRS file missing expected variables.")
        glat = np.asarray(geo.variables[latn][:])
        glon = np.asarray(geo.variables[lonn][:])
        mask = np.asarray(phys.variables[cmn][:])

    # crop to bounding box
    ok = (
        (glat >= SITE_LAT - deg) & (glat <= SITE_LAT + deg) &
        (glon >= SITE_LON - deg) & (glon <= SITE_LON + deg)
    )
    if not ok.any():
        sys.exit("VIIRS granule covers Calgary in CMR but pixels not in bounding box — "
                 "try a different date.")

    # trim to row range that has any valid data
    rows_ok = ok.any(axis=1)
    r0, r1 = np.argmax(rows_ok), len(rows_ok) - np.argmax(rows_ok[::-1])
    glat = glat[r0:r1]; glon = glon[r0:r1]; mask = mask[r0:r1]
    ok   = ok[r0:r1]

    # mask pixels outside bounding box as nan
    mask_f = mask.astype("float32")
    mask_f[~ok] = np.nan

    return mask_f, glat, glon, gts


# ── figure helpers ─────────────────────────────────────────────────────────────
def _goes_bucket_for(ts: dt.datetime) -> str:
    cutover19 = dt.datetime(2025, 4, 4, tzinfo=dt.timezone.utc)
    return "noaa-goes19" if ts >= cutover19 else "noaa-goes16"


def _set_bucket(ts: dt.datetime):
    _fg.GOES_BUCKET   = _goes_bucket_for(ts)
    _fg.S3_HTTPS_BASE = f"https://{_fg.GOES_BUCKET}.s3.amazonaws.com"


def _map_extent(g_lons, g_lats):
    return [np.nanmin(g_lons) - 0.3, np.nanmax(g_lons) + 0.3,
            np.nanmin(g_lats) - 0.3, np.nanmax(g_lats) + 0.3]


def _draw_map(ax, geo):
    ax.add_feature(cfeature.BORDERS,   lw=0.4, edgecolor="#555555")
    ax.add_feature(cfeature.STATES,    lw=0.3, edgecolor="#888888")
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="#555555")


def nearest_metar_label(ts: dt.datetime) -> str:
    """Return a short sky-condition string from the nearest CYYC METAR to ts."""
    OKTA_LABEL = {0: "SKC", 1: "FEW", 2: "FEW", 3: "SCT", 4: "SCT",
                  5: "BKN", 6: "BKN", 7: "OVC", 8: "OVC"}
    try:
        obs = fetch_metar_csv("CYYC", ts.date(), ts.date())
    except Exception:
        return "METAR unavailable"
    if not obs:
        return "METAR: no data"
    nearest = min(obs, key=lambda o: abs((o.timestamp - ts).total_seconds()))
    okta = nearest.coverage_okta
    cov  = OKTA_LABEL.get(okta, "?")
    delta = int((nearest.timestamp - ts).total_seconds() / 60)
    sign  = "+" if delta >= 0 else "−"
    return (f"METAR CYYC {nearest.timestamp.strftime('%H:%M')} UTC  "
            f"{cov} ({okta}/8 okta)")


INSET_DEG = 0.15   # ±0.15° ≈ ±17 km around station for pixel inset

def _draw_pixel_inset(ax, data2d, lons2d, lats2d, geo, cmap, vmin, vmax,
                      inset_pos=(0.60, 0.00, 0.39, 0.39),
                      overlay_color="#222222"):
    """Add a zoomed inset showing individual pixels around CYYC.

    overlay_color controls the indicator rectangle and sampling-box outline.
    Pass 'white' for panels with predominantly dark pixels (e.g. VIIRS snow season).
    """
    # When overlay_color is white, use a halo (white thick + black thin) so the
    # box reads on both dark and light pixel backgrounds.
    halo = overlay_color == "white"
    ink  = "#000000"

    def _rect(patch_ax, x, y, w, h, lw, ec, **kw):
        patch_ax.add_patch(mpatches.Rectangle((x, y), w, h,
                           linewidth=lw, edgecolor=ec, facecolor="none", **kw))

    # rectangle on main map showing inset area
    if halo:
        _rect(ax, SITE_LON - INSET_DEG, SITE_LAT - INSET_DEG,
              2*INSET_DEG, 2*INSET_DEG, 2.2, "white", transform=geo, zorder=8)
    _rect(ax, SITE_LON - INSET_DEG, SITE_LAT - INSET_DEG,
          2*INSET_DEG, 2*INSET_DEG, 0.9, ink, transform=geo, zorder=8)

    axin = ax.inset_axes(inset_pos, projection=ax.projection)
    axin.set_extent([SITE_LON - INSET_DEG, SITE_LON + INSET_DEG,
                     SITE_LAT - INSET_DEG, SITE_LAT + INSET_DEG], crs=geo)
    axin.pcolormesh(lons2d, lats2d, data2d,
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    transform=geo, shading="nearest", rasterized=True)
    for spine in axin.spines.values():
        spine.set_linewidth(0.9)
        spine.set_edgecolor(ink)
    return axin


def _draw_goes_footprint(ax, g_lons, g_lats, geo):
    """Draw a dashed outline around the GOES ABI scan tile.

    The tile is a rectangle in scan-angle space; projected onto lat/lon it
    appears as a parallelogram because of the oblique viewing geometry at 51°N.
    Drawing the boundary makes this look deliberate rather than a display artefact.
    """
    lon = np.concatenate([
        g_lons[0,  :],           # top row  (left → right)
        g_lons[1:-1, -1],        # right col (excl corners)
        g_lons[-1, ::-1],        # bottom row (right → left)
        g_lons[-2:0:-1, 0],      # left col  (excl corners)
        [g_lons[0, 0]],          # close
    ])
    lat = np.concatenate([
        g_lats[0,  :],
        g_lats[1:-1, -1],
        g_lats[-1, ::-1],
        g_lats[-2:0:-1, 0],
        [g_lats[0, 0]],
    ])
    valid = ~(np.isnan(lon) | np.isnan(lat))
    ax.plot(lon[valid], lat[valid],
            color="#222222", lw=0.9, ls="--", transform=geo, zorder=7)
    # Label at top-right corner of the tile
    tr_lon = g_lons[0, -1]
    tr_lat = g_lats[0, -1]
    if not (np.isnan(tr_lon) or np.isnan(tr_lat)):
        lbl = ax.text(tr_lon + 0.3, tr_lat, "ABI scan\nfootprint",
                      transform=geo, fontsize=5.5, color="#333333",
                      ha="left", va="center", zorder=8)
        lbl.set_path_effects([mpe.withStroke(linewidth=1.5, foreground="white")])


def _draw_viirs_box(ax, geo):
    """Draw the 7×7 VIIRS pixel sampling box (~5 km × 5 km) around the station."""
    from matplotlib.patches import Rectangle
    import cartopy.crs as _ccrs
    lon0 = SITE_LON - VIIRS_BOX_DLON
    lat0 = SITE_LAT - VIIRS_BOX_DLAT
    w = 2 * VIIRS_BOX_DLON
    h = 2 * VIIRS_BOX_DLAT
    # halo: thick white then thin black — readable on both dark and light pixels
    ax.add_patch(Rectangle((lon0, lat0), w, h,
                            linewidth=2.5, edgecolor="white",
                            facecolor="none", transform=geo, zorder=9))
    ax.add_patch(Rectangle((lon0, lat0), w, h,
                            linewidth=1.2, edgecolor="black",
                            facecolor="none", transform=geo, zorder=10))
    txt = ax.text(SITE_LON + VIIRS_BOX_DLON + 0.15, SITE_LAT,
                  "7×7 px\n(~5 km)", fontsize=5.5, color="black",
                  va="center", ha="left", transform=geo, zorder=10)
    txt.set_path_effects([mpe.withStroke(linewidth=2, foreground="white")])


def _plot_goes(ax, g_lons, g_lats, bcm, ts, geo, col_title):
    ax.pcolormesh(g_lons, g_lats, bcm,
                  cmap=_CMAP_GOES, vmin=0, vmax=1,
                  transform=geo, shading="nearest", rasterized=True)
    ax.plot(SITE_LON, SITE_LAT, "r+", ms=10, mew=1.5, transform=geo, zorder=8)
    ax.set_title(f"GOES-East ABI  BCM  (~2 km)\n{ts.strftime('%Y-%m-%d %H:%M')} UTC",
                 fontsize=8)
    ax.text(-0.06, 0.5, col_title, transform=ax.transAxes,
            fontsize=8, fontweight="bold", rotation=90,
            va="center", ha="right", color="#333333")


def _plot_viirs(ax, v_mask, v_lats, v_lons, ts, geo):
    flat_lat = v_lats.ravel(); flat_lon = v_lons.ravel(); flat_m = v_mask.ravel()
    ok = ~np.isnan(flat_m)
    ax.scatter(flat_lon[ok], flat_lat[ok], c=flat_m[ok],
               cmap=_CMAP_VIIRS, vmin=0, vmax=3,
               s=0.8, linewidths=0, transform=geo, zorder=4, rasterized=True)
    ax.plot(SITE_LON, SITE_LAT, "r+", ms=10, mew=1.5, transform=geo, zorder=8)
    ax.set_title(f"VIIRS NOAA-20  MVCM  (~750 m)\n{ts.strftime('%Y-%m-%d %H:%M')} UTC",
                 fontsize=8)


# ── figure ─────────────────────────────────────────────────────────────────────
def make_fig(snow_free_ts: dt.datetime, snow_ts: dt.datetime,
             sf_plat: str = VIIRS_PLAT, sn_plat: str = VIIRS_PLAT):
    listing_cache: dict = {}
    token = earthdata_token()
    geo   = ccrs.PlateCarree()
    proj  = ccrs.LambertConformal(central_longitude=SITE_LON,
                                  central_latitude=SITE_LAT,
                                  standard_parallels=(45, 55))

    # ── fetch all four datasets ───────────────────────────────────────────────
    print("Fetching snow-free GOES …")
    _set_bucket(snow_free_ts)
    bcm_sf, g_lons_sf, g_lats_sf, gts_sf = fetch_goes_tile(snow_free_ts, listing_cache)

    print(f"Fetching snow-free VIIRS …")
    vm_sf, vla_sf, vlo_sf, vts_sf = fetch_viirs_tile(snow_free_ts, token, sf_plat)

    print("Fetching snow-season GOES …")
    _set_bucket(snow_ts)
    listing_cache.clear()
    bcm_sn, g_lons_sn, g_lats_sn, gts_sn = fetch_goes_tile(snow_ts, listing_cache)

    print(f"Fetching snow-season VIIRS …")
    vm_sn, vla_sn, vlo_sn, vts_sn = fetch_viirs_tile(snow_ts, token, sn_plat)

    # ── METAR ─────────────────────────────────────────────────────────────────
    print("Fetching METAR CYYC …")
    metar_sf = nearest_metar_label(snow_free_ts)
    metar_sn = nearest_metar_label(snow_ts)
    print(f"  snow-free : {metar_sf}")
    print(f"  snow-season: {metar_sn}")

    # use snow-free GOES extent for all panels (consistent view)
    extent = _map_extent(g_lons_sf, g_lats_sf)

    # ── 2×2 layout: rows = snow-free / snow-season; cols = GOES / VIIRS ──────
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5),
                             subplot_kw={"projection": proj})
    fig.subplots_adjust(wspace=0.05, hspace=0.28,
                        left=0.08, right=0.97, top=0.92, bottom=0.09)

    col_labels = ["GOES-East ABI  BCM  (~2 km, oblique ~69°)",
                  "VIIRS MVCM  (~750 m, near-nadir)"]
    for j, lbl in enumerate(col_labels):
        axes[0, j].set_title(lbl, fontsize=8, fontweight="bold", pad=4)

    for row_axes in axes:
        for ax in row_axes:
            ax.set_extent(extent, crs=geo)
            _draw_map(ax, geo)

    # row labels (rotated, left side)
    # Derive the row labels from the scenes actually plotted, so they cannot drift
    # out of step with --date/--snow-date the way hardcoded labels did.
    row_labels = [f"Snow-free\n({snow_free_ts:%b %Y})", f"Snow season\n({snow_ts:%b %Y})"]
    for i, lbl in enumerate(row_labels):
        axes[i, 0].text(-0.08, 0.5, lbl, transform=axes[i, 0].transAxes,
                        fontsize=8, fontweight="bold", rotation=90,
                        va="center", ha="right", color="#333333")

    flat = lambda a: a.ravel()

    def _fill_row(row, bcm, g_lons, g_lats, gts, vm, vla, vlo, vts, metar_lbl,
                  vplat=""):
        # GOES panel
        axes[row, 0].pcolormesh(g_lons, g_lats, bcm,
                                cmap=_CMAP_GOES, vmin=0, vmax=1,
                                transform=geo, shading="nearest", rasterized=True)
        _draw_goes_footprint(axes[row, 0], g_lons, g_lats, geo)
        axes[row, 0].plot(SITE_LON, SITE_LAT, "k+", ms=10, mew=1.5,
                          transform=geo, zorder=8)
        axes[row, 0].text(0.02, 0.02, gts.strftime("%Y-%m-%d %H:%M UTC"),
                          transform=axes[row, 0].transAxes, fontsize=6.5,
                          va="bottom", color="#222222")
        _draw_pixel_inset(axes[row, 0], bcm, g_lons, g_lats, geo,
                          _CMAP_GOES, 0, 1)

        # VIIRS panel
        ok = ~np.isnan(flat(vm))
        axes[row, 1].scatter(flat(vlo)[ok], flat(vla)[ok], c=flat(vm)[ok],
                             cmap=_CMAP_VIIRS, vmin=0, vmax=3,
                             s=0.8, linewidths=0, transform=geo,
                             zorder=4, rasterized=True)
        axes[row, 1].plot(SITE_LON, SITE_LAT, "+", color="white", ms=14, mew=2.5,
                          transform=geo, zorder=8)
        axes[row, 1].plot(SITE_LON, SITE_LAT, "+", color="black", ms=10, mew=1.5,
                          transform=geo, zorder=9)
        ts_txt = axes[row, 1].text(0.02, 0.02,
                                   (f"{vplat}\n" if vplat else "")
                                   + vts.strftime("%Y-%m-%d %H:%M UTC"),
                                   transform=axes[row, 1].transAxes, fontsize=6.5,
                                   va="bottom", color="#222222")
        ts_txt.set_path_effects([mpe.withStroke(linewidth=2, foreground="white")])
        _draw_viirs_box(axes[row, 1], geo)
        _draw_pixel_inset(axes[row, 1], vm, vlo, vla, geo,
                          _CMAP_VIIRS, 0, 3, overlay_color="white")

        # METAR strip — once per row, centred below the left panel
        axes[row, 0].text(1.02, -0.09, metar_lbl,
                          transform=axes[row, 0].transAxes,
                          fontsize=7, ha="center", va="top",
                          color="#222222",
                          bbox=dict(facecolor="#f0f0f0", edgecolor="#999999",
                                    boxstyle="round,pad=0.3", lw=0.7))

    _fill_row(0, bcm_sf, g_lons_sf, g_lats_sf, gts_sf,
              vm_sf, vla_sf, vlo_sf, vts_sf, metar_sf, sf_plat)
    _fill_row(1, bcm_sn, g_lons_sn, g_lats_sn, gts_sn,
              vm_sn, vla_sn, vlo_sn, vts_sn, metar_sn, sn_plat)


    # ── shared legend ─────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor="#2a2a2a", edgecolor="#aaaaaa", label="Cloudy"),
        mpatches.Patch(facecolor="#d8d8d8", edgecolor="#aaaaaa", label="Clear"),
        plt.Line2D([0],[0], marker="+", color="black", lw=0,
                   markersize=8, markeredgewidth=1.5, label=SITE_NAME),
    ]
    fig.legend(handles=legend_handles, fontsize=7.5, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               framealpha=0.92, edgecolor="#cccccc")

    fig.suptitle("Cloud mask comparison: GOES ABI vs VIIRS MVCM — Calgary area",
                 fontsize=9.5, y=0.96)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"figure_masks.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  → figures/figure_masks.pdf/.png")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Both published scenes are NIGHT overpasses with METAR OVC at CYYC, so the surface
    # truth is identical and the only difference between the rows is the ground state.
    # Minutes matter: ABI scans every 5 min, and the false-clear is a property of the
    # scene nearest the VIIRS overpass, not of the hour.
    ap.add_argument("--date",      default="2020-10-07",
                    help="Snow-free UTC date (default: 2020-10-07)")
    ap.add_argument("--hour",      type=int, default=9,
                    help="UTC hour for snow-free GOES scene (default: 9)")
    ap.add_argument("--minute",    type=int, default=6,
                    help="UTC minute for snow-free GOES scene (default: 6)")
    ap.add_argument("--snow-date", default="2021-02-12",
                    help="Snow-season UTC date (default: 2021-02-12)")
    ap.add_argument("--snow-hour", type=int, default=9,
                    help="UTC hour for snow-season GOES scene (default: 9)")
    ap.add_argument("--snow-minute", type=int, default=54,
                    help="UTC minute for snow-season GOES scene (default: 54)")
    # The rows may need different VIIRS satellites: the published snow scene is the
    # SNPP overpass, which is the one on which both masks fail together.
    ap.add_argument("--viirs-plat", default="NOAA20",
                    help="VIIRS platform for the snow-free row (default: NOAA20)")
    ap.add_argument("--snow-viirs-plat", default="SNPP",
                    help="VIIRS platform for the snow-season row (default: SNPP)")
    args = ap.parse_args()

    def _ts(date_str, hour, minute=0):
        d = dt.date.fromisoformat(date_str)
        return dt.datetime(d.year, d.month, d.day, hour, minute, 0,
                           tzinfo=dt.timezone.utc)

    snow_free_ts = _ts(args.date,      args.hour,      args.minute)
    snow_ts      = _ts(args.snow_date, args.snow_hour, args.snow_minute)

    print(f"Snow-free scene : {snow_free_ts.strftime('%Y-%m-%d %H:%M')} UTC  ({args.viirs_plat})")
    print(f"Snow season     : {snow_ts.strftime('%Y-%m-%d %H:%M')} UTC  ({args.snow_viirs_plat})")
    make_fig(snow_free_ts, snow_ts, args.viirs_plat, args.snow_viirs_plat)
