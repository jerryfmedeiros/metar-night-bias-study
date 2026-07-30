"""
make_figures.py — figures for Paper 1 (methods-paper-draft.tex)

Generates:
  figures/fig1_map.pdf      — map of 8 Canadian study sites
  figures/fig3_forest.pdf   — forest plot: Human Δ vs GOES Δ, all 8 cities
  figures/fig4_diurnal.pdf  — diurnal cloud fraction by local hour

Run from the project root:
  python3 make_figures.py
"""

import csv
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Site data (from the paper's main results table / server goes.txt outputs)
# (name, lat, lon, human_delta, h_ci_lo, h_ci_hi,
#  goes_delta, g_ci_lo, g_ci_hi, goes_family)
# GOES CIs: night-unit bootstrap from the logged rerun (goes_nightly.py).
# Vancouver uses GOES-West family.
# ---------------------------------------------------------------------------
# Geography and station identity are fixed; the deltas are not. They are read from
# data/derived/dynamic_tables.csv (1 cm snow threshold, night-unit bootstrap CIs, snow-free
# on both sides) so this figure cannot drift from the table the way a transcribed copy can.
# goes_nearest is already each site's primary family -- West at Vancouver, East elsewhere.
_GEO = [
    # (name, lat, lon, icao, goes_family)
    ("Toronto",   43.7,  -79.6, "CYYZ", "East"),
    ("Halifax",   44.9,  -63.5, "CYHZ", "East"),
    ("Ottawa",    45.3,  -75.7, "CYOW", "East"),
    ("Montréal",  45.5,  -73.7, "CYUL", "East"),
    ("Vancouver", 49.2, -123.2, "CYVR", "West"),
    ("Winnipeg",  49.9,  -97.2, "CYWG", "East"),
    ("Calgary",   51.1, -114.0, "CYYC", "East"),
    ("Edmonton",  53.3, -113.6, "CYEG", "East"),
]


def _load_sites(path="data/derived/dynamic_tables.csv", thresh="1.0"):
    """(name, lat, lon, human_d, h_lo, h_hi, goes_d, g_lo, g_hi, family) from the table."""
    vals = {}
    for r in csv.DictReader(open(path)):
        if r["threshold_cm"] != thresh:
            continue
        vals[(r["site"], r["record"])] = (
            float(r["delta_pp"]), float(r["ci_lo"]), float(r["ci_hi"]))
    out = []
    for name, lat, lon, icao, fam in _GEO:
        slug = name.lower().replace("é", "e")
        try:
            h = vals[(slug, f"metar_{icao}")]
            g = vals[(slug, "goes_nearest")]
        except KeyError as e:
            raise SystemExit(f"make_figures: missing {e} in {path}")
        out.append((name, lat, lon, *h, *g, fam))
    return out


SITES = _load_sites()

ICAO = {
    "Toronto":   "CYYZ", "Halifax":   "CYHZ", "Ottawa":    "CYOW",
    "Montréal":  "CYUL", "Vancouver": "CYVR", "Winnipeg":  "CYWG",
    "Calgary":   "CYYC", "Edmonton":  "CYEG",
}

# Okabe-Ito colorblind-safe palette.
# Circle  + vermillion = GOES Δ significantly < 0 (nights genuinely cloudier)
# Diamond + blue       = GOES Δ significantly > 0 (nights genuinely clearer)
# Square  + gray       = CI spans zero (no resolvable diurnal signal)
GOES_COLOR  = {True: "#D55E00", False: "#0072B2"}   # True = GOES Δ < 0
GOES_MARKER = {True: "o",       False: "D"}
FLAT_COLOR, FLAT_MARKER = "#888888", "s"
# The GOES-East and GOES-West families agree in sign at every dual-view site
# (Vancouver, Calgary, Edmonton), so no site needs a "families disagree" class.
CONTESTED = set()
CONTESTED_COLOR, CONTESTED_MARKER = "#000000", "X"


def goes_class(gd, g_lo, g_hi):
    """Sign of the GOES delta, but only where the CI actually resolves it.

    A binary test on the point estimate alone would label Vancouver (+0.4 pp, CI spanning
    zero) as 'nights genuinely clearer', which contradicts the text.
    """
    if g_hi < 0:
        return "cloudier"
    if g_lo > 0:
        return "clearer"
    return "flat"


def goes_style(gd, g_lo, g_hi, name=None):
    """(marker, color) for a site's snow-free GOES night-day signal."""
    if name in CONTESTED:
        return CONTESTED_MARKER, CONTESTED_COLOR
    cls = goes_class(gd, g_lo, g_hi)
    if cls == "flat":
        return FLAT_MARKER, FLAT_COLOR
    neg = cls == "cloudier"
    return GOES_MARKER[neg], GOES_COLOR[neg]

# ---------------------------------------------------------------------------
# Figure 1 — Map (cartopy, Lambert Conformal centred on Canada)
# ---------------------------------------------------------------------------
def make_map():
    print("Figure 1: map …")

    proj = ccrs.LambertConformal(
        central_longitude=-96, central_latitude=49,
        standard_parallels=(49, 77),
    )
    data_crs = ccrs.PlateCarree()

    # Leave bottom 18% of figure for legend + footnote outside the map
    fig = plt.figure(figsize=(5.5, 4.2))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.18)

    ax.set_extent([-135, -52, 41, 61], crs=data_crs)

    # Draw order: BORDERS first, then OCEAN/LAKES on top to mask any
    # border lines that cross water bodies (per AMS figure policy).
    ax.add_feature(cfeature.LAND.with_scale("50m"),
                   facecolor="#f5f5f0", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#cccccc", lw=0.4, zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                   edgecolor="#888888", lw=0.7, zorder=2)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),
                   facecolor="#d0d0d0", zorder=3)
    ax.add_feature(cfeature.LAKES.with_scale("50m"),
                   facecolor="#d0d0d0", zorder=4)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#888888", lw=0.5, zorder=5)

    # Subtle gridlines — no labels (labels in Lambert Conformal appear
    # at odd positions; the coast and city names orient the reader)
    ax.gridlines(lw=0.25, color="#bbbbbb", linestyle="--",
                 xlocs=range(-130, -50, 10), ylocs=range(42, 62, 4))

    # ------------------------------------------------------------------
    # Pure network map: marker shape encodes station pairing, nothing else.
    # Square = city with a qualified automated partner (auto_dod row in the
    #          derived tables); circle = human-augmented station only.
    # The GOES night-day verdicts live in the forest plot (fig3), with CIs,
    # where they belong; encoding them here too invited legend rot.
    # ------------------------------------------------------------------
    paired = {r["site"] for r in csv.DictReader(open("data/derived/dynamic_tables.csv"))
              if r["record"] == "auto_dod"}
    is_paired = lambda name: name.lower().replace("é", "e") in paired
    for name, lat, lon, hd, lo, hi, gd, g_lo, g_hi, fam in SITES:
        marker = "s" if is_paired(name) else "o"
        ax.plot(lon, lat, marker, ms=8 if marker == "s" else 7, color="#000000",
                markeredgecolor="white", markeredgewidth=0.6,
                transform=data_crs, zorder=6)

    # ------------------------------------------------------------------
    # Site labels — offsets tuned to avoid overlap at 5.5-in width
    # (dlon, dlat, ha, va)
    # ------------------------------------------------------------------
    label_cfg = {
        "Vancouver": (-2.5,  +1.2, "right", "bottom"),
        "Calgary":   (+1.3,  -1.5, "left",  "top"),
        "Edmonton":  (+1.3,  +0.9, "left",  "bottom"),
        "Winnipeg":  (+1.8,  +0.5, "left",  "center"),
        "Toronto":   (+1.5,  -1.3, "left",  "top"),
        "Ottawa":    (-1.0,  +1.2, "right", "bottom"),
        "Montréal":  (+1.3,  +0.8, "left",  "bottom"),
        "Halifax":   (+1.5,   0.0, "left",  "center"),
    }

    for name, lat, lon, *rest in SITES:
        dlon, dlat, ha, va = label_cfg[name]
        fam = rest[-1]
        suffix = "*" if fam == "West" else ""
        ax.text(
            lon + dlon, lat + dlat,
            f"{name}\n({ICAO[name]}){suffix}",
            transform=data_crs,
            fontsize=6, ha=ha, va=va, zorder=7,
            linespacing=1.2,
        )

    ax.set_title(
        "Study sites: 8 staffed Canadian airports, 2020–2025",
        fontsize=8, pad=4,
    )

    # ------------------------------------------------------------------
    # Legend and footnote OUTSIDE the map (in the bottom margin)
    # ------------------------------------------------------------------
    leg_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#000000",
               markeredgecolor="#000000", markersize=7,
               label="human-augmented station"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#000000",
               markeredgecolor="#000000", markersize=8,
               label="+ qualified automated partner (20–44 km)"),
    ]
    fig.legend(handles=leg_handles, fontsize=7, loc="lower center",
               bbox_to_anchor=(0.5, 0.045), ncol=2,
               framealpha=0.9, edgecolor="#cccccc")

    fig.text(0.5, 0.01,
             "* Vancouver (CYVR) is compared against GOES-West; all other sites"
             " against GOES-East.",
             ha="center", va="bottom", fontsize=6, color="#555555")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig1_map.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → figures/fig1_map.pdf/.png")


# ---------------------------------------------------------------------------
# Figure 3 — Forest plot
# ---------------------------------------------------------------------------
def make_forest():
    print("Figure 3: forest plot …")

    # Ordered south → north (bottom to top)
    ordered = sorted(SITES, key=lambda r: r[1])

    names   = [r[0]  for r in ordered]
    hd      = np.array([r[3]  for r in ordered])
    ci_lo   = np.array([r[4]  for r in ordered])
    ci_hi   = np.array([r[5]  for r in ordered])
    gd      = np.array([r[6]  for r in ordered])
    g_ci_lo = np.array([r[7]  for r in ordered])
    g_ci_hi = np.array([r[8]  for r in ordered])

    n = len(names)
    y = np.arange(n)

    # AMS two-column width = 5.5 in
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Grayscale encoding: shape + line weight discriminate the two series.
    # GOES  → open square,   medium gray (#666666), thin CI bars  (lw=1.0)
    # Human → filled circle, black,                 thick CI bars (lw=1.8)
    _GOES_C  = "#666666"
    _HUMAN_C = "#000000"

    # --- GOES Δ (open gray square + Wilson 95% CI) ---
    ax.scatter(gd, y, marker="s", s=48, facecolors="none",
               edgecolors=_GOES_C, lw=1.5, zorder=4,
               label="GOES Δ, snow-free ground (night-unit bootstrap 95% CI)")
    for i in range(n):
        ax.plot([g_ci_lo[i], g_ci_hi[i]], [y[i], y[i]],
                color=_GOES_C, lw=1.0, zorder=3)
        for x_cap in (g_ci_lo[i], g_ci_hi[i]):
            ax.plot([x_cap, x_cap], [y[i]-0.09, y[i]+0.09],
                    color=_GOES_C, lw=1.0, zorder=3)

    # --- Human Δ with bootstrap CI ---
    ax.scatter(hd, y, marker="o", s=52, color=_HUMAN_C,
               zorder=5, label="Human Δ (bootstrap 95% CI)")
    for i in range(n):
        ax.plot([ci_lo[i], ci_hi[i]], [y[i], y[i]],
                color=_HUMAN_C, lw=1.8, zorder=4)
        for x_cap in (ci_lo[i], ci_hi[i]):
            ax.plot([x_cap, x_cap], [y[i]-0.13, y[i]+0.13],
                    color=_HUMAN_C, lw=1.8, zorder=4)

    # --- Dotted connector: GOES → Human (shows the H-G gap) ---
    for i in range(n):
        ax.plot([gd[i], hd[i]], [y[i], y[i]],
                color="#aaaaaa", lw=0.9, ls=":", zorder=3)

    # --- Reference line at 0 ---
    ax.axvline(0, color="black", lw=1.0, zorder=2)

    # H-G = +21.0 pp for Calgary (snow-free) is called out in the figure caption.

    # --- Y-axis labels ---
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{name}  ({ICAO[name]})" for name in names],
        fontsize=8,
    )

    # --- X-axis ---
    ax.set_xlim(-13, 22)
    ax.set_xlabel("Night − Day clear-or-few fraction (pp)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(2))
    ax.grid(axis="x", lw=0.4, color="#dddddd", zorder=0)

    # --- Legend ---
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=2, framealpha=0.92, edgecolor="#cccccc")

    ax.set_title(
        "Human-augmented vs GOES satellite: night−day clear-sky bias\n"
        "All 8 cities. Human 2020–2025; GOES Enterprise mask, Dec 2021–2025.\n"
        "Dotted line = human − satellite difference",
        fontsize=8,
    )

    fig.tight_layout(pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig3_forest.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → figures/fig3_forest.pdf/.png")


# ---------------------------------------------------------------------------
# Figure 4 — Diurnal cloud fraction
# ---------------------------------------------------------------------------

# Grayscale site styles: shade × line style × marker shape — three discriminants.
# (color, linestyle, marker, markevery_offset)
# markevery period=6 → 4 marks per 24-h trace; staggered offsets avoid pileup.
_SITE_STYLES = [
    ("#000000", "-",  "o", 0),   # black solid,       circle
    ("#000000", "--", "s", 3),   # black dashed,      square
    ("#3d3d3d", "-",  "^", 1),   # dark gray solid,   triangle-up
    ("#3d3d3d", "--", "D", 4),   # dark gray dashed,  diamond
    ("#737373", "-",  "v", 2),   # medium gray solid, triangle-down
    ("#737373", "--", "p", 5),   # medium gray dashed,pentagon
    ("#a0a0a0", "-",  "x", 0),   # light gray solid,  cross
    ("#a0a0a0", "--", "*", 3),   # light gray dashed, star
]

def _parse_clear_few(metar_str: str):
    """True if the sky is clear-or-few (okta <= 2), via the study parser.

    Uses fetch_metar.parse_metar_cloud so this figure's clear/few decision is
    the same one behind every table (an earlier local regex dropped reports
    with no parseable sky group, which the study parser scores clear)."""
    from fetch_metar import parse_metar_cloud
    return parse_metar_cloud(metar_str)[0] <= 2


def _load_hourly_clear(icao: str, tz_name: str):
    """
    Read all annual METAR CSVs for *icao* from the METAR cache and return
    a Series of clear-or-few fraction (0–100) indexed by local hour 0–23.
    Returns None if no files found.

    The cache sits at data/metar_cache/ in the packaged release and metar_cache/ in the
    working tree; check both so this figure builds in either layout rather than silently
    skipping, which is how it went missing from a release build.
    """
    try:
        import pandas as pd
        from zoneinfo import ZoneInfo
    except ImportError:
        import pandas as pd
        from datetime import timezone
        ZoneInfo = None

    files = []
    for cache_dir in ("data/metar_cache", "metar_cache"):
        files = sorted(glob.glob(f"{cache_dir}/{icao}_*.csv"))
        if files:
            break
    if not files:
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=["valid", "metar"])
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True)
    data.drop_duplicates(subset="valid", inplace=True)

    data["valid"] = pd.to_datetime(data["valid"], utc=True)
    if ZoneInfo:
        tz = ZoneInfo(tz_name)
        data["local_hour"] = data["valid"].dt.tz_convert(tz).dt.hour
    else:
        data["local_hour"] = data["valid"].dt.hour   # fallback UTC

    data["clear_few"] = data["metar"].apply(_parse_clear_few)
    data = data.dropna(subset=["clear_few"])
    if data.empty:
        return None

    hourly = data.groupby("local_hour")["clear_few"].mean() * 100
    return hourly.reindex(range(24))


def make_diurnal():
    """
    Five-panel figure (gridspec 3×2, top panel spans both columns).
    (a) All human sites overlaid.
    (b)-(e) each paired city: human station vs. its automated partner
    (Calgary/CYBW, Montréal/CYHU, Toronto/CYTZ, Edmonton/CZVL).
    All legends placed outside the chart area.
    """
    print("Figure 4: diurnal …")
    try:
        from sites import SITES as SITE_REGISTRY
    except ImportError:
        print("  sites.py not found; skipping diurnal figure.")
        return

    hours = np.arange(24)

    # ── collect curves ────────────────────────────────────────────────────────
    human_curves = {}
    auto_curves  = {}

    for i, (slug, site) in enumerate(SITE_REGISTRY.items()):
        color, ls, marker, moff = _SITE_STYLES[i % len(_SITE_STYLES)]
        h = _load_hourly_clear(site.human, site.iana_tz)
        if h is not None:
            human_curves[site.name] = (h, color, ls, marker, moff, site.human)
        if site.auto:
            a = _load_hourly_clear(site.auto, site.iana_tz)
            if a is not None:
                auto_curves[site.name] = (a, site.auto)

    if not human_curves:
        print("  no METAR data found in data/metar_cache/ or metar_cache/; "
              "skipping diurnal figure.")
        return

    # ── layout ────────────────────────────────────────────────────────────────
    # Right margin left open for the 8-site legend (avoids collision with
    # bottom panel titles that would occur if the legend sat below the top panel).
    fig = plt.figure(figsize=(7.5, 6.4))
    fig.subplots_adjust(left=0.07, right=0.72, top=0.93, bottom=0.10)
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 1, 1],
        hspace=0.55,
        wspace=0.35,
    )
    ax_all = fig.add_subplot(gs[0, :])   # top row, both columns
    ax_yyc = fig.add_subplot(gs[1, 0])   # (b) Calgary
    ax_yul = fig.add_subplot(gs[1, 1])   # (c) Montréal
    ax_yyz = fig.add_subplot(gs[2, 0])   # (d) Toronto
    ax_yeg = fig.add_subplot(gs[2, 1])   # (e) Edmonton

    # ── (a) all human sites ───────────────────────────────────────────────────
    handles_all = []
    for name, (series, color, ls, marker, moff, icao) in human_curves.items():
        ax_all.plot(hours, series.values, color=color, ls=ls, lw=1.2,
                    marker=marker, markersize=4, markevery=(moff, 6),
                    markeredgewidth=0.6, markerfacecolor=color)
        handles_all.append(
            Line2D([0],[0], color=color, ls=ls, lw=1.2,
                   marker=marker, markersize=4, label=f"{name} ({icao})")
        )
    ax_all.set_xlim(0, 23)
    ax_all.set_xticks([0, 6, 12, 18, 23])
    ax_all.set_xlabel("Local hour", fontsize=7)
    ax_all.set_ylabel("Clear-or-few fraction (%)", fontsize=7)
    ax_all.set_title("(a) Human METAR — all sites", fontsize=7.5)
    ax_all.tick_params(labelsize=6.5)
    ax_all.grid(lw=0.3, color="#dddddd")
    # legend to the RIGHT of the top panel (avoids vertical collision below)
    ax_all.legend(
        handles=handles_all, fontsize=6, ncol=1,
        loc="center left", bbox_to_anchor=(1.03, 0.5),
        framealpha=0.9, edgecolor="#cccccc", handlelength=2.2,
        borderpad=0.6,
    )

    # ── helper: plot one human-vs-auto comparison panel (no per-panel legend) ─
    def _comparison_panel(ax, city_name, h_icao, a_icao, title):
        entry_h = human_curves.get(city_name)
        entry_a = auto_curves.get(city_name)
        if entry_h:
            ax.plot(hours, entry_h[0].values, color="#000000", lw=1.5,
                    marker="o", markersize=4, markevery=(0, 6),
                    markeredgewidth=0.6, label=f"{h_icao} (human-augmented)")
        if entry_a:
            ax.plot(hours, entry_a[0].values, color="#555555", lw=1.5,
                    ls="--", marker="s", markersize=4, markevery=(3, 6),
                    markeredgewidth=0.6, label=f"{a_icao} (automated)")
        ax.set_xlim(0, 23)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("Local hour", fontsize=7)
        ax.set_ylabel("Clear-or-few fraction (%)", fontsize=7)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(lw=0.3, color="#dddddd")

    _comparison_panel(ax_yyc, "Calgary",  "CYYC", "CYBW",
                      "(b) Calgary: human vs. automated")
    _comparison_panel(ax_yul, "Montreal", "CYUL", "CYHU",
                      "(c) Montréal: human vs. automated")
    _comparison_panel(ax_yyz, "Toronto",  "CYYZ", "CYTZ",
                      "(d) Toronto: human vs. automated")
    _comparison_panel(ax_yeg, "Edmonton", "CYEG", "CZVL",
                      "(e) Edmonton: human vs. automated")

    # Shared legend for (b)-(e) — colour coding is identical
    shared = [
        Line2D([0],[0], color="#000000", lw=1.5, marker="o", markersize=4,
               label="Human-augmented station"),
        Line2D([0],[0], color="#555555", lw=1.5, ls="--", marker="s", markersize=4,
               label="Automated partner"),
    ]
    fig.legend(handles=shared, fontsize=6.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.01),
               framealpha=0.9, edgecolor="#cccccc", handlelength=1.8)

    fig.suptitle(
        "Diurnal clear-or-few fraction (SKC/CLR/FEW), 2020–2025",
        fontsize=8.5, y=0.98,
    )

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig4_diurnal.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  → figures/fig4_diurnal.pdf/.png")


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Artifact panel: two independent estimates of the human nighttime excess (pp),
# snow-free ground (1 cm). Read from the derived tables so the figure can never
# drift from them -- H-G (paired night/day-unit bootstrap) from dynamic_tables.csv,
# VIIRS matched difference-of-deltas (night/day-unit bootstrap) from viirs_scenes.csv.
# ---------------------------------------------------------------------------
ARTIFACT_SITES = ["Toronto", "Halifax", "Ottawa", "Montréal", "Vancouver",
                  "Winnipeg", "Calgary", "Edmonton"]


def load_artifact():
    """(name, hg, hg_lo, hg_hi, vdod, v_lo, v_hi) per site, from the derived CSVs."""
    hg, vd = {}, {}
    for r in csv.DictReader(open("data/derived/dynamic_tables.csv")):
        if r["record"] == "human_minus_goes" and r["threshold_cm"] == "1.0":
            hg[r["site"]] = (float(r["delta_pp"]), float(r["ci_lo"]), float(r["ci_hi"]))
    for r in csv.DictReader(open("data/derived/viirs_scenes.csv")):
        if r["window"] == "dynamic" and r["snow_thresh_cm"] == "1.0":
            vd[r["site"]] = (float(r["dod_unit"]), float(r["dod_unit_lo"]), float(r["dod_unit_hi"]))
    return [(name, *hg[name.lower().replace("é", "e")], *vd[name.lower().replace("é", "e")])
            for name in ARTIFACT_SITES]


LATS = {name: lat for name, lat, *_ in SITES}


def make_artifact():
    print("Figure: artifact two-estimate forest …")
    ordered = sorted(load_artifact(), key=lambda r: LATS[r[0]])
    names = [r[0] for r in ordered]
    hg   = np.array([r[1] for r in ordered])
    hglo = np.array([r[2] for r in ordered]); hghi = np.array([r[3] for r in ordered])
    vd   = np.array([r[4] for r in ordered])
    vlo  = np.array([r[5] for r in ordered]); vhi = np.array([r[6] for r in ordered])
    n = len(names)
    y = np.arange(n)
    off = 0.17

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter(hg, y + off, marker="o", s=52, color="#000000", zorder=5,
               label="Human excess over GOES (H−G, paired night-unit 95% CI)")
    for i in range(n):
        ax.plot([hglo[i], hghi[i]], [y[i] + off] * 2, color="#000000", lw=1.6, zorder=4)
    ax.scatter(vd, y - off, marker="s", s=48, facecolors="none", edgecolors="#666666",
               lw=1.5, zorder=4,
               label="Human excess over VIIRS (matched DoD, night-unit 95% CI)")
    for i in range(n):
        ax.plot([vlo[i], vhi[i]], [y[i] - off] * 2, color="#666666", lw=1.1, zorder=3)
    ax.axvline(0, color="black", lw=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{nm}\n({ICAO[nm]})" for nm in names], fontsize=8)
    ax.set_xlabel("Human nighttime clear-sky excess over satellite (pp)", fontsize=9)
    ax.set_title("Two independent estimates of the observer artifact\n"
                 "snow-free ground, 2020–2025", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", lw=0.3, color="#dddddd", zorder=1)
    fig.legend(fontsize=7, loc="lower center", ncol=1, framealpha=0.9,
               edgecolor="#cccccc", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig5_artifact.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  → figures/fig5_artifact.pdf/.png")


if __name__ == "__main__":
    make_map()
    make_forest()
    make_artifact()
    make_diurnal()
    print("Done.")
