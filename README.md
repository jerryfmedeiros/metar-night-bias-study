[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21698181-blue)](https://doi.org/10.5281/zenodo.21698181)

# Human cloud reports overstate nighttime clearing

Code and data behind:

**Medeiros (2026).** *Human cloud reports overstate nighttime clearing: a ceilometer,
GOES, and VIIRS comparison at eight Canadian airports* (submitted to
J. Atmos. Oceanic Technol.). At all eight major Canadian airports, the human-augmented
record reports 7–14 pp more clear sky at night than by day, while a paired
automated ceilometer (at four of the cities), the GOES ABI clear-sky mask, and
scene-matched VIIRS overpasses see at most a few points of real clearing.

Every table value in the paper is reproducible from the data shipped here, with no
downloads. So is every figure except Figure 2, which plots two individual satellite
scenes and re-fetches them from S3 and LAADS (see *Rebuilding from scratch*). All
random steps (satellite sampling, bootstraps) are seeded; the default seed is 42 and
bootstraps use B = 2000.

## Layout

```
sites.py                        site registry: 8 cities, station coords, timezones
fetch_metar.py                  IEM ASOS METAR download + parser (disk cache)
fetch_goes.py                   GOES ABI L2 file location, download, pixel sampling
fetch_viirs.py                  VIIRS MVCM granule search (CMR), download, sampling
fetch_snow_cds.py               daily ERA5-Land snow depth per site, from the CDS

climatology_goes.py             GOES clear-sky sampler; --log-draws writes per-draw CSVs
edmonton_acmf_resample.py       reads the Edmonton draw logs' timestamps from the full-disk
                                ACMF product (Edmonton sits outside the ABI CONUS sector;
                                fetch_goes refuses out-of-sector stations rather than
                                clamping to the sector edge)
climatology_viirs.py            VIIRS sampler + same-scene METAR collocation
viirs_restore_granules.py       attaches per-scene granule provenance from CMR and
                                re-samples catalogue-version-conflicted scenes from the
                                current granule (the shipped logs are its output)
metar_climatology.py            METAR sky-state climatology, solar regimes, seasons
observable_nights.py            night-unit bias + the full-year difference-of-differences
fetch_all_sites.py              driver: AUTO screen + per-site analyses

human_deltas.py          human night-day deltas (full-year/snow-free/seasonal)
goes_nightly.py          night-unit GOES deltas from the per-draw logs
dynamic_tables.py               snow-free human/GOES deltas, H-G, DoD + threshold sweep;
                                --era {enterprise,baseline,all} selects the cloud-mask algorithm
                                behind the GOES columns (default enterprise, the paper's primary)
viirs_scenes.py                 matched VIIRS analysis from the per-scene logs
snow_falseclear.py              snow-season false-clear rate of both masks on
                                human-cloud night scenes (bare vs snow); the snow false-clear table
calgary_drift.py                per-year Calgary DoD + its 2020->2025 decomposition
                                and CYYC night-AUTO check (drift discussion)
discussion_stats.py             the Discussion/Limitations one-liners: cross-site night
                                correlations + effective N, Spearman latitude-vs-H-G,
                                48 site-year deltas + pair DoD slopes, diurnal
                                pre-dawn/afternoon stats, >12,000-ft layer shares,
                                VIIRS own night-day deltas
fetch_snow_insitu.py            airport daily snow-on-ground from ECCC (snow-filter check)
snow_insitu_validation.py       in-situ completeness + ERA5-vs-in-situ agreement,
                                behind the ERA5-Land snow-screen justification
make_tables.py                  regenerates the paper's main results table (tab:main) rows from the derived
                                tables; --check verifies a .tex against them
make_figures.py / figure_masks.py   figures

data/metar_cache/               raw METAR CSVs, 14 stations, 2020–2025 (from IEM): the 12
                                analysis stations (8 human-augmented + 4 automated partners
                                CYBW, CYHU, CYTZ, CZVL) plus CYXX and CYAW, two candidate
                                partners the night-%AUTO screen rejected as human-augmented.
                                CYKZ (Buttonville) has no IEM feed for 2020–2025, so nothing
                                to ship.
data/goes_draws/                per-draw GOES logs: timestamp, station, nearest-pixel
                                class, 3x3/5x5 box means. 110,000 draws over 11 site-family
                                runs (5,000 per regime per run; a draw whose scan is missing
                                from the archive is skipped and redrawn), including the
                                Vancouver-East/Calgary-West/Edmonton-West cross-checks. Each
                                draw is logged at every station pixel that shares it (human,
                                automated partner, and any candidate), so a single draw can
                                produce two or three rows. Edmonton lies outside the ABI
                                CONUS sector, so its two logs are read from the full-disk
                                ACMF product and carry a pix_km audit column (great-circle
                                distance station -> sampled pixel centre; <= 2.7 km).
                                The CYTZ and CZVL partner pixels were sampled at the
                                Toronto/Edmonton draw timestamps (see
                                partner_pixel_backfill.py) and merged into the east logs.
                                acmf_resample_edmonton_{east,west}.csv are the raw per-pixel
                                stage of the Edmonton full-disk read, kept as provenance
                                (edmonton_acmf_resample.py assembles them into the two logs).
data/viirs_matches/             per-scene VIIRS logs: timestamp, platform, decisions,
                                Moon state, METAR offset, and source granule (current
                                LAADS catalogue). One row per unique (timestamp, platform):
                                6,664–6,742 scenes per site, 53,514 total. Scenes the
                                catalogue listed under conflicting processing versions
                                during sampling (~100–170 per site) carry values re-read
                                from the current granule; granule_index_<site>.csv and
                                resolved_<site>.csv are the per-scene provenance.
data/snow_cache/                daily ERA5-Land snow depth per site, fetched from the
                                Copernicus CDS (snow filters). Each series is the valid
                                ERA5-Land land cell nearest the station; at Vancouver the
                                airport's own cell is water-masked and the nearest land
                                cell stands in (fetch_snow_cds.py prints the choice and
                                refuses to write an all-null cache)
data/snow_insitu/               airport daily snow-on-ground (ECCC), 8 sites, for the
                                snow-screen validation only (not used in the analysis itself)
data/derived/                   the tables behind the paper's numbers:
                                human_deltas.csv, goes_nightly.csv,
                                dynamic_tables.csv (+ _baseline / _pooled era variants),
                                viirs_scenes.csv, viirs_moon_split.csv,
                                viirs_threshold_sweep.csv, snow_falseclear.csv,
                                calgary_drift.csv, snow_insitu_validation.csv,
                                discussion_stats.csv
figures/                        final figure PDFs as submitted
checkpoints/                    gitignored sampler resume-state (goes_progress*.json,
                                viirs_progress*.json); not needed to reproduce the paper's
                                numbers from the shipped data/, only regenerated if you
                                rebuild a sampler run from scratch
```

## Reproducing the headline numbers (no downloads needed)

The shipped caches and logs are enough to rebuild every number in the paper.
One-time setup (all readers point at `data/` directly):

```bash
pip install numpy ephem            # netCDF4/matplotlib/pandas/cartopy only for sampling/figures
mkdir -p results
```

Then:

```bash
# Human night-day deltas, all sites and windows, + auto-partner DoD
python3 human_deltas.py

# Night-unit GOES deltas by month windows (full-year/seasonal)
python3 goes_nightly.py --glob 'data/goes_draws/goes_draws_*.csv'

# Snow-free (bare-ground) human/GOES deltas, H-G, DoD, with the 0.5-5 cm threshold sweep.
# The ABI cloud mask switched algorithm on 2021-11-29 (baseline threshold mask -> naive Bayesian
# Enterprise Cloud Mask). --era picks which one backs the GOES columns; the default, enterprise,
# is the paper's primary reference and reproduces its tables. The other two are reported in the
# paper alongside it.
python3 dynamic_tables.py                                            # -> results/dynamic_tables.csv
python3 dynamic_tables.py --era all      --csv results/dynamic_tables_pooled.csv
python3 dynamic_tables.py --era baseline --csv results/dynamic_tables_baseline.csv

# VIIRS matched-scene gaps (scene-level z + cluster-honest night-unit CIs),
# difference-of-deltas on both footprints, moon split, snow windows
python3 viirs_scenes.py

# Discussion/Limitations one-liners (correlations, Spearman, site-years, diurnal, layers)
python3 discussion_stats.py

# The 0.5-5 cm snow-threshold sweep for the VIIRS side
python3 viirs_scenes.py --snow-thresh 0.005 0.01 0.02 0.05 --csv-out results/viirs_threshold_sweep.csv

# Snow-season false-clear rate of both masks (the snow false-clear table): GOES draws matched to
# nearest METAR within +-30 min, VIIRS from the per-scene logs
python3 snow_falseclear.py

# Per-year Calgary DoD, its 2020->2025 decomposition, and the night-AUTO check
python3 calgary_drift.py

# Snow-screen justification: in-situ completeness + ERA5-vs-in-situ agreement (offline,
# from the shipped data/snow_insitu/ cache; fetch_snow_insitu.py rebuilds that from ECCC)
python3 snow_insitu_validation.py

# Full-year difference-of-differences against the automated partners
python3 observable_nights.py --site calgary --no-era5 --no-pm25
python3 observable_nights.py --site montreal --no-era5 --no-pm25
python3 observable_nights.py --site toronto --no-era5 --no-pm25
python3 observable_nights.py --site edmonton --no-era5 --no-pm25

# Regenerate the paper's main results table rows from the derived tables
python3 make_tables.py
```

Expected output is exactly `data/derived/*.csv` (same seeds, same data). The full-year
difference-of-differences printed by `observable_nights.py` matches the `auto_dod` rows in
`human_deltas.csv`: both resample the two stations from one shared rng stream, so the two
records are drawn independently of each other.

## Rebuilding from scratch

The samplers re-download from the public archives. This is slow (a few hours per
site for GOES; VIIRS granules are large) but fully resumable via checkpoints:

```bash
# METAR (fills data/metar_cache/)
python3 fetch_all_sites.py --sites all --screen-only

# GOES, with per-draw logging (writes goes_draws_<site>_<family>.csv)
python3 climatology_goes.py --site calgary --samples 5000 --seed 42 --log-draws --discard

# VIIRS, with per-scene logging (needs a NASA Earthdata token; see fetch_viirs.py)
python3 climatology_viirs.py --site calgary --samples 5000 --seed 42 --log-matches --discard
```

Fresh GOES/VIIRS draws are new random samples: point estimates move by well under
1 pp, and the shipped per-draw logs remain the record of the published runs.

## Data sources and terms

| Source | Provider | Access |
|---|---|---|
| METAR (12 stations, 2020–2025) | Iowa Environmental Mesonet ASOS archive | public |
| GOES-16/17/18/19 ABI L2 ACMC | NOAA Open Data Dissemination, AWS S3 | public, anonymous |
| VIIRS MVCM (CLDMSK_L2_VIIRS, C002) | NASA LAADS DAAC | free Earthdata login |

The shipped METAR cache is redistributed unmodified from the IEM archive to pin the
exact inputs of the published analysis.

## License and citation

Code is MIT-licensed (see LICENSE). The shipped data files remain under their
providers' terms above. If you use this, cite the JTECH paper (CITATION.cff has the
metadata; DOI to follow on acceptance).
