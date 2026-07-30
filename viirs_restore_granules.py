"""
viirs_restore_granules.py — restore per-scene granule provenance to the VIIRS
match logs, and resolve the version-conflicted duplicate scenes.

Why: the released viirs_matches_<site>.csv logs predate the granule column the
current climatology_viirs.py writes, and the LAADS catalogue listed some scenes
under successive processing versions during the sampling runs. About 1,060
(ts_utc, platform) keys across the eight sites therefore carry TWO rows with
conflicting satellite decisions and nothing to say which processing version
produced which. The analysis kept the first row in file order, an arbitrary
choice (headline effect <= 0.5 pp on snow-free ground, but it should not be
arbitrary at all).

Three stages, all resumable, run in order:

  # 1. metadata (anonymous CMR, no downloads): attach the current-catalogue
  #    granule file name + URL to every unique logged scene
  python3 viirs_restore_granules.py --stage metadata [--sites all]

  # 2. resolve (LAADS downloads, needs EARTHDATA_TOKEN or ~/.edl_token):
  #    re-sample ONLY the conflicted keys from the current granule, with the
  #    same nearest-pixel and 7x7-box rules as the published run
  python3 viirs_restore_granules.py --stage resolve [--sites all] [--keep-cache]

  # 3. emit (offline): write viirs_matches_<site>_v2.csv — one row per unique
  #    (ts_utc, platform), granule column on every row, conflicted keys carrying
  #    the stage-2 authoritative values
  python3 viirs_restore_granules.py --stage emit [--sites all]

Outputs (data/viirs_matches/):
  granule_index_<site>.csv    ts_utc, platform, granule, url        (stage 1)
  resolved_<site>.csv         authoritative values for conflicts    (stage 2)
  viirs_matches_<site>_v2.csv patched, deduplicated log             (stage 3)

The v2 logs collapse the ~3,300 byte-identical duplicate rows per site as well;
consumers already dedup on (ts_utc, platform), so their numbers change only via
the ~100-170 resolved conflicts per site.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

import fetch_viirs as fv
from sites import SITES

MATCH_DIR = Path("data/viirs_matches")
SLUGS = ["calgary", "edmonton", "halifax", "montreal", "ottawa", "toronto",
         "vancouver", "winnipeg"]


def load_log(slug):
    with open(MATCH_DIR / f"viirs_matches_{slug}.csv") as fh:
        return list(csv.DictReader(fh))


def unique_keys(rows):
    """(ts_utc, platform) -> list of rows, in first-appearance order."""
    by = defaultdict(list)
    order = []
    for r in rows:
        k = (r["ts_utc"], r["platform"])
        if k not in by:
            order.append(k)
        by[k].append(r)
    return by, order


def conflicted(by):
    return {k: v for k, v in by.items()
            if len(v) > 1 and len({(r["pixel_clear"], r["box_clear"]) for r in v}) > 1}


# ---------------------------------------------------------------------------
# stage 1: metadata
# ---------------------------------------------------------------------------

def stage_metadata(slug):
    site = SITES[slug]
    lat, lon = site.coords[site.human]
    rows = load_log(slug)
    by, order = unique_keys(rows)

    out = MATCH_DIR / f"granule_index_{slug}.csv"
    done = set()
    if out.exists():
        with open(out) as fh:   # only non-empty hits count as done; misses retry
            done = {(r["ts_utc"], r["platform"])
                    for r in csv.DictReader(fh) if r["granule"]}
    is_new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if is_new:
        w.writerow(["ts_utc", "platform", "granule", "url"])

    todo = [k for k in order if k not in done]
    # group needed keys by (platform, year, month): one CMR query per group
    groups = defaultdict(list)
    for ts_s, plat in todo:
        t = dt.datetime.fromisoformat(ts_s)
        groups[(plat, t.year, t.month)].append((ts_s, t))
    print(f"[{slug}] {len(by)} unique scenes, {len(done)} indexed, "
          f"{len(todo)} to do in {len(groups)} CMR queries", flush=True)

    n_hit = n_miss = 0
    for (plat, y, m), keys in sorted(groups.items()):
        start = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
        end = (dt.datetime(y + 1, 1, 1, tzinfo=dt.timezone.utc) if m == 12
               else dt.datetime(y, m + 1, 1, tzinfo=dt.timezone.utc))
        grans = fv.cmr_granules(plat, lat, lon, start, end, page_size=2000)
        by_time = {t.replace(second=0): u for u, t in grans}
        for ts_s, t in keys:
            url = by_time.get(t.replace(second=0))
            if url is None and grans:   # tolerate <=3 min drift
                u, tt = min(grans, key=lambda g: abs((g[1] - t).total_seconds()))
                if abs((tt - t).total_seconds()) <= 180:
                    url = u
            if url:
                w.writerow([ts_s, plat, url.rsplit("/", 1)[-1], url])
                n_hit += 1
            else:
                w.writerow([ts_s, plat, "", ""])
                n_miss += 1
        fh.flush()
    fh.close()
    print(f"[{slug}] indexed {n_hit} granules, {n_miss} not found in CMR", flush=True)


# ---------------------------------------------------------------------------
# stage 2: resolve conflicts
# ---------------------------------------------------------------------------

def stage_resolve(slug, keep_cache):
    site = SITES[slug]
    lat, lon = site.coords[site.human]
    rows = load_log(slug)
    by, _ = unique_keys(rows)
    conf = conflicted(by)

    idx = {}
    with open(MATCH_DIR / f"granule_index_{slug}.csv") as fh:
        for r in csv.DictReader(fh):
            idx[(r["ts_utc"], r["platform"])] = (r["granule"], r["url"])

    out = MATCH_DIR / f"resolved_{slug}.csv"
    done = set()
    if out.exists():
        with open(out) as fh:
            done = {(r["ts_utc"], r["platform"]) for r in csv.DictReader(fh)}
    is_new = not out.exists()
    fh = open(out, "a", newline="")
    w = csv.writer(fh)
    if is_new:
        w.writerow(["ts_utc", "platform", "pixel_clear", "box_clear",
                    "granule", "pixel_dist_km"])

    token = fv.earthdata_token()
    todo = [k for k in conf if k not in done]
    print(f"[{slug}] {len(conf)} conflicted keys, {len(done)} resolved, "
          f"{len(todo)} to do", flush=True)
    n_done = n_fail = 0
    for k in todo:
        granule, url = idx.get(k, ("", ""))
        if not url:
            n_fail += 1
            continue
        try:
            local = fv.download_cached(url, token)
            try:
                px = fv.sample_granule(local, lat, lon)
                bx = fv.sample_granule_box(local, lat, lon)
            finally:
                if not keep_cache:
                    local.unlink(missing_ok=True)
            if px is None or bx is None:
                n_fail += 1
                continue
            w.writerow([k[0], k[1], int(px[0] == "CLEAR"), int(bx[0] == "CLEAR"),
                        granule, f"{px[1]:.2f}"])
            n_done += 1
            if n_done % 20 == 0:
                fh.flush()
                print(f"  [{slug}] {n_done}/{len(todo)} resolved ({n_fail} failed)",
                      flush=True)
        except KeyboardInterrupt:
            print("interrupted; resume by re-running", flush=True)
            break
        except Exception as e:
            n_fail += 1
            if n_fail <= 5:
                print(f"  [{slug}] fail {k}: {type(e).__name__} {e}", flush=True)
    fh.close()
    print(f"[{slug}] resolve done: {n_done} new, {n_fail} failed", flush=True)


# ---------------------------------------------------------------------------
# stage 3: emit patched logs
# ---------------------------------------------------------------------------

def stage_emit(slug):
    rows = load_log(slug)
    by, order = unique_keys(rows)
    conf = conflicted(by)

    idx = {}
    with open(MATCH_DIR / f"granule_index_{slug}.csv") as fh:
        for r in csv.DictReader(fh):
            idx[(r["ts_utc"], r["platform"])] = r["granule"]
    res = {}
    p = MATCH_DIR / f"resolved_{slug}.csv"
    if p.exists():
        with open(p) as fh:
            for r in csv.DictReader(fh):
                res[(r["ts_utc"], r["platform"])] = r
    missing = [k for k in conf if k not in res]
    if missing:
        sys.exit(f"[{slug}] {len(missing)} conflicted keys unresolved — "
                 f"run --stage resolve first (e.g. {missing[0]})")

    out = MATCH_DIR / f"viirs_matches_{slug}_v2.csv"
    fields = list(rows[0].keys()) + ["granule"]
    n_conf = 0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k in order:
            row = dict(by[k][0])           # first row carries the shared fields
            row["granule"] = idx.get(k, "")
            if k in conf:
                row["pixel_clear"] = res[k]["pixel_clear"]
                row["box_clear"] = res[k]["box_clear"]
                row["granule"] = res[k]["granule"]
                n_conf += 1
            w.writerow(row)
    print(f"[{slug}] wrote {out}: {len(order)} rows "
          f"({len(rows) - len(order)} duplicates collapsed, "
          f"{n_conf} conflicts resolved authoritatively)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("metadata", "resolve", "emit"))
    ap.add_argument("--sites", nargs="+", default=["all"])
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()
    slugs = SLUGS if args.sites == ["all"] else args.sites
    for slug in slugs:
        if args.stage == "metadata":
            stage_metadata(slug)
        elif args.stage == "resolve":
            stage_resolve(slug, args.keep_cache)
        else:
            stage_emit(slug)


if __name__ == "__main__":
    main()
