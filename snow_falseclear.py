#!/usr/bin/env python3
"""snow_falseclear.py — snow-season false-clear rate for both satellite masks.

On night scenes where the observer reported cloud (okta > 2), how often does each
mask call the station clear, split by bare vs snow-covered ground? This is the rate
behind Figure 2, which shows the two masks failing together over snow on one pair of
scenes, and behind the snow-free restriction that removes it.

VIIRS reads straight from the per-scene match logs (data/viirs_matches/), which
already carry the human okta<=2 flag and the 7x7-box decision. GOES draws are not
logged against a report, so each night draw at the human pixel is matched to the
nearest METAR within +/-30 min (VIIRS parity), okta parsed with the study's mapping
(fetch_metar). Each mask is scored at its primary footprint: GOES nearest pixel,
VIIRS box. Snow days and the local-noon-to-noon night unit follow viirs_scenes.

Writes one row per (site, mask, ground) to results/snow_falseclear.csv.
"""
import argparse
import bisect
import csv
import datetime as dt
import glob
from zoneinfo import ZoneInfo

from sites import SITES
from fetch_metar import parse_metar_cloud
from viirs_scenes import snow_days

SNOW_THRESH_M = 0.01          # > 1 cm daily-max ERA5-Land snow depth = snow day
MATCH_TOL_S = 1800            # +-30 min GOES-draw-to-METAR tolerance (VIIRS parity)


def night_unit(ts_utc, tz):
    """Local-noon-to-noon night-unit date for a UTC scene timestamp."""
    return (ts_utc.astimezone(tz) - dt.timedelta(hours=12)).date()


def metar_index(human):
    """Sorted (epoch, okta) index of all cached reports for one station."""
    recs = []
    for path in glob.glob(f"data/metar_cache/{human}_*.csv"):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                raw = (r.get("metar") or "").strip()
                if not raw:
                    continue
                t = dt.datetime.strptime(r["valid"], "%Y-%m-%d %H:%M").replace(
                    tzinfo=dt.timezone.utc)
                recs.append((t.timestamp(), parse_metar_cloud(raw)[0]))
    recs.sort()
    return [x[0] for x in recs], [x[1] for x in recs]


def nearest_okta(ts_list, okta_list, ts, tol=MATCH_TOL_S):
    i = bisect.bisect_left(ts_list, ts)
    best, bd = None, tol + 1
    for j in (i - 1, i):
        if 0 <= j < len(ts_list) and abs(ts_list[j] - ts) < bd:
            bd, best = abs(ts_list[j] - ts), okta_list[j]
    return best if bd <= tol else None


def viirs_rates(slug, tz, snow):
    """{ground: [n_mask_clear, n_human_cloud]} from the per-scene VIIRS log."""
    d = {"bare": [0, 0], "snow": [0, 0]}
    seen = set()
    with open(f"data/viirs_matches/viirs_matches_{slug}.csv") as fh:
        for r in csv.DictReader(fh):
            key = (r["ts_utc"], r["platform"])
            if key in seen:
                continue
            seen.add(key)
            if r["regime"] != "NIGHT" or r["human_usable"] == "" or r["box_clear"] == "":
                continue
            if int(r["human_usable"]) == 1:          # keep human-reported-cloud only
                continue
            ts = dt.datetime.fromisoformat(r["ts_utc"])
            g = "snow" if night_unit(ts, tz) in snow else "bare"
            d[g][1] += 1
            d[g][0] += int(r["box_clear"])
    return d


def goes_rates(slug, human, tz, snow):
    """{ground: [n_mask_clear, n_human_cloud]} from GOES draws matched to METAR."""
    fam = "west" if slug == "vancouver" else "east"   # primary family (as in the main results table)
    tsl, okl = metar_index(human)
    d = {"bare": [0, 0], "snow": [0, 0]}
    seen = set()
    with open(f"data/goes_draws/goes_draws_{slug}_{fam}.csv") as fh:
        for r in csv.DictReader(fh):
            if r["regime"] != "NIGHT" or r["station"] != human:
                continue
            if r["ts_utc"] in seen:
                continue
            seen.add(r["ts_utc"])
            ts = dt.datetime.fromisoformat(r["ts_utc"])
            ok = nearest_okta(tsl, okl, ts.timestamp())
            if ok is None or ok <= 2:                # keep human-reported-cloud only
                continue
            g = "snow" if night_unit(ts, tz) in snow else "bare"
            d[g][1] += 1
            d[g][0] += int(r["nearest_clear"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/snow_falseclear.csv")
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    args = ap.parse_args()

    rows = []
    for slug in args.sites:
        site = SITES[slug]
        tz = ZoneInfo(site.iana_tz)
        snow = snow_days(slug, SNOW_THRESH_M)
        for mask, rates in (("goes", goes_rates(slug, site.human, tz, snow)),
                            ("viirs", viirs_rates(slug, tz, snow))):
            for ground in ("bare", "snow"):
                nclear, n = rates[ground]
                pct = round(100 * nclear / n, 1) if n else ""
                rows.append((slug, mask, ground, n, nclear, pct))
                print(f"{slug:10} {mask:5} {ground:4} "
                      f"{'n/a' if not n else f'{pct:4.1f}% ({nclear}/{n})'}")

    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "mask", "ground", "n_human_cloud",
                    "n_mask_clear", "false_clear_pct"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
