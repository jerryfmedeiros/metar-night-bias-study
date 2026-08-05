"""
make_tables.py — emit the paper's main results table (tab:main) body rows as LaTeX, straight from the results CSVs.

The main results table (tab:main) previously had every figure typed in by hand, which is the one place a
transcription slip would survive every other check in this repo. This regenerates the row
bodies from:

    results/dynamic_tables.csv   human/GOES deltas, H-G, auto DoD, night/day levels
    results/viirs_scenes.csv     matched-scene night gap and McNemar z

Usage:
  python3 make_tables.py                      # print rows to stdout
  python3 make_tables.py --check docs/methods-paper-body.tex
                                              # verify the .tex matches, exit 1 if not
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict

THRESHOLD_CM = "1.0"          # the published snow cutoff
VIIRS_WINDOW = "dynamic"      # ERA5-Land snow filter, not the calendar proxy

# Display order and labels, south to north (as the paper orders them).
SITES = [
    ("toronto",   "Toronto",   "CYYZ"),
    ("halifax",   "Halifax",   "CYHZ"),
    ("ottawa",    "Ottawa",    "CYOW"),
    ("montreal",  "Montréal",  "CYUL"),
    ("vancouver", "Vancouver", "CYVR"),
    ("winnipeg",  "Winnipeg",  "CYWG"),
    ("calgary",   "Calgary",   "CYYC"),
    ("edmonton",  "Edmonton",  "CYEG"),
]


def load_dynamic(path="data/derived/dynamic_tables.csv"):
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["threshold_cm"] == THRESHOLD_CM:
            out[r["site"]][r["record"]] = r
    return out


def load_viirs(path="data/derived/viirs_scenes.csv"):
    return {r["site"]: r for r in csv.DictReader(open(path))
            if r["window"] == VIIRS_WINDOW}


def load_moon(path="data/derived/viirs_moon_split.csv"):
    out = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if r["window"] == VIIRS_WINDOW and r["snow_thresh_cm"] == THRESHOLD_CM:
            out[r["site"]][r["stratum"]] = r
    return out


def build_moon_rows():
    """Rows for the moonlit/dark table (tab:moon). Edmonton's strata are snow-
    contaminated; they print like every other site's, and the table caption
    says they are shown for completeness but not read."""
    moon = load_moon()
    lines = []
    for slug, name, icao in SITES:
        m = moon[slug]
        d, l = m["dark"], m["moonlit"]
        lines.append(
            f"{name:9s} & ${float(d['night_gap_pp']):+.1f}\\;(z={float(d['mcnemar_z']):.1f})$ & "
            f"${float(l['night_gap_pp']):+.1f}\\;(z={float(l['mcnemar_z']):.1f})$ \\\\"
        )
    return lines


def ci(r):
    return f"${float(r['delta_pp']):+.1f}\\;[{float(r['ci_lo']):+.1f},{float(r['ci_hi']):+.1f}]$"


def build_rows():
    dyn, vii = load_dynamic(), load_viirs()
    lines = []
    for slug, name, icao in SITES:
        d, v = dyn[slug], vii[slug]
        human = d[f"metar_{icao}"]
        dod = d.get("auto_dod")
        dod_txt = ci(dod).replace("$", "$") if dod else "—"
        lines.append(
            f"{name:9s} & {icao} & {float(human['day_pct']):.1f} & "
            f"{float(human['night_pct']):.1f} & {ci(human)} & {ci(d['goes_nearest'])} & "
            f"{ci(d['human_minus_goes'])} & "
            f"${float(v['night_gap_pp']):+.1f}\\;(z={float(v['mcnemar_z']):.1f})$ "
            f"\\hfill {dod_txt} \\\\"
        )
    return lines


# Numbers as they must appear in the .tex, keyed for a readable failure message.
def expected_values():
    dyn, vii = load_dynamic(), load_viirs()
    exp = {}
    for slug, name, icao in SITES:
        d, v = dyn[slug], vii[slug]
        for label, rec in (("human", f"metar_{icao}"), ("GOES", "goes_nearest"),
                           ("H-G", "human_minus_goes"), ("autoDoD", "auto_dod")):
            if rec in d:
                r = d[rec]
                exp[f"{name} {label}"] = (float(r["delta_pp"]),
                                          float(r["ci_lo"]), float(r["ci_hi"]))
        exp[f"{name} VIIRS"] = (float(v["night_gap_pp"]), float(v["mcnemar_z"]))
        exp[f"{name} levels"] = (float(d[f"metar_{icao}"]["day_pct"]),
                                 float(d[f"metar_{icao}"]["night_pct"]))
    return exp


def check(tex_path):
    """Confirm every generated row body appears verbatim in the .tex."""
    tex = open(tex_path, encoding="utf-8").read()
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    body = norm(tex)
    bad = [ln for ln in build_rows() if norm(ln) not in body]
    if bad:
        print(f"MISMATCH: {len(bad)} of {len(SITES)} main-table rows differ from the CSVs.\n")
        for ln in bad:
            print("  expected:", norm(ln))
        return 1
    bad_moon = [ln for ln in build_moon_rows() if norm(ln) not in body]
    if bad_moon:
        print(f"MISMATCH: {len(bad_moon)} moon-table rows differ from the CSVs.\n")
        for ln in bad_moon:
            print("  expected:", norm(ln))
        return 1
    print(f"OK: all {len(SITES)} main-table rows and {len(SITES)} moon-table rows "
          f"in {tex_path} match the results CSVs.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", metavar="TEXFILE",
                    help="verify the .tex matches the CSVs instead of printing")
    args = ap.parse_args()
    if args.check:
        sys.exit(check(args.check))
    for ln in build_rows():
        print(ln)
    print("% --- moon-split table (tab:moon) rows ---")
    for ln in build_moon_rows():
        print(ln)


if __name__ == "__main__":
    main()
