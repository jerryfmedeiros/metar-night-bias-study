#!/usr/bin/env python3
"""fetch_snow_insitu.py — airport daily snow-on-ground from ECCC, for the snow-filter check.

Downloads Environment and Climate Change Canada daily climate data for each study
airport (2020-2025) and caches the reported "Snow on Grnd (cm)" per day. Used by
snow_insitu_validation.py to show (a) that the in-situ record is too incomplete over
2020-2025 to screen on, station-automation having thinned it, and (b) that where it does
exist the ERA5-Land screen agrees with it and errs conservatively.

Station IDs are ECCC's internal index for the current-era airport station (the one whose
daily record covers 2020-2025), from the ECCC Station Inventory. Writes one CSV per site
to data/snow_insitu/<slug>.csv (date, snow_on_grnd_cm) for days with a reported value.

Usage:
  python3 fetch_snow_insitu.py                 # all sites with an ECCC airport station
"""
import argparse
import csv
import io
import os
import urllib.request

from sites import SITES

# ECCC internal Station ID for the current-era airport climate station (covers 2020-2025).
STATION_ID = {
    "CYYZ": 51459, "CYHZ": 50620, "CYOW": 49568, "CYUL": 51157,
    "CYVR": 51442, "CYWG": 51097, "CYYC": 50430, "CYEG": 50149,
}
BULK = ("https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        "?format=csv&stationID={sid}&Year={year}&Month=1&Day=1&timeframe=2&submit=Download")


def fetch_year(sid, year):
    with urllib.request.urlopen(BULK.format(sid=sid, year=year), timeout=90) as resp:
        return resp.read().decode("utf-8-sig")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", default=list(SITES))
    ap.add_argument("--years", type=int, nargs=2, default=(2020, 2025))
    ap.add_argument("--out-dir", default="data/snow_insitu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for slug in args.sites:
        icao = SITES[slug].human
        sid = STATION_ID.get(icao)
        if sid is None:
            continue
        rows = []
        for year in range(args.years[0], args.years[1] + 1):
            for r in csv.DictReader(io.StringIO(fetch_year(sid, year))):
                depth = (r.get("Snow on Grnd (cm)") or "").strip()
                if depth != "":
                    rows.append((r["Date/Time"], depth))
        out = os.path.join(args.out_dir, f"{slug}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "snow_on_grnd_cm"])
            w.writerows(rows)
        print(f"{slug:10} {icao} (ECCC {sid}): {len(rows)} days with a reported snow depth")


if __name__ == "__main__":
    main()
