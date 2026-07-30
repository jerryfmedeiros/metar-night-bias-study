"""
metar_climatology.py — Advanced climatology analysis for Allsky Cloud Research.

Analyzes:
1. Day vs. Night "Astronomy Bias"
2. Seasonal trends
3. Multi-year aggregates (2020-2025)
4. Cloud Genus & Altitude frequencies

Usage:
  python metar_climatology.py --years 2020 2025 --stations CYYC CYBW
"""
import argparse
import datetime as dt
import math
import sys
from collections import defaultdict
import ephem

# Reuse functions from fetch_metar.py
try:
    from fetch_metar import fetch_metar_csv, parse_metar_cloud, altitude_bucket_from_base_m
except ImportError:
    sys.exit("Error: Could not import fetch_metar.py. Ensure it is in the current directory.")

# Default coordinates for Calgary
LAT = 51.05
LON = -114.07
ELEVATION_M = 1043

def calculate_sun_alt(timestamp: dt.datetime, lat: float, lon: float, elevation: float) -> float:
    obs = ephem.Observer()
    obs.lat, obs.lon = str(lat), str(lon)
    obs.elevation = elevation
    obs.date = timestamp
    sun = ephem.Sun(obs)
    return math.degrees(sun.alt)

def get_solar_regime(sun_alt: float) -> str:
    if sun_alt >= 6.0: return "DAY"
    if sun_alt <= -12.0: return "NIGHT"
    return "TWILIGHT"

def get_cloud_label(okta: int) -> str:
    if okta == 0: return "SKC"
    if okta <= 2: return "FEW"
    if okta <= 4: return "SCT"
    if okta <= 6: return "BKN"
    return "OVC"

def get_season(month: int) -> str:
    if month in [12, 1, 2]: return "WINTER"
    if month in [3, 4, 5]: return "SPRING"
    if month in [6, 7, 8]: return "SUMMER"
    return "AUTUMN"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs=2, type=int, default=[2020, 2025], help="Start and End year inclusive")
    parser.add_argument("--stations", nargs="+", default=None, help="ICAO station IDs (default: site's human+auto, or CYYC CYBW)")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--site", default=None,
                        help="site slug from sites.py; fills --lat/--lon/--stations from the registry")
    parser.add_argument("--okta-thresh", type=int, default=2,
                        help="max okta counted as usable/clear (default 2 = SKC+FEW; use 0 for SKC/CLR only)")
    args = parser.parse_args()

    # --site is a convenience that fills coords + stations from the registry;
    # explicit --lat/--lon/--stations still override. Default stays Calgary.
    global ELEVATION_M
    if args.site:
        from sites import get_site
        site = get_site(args.site)
        if args.lat is None: args.lat = site.lat
        if args.lon is None: args.lon = site.lon
        if args.stations is None: args.stations = list(site.stations)
        ELEVATION_M = site.elevation_m   # was silently left at the Calgary default
    if args.lat is None: args.lat = LAT
    if args.lon is None: args.lon = LON
    if args.stations is None: args.stations = ["CYYC", "CYBW"]

    y_start, y_end = args.years
    print(f"Deep Climatology Analysis: {y_start} to {y_end}")
    print(f"Stations: {', '.join(args.stations)}")
    print("-" * 100)

    for station in args.stations:
        print(f"\nProcessing {station}...")
        
        # Aggregators
        overall_stats = defaultdict(lambda: defaultdict(int))
        seasonal_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        yearly_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int))) 
        cloud_types = defaultdict(int)
        cloudy_obs_count = 0
        # Seasonal cloud character: among cloudy obs, count genus/altitude per season.
        season_cloud = defaultdict(lambda: defaultdict(int))
        # AUTO-flag split: aug_stats[regime][AUTO|MAN] -> {TOTAL, USABLE}
        aug_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        above_ceiling = 0   # obs reporting a layer above the ASOS ceilometer limit
        total_obs = 0
        CEILING_FT = 12000  # high-layer marker: dome-scanning observers report layers above
        # this far more often than the single-beam ceilometer (its range runs to 25,000 ft)

        for year in range(y_start, y_end + 1):
            start_date = dt.date(year, 1, 1)
            end_date = dt.date(year, 12, 31)
            
            try:
                obs_list = fetch_metar_csv(station, start_date, end_date)
            except Exception as e:
                print(f"  Error fetching {year}: {e}")
                continue

            for obs in obs_list:
                sun_alt = calculate_sun_alt(obs.timestamp, args.lat, args.lon, ELEVATION_M)
                regime = get_solar_regime(sun_alt)
                label = get_cloud_label(obs.coverage_okta)
                season = get_season(obs.timestamp.month)
                
                overall_stats[regime][label] += 1
                overall_stats[regime]["TOTAL"] += 1
                seasonal_stats[season][regime][label] += 1
                seasonal_stats[season][regime]["TOTAL"] += 1
                yearly_stats[year][regime][label] += 1
                yearly_stats[year][regime]["TOTAL"] += 1

                # AUTO vs human-augmented split, and clouds above the ASOS ceiling
                total_obs += 1
                aug = "AUTO" if obs.is_auto else "MAN"
                aug_stats[regime][aug]["TOTAL"] += 1
                if obs.coverage_okta <= args.okta_thresh:
                    aug_stats[regime][aug]["USABLE"] += 1
                if any(b > CEILING_FT for _, b in obs.layers if b):
                    above_ceiling += 1

                # Cloud Type Analysis (only for non-SKC)
                if obs.coverage_okta > 0:
                    cloudy_obs_count += 1
                    sc = season_cloud[season]
                    sc["CLOUDY"] += 1
                    # Altitude buckets
                    bucket = altitude_bucket_from_base_m(obs.cloud_base_m)
                    if bucket:
                        cloud_types[f"ALT_{bucket.upper()}"] += 1
                        sc[f"ALT_{bucket.upper()}"] += 1
                    # Genus hints (CB/TCU = convection)
                    if obs.genus_hint:
                        cloud_types[f"GENUS_{obs.genus_hint}"] += 1
                        sc[f"GENUS_{obs.genus_hint}"] += 1

        # --- Report 1: Overall Summary ---
        print(f"\n[SUMMARY] {station} ({y_start}-{y_end})")
        print(f"{'Regime':<10} | {'SKC':<7} | {'FEW':<7} | {'SCT':<7} | {'BKN':<7} | {'OVC':<7} | {'Total':<8}")
        print("-" * 75)
        labels = ["SKC", "FEW", "SCT", "BKN", "OVC"]
        for regime in ["DAY", "NIGHT", "TWILIGHT"]:
            r = overall_stats[regime]
            tot = r["TOTAL"]
            if tot == 0: continue
            row = [f"{regime:<10}"]
            for lbl in labels:
                pct = (r[lbl] / tot) * 100
                row.append(f"{pct:5.1f}%")
            row.append(f"{tot:<8}")
            print(" | ".join(row))

        # --- Report 2: Yearly Breakdown ---
        print(f"\n[YEARLY] Night vs Day 'Usable' (SKC+FEW) Frequency")
        print(f"{'Year':<10} | {'Day':<10} | {'Night':<10} | {'Diff':<10}")
        print("-" * 45)
        for year in range(y_start, y_end + 1):
            y = yearly_stats[year]
            if y["DAY"]["TOTAL"] == 0 or y["NIGHT"]["TOTAL"] == 0: continue
            day_p = ((y["DAY"]["SKC"] + y["DAY"]["FEW"]) / y["DAY"]["TOTAL"]) * 100
            night_p = ((y["NIGHT"]["SKC"] + y["NIGHT"]["FEW"]) / y["NIGHT"]["TOTAL"]) * 100
            diff = night_p - day_p
            print(f"{year:<10} | {day_p:5.1f}%     | {night_p:5.1f}%     | {diff:+5.1f}%")

        # --- Report 3: Seasonal Astronomy Bias ---
        print(f"\n[SEASONAL] Night vs Day 'Usable' (SKC+FEW) Frequency")
        print(f"{'Season':<10} | {'Day':<10} | {'Night':<10} | {'Diff':<10}")
        print("-" * 45)
        for season in ["WINTER", "SPRING", "SUMMER", "AUTUMN"]:
            s = seasonal_stats[season]
            if s["DAY"]["TOTAL"] == 0 or s["NIGHT"]["TOTAL"] == 0: continue
            day_p = ((s["DAY"]["SKC"] + s["DAY"]["FEW"]) / s["DAY"]["TOTAL"]) * 100
            night_p = ((s["NIGHT"]["SKC"] + s["NIGHT"]["FEW"]) / s["NIGHT"]["TOTAL"]) * 100
            diff = night_p - day_p
            print(f"{season:<10} | {day_p:5.1f}%     | {night_p:5.1f}%     | {diff:+5.1f}%")

        # --- Report 4: Cloud Characterization ---
        print(f"\n[CLOUDS] Frequency of types (when cloudy, N={cloudy_obs_count})")
        sorted_types = sorted(cloud_types.items(), key=lambda x: x[1], reverse=True)
        for t, count in sorted_types:
            pct = (count / cloudy_obs_count) * 100
            print(f"  - {t:<15}: {pct:5.1f}% ({count})")

        # --- Report 4b: Seasonal cloud character (share of cloudy obs) ---
        print(f"\n[CLOUD CHARACTER BY SEASON] {station} — share of cloudy obs")
        print(f"{'Season':<8} | {'cloudy N':>8} | {'low':>6} | {'mid':>6} | {'high':>6} | "
              f"{'CB':>5} | {'TCU':>5}")
        print("-" * 60)
        for season in ["WINTER", "SPRING", "SUMMER", "AUTUMN"]:
            sc = season_cloud[season]
            n = sc["CLOUDY"]
            if n == 0:
                continue
            def pc(key):
                return f"{sc[key] / n * 100:5.1f}%"
            print(f"{season:<8} | {n:8d} | {pc('ALT_LOW')} | {pc('ALT_MID')} | {pc('ALT_HIGH')} | "
                  f"{pc('GENUS_CB')} | {pc('GENUS_TCU')}")

        # --- Report 5: Augmentation (AUTO flag) ---
        # Is this station's NIGHT data actually human-augmented, or AUTO? And does
        # the "nighttime clearing" show up in the manual subset vs the auto subset?
        print(f"\n[AUGMENTATION] AUTO-flag prevalence & 'usable' (SKC+FEW) by regime")
        print(f"{'Regime':<10} | {'%AUTO':<8} | {'Manual usable':<14} | {'AUTO usable':<12}")
        print("-" * 55)
        for regime in ["DAY", "NIGHT", "TWILIGHT"]:
            a = aug_stats[regime]
            tot = a["AUTO"]["TOTAL"] + a["MAN"]["TOTAL"]
            if tot == 0:
                continue
            pct_auto = a["AUTO"]["TOTAL"] / tot * 100
            man_u = (a["MAN"]["USABLE"] / a["MAN"]["TOTAL"] * 100) if a["MAN"]["TOTAL"] else float("nan")
            auto_u = (a["AUTO"]["USABLE"] / a["AUTO"]["TOTAL"] * 100) if a["AUTO"]["TOTAL"] else float("nan")
            print(f"{regime:<10} | {pct_auto:5.1f}%   | {man_u:12.1f}% | {auto_u:10.1f}%")

        # --- Report 6: SKC-only (apples-to-apples vs satellite binary clear) ---
        # FEW is cloud; a binary satellite mask scores it "cloudy". Compare SKC-only
        # to the satellite clear rate, not SKC+FEW.
        print(f"\n[SKC-ONLY] Day vs Night true-clear (SKC) frequency")
        d, n = overall_stats["DAY"], overall_stats["NIGHT"]
        if d["TOTAL"] and n["TOTAL"]:
            d_skc = d["SKC"] / d["TOTAL"] * 100
            n_skc = n["SKC"] / n["TOTAL"] * 100
            print(f"  Day SKC: {d_skc:5.1f}%   Night SKC: {n_skc:5.1f}%   Diff: {n_skc - d_skc:+5.1f}%")

        # --- Report 7: clouds above the ASOS ceilometer limit ---
        # Quantifies what CYBW's automated ceilometer structurally cannot report
        # (expect ~0% at CYBW; non-zero at CYYC exposes the instrument confound).
        if total_obs:
            print(f"\n[CEILOMETER] Obs reporting a layer above {CEILING_FT} ft AGL: "
                  f"{above_ceiling / total_obs * 100:.1f}% ({above_ceiling}/{total_obs})")

        print("\n" + "="*100)

if __name__ == "__main__":
    main()
