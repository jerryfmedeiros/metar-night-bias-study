"""sites.py — registry of Canadian observing sites for the multi-site cloud-bias study.

Single source of truth for each site's coordinates, timezone, NAPS city filter, and
GOES satellite family, consumed by every fetcher and analysis script so the Calgary
constants stop being hardcoded module-by-module.

Design notes
------------
* `auto` (the co-located automated station for the human-vs-auto difference-of-
  differences) is OPTIONAL. Clean 24/7 human+auto pairs with a full 2020-2025 IEM
  archive are scarce (e.g. Boundary Bay, Rockcliffe, St-Andrews have no IEM METAR
  feed; Buttonville closed Nov-2023). Where there is no clean pair, `auto` is None
  and GOES is the objective reference — every site still gets the human-vs-satellite
  night/day test, which is the core cross-site replication.
* Whether a "human" station is genuinely human-augmented AT NIGHT (vs silently
  reverting to AUTO) is decided EMPIRICALLY by the night-%AUTO screen in the driver,
  not asserted here. Entries below are CANDIDATES to screen.
* GOES family is chosen by view geometry: West (GOES-17->18) for Pacific/BC sites,
  East (GOES-16->19) elsewhere. Vancouver samples BOTH as a cross-check (East is
  very oblique there). The satellite within a family is chosen per-timestamp by the
  cutover dates in climatology_goes.select_goes_bucket.
* Every site, Calgary included, now uses the IEM station metadata. Calgary was once
  pinned to the earlier single-site reference point (51.05, -114.07, 1043 m) inherited
  from the original scripts; it is not any more. The difference is immaterial to every
  published result (it shifts the solar-altitude cut by a few reports out of ~23,000:
  CYYC night N 22,879 -> 22,866, CYBW 33,602 -> 33,577) but it does mean the night-N
  column of the stations table must be regenerated from this registry, not copied from
  the earlier paper.

Station coordinates were taken from the IEM station metadata
(json/network.py?network=CA_<prov>_ASOS).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    slug: str
    name: str
    human: str                              # human-augmented station ICAO
    auto: str | None                        # co-located automated ICAO, or None
    lat: float                              # site reference point: ephem / ERA5 / darkness
    lon: float
    elevation_m: int
    iana_tz: str                            # night-keying & filename local time (handles DST)
    coords: dict[str, tuple[float, float]]  # ICAO -> (lat, lon), for the GOES fixed-grid pixel
    goes: tuple[str, ...]                   # ("east",) / ("west",) / ("east", "west")
    naps_city: str | None                   # lowercase substring matched against the NAPS city column
    naps_std_offset_h: int                  # NAPS local STANDARD-time UTC offset (no DST)
    # VIIRS (polar) platforms to query for the second-satellite cross-check. All sites
    # use all three by default; NOAA-21 just returns nothing before ~2023.
    viirs_platforms: tuple[str, ...] = ("SNPP", "NOAA20", "NOAA21")
    # ABI L2 clear-sky-mask product (sector). ACMC (CONUS, 5-min) covers every site
    # except Edmonton: the CONUS north edge crosses ~52.9N at Edmonton's longitude, so
    # CYEG/CZVL need the full-disk ACMF (10-min). fetch_goes raises OutOfSector rather
    # than clamping if a site/sector combination is wrong.
    goes_product: str = "ACMC"

    @property
    def stations(self) -> tuple[str, ...]:
        """Station ICAOs to analyse: (human,) or (human, auto)."""
        return (self.human,) if self.auto is None else (self.human, self.auto)

    def era5_area(self, pad: float = 0.15) -> list[float]:
        """ERA5 request box [N, W, S, E] around the reference point (0.25 deg grid)."""
        return [self.lat + pad, self.lon - pad, self.lat - pad, self.lon + pad]


# Keyed by short slug. `auto`/coords/goes per the notes above; screen before trusting.
SITES: dict[str, Site] = {
    # --- authoritative: reproduces the published Calgary study ---
    "calgary": Site(
        slug="calgary", name="Calgary", human="CYYC", auto="CYBW",
        # Reference point is the human station (CYYC), matching every other site here.
        lat=51.1139, lon=-114.0203, elevation_m=1084, iana_tz="America/Edmonton",
        coords={"CYYC": (51.1139, -114.0203), "CYBW": (51.1039, -114.3700)},
        goes=("east",), naps_city="calgary", naps_std_offset_h=-7,
    ),
    # --- Pacific / BC: samples BOTH GOES families (East is steeply oblique here) ---
    "vancouver": Site(
        slug="vancouver", name="Vancouver", human="CYVR", auto="CYXX",
        lat=49.1830, lon=-123.1682, elevation_m=4, iana_tz="America/Vancouver",
        coords={"CYVR": (49.1830, -123.1682), "CYXX": (49.0274, -122.3771)},
        goes=("east", "west"), naps_city="vancouver", naps_std_offset_h=-8,
    ),
    # --- Prairie / continental ---
    "edmonton": Site(
        # CZVL (Villeneuve), ~44 km NW, is a genuine 100%-night-AUTO ceilometer partner
        # (found by the automated-partner search; reports sky condition, full 2020-2025).
        slug="edmonton", name="Edmonton", human="CYEG", auto="CZVL",
        lat=53.3097, lon=-113.5797, elevation_m=723, iana_tz="America/Edmonton",
        coords={"CYEG": (53.3097, -113.5797), "CZVL": (53.6675, -113.8544)},
        goes=("east",), naps_city="edmonton", naps_std_offset_h=-7,
        goes_product="ACMF",  # outside the CONUS sector; see Site.goes_product
    ),
    "winnipeg": Site(
        slug="winnipeg", name="Winnipeg", human="CYWG", auto=None,
        lat=49.9167, lon=-97.2333, elevation_m=239, iana_tz="America/Winnipeg",
        coords={"CYWG": (49.9167, -97.2333)},
        goes=("east",), naps_city="winnipeg", naps_std_offset_h=-6,
    ),
    # --- Central / Great Lakes / St. Lawrence ---
    "toronto": Site(
        # CYKZ (Buttonville) closed Nov-2023 and has no IEM feed; CYTZ (Billy Bishop,
        # ~20 km SE) is the genuine 100%-night-AUTO ceilometer partner used here.
        slug="toronto", name="Toronto", human="CYYZ", auto="CYTZ",
        lat=43.6772, lon=-79.6306, elevation_m=173, iana_tz="America/Toronto",
        coords={"CYYZ": (43.6772, -79.6306), "CYTZ": (43.6286, -79.3950)},
        goes=("east",), naps_city="toronto", naps_std_offset_h=-5,
    ),
    "ottawa": Site(
        slug="ottawa", name="Ottawa", human="CYOW", auto=None,
        lat=45.3225, lon=-75.6692, elevation_m=114, iana_tz="America/Toronto",
        coords={"CYOW": (45.3225, -75.6692)},
        goes=("east",), naps_city="ottawa", naps_std_offset_h=-5,
    ),
    "montreal": Site(
        slug="montreal", name="Montreal", human="CYUL", auto="CYHU",
        lat=45.4683, lon=-73.7414, elevation_m=36, iana_tz="America/Toronto",
        coords={"CYUL": (45.4683, -73.7414), "CYHU": (45.5167, -73.4167)},
        goes=("east",), naps_city="montr", naps_std_offset_h=-5,  # matches montreal / montréal
    ),
    # --- Atlantic / maritime ---
    "halifax": Site(
        slug="halifax", name="Halifax", human="CYHZ", auto="CYAW",
        lat=44.8808, lon=-63.5086, elevation_m=145, iana_tz="America/Halifax",
        coords={"CYHZ": (44.8808, -63.5086), "CYAW": (44.6397, -63.4994)},
        goes=("east",), naps_city="halifax", naps_std_offset_h=-4,
    ),
}


def get_site(slug: str) -> Site:
    """Look up a site by slug, with a helpful error listing valid slugs."""
    try:
        return SITES[slug]
    except KeyError:
        raise SystemExit(
            f"Unknown site {slug!r}. Known sites: {', '.join(sorted(SITES))}"
        )


def resolve_sites(slugs: list[str]) -> list[Site]:
    """Resolve a list of slugs (or ['all']) to Site objects, preserving registry order."""
    if slugs == ["all"]:
        return list(SITES.values())
    return [get_site(s) for s in slugs]
