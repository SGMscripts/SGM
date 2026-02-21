#!/usr/bin/env python3
"""
Update a local birds list from eBird observations around a location.

Output format (one per line):
  Genus species_Common Name
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


EBIRD_RECENT_GEO_URL = "https://api.ebird.org/v2/data/obs/geo/recent"
EBIRD_HOTSPOT_GEO_URL = "https://api.ebird.org/v2/ref/hotspot/geo"
EBIRD_REGION_HISTORIC_TEMPLATE = "https://api.ebird.org/v2/data/obs/{region}/historic/{yyyy}/{mm}/{dd}"


def sanitize(text: str) -> str:
    s = str(text or "")
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return " ".join(s.split())


def fetch_json(api_key: str, url: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "X-eBirdApiToken": api_key,
            "Accept": "application/json",
            "User-Agent": "SFX-BirdDetect-LocalList-Updater/1.0",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"eBird HTTP {resp.status}")
    return json.loads(raw.decode("utf-8", errors="replace"))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl / 2.0) ** 2)
    return 2.0 * r * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def select_region_code(api_key: str, lat: float, lon: float, radius_km: int) -> Optional[str]:
    params = {
        "lat": f"{lat:.6f}",
        "lng": f"{lon:.6f}",
        "dist": str(int(max(1, min(50, radius_km)))),
        "fmt": "json",
    }
    url = f"{EBIRD_HOTSPOT_GEO_URL}?{urllib.parse.urlencode(params)}"
    try:
        data = fetch_json(api_key, url)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None

    first = data[0] if isinstance(data[0], dict) else {}
    for key in ("subnational2Code", "subnational1Code", "countryCode"):
        val = sanitize(first.get(key, ""))
        if val:
            return val
    return None


def fetch_recent_observations(
    api_key: str,
    lat: float,
    lon: float,
    radius_km: int,
    back_days: int,
    max_results: int,
) -> List[dict]:
    params = {
        "lat": f"{lat:.6f}",
        "lng": f"{lon:.6f}",
        "dist": str(int(radius_km)),
        "back": str(int(back_days)),
        "maxResults": str(int(max_results)),
    }
    url = f"{EBIRD_RECENT_GEO_URL}?{urllib.parse.urlencode(params)}"
    data = fetch_json(api_key, url)
    if not isinstance(data, list):
        raise RuntimeError("Unexpected eBird response format")
    return data


def fetch_region_historic_observations(
    api_key: str,
    lat: float,
    lon: float,
    radius_km: int,
    back_days: int,
    max_results: int,
) -> Tuple[List[dict], str, int, int]:
    region_code = select_region_code(api_key, lat, lon, radius_km)
    if not region_code:
        raise RuntimeError("Could not resolve eBird region code near this location")

    out: List[dict] = []
    day_ok = 0
    day_fail = 0
    today = dt.date.today()

    for offset in range(back_days):
        d = today - dt.timedelta(days=offset)
        url = EBIRD_REGION_HISTORIC_TEMPLATE.format(
            region=urllib.parse.quote(region_code, safe="-_"),
            yyyy=f"{d.year:04d}",
            mm=f"{d.month:02d}",
            dd=f"{d.day:02d}",
        )
        try:
            data = fetch_json(api_key, url)
            if not isinstance(data, list):
                day_fail += 1
                continue
            day_ok += 1
        except Exception:
            day_fail += 1
            continue

        for row in data:
            if not isinstance(row, dict):
                continue
            try:
                rlat = float(row.get("lat"))
                rlon = float(row.get("lng"))
            except Exception:
                continue
            if haversine_km(lat, lon, rlat, rlon) <= float(radius_km):
                out.append(row)
                if len(out) >= max_results:
                    return out, region_code, day_ok, day_fail

        # Keep request burst lower when doing hundreds of daily calls.
        if back_days > 30:
            time.sleep(0.015)

    return out, region_code, day_ok, day_fail


def build_species_lines(observations: List[dict]) -> Tuple[List[str], int]:
    by_scientific: Dict[str, Dict[str, int]] = {}

    for row in observations:
        if not isinstance(row, dict):
            continue
        sci = sanitize(row.get("sciName", ""))
        com = sanitize(row.get("comName", ""))
        if not sci:
            continue
        if sci not in by_scientific:
            by_scientific[sci] = {}
        if com:
            by_scientific[sci][com] = by_scientific[sci].get(com, 0) + 1

    lines: List[str] = []
    for sci in sorted(by_scientific.keys(), key=lambda x: x.lower()):
        com_counts = by_scientific[sci]
        if com_counts:
            # Prefer the most frequently observed common name for this scientific name.
            common = sorted(com_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))[0][0]
            lines.append(f"{sci}_{common}")
        else:
            lines.append(sci)

    return lines, len(by_scientific)


def write_local_list(
    out_file: str,
    lines: List[str],
    lat: float,
    lon: float,
    radius_km: int,
    back_days: int,
    obs_count: int,
    source_mode: str,
) -> None:
    out_file = os.path.abspath(out_file)
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    now_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    header = [
        "# Auto-generated from eBird observations",
        f"# Generated UTC: {now_utc}",
        f"# Location: lat={lat:.6f}, lon={lon:.6f}",
        f"# Radius km: {radius_km}, Back days: {back_days}",
        f"# Source mode: {source_mode}",
        f"# Source observations: {obs_count}",
        "",
    ]

    with open(out_file, "w", encoding="utf-8") as f:
        for line in header:
            f.write(line + "\n")
        for line in lines:
            f.write(line + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--radius-km", type=int, default=50)
    p.add_argument("--back-days", type=int, default=30)
    p.add_argument("--max-results", type=int, default=10000)
    p.add_argument("--api-key", default="")
    p.add_argument("--api-key-env", default="EBIRD_API_KEY")
    p.add_argument("--out-file", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    lat = float(args.lat)
    lon = float(args.lon)
    radius_km = max(1, min(500, int(args.radius_km)))
    back_days = max(1, min(365, int(args.back_days)))
    max_results = max(100, min(20000, int(args.max_results)))
    out_file = str(args.out_file)

    api_key = sanitize(args.api_key)
    if not api_key and args.api_key_env:
        api_key = sanitize(os.getenv(str(args.api_key_env), ""))

    if not api_key:
        print("ERROR: eBird API key is missing.", file=sys.stderr)
        return 2
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        print("ERROR: invalid latitude/longitude.", file=sys.stderr)
        return 2

    observations: List[dict] = []
    source_mode = ""
    region_code = ""
    day_ok = 0
    day_fail = 0

    try:
        if back_days <= 30:
            observations = fetch_recent_observations(
                api_key=api_key,
                lat=lat,
                lon=lon,
                radius_km=radius_km,
                back_days=back_days,
                max_results=max_results,
            )
            source_mode = "geo_recent"
        else:
            observations, region_code, day_ok, day_fail = fetch_region_historic_observations(
                api_key=api_key,
                lat=lat,
                lon=lon,
                radius_km=radius_km,
                back_days=back_days,
                max_results=max_results,
            )
            source_mode = f"region_historic:{region_code}"
    except Exception as ex:
        print(f"ERROR: eBird fetch failed: {ex}", file=sys.stderr)
        return 2

    lines, species_count = build_species_lines(observations)
    if not lines:
        print("ERROR: no species returned by eBird for this location/time window.", file=sys.stderr)
        return 2

    try:
        write_local_list(
            out_file=out_file,
            lines=lines,
            lat=lat,
            lon=lon,
            radius_km=radius_km,
            back_days=back_days,
            obs_count=len(observations),
            source_mode=source_mode,
        )
    except Exception as ex:
        print(f"ERROR: failed writing local list: {ex}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "ok",
                "out_file": os.path.abspath(out_file),
                "species_count": species_count,
                "observation_rows": len(observations),
                "source_mode": source_mode,
                "region_code": region_code,
                "historic_days_ok": day_ok,
                "historic_days_failed": day_fail,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
