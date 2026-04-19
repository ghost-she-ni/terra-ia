from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests


def plu_zone_counts(gdf: gpd.GeoDataFrame) -> dict[str, int]:
    if gdf.empty or "geometry" not in gdf.columns:
        return {"U": 0, "AU": 0, "N": 0, "A": 0}

    counts = {"U": 0, "AU": 0, "N": 0, "A": 0}
    for _, row in gdf.iterrows():
        zone_val = None
        for col in gdf.columns:
            if col.lower() in ["typezone", "zone", "libelle", "lib_zone", "code_zone", "nom_zone", "destdomi"]:
                zone_val = row.get(col)
                if pd.notna(zone_val):
                    break
        zone_str = str(zone_val).strip().upper() if zone_val is not None else ""
        zone_str = zone_str.replace(" ", "")
        if zone_str.startswith("AU"):
            counts["AU"] += 1
        elif zone_str.startswith("U"):
            counts["U"] += 1
        elif zone_str.startswith("N"):
            counts["N"] += 1
        elif zone_str.startswith("A"):
            counts["A"] += 1
    return counts


def download_plu(dest: Path, *, target_crs: str, commune_code: str) -> bool:
    if dest.exists():
        try:
            gdf = gpd.read_file(dest)
            if gdf.crs is None or str(gdf.crs.to_epsg()) != "2154":
                gdf = gdf.to_crs(target_crs)
            gdf.to_file(dest, driver="GeoJSON")
            counts = plu_zone_counts(gdf)
            print(
                f"  OK PLU present : {dest.name} ({len(gdf)} entites) "
                f"(U: {counts['U']}, AU: {counts['AU']}, N: {counts['N']}, A: {counts['A']})"
            )
            return True
        except Exception as exc:
            print(f"  WARNING PLU existant illisible ({exc}) - tentative telechargement")

    urls = [
        (
            "WFS Geoportail Urbanisme",
            "https://wxs.ign.fr/geoportail/wfs?"
            "SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
            "&TYPENAMES=ms:zone_urba"
            f"&CQL_FILTER=code_com='{commune_code}'"
            "&OUTPUTFORMAT=application/json"
            "&COUNT=5000",
        ),
        (
            "API Geoportail Urbanisme REST",
            "https://www.geoportail-urbanisme.gouv.fr/api/document/"
            "200069110_PLUi_20260309/file/DU_200069110",
        ),
        (
            "GeoJSON direct (ID document)",
            "https://wxs.ign.fr/urbanisme/geoportail/wfs?"
            "SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
            "&TYPENAMES=BDPU_ZONE_URBA"
            "&propertyName=libelle,libelong,typezone,nomfic,urlfic"
            f"&CQL_FILTER=code_insee='{commune_code}'"
            "&OUTPUTFORMAT=application/json",
        ),
    ]

    last_columns: list[str] | None = None

    for label, url in urls:
        response: requests.Response | None = None
        try:
            print(f"  Telechargement PLU ({label}) ...", end=" ", flush=True)
            response = requests.get(url, timeout=180)
            print(f"status={response.status_code}", end=" ", flush=True)
            response.raise_for_status()
            dest.write_bytes(response.content)
            gdf = gpd.read_file(dest)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            if str(gdf.crs.to_epsg()) != "2154":
                gdf = gdf.to_crs(target_crs)
            gdf.to_file(dest, driver="GeoJSON")
            counts = plu_zone_counts(gdf)
            print(
                f"OK - PLU Chambery: {len(gdf)} zones "
                f"(U: {counts['U']}, AU: {counts['AU']}, N: {counts['N']}, A: {counts['A']})"
            )
            return True
        except Exception as exc:
            if response is not None:
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        if "features" in data and isinstance(data["features"], list) and data["features"]:
                            props = data["features"][0].get("properties", {}) or {}
                            last_columns = sorted({str(key) for key in props.keys()})
                        else:
                            last_columns = sorted({str(key) for key in data.keys()})
                except Exception:
                    pass
            print(f"ERREUR {label} - {exc}")

    if last_columns:
        print(f"  INFO Colonnes disponibles dans la derniere reponse JSON : {', '.join(last_columns)}")

    print("  WARNING PLU indisponible - zone filtering disabled")
    dest.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    return False


def download_bd_topo_batiments(dest: Path, *, target_crs: str) -> bool:
    if dest.exists():
        try:
            gdf = gpd.read_file(dest)
            if not gdf.empty and (gdf.crs is None or str(gdf.crs.to_epsg()) != "2154"):
                gdf = gdf.to_crs(target_crs)
                gdf.to_file(dest, driver="GeoJSON")
            print(f"  OK BD TOPO batiments present : {len(gdf)} polygones")
            return True
        except Exception as exc:
            print(f"  WARNING BD TOPO existant illisible ({exc}) - tentative telechargement")

    urls = [
        (
            "WFS BBOX batiment",
            "https://data.geopf.fr/wfs?"
            "SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
            "&TYPENAMES=BDTOPO_V3:batiment"
            "&BBOX=925000,6499000,929000,6505000,EPSG:2154"
            "&OUTPUTFORMAT=application/json"
            "&COUNT=10000",
        ),
        (
            "WFS BBOX construction (alt)",
            "https://data.geopf.fr/wfs?"
            "SERVICE=WFS&VERSION=2.0.0&REQUEST=GetFeature"
            "&TYPENAMES=BDTOPO_V3:construction"
            "&BBOX=925000,6499000,929000,6505000,EPSG:2154"
            "&OUTPUTFORMAT=application/json",
        ),
    ]

    for label, url in urls:
        try:
            print(f"  Telechargement BD TOPO batiments ({label}) ...", end=" ", flush=True)
            response = requests.get(url, timeout=180)
            print(f"status={response.status_code}", end=" ", flush=True)
            response.raise_for_status()
            dest.write_bytes(response.content)
            gdf = gpd.read_file(dest)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            if str(gdf.crs.to_epsg()) != "2154":
                gdf = gdf.to_crs(target_crs)
            gdf.to_file(dest, driver="GeoJSON")
            print(f"OK ({len(gdf)} batiments)")
            return True
        except Exception as exc:
            print(f"ERREUR BD TOPO ({label}) - {exc}")

    return False
