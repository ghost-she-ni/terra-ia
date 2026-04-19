from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.merge import merge


def merge_dalles(paths: list[Path], output: Path, label: str) -> bool:
    valid_paths = [path for path in paths if path.exists()]
    if not valid_paths:
        return False
    if output.exists():
        print(f"  OK fusion deja faite : {output.name}")
        return True
    if len(valid_paths) == 1:
        shutil.copy(valid_paths[0], output)
        print(f"  OK une dalle {label} copiee.")
        return True

    print(f"  Fusion {len(valid_paths)} dalles {label} ...", end=" ", flush=True)
    try:
        datasets = [rasterio.open(path) for path in valid_paths]
        mosaic, transform = merge(datasets)
        meta = datasets[0].meta.copy()
        meta.update(
            {
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "lzw",
                "nodata": -9999.0,
            }
        )
        with rasterio.open(output, "w", **meta) as dst:
            dst.write(mosaic)
        for dataset in datasets:
            dataset.close()
        size_mb = output.stat().st_size / 1e6
        print(f"OK ({size_mb:.1f} Mo, {len(valid_paths)} dalles)")
        return True
    except Exception as exc:
        print(f"ERREUR - {exc}")
        return False


def validate_raster(path: Path, label: str) -> bool:
    print(f"\n  Validation {label} : {path.name}")
    try:
        with rasterio.open(path) as src:
            crs_wkt = str(src.crs)
            crs_epsg = src.crs.to_epsg()
            crs_ok = (str(crs_epsg) == "2154") or (
                "Lambert_Conformal_Conic" in crs_wkt and "700000" in crs_wkt and "6600000" in crs_wkt
            )
            res_ok = abs(src.res[0] - 0.5) < 0.01
            km_w = src.width * 0.5 / 1000
            km_h = src.height * 0.5 / 1000
            print(f"    CRS     : Lambert 93 / EPSG:2154  {'OK' if crs_ok else 'KO'}")
            print(f"    Resol.  : {src.res}  {'OK' if res_ok else 'KO pas 50cm'}")
            print(
                f"    Taille  : {src.width:,} x {src.height:,} px  "
                f"({km_w:.1f} x {km_h:.1f} km = {km_w * km_h:.1f} km2)"
            )
            print(
                f"    Emprise : {src.bounds.left:.0f}, {src.bounds.bottom:.0f}"
                f" -> {src.bounds.right:.0f}, {src.bounds.top:.0f}"
            )
            print(f"    NoData  : {src.nodata}")
        return True
    except Exception as exc:
        print(f"    KO Erreur : {exc}")
        return False


def load_parcelles(path: Path, *, target_crs: str) -> gpd.GeoDataFrame:
    print(f"\n  Chargement parcelles : {path.name}")
    gdf = gpd.read_file(path)
    print(f"    {len(gdf)} parcelles - CRS: {gdf.crs}")
    if gdf.crs is None or str(gdf.crs.to_epsg()) != "2154":
        gdf = gdf.to_crs(target_crs)
        print("    Reprojection -> EPSG:2154 OK")
    valid = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].reset_index(drop=True)
    print(f"    {len(valid)} parcelles valides geometriquement")
    return valid


def join_plu_to_parcelles(parcelles: gpd.GeoDataFrame, plu_path: Path, *, target_crs: str) -> pd.Series:
    n_rows = len(parcelles)
    if not plu_path.exists():
        print("  WARNING PLU manquant - zone_plu=inconnu")
        return pd.Series(["inconnu"] * n_rows, index=parcelles.index, name="zone_plu")

    try:
        plu = gpd.read_file(plu_path)
        if plu.empty:
            print("  WARNING PLU vide - zone_plu=inconnu")
            return pd.Series(["inconnu"] * n_rows, index=parcelles.index, name="zone_plu")
        if plu.crs is None or str(plu.crs.to_epsg()) != "2154":
            plu = plu.to_crs(target_crs)

        zone_col = None
        for candidate in ["typezone", "zone", "libelle", "lib_zone", "code_zone", "nom_zone", "destdomi", "type"]:
            if candidate in plu.columns:
                zone_col = candidate
                break

        if zone_col is None:
            print("  WARNING PLU : aucune colonne de zone trouvee - zone_plu=inconnu")
            return pd.Series(["inconnu"] * n_rows, index=parcelles.index, name="zone_plu")

        def normalize_zone(value: object) -> str:
            if pd.isna(value):
                return "inconnu"
            zone = str(value).strip().upper().replace(" ", "")
            if zone.startswith("AU"):
                return zone
            if zone.startswith("U"):
                return zone
            if zone.startswith("N"):
                return "N"
            if zone.startswith("A"):
                return "A"
            return "inconnu"

        def zone_priority(zone: str) -> int:
            if zone.startswith("AU"):
                return 0
            if zone.startswith("U"):
                return 1
            if zone.startswith("N"):
                return 2
            if zone.startswith("A"):
                return 3
            return 4

        joined = gpd.sjoin(
            parcelles[["geometry"]].reset_index(),
            plu[[zone_col, "geometry"]],
            how="left",
            predicate="intersects",
        )
        joined["zone_norm"] = joined[zone_col].apply(normalize_zone)

        zone_series = joined.groupby("index")["zone_norm"].agg(
            lambda series: series.iloc[series.map(zone_priority).argmin()] if len(series) else "inconnu"
        ).reindex(range(n_rows), fill_value="inconnu")

        total = len(zone_series)
        count_u = zone_series.str.startswith("U") & ~zone_series.str.startswith("AU")
        count_au = zone_series.str.startswith("AU")
        count_n = zone_series == "N"
        count_a = zone_series == "A"
        count_unknown = zone_series == "inconnu"

        def pct(value: int) -> float:
            return value / total * 100 if total else 0

        print("  Zone PLU distribution:")
        print(f"    U (urbanise):        {count_u.sum():,} ({pct(int(count_u.sum())):.1f}%)")
        print(f"    AU (a urbaniser):    {count_au.sum():,} ({pct(int(count_au.sum())):.1f}%)")
        print(f"    N (naturel):         {count_n.sum():,} ({pct(int(count_n.sum())):.1f}%) - exclues du ML")
        print(f"    A (agricole):        {count_a.sum():,} ({pct(int(count_a.sum())):.1f}%) - exclues du ML")
        print(f"    inconnu:             {count_unknown.sum():,} ({pct(int(count_unknown.sum())):.1f}%)")

        return zone_series.rename("zone_plu")
    except Exception as exc:
        print(f"  WARNING PLU join erreur : {exc}")
        return pd.Series(["inconnu"] * n_rows, index=parcelles.index, name="zone_plu")
