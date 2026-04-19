from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


def load_brgm_local(brgm_dir: Path) -> dict:
    hazard_files = {}
    subdirs = {
        "argiles": brgm_dir / "argiles",
        "mvt_terrain": brgm_dir / "mvt",
    }

    for name, folder in subdirs.items():
        if not folder.exists():
            print(f"  Info BRGM {name}: dossier {folder} absent - ignore")
            continue

        shapefiles = list(folder.glob("*.shp"))
        if shapefiles:
            try:
                gdf = gpd.read_file(shapefiles[0])
                print(f"  BRGM {name}: {shapefiles[0].name} ({len(gdf)} entites)")
                hazard_files[name] = gdf
            except Exception as exc:
                print(f"  Warning BRGM {name}: erreur lecture {shapefiles[0].name} - {exc}")
            continue

        csv_files = list(folder.glob("*.csv"))
        if csv_files:
            csv_path = csv_files[0]
            df_csv = None
            for read_kwargs in (
                {"low_memory": False},
                {"skiprows": 3, "low_memory": False},
                {"sep": ";", "low_memory": False},
                {"sep": ";", "skiprows": 3, "low_memory": False},
            ):
                try:
                    df_csv = pd.read_csv(csv_path, **read_kwargs)
                    break
                except Exception:
                    continue

            if df_csv is None:
                print(f"  Warning BRGM {name}: impossible de lire le CSV {csv_path.name}")
                continue

            print(f"    Colonnes CSV {name}: {list(df_csv.columns[:8])}")
            lon_candidates = [
                "longitude",
                "lon",
                "x",
                "X",
                "LONGITUDE",
                "LON",
                "coord_x",
                "coordx",
                "wgs84_x",
                "longitudeDoublePrec",
                "xsaisi",
            ]
            lat_candidates = [
                "latitude",
                "lat",
                "y",
                "Y",
                "LATITUDE",
                "LAT",
                "coord_y",
                "coordy",
                "wgs84_y",
                "latitudeDoublePrec",
                "ysaisi",
            ]
            lon_col = next((col for col in lon_candidates if col in df_csv.columns), None)
            lat_col = next((col for col in lat_candidates if col in df_csv.columns), None)

            if lon_col and lat_col:
                gdf = gpd.GeoDataFrame(
                    df_csv,
                    geometry=gpd.points_from_xy(df_csv[lon_col], df_csv[lat_col]),
                    crs="EPSG:4326",
                )
                print(f"  BRGM {name}: {csv_path.name} ({len(gdf)} points, colonnes {lon_col}/{lat_col})")
                hazard_files[name] = gdf
            else:
                print(f"  Warning BRGM {name}: colonnes geo introuvables dans CSV")
                print(f"       Colonnes disponibles : {list(df_csv.columns)}")
            continue

        print(f"  Info BRGM {name}: aucun .shp ni .csv dans {folder} - ignore")

    if not hazard_files:
        print("  Info BRGM: aucune donnee locale - overlay ignore ce run")
        print("            Pour activer : telecharger fichiers sur georisques.gouv.fr")
        print(f"            -> {brgm_dir}/argiles/*.shp  (ou *.csv)")
        print(f"            -> {brgm_dir}/mvt/*.shp      (ou *.csv)")

    return hazard_files


def join_brgm_to_parcelles(parcelles: gpd.GeoDataFrame, hazard_files: dict) -> dict:
    result = {}

    for name, source in hazard_files.items():
        try:
            if isinstance(source, gpd.GeoDataFrame):
                hazard_gdf = source
            else:
                if not Path(source).exists():
                    continue
                hazard_gdf = gpd.read_file(source)

            if hazard_gdf.crs != parcelles.crs:
                hazard_gdf = hazard_gdf.to_crs(parcelles.crs)

            joined = gpd.sjoin(
                parcelles[["geometry"]].reset_index(),
                hazard_gdf[["geometry"]],
                how="left",
                predicate="intersects",
            )
            flag = joined.groupby("index")["index_right"].count() > 0
            result[f"brgm_{name}_flag"] = flag.reindex(range(len(parcelles)), fill_value=False).values
            n_flagged = int(flag.sum())
            print(
                f"  BRGM {name}: {n_flagged} parcelles en zone a risque "
                f"({n_flagged / max(len(parcelles), 1) * 100:.1f}%)"
            )
        except Exception as exc:
            print(f"  Warning BRGM join {name}: {exc}")

    return result


__all__ = ["join_brgm_to_parcelles", "load_brgm_local"]
