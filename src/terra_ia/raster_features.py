from __future__ import annotations

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats
from scipy.ndimage import generic_filter, laplace, uniform_filter


DEFAULT_ZONAL_WORKERS = max(1, min(16, max(4, (os.cpu_count() or 1) * 2)))
DEFAULT_ZONAL_CHUNK_SIZE = 128


def _zonal_stats_chunk(args: tuple[list[dict], str, list[str], float]) -> list[dict]:
    geoms, raster_path, stats, nodata = args
    return zonal_stats(geoms, raster_path, stats=stats, nodata=nodata)


def _run_zonal_stats(
    geoms: list[dict],
    raster_path: Path,
    stats: list[str],
    *,
    nodata: float = -9999.0,
) -> list[dict]:
    if not geoms:
        return []

    worker_count = max(1, int(os.environ.get("TERRA_IA_ZONAL_WORKERS", DEFAULT_ZONAL_WORKERS)))
    chunk_size = max(32, int(os.environ.get("TERRA_IA_ZONAL_CHUNK_SIZE", DEFAULT_ZONAL_CHUNK_SIZE)))
    raster_path_str = str(raster_path)

    if worker_count == 1 or len(geoms) < chunk_size:
        return zonal_stats(geoms, raster_path_str, stats=stats, nodata=nodata)

    chunks = [geoms[idx : idx + chunk_size] for idx in range(0, len(geoms), chunk_size)]
    chunk_args = [(chunk, raster_path_str, stats, nodata) for chunk in chunks]

    try:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(chunks))) as executor:
            futures = {
                executor.submit(_zonal_stats_chunk, chunk_arg): idx
                for idx, chunk_arg in enumerate(chunk_args)
            }
            ordered_results: list[list[dict] | None] = [None] * len(chunks)
            completed = 0
            total = len(chunks)
            progress_every = max(1, total // 8)

            for future in as_completed(futures):
                idx = futures[future]
                ordered_results[idx] = future.result()
                completed += 1
                if completed == total or completed % progress_every == 0:
                    print(
                        f"    zonal progress: {completed}/{total} chunks "
                        f"({worker_count} workers, chunk={chunk_size})",
                        flush=True,
                    )

        return [row for chunk in ordered_results if chunk is not None for row in chunk]
    except Exception as exc:
        print(f"    Warning: parallel zonal fallback serial : {exc}")
        return zonal_stats(geoms, raster_path_str, stats=stats, nodata=nodata)


def compute_slope_raster(mnt_data: np.ndarray, cellsize: float = 0.5) -> np.ndarray:
    def horn(window: np.ndarray) -> float:
        if np.any(np.isnan(window)):
            return np.nan
        z = window.reshape(3, 3)
        dzdx = ((z[0, 2] + 2 * z[1, 2] + z[2, 2]) - (z[0, 0] + 2 * z[1, 0] + z[2, 0])) / (
            8 * cellsize
        )
        dzdy = ((z[2, 0] + 2 * z[2, 1] + z[2, 2]) - (z[0, 0] + 2 * z[0, 1] + z[0, 2])) / (
            8 * cellsize
        )
        return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

    return generic_filter(mnt_data.astype(float), horn, size=3, mode="nearest").astype(np.float32)


def compute_tri_riley(mnt_data: np.ndarray) -> np.ndarray:
    def tri(window: np.ndarray) -> float:
        if np.any(np.isnan(window)):
            return np.nan
        return float(np.sqrt(np.sum((np.delete(window, 4) - window[4]) ** 2)))

    return generic_filter(mnt_data.astype(float), tri, size=3, mode="nearest").astype(np.float32)


def compute_twi_and_thalweg(
    mnt_path: Path,
    *,
    beta_min_deg: float = 0.5,
    thalweg_cells: int = 500,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        import richdem as rd

        dem = rd.LoadGDAL(str(mnt_path), no_data=-9999.0)
        rd.FillDepressions(dem, epsilon=True, in_place=True)

        accum_rd = rd.FlowAccumulation(dem, method="D8")
        accum = np.array(accum_rd).astype(np.float64)

        with rasterio.open(mnt_path) as src:
            mnt_arr = src.read(1).astype(float)
            cellsize = src.res[0]
            nodata = src.nodata or -9999.0
        mnt_arr[mnt_arr == nodata] = np.nan

        slope_deg = compute_slope_raster(mnt_arr, cellsize)
        slope_rad = np.deg2rad(slope_deg.astype(float))

        beta_min_rad = np.deg2rad(beta_min_deg)
        slope_eff = np.maximum(slope_rad, beta_min_rad)
        area = accum * (cellsize**2)
        area = np.maximum(area, cellsize**2)
        twi = np.log(area / np.tan(slope_eff))
        twi = np.clip(twi, None, 20.0)
        twi[np.isnan(mnt_arr)] = np.nan

        thalweg_mask = (accum >= thalweg_cells).astype(np.float32)
        thalweg_mask[np.isnan(mnt_arr)] = np.nan

        print(
            f"OK  TWI moy={np.nanmean(twi):.2f}  "
            f"thalweg={np.nansum(thalweg_mask == 1):.0f} pixels "
            f"({np.nanmean(thalweg_mask == 1) * 100:.1f}% raster)"
        )
        return twi.astype(np.float32), thalweg_mask

    except ImportError:
        print("Warning: richdem non disponible - TWI et thalweg ignores")
        print("   conda install -c conda-forge richdem -y")
        return None, None
    except Exception as exc:
        print(f"Warning: TWI/thalweg : {exc}")
        return None, None


def compute_svf(mnh_path: Path) -> np.ndarray | None:
    try:
        import rvt.vis

        with rasterio.open(mnh_path) as src:
            arr = src.read(1).astype(float)
            res = src.res[0]
            nodata = src.nodata or -9999
        arr[arr == nodata] = np.nan

        result = rvt.vis.sky_view_factor(
            dem=arr,
            resolution=res,
            compute_svf=True,
            compute_asvf=False,
            compute_opns=False,
        )
        svf = result["svf"]
        print(f"OK  moy={np.nanmean(svf):.3f}  min={np.nanmin(svf):.3f}")
        return svf.astype(np.float32)
    except ImportError:
        print("Warning: rvt-py non disponible (pip install rvt-py)")
        return None
    except Exception as exc:
        print(f"Warning: SVF : {exc}")
        return None


def compute_hillshade_winter(
    mnt_data: np.ndarray,
    cellsize: float = 0.5,
    lat_deg: float = 45.57,
) -> np.ndarray:
    decl = -23.45
    elev_rad = np.deg2rad(90.0 - lat_deg + decl)
    zen_rad = np.pi / 2 - elev_rad
    az_rad = np.pi

    dy, dx = np.gradient(mnt_data.astype(float), cellsize)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect_rad = np.arctan2(-dy, dx)

    hs = np.cos(zen_rad) * np.cos(slope_rad) + np.sin(zen_rad) * np.sin(slope_rad) * np.cos(
        aspect_rad - az_rad
    )
    hs = np.clip(hs, 0, 1).astype(np.float32)
    hs[np.isnan(mnt_data)] = np.nan
    return hs


def compute_aspect_south(mnt_data: np.ndarray, cellsize: float = 0.5) -> np.ndarray:
    dy, dx = np.gradient(mnt_data.astype(float), cellsize)
    aspect_deg = np.degrees(np.arctan2(-dy, dx)) % 360
    is_south = ((aspect_deg >= 135) & (aspect_deg <= 225)).astype(np.float32)
    is_south[np.isnan(mnt_data)] = np.nan
    return is_south


def compute_profile_curvature(mnt_data: np.ndarray, cellsize: float = 0.5) -> np.ndarray:
    mask = np.isnan(mnt_data)
    z = np.asarray(mnt_data, dtype=np.float32)

    if mask.any():
        valid = z[~mask]
        fill_value = float(np.nanmedian(valid)) if valid.size else 0.0
        z = z.copy()
        z[mask] = fill_value

    # A local Laplacian is a good low-memory proxy for profile curvature here.
    curv = laplace(z, mode="nearest") / np.float32(cellsize**2)
    curv = uniform_filter(curv.astype(np.float32), size=5)
    curv[mask] = np.nan
    return curv


def save_tif(array: np.ndarray, ref_path: Path, out_path: Path) -> None:
    with rasterio.open(ref_path) as ref:
        meta = ref.meta.copy()
    meta.update({"count": 1, "dtype": "float32", "compress": "lzw", "nodata": -9999.0})
    arr = array.astype("float32")
    arr[np.isnan(arr)] = -9999.0
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr, 1)


def zonal(
    raster_path: Path,
    parcelles: gpd.GeoDataFrame,
    prefix: str,
    stats: list | None = None,
) -> dict:
    stats = stats or ["mean", "max", "std"]
    geoms = [geom.__geo_interface__ for geom in parcelles.geometry]
    try:
        res = _run_zonal_stats(geoms, raster_path, stats, nodata=-9999.0)
        return {f"{prefix}_{stat}": [row.get(stat) for row in res] for stat in stats}
    except Exception as exc:
        print(f"    Warning: zonal {prefix} : {exc}")
        return {}


def zonal_percentile(
    raster_path: Path,
    parcelles: gpd.GeoDataFrame,
    prefix: str,
    percentiles: list[int],
) -> dict:
    stat_names = [f"percentile_{p}" for p in percentiles]
    geoms = [geom.__geo_interface__ for geom in parcelles.geometry]
    try:
        res = _run_zonal_stats(geoms, raster_path, stat_names + ["std"], nodata=-9999.0)
        out = {}
        for percentile in percentiles:
            out[f"{prefix}_p{percentile}"] = [row.get(f"percentile_{percentile}") for row in res]
        out[f"{prefix}_std"] = [row.get("std") for row in res]
        return out
    except Exception as exc:
        print(f"    Warning: zonal percentiles {prefix} : {exc}")
        return {}


def compute_flat_platform(
    slope_data: np.ndarray,
    cellsize: float = 0.5,
    theta_constructible: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import label as ndlabel

    _ = cellsize
    flat_mask = (slope_data <= theta_constructible).astype(np.float32)
    flat_mask[np.isnan(slope_data)] = np.nan

    flat_binary = flat_mask == 1
    labeled_arr, _ = ndlabel(flat_binary)
    return flat_mask, labeled_arr.astype(np.float32)


def compute_max_flat_area_per_parcel(
    flat_mask_path: Path,
    flat_labeled_path: Path,
    parcelles: gpd.GeoDataFrame,
    cellsize: float = 0.5,
) -> dict:
    _ = flat_labeled_path
    _ = cellsize

    geoms = [geom.__geo_interface__ for geom in parcelles.geometry]
    ratio_stats = _run_zonal_stats(geoms, flat_mask_path, ["mean", "count"], nodata=-9999.0)

    flat_ratio = [row.get("mean") if row.get("mean") is not None else np.nan for row in ratio_stats]
    surface_m2 = parcelles.geometry.area.values
    max_flat_areas = [
        float(surface_m2[idx] * flat_ratio[idx]) if not np.isnan(flat_ratio[idx]) else np.nan
        for idx in range(len(parcelles))
    ]

    valid_flat = [value for value in max_flat_areas if not np.isnan(value)]
    if valid_flat:
        print(
            f"OK  mediane max_flat={np.median(valid_flat):.0f}m2  "
            f"ratio_plat_moy={np.nanmean(flat_ratio):.3f}"
        )

    return {
        "max_flat_area_m2": max_flat_areas,
        "flat_area_ratio": flat_ratio,
    }


__all__ = [
    "compute_aspect_south",
    "compute_flat_platform",
    "compute_hillshade_winter",
    "compute_max_flat_area_per_parcel",
    "compute_profile_curvature",
    "compute_slope_raster",
    "compute_svf",
    "compute_tri_riley",
    "compute_twi_and_thalweg",
    "save_tif",
    "zonal",
    "zonal_percentile",
]
