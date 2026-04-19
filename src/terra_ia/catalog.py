from __future__ import annotations

COMMUNE_CODE = "73065"
TARGET_CRS = "EPSG:2154"
LAT_DEG = 45.57

PLU_CONSTRUCTIBLE_ZONES = ["U", "AU", "UB", "UC", "UD", "UE", "UF", "AUB"]


def build_mnt_url(x: int, y: int) -> str:
    return (
        f"https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&EXCEPTIONS=text/xml"
        f"&REQUEST=GetMap"
        f"&LAYERS=IGNF_LIDAR-HD_MNT_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
        f"&FORMAT=image/geotiff&STYLES=&CRS=EPSG:2154"
        f"&BBOX={x-0.25},{y-0.25},{x+999.75},{y+999.75}"
        f"&WIDTH=2000&HEIGHT=2000"
        f"&FILENAME=LHD_FXX_{x//1000:04d}_{y//1000:04d}_MNT_O_0M50_LAMB93_IGN69.tif"
    )


def build_mnh_url(x: int, y: int) -> str:
    return (
        f"https://data.geopf.fr/wms-r?SERVICE=WMS&VERSION=1.3.0&EXCEPTIONS=text/xml"
        f"&REQUEST=GetMap"
        f"&LAYERS=IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
        f"&FORMAT=image/geotiff&STYLES=&CRS=EPSG:2154"
        f"&BBOX={x-0.25},{y-0.25},{x+999.75},{y+999.75}"
        f"&WIDTH=2000&HEIGHT=2000"
        f"&FILENAME=LHD_FXX_{x//1000:04d}_{y//1000:04d}_MNH_O_0M50_LAMB93_IGN69.tif"
    )


GRID_X = [925000, 926000, 927000, 928000]
GRID_Y = [6499000, 6500000, 6501000, 6502000, 6503000, 6504000]

URLS_MNT = [build_mnt_url(x, y) for y in GRID_Y for x in GRID_X]
URLS_MNH = [build_mnh_url(x, y) for y in GRID_Y for x in GRID_X]

FEATURE_GROUPS = {
    "SLOPE": ["slope_p50", "slope_p90", "slope_std"],
    "HYDROLOGY": ["twi_mean", "has_thalweg_mean"],
    "MORPHOLOGY": ["profile_curvature_mean", "tri_mean"],
    "SUNLIGHT": ["hillshade_winter_mean", "aspect_south_ratio_mean", "svf_mean"],
}
ALL_FEATURES = [feature for features in FEATURE_GROUPS.values() for feature in features]

FEATURES_CPI_ONLY = [
    "max_flat_area_m2",
    "flat_area_ratio",
    "compactness_ratio",
    "elongation_ratio",
    "ces_existant",
    "ces_residuel",
    "emprise_residuelle_m2",
]

GROUP_WEIGHTS_TECHNIQUE = {"SLOPE": 0.40, "HYDROLOGY": 0.35, "MORPHOLOGY": 0.25}
GROUP_WEIGHTS_VALEUR = {"SUNLIGHT": 1.0}
GROUP_WEIGHTS = {"SLOPE": 0.35, "HYDROLOGY": 0.30, "MORPHOLOGY": 0.20, "SUNLIGHT": 0.15}

THETA_CONSTRUCTIBLE = 7.0
BUFFER_DISTANCE_M = 15.0

THRESHOLD_DISCLAIMER = """
SEUILS TERRA-IA V4 - NOTE METHODOLOGIQUE
Les seuils numeriques (8°, 15°, 25°, 20% thalweg) sont des conventions
internes calibrees sur donnees Chambery et inspirees de la pratique
professionnelle francaise. Ils ne constituent pas des valeurs normatives
directement prescrites par Eurocode 7, NF P 94-500 ou Code de l'Urbanisme.
Validation expert en cours avant soutenance finale.
"""
