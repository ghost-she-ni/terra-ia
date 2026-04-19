"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TERRA-IA — Pipeline complet v6.0                                           ║
║  Scoring de constructibilité morphologique — LiDAR HD IGN                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

CORRECTIONS V2 → V3
-------------------
  ✓ CRITIQUE : 24 dalles IGN (vs 4) → couverture ~80% commune Chambéry
  ✓ CRITIQUE : DVF Savoie depuis data.gouv.fr CSV direct (plus API Cerema)
  ✓ HAUTE    : Blocs spatiaux 300m (vs 200m) — meilleure indépendance CV
  ✓ HAUTE    : XGBRanker qid=block_id (groupes par contexte local, pas global)
  ✓ HAUTE    : GridSearch hyperparamètres XGBRanker
  ✓ MOYENNE  : Matrice de confusion par fold (diagnostic RF NDCG=1.000)
  ✓ MOYENNE  : Calibration τ softmin auto sur distribution CPI
  ✓ MOYENNE  : Analyse équilibre classes par bloc (blocs monoclasses détectés)
  ✓ BONUS    : svf_mean et aspect_south_ratio_mean ajoutés aux features ML
  ✓ BONUS    : Rapport statistique complet exporté en JSON

CORRECTIONS V3 → V4
-------------------
  ✓ CRITIQUE : Split CPI → CPI_technique (géotechnique) + CPI_valeur (commercial)
  ✓ HAUTE    : Fix jointure DVF — normalisation ID 14 chars + percentiles Chambéry
  ✓ HAUTE    : Nouvelles features max_flat_area_m2 et flat_area_ratio
  ✓ HAUTE    : Overlay BRGM géorisques (argiles, mouvements terrain)
  ✓ MOYENNE  : Relabellisation seuils comme heuristiques (pas normes directes)
  ✓ MOYENNE  : Buffer 15m autour parcelles pour features contextuelles

CORRECTIONS V4 → V5
-------------------
  ✓ CRITIQUE : Masque PLU — zones N/A exclues (Code Urb. L151-9)
  ✓ CRITIQUE : Compacité parcelle — détection formes allongées (Polsby-Popper)
  ✓ CRITIQUE : CES résiduel — emprise bâtie existante (BD TOPO IGN)
  ✓ HAUTE    : Validation clustering labels Snorkel (anti-circularity)
  ✓ HAUTE    : SHAP export enrichi (PLU + forme + CES + cluster_score)

CORRECTIONS V5 → V6
-------------------
  ✓ CRITIQUE : Snorkel V6 — H1/H2 basés sur 3 features stables validées (4 approches)
  ✓ CRITIQUE : H5 clustering — signal non supervisé dans le vote Snorkel (cluster_scores_v6.csv)
  ✓ HAUTE    : Intervalles de confiance CPI par bootstrap (100 runs)
  ✓ HAUTE    : Score consensus 4 approches (consensus_score)
  ✓ MOYENNE  : Outputs renommés v6

    Ref: Nicoletti M. (2025) — Rapport de stage Licence SIG, ECOGEO Nice
         Identification potentiel constructible par SIG — Nice
    Ref: Polsby & Popper (1991) — compactness index
    Ref: Code de l'Urbanisme L151-9 — zones N et A

PRÉREQUIS
---------
    conda install -c conda-forge rasterio geopandas rasterstats numpy scipy richdem -y
    pip install scikit-learn xgboost shap rvt-py osmnx requests

USAGE
-----
    # Run complet depuis zéro
    python pipeline.py

    # Skip téléchargement (données déjà présentes)
    $env:SKIP_DOWNLOAD="True"; python pipeline.py                      # Windows PS
    SKIP_DOWNLOAD=True python pipeline.py                              # Linux/Mac

    # Skip téléchargement + recalcul features seulement
    $env:SKIP_DOWNLOAD="True"; $env:SKIP_FEATURES="True"; python pipeline.py

RÉFÉRENCES SCIENTIFIQUES
------------------------
    Horn (1981)              Proceedings IEEE 69(1):14-47
    Riley et al. (1999)      Intermountain J. Sciences 5(1-4):23-27
    Beven & Kirkby (1979)    Hydrological Sciences Bulletin 24(1):43-69
    Barnes et al. (2014)     Computers & Geosciences 62:117-127
    Zevenbergen & Thorne (1987) Earth Surface Processes 12(1):47-56
    Zakšek et al. (2011)     Remote Sensing 3(2):398-415
    Ratner et al. (2017)     VLDB Journal 26:793-817   [Snorkel]
    Chen & Guestrin (2016)   KDD 2016                  [XGBoost]
    Burges et al. (2006)     ICML 2006                 [LambdaMART]
    Lundberg & Lee (2017)    NeurIPS 30                [SHAP]
    Valavi et al. (2019)     Methods Ecol Evol 10(2):225-232
"""

# ── Stdlib
import os, sys, time, warnings
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terra_ia import catalog as shared_catalog
from terra_ia.consensus import (
    compute_consensus_score as shared_compute_consensus_score,
    normalize_parcelle_ids as shared_normalize_parcelle_ids,
)
from terra_ia.downloads import (
    download_all_dalles as shared_download_all_dalles,
    download_dvf_latest_for_commune,
    download_file as shared_download_file,
    download_parcelles as shared_download_parcelles,
)
from terra_ia.hazards import (
    join_brgm_to_parcelles as shared_join_brgm_to_parcelles,
    load_brgm_local as shared_load_brgm_local,
)
from terra_ia.labeling import (
    analyze_blocks as shared_analyze_blocks,
    create_snorkel_labels_v6 as shared_create_snorkel_labels_v6,
    create_spatial_blocks as shared_create_spatial_blocks,
    load_cluster_scores as shared_load_cluster_scores,
    prepare_dvf as shared_prepare_dvf,
    validate_labels_with_clustering as shared_validate_labels_with_clustering,
)
from terra_ia.ml import (
    compare_models as shared_compare_models,
    compute_cpi_bootstrap as shared_compute_cpi_bootstrap,
    train_and_explain as shared_train_and_explain,
)
from terra_ia.pipeline_resilience import (
    PipelineCheckpointManager,
    load_osm_roads_union,
    module_available,
)
from terra_ia.pipeline_runtime import build_pipeline_runtime_config
from terra_ia.reporting import export_results as shared_export_results
from terra_ia.raster_features import (
    compute_aspect_south as shared_compute_aspect_south,
    compute_flat_platform as shared_compute_flat_platform,
    compute_hillshade_winter as shared_compute_hillshade_winter,
    compute_max_flat_area_per_parcel as shared_compute_max_flat_area_per_parcel,
    compute_profile_curvature as shared_compute_profile_curvature,
    compute_slope_raster as shared_compute_slope_raster,
    compute_svf as shared_compute_svf,
    compute_tri_riley as shared_compute_tri_riley,
    compute_twi_and_thalweg as shared_compute_twi_and_thalweg,
    save_tif as shared_save_tif,
    zonal as shared_zonal,
    zonal_percentile as shared_zonal_percentile,
)
from terra_ia.scoring import (
    calibrate_tau as shared_calibrate_tau,
    compute_ces_residuel as shared_compute_ces_residuel,
    compute_cpi_v3 as shared_compute_cpi_v3,
    compute_parcel_compactness as shared_compute_parcel_compactness,
    filter_parcelles as shared_filter_parcelles,
    normalize as shared_normalize,
    precompute_group_scores_for_tau as shared_precompute_group_scores_for_tau,
    softmin as shared_softmin,
)
from terra_ia.spatial_data import (
    join_plu_to_parcelles as shared_join_plu_to_parcelles,
    load_parcelles as shared_load_parcelles,
    merge_dalles as shared_merge_dalles,
    validate_raster as shared_validate_raster,
)
from terra_ia.urban_data import (
    download_bd_topo_batiments as shared_download_bd_topo_batiments,
    download_plu as shared_download_plu,
    plu_zone_counts as shared_plu_zone_counts,
)

# ── Scientific stack
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 0 - CONFIGURATION
# ==============================================================================

# All paths, flags, and output names are resolved in src/terra_ia/pipeline_runtime.py.
RUNTIME = build_pipeline_runtime_config()
PROJECT_ROOT = RUNTIME.project_root
DATA_DIR = RUNTIME.data_dir
RASTER_DIR = RUNTIME.raster_dir
OUTPUT_DIR = RUNTIME.output_dir
MNT_PATH = RUNTIME.mnt_path
MNH_PATH = RUNTIME.mnh_path
PARCELLES_PATH = RUNTIME.parcelles_path
DVF_PATH = RUNTIME.dvf_path
PLU_PATH = RUNTIME.plu_path
BRGM_DIR = RUNTIME.brgm_dir
SITADEL_PATH = RUNTIME.sitadel_path
BD_TOPO_PATH = RUNTIME.bd_topo_path
OUTPUT_CSV_V3 = RUNTIME.output_csv_v3
OUTPUT_CSV_ML = RUNTIME.output_csv_ml_v3
OUTPUT_REPORT = RUNTIME.output_report
OUTPUT_CSV_V4 = RUNTIME.output_csv_v4
OUTPUT_CSV_ML_V4 = RUNTIME.output_csv_ml_v4
OUTPUT_CSV_V6 = RUNTIME.output_csv_v6
OUTPUT_CSV_ML_V6 = RUNTIME.output_csv_ml_v6
OUTPUT_SHAP_PARCELLE = RUNTIME.output_shap_parcelle
OUTPUT_CLUSTER_SCORES = RUNTIME.output_cluster_scores
CLUSTER_SCORES_PATH = RUNTIME.cluster_scores_path
APPROACH3_PATH = RUNTIME.approach3_path
APPROACH4_PATH = RUNTIME.approach4_path
OUTPUT_README_ML_V6 = RUNTIME.output_readme_ml_v6
CHECKPOINT_DIR = RUNTIME.checkpoint_dir
PIPELINE_STATE_PATH = RUNTIME.pipeline_state_path
STAGE3_FEATURES_CHECKPOINT = RUNTIME.stage3_features_checkpoint
STAGE6_LABELS_CHECKPOINT = RUNTIME.stage6_labels_checkpoint
STAGE9_SCORES_CHECKPOINT = RUNTIME.stage9_scores_checkpoint
LOGS_DIR = RUNTIME.logs_dir
LOG_FILE = RUNTIME.log_file
OSM_ROADS_CACHE_PATH = RUNTIME.osm_roads_cache_path
SKIP_DOWNLOAD = RUNTIME.skip_download
SKIP_FEATURES = RUNTIME.skip_features
SKIP_PLU = RUNTIME.skip_plu
SKIP_BATI = RUNTIME.skip_bati
SKIP_BOOTSTRAP = RUNTIME.skip_bootstrap
SKIP_ZONAL = RUNTIME.skip_zonal
RESUME = RUNTIME.resume
REFRESH_OSM = RUNTIME.refresh_osm
URLS_MNT = shared_catalog.URLS_MNT
URLS_MNH = shared_catalog.URLS_MNH
COMMUNE_CODE = shared_catalog.COMMUNE_CODE
TARGET_CRS = shared_catalog.TARGET_CRS
LAT_DEG = shared_catalog.LAT_DEG
PLU_CONSTRUCTIBLE_ZONES = shared_catalog.PLU_CONSTRUCTIBLE_ZONES

# ── Paramètres filtrage
SEUIL_NAN      = 0.40    # V3 : assoupli à 40% (vs 50% V2) pour plus de parcelles
SEUIL_SURFACE  = 5000.0  # m² max résidentiel
SEUIL_SURF_MIN = 20.0    # m² min

# ── Paramètres score CPI V3
TAU_SOFTMIN    = None    # Auto-calibré sur distribution (voir calibrate_tau)
BETA_MIN_DEG   = 0.5     # Seuil anti-singularité TWI (Beven & Kirkby)

# ── Paramètres spatial CV V3
GRID_SIZE_M    = 300     # V3 : 300m (vs 200m V2) — meilleure indépendance
N_FOLDS        = 5

# ── Features organisées par groupe (pour CPI et ML)
FEATURE_GROUPS = shared_catalog.FEATURE_GROUPS
ALL_FEATURES = shared_catalog.ALL_FEATURES

# Features calculées (ÉTAPE 2/3) mais exclues du ML — corrélées à surface_m2,
# qui avait causé un overfitting via la condition H1 (surface > 200m²).
# Elles restent utilisées dans compute_cpi_v3() et exportées dans le CSV.
FEATURES_CPI_ONLY = shared_catalog.FEATURES_CPI_ONLY

# Poids spécifiés avant entraînement ML, inspirés de la pratique géotechnique
# alpine (Şatır & Berberoğlu 2016) et validés par SHAP V3.
# IMPORTANT : ces poids sont des engineering choices, pas des valeurs normatives.
# Validation expert en cours (IGN, Dr. Poux).
GROUP_WEIGHTS_TECHNIQUE = shared_catalog.GROUP_WEIGHTS_TECHNIQUE
GROUP_WEIGHTS_VALEUR    = shared_catalog.GROUP_WEIGHTS_VALEUR
GROUP_WEIGHTS = shared_catalog.GROUP_WEIGHTS

# ── Seuils V4 (engineering choices, voir THRESHOLD_DISCLAIMER)
THETA_CONSTRUCTIBLE = shared_catalog.THETA_CONSTRUCTIBLE
BUFFER_DISTANCE_M   = shared_catalog.BUFFER_DISTANCE_M
THRESHOLD_DISCLAIMER = shared_catalog.THRESHOLD_DISCLAIMER


# ==============================================================================
# SECTION 1 — TÉLÉCHARGEMENT
# ==============================================================================

def download_file(url: str, dest: Path, label: str = "") -> bool:
    """Télécharge un fichier. Skip si déjà présent."""
    return shared_download_file(url, dest, label)


def download_all_dalles(urls: list, prefix: str) -> list:
    """Télécharge toutes les dalles. Extrait le nom depuis FILENAME= dans l'URL."""
    return shared_download_all_dalles(urls, prefix, DATA_DIR)


def download_parcelles(code: str, dest: Path) -> bool:
    """Télécharge les parcelles cadastrales depuis cadastre.data.gouv.fr."""
    return shared_download_parcelles(code, dest)


def download_dvf_v3(dest: Path) -> bool:
    """
    Télécharge les DVF (Demandes de Valeur Foncière) Savoie depuis data.gouv.fr.

    Source : https://files.data.gouv.fr/geo-dvf/latest/csv/
    Contient toutes les transactions immobilières du département 73.
    Licence : Etalab 2.0 — données open data depuis 2014.

    Les fichiers sont en gzip (.csv.gz) — pandas les lit nativement.

    Ref : Ministère de l'Économie et des Finances (2023).
          Demandes de valeur foncière (DVF).
          data.gouv.fr, dataset #5c4ae55a634f4117716d5656
    """
    return download_dvf_latest_for_commune(
        dest,
        data_dir=DATA_DIR,
        commune_code=COMMUNE_CODE,
    )


def _plu_zone_counts(gdf: gpd.GeoDataFrame) -> dict:
    """Compte les zones principales U/AU/N/A dans un PLU."""
    return shared_plu_zone_counts(gdf)


def download_plu(dest: Path) -> bool:
    """
    Télécharge le PLU de Chambéry (INSEE 73065) depuis Géoportail Urbanisme.
    WFS prioritaire, fallback API direct. Non bloquant.
    """
    return shared_download_plu(
        dest,
        target_crs=TARGET_CRS,
        commune_code=COMMUNE_CODE,
    )


def download_bd_topo_batiments(dest: Path) -> bool:
    """
    Télécharge les bâtiments existants (BD TOPO) via WFS IGN.
    Non bloquant : en cas d'échec, retourne False.
    """
    return shared_download_bd_topo_batiments(dest, target_crs=TARGET_CRS)


def merge_dalles(paths: list, output: Path, label: str) -> bool:
    """Fusionne des dalles GeoTIFF. Gère les dalles manquantes gracieusement."""
    return shared_merge_dalles(paths, output, label)


# ==============================================================================
# SECTION 2 — VALIDATION DES DONNÉES
# ==============================================================================

def validate_raster(path: Path, label: str) -> bool:
    """Valide CRS, résolution et intégrité d'un raster IGN."""
    return shared_validate_raster(path, label)


def load_parcelles(path: Path) -> gpd.GeoDataFrame:
    """Charge les parcelles cadastrales et les reprojette en Lambert 93."""
    return shared_load_parcelles(path, target_crs=TARGET_CRS)


def join_plu_to_parcelles(parcelles: gpd.GeoDataFrame,
                          plu_path: Path) -> pd.Series:
    """
    Joint le PLU aux parcelles. Retourne une Series zone_plu (U, AU, N, A, inconnu).
    """
    return shared_join_plu_to_parcelles(
        parcelles,
        plu_path,
        target_crs=TARGET_CRS,
    )


# ==============================================================================
# SECTION 3 — CALCUL DES FEATURES RASTER
# ==============================================================================

def compute_slope_raster(mnt_data: np.ndarray,
                          cellsize: float = 0.5) -> np.ndarray:
    return shared_compute_slope_raster(mnt_data, cellsize)


def compute_tri_riley(mnt_data: np.ndarray) -> np.ndarray:
    return shared_compute_tri_riley(mnt_data)


def compute_twi_and_thalweg(mnt_path: Path,
                              beta_min_deg: float = 0.5,
                              thalweg_cells: int = 500
                              ) -> tuple:
    return shared_compute_twi_and_thalweg(
        mnt_path,
        beta_min_deg=beta_min_deg,
        thalweg_cells=thalweg_cells,
    )


def compute_svf(mnh_path: Path) -> np.ndarray | None:
    return shared_compute_svf(mnh_path)


def compute_hillshade_winter(mnt_data: np.ndarray,
                              cellsize: float = 0.5,
                              lat_deg: float = 45.57) -> np.ndarray:
    return shared_compute_hillshade_winter(mnt_data, cellsize, lat_deg)


def compute_aspect_south(mnt_data: np.ndarray,
                          cellsize: float = 0.5) -> np.ndarray:
    return shared_compute_aspect_south(mnt_data, cellsize)


def compute_profile_curvature(mnt_data: np.ndarray,
                               cellsize: float = 0.5) -> np.ndarray:
    return shared_compute_profile_curvature(mnt_data, cellsize)


def save_tif(array: np.ndarray, ref_path: Path, out_path: Path) -> None:
    return shared_save_tif(array, ref_path, out_path)


# ==============================================================================
# SECTION 4 — EXTRACTION ZONAL STATS
# ==============================================================================

def zonal(raster_path: Path, parcelles: gpd.GeoDataFrame,
          prefix: str, stats: list | None = None) -> dict:
    return shared_zonal(raster_path, parcelles, prefix, stats)


def zonal_percentile(raster_path: Path, parcelles: gpd.GeoDataFrame,
                     prefix: str, percentiles: list) -> dict:
    return shared_zonal_percentile(raster_path, parcelles, prefix, percentiles)


# ==============================================================================
# SECTION 5 — FILTRAGE ET NETTOYAGE
# ==============================================================================

def filter_parcelles(df: pd.DataFrame,
                     parcelles: gpd.GeoDataFrame) -> pd.DataFrame:
    return shared_filter_parcelles(
        df,
        parcelles,
        all_features=ALL_FEATURES,
        plu_constructible_zones=PLU_CONSTRUCTIBLE_ZONES,
        seuil_nan=SEUIL_NAN,
        seuil_surface=SEUIL_SURFACE,
        seuil_surf_min=SEUIL_SURF_MIN,
    )
    """
    Filtre les parcelles problématiques identifiées en V1/V2.
    Seuil NAN assoupli à 40% en V3 (vs 50% V2) pour récupérer plus de parcelles.
    """
    lidar_cols = [c for c in ALL_FEATURES if c in df.columns]
    df["nan_ratio"]  = df[lidar_cols].isna().mean(axis=1) if lidar_cols else 0.0
    df["surface_m2"] = parcelles.geometry.area.values
    plu_constructible = (
        df["zone_plu"].isin(PLU_CONSTRUCTIBLE_ZONES + ["inconnu"])
        if "zone_plu" in df.columns else pd.Series(True, index=df.index)
    )

    df["is_valid"] = (
        (df["nan_ratio"]  <= SEUIL_NAN) &
        (df["surface_m2"] <= SEUIL_SURFACE) &
        (df["surface_m2"] >= SEUIL_SURF_MIN) &
        plu_constructible
    )

    n_tot   = len(df)
    n_valid = df["is_valid"].sum()
    n_nan   = (df["nan_ratio"] > SEUIL_NAN).sum()
    n_big   = (df["surface_m2"] > SEUIL_SURFACE).sum()
    n_small = (df["surface_m2"] < SEUIL_SURF_MIN).sum()
    n_plu   = (~plu_constructible).sum()

    print(f"    Total               : {n_tot:,}")
    print(f"    ✓ Valides           : {n_valid:,} ({n_valid/n_tot*100:.1f}%)")
    print(f"    ✗ NaN > {SEUIL_NAN*100:.0f}%        : {n_nan:,} ({n_nan/n_tot*100:.1f}%)")
    print(f"    ✗ > {SEUIL_SURFACE:.0f}m²       : {n_big:,}")
    print(f"    ✗ < {SEUIL_SURF_MIN:.0f}m²          : {n_small:,}")
    print(f"    ✗ Zone N/A (PLU)    : {n_plu:,} ({n_plu/n_tot*100:.1f}%) — non constructible légalement")

    return df


# ==============================================================================
# SECTION 6 — SCORE CPI V3 DÉTERMINISTE
# ==============================================================================

def calibrate_tau(group_scores_df: pd.DataFrame,
                  target_std: float = 18.0) -> float:
    return shared_calibrate_tau(group_scores_df, target_std)
    """
    Calibration automatique de τ (température softmin).

    Principe : on cherche τ tel que l'écart-type du score final soit ~18 pts,
    ce qui donne une distribution [0,100] raisonnablement étalée.
    - τ trop faible → scores extrêmes, distribution bimodale
    - τ trop élevé  → scores trop lisses, tout proche de la moyenne

    Méthode : recherche binaire sur τ ∈ [0.1, 50].

    V3 améliore la V2 où τ=15 donnait 94.8% en "Moyen" — trop lisse.
    """
    groups = group_scores_df.values  # shape (n, 4)

    def score_std(tau):
        v = groups.astype(float)
        v_s = v - np.max(v, axis=1, keepdims=True)
        exp_neg = np.exp(-v_s / tau)
        w = exp_neg / exp_neg.sum(axis=1, keepdims=True)
        sm = (w * v).sum(axis=1)
        combined = 0.70 * v.mean(axis=1) * 0 + 0.30 * sm  # simplified
        combined = 0.70 * groups.mean(axis=1) + 0.30 * sm
        return combined.std()

    lo, hi = 0.5, 100.0
    for _ in range(50):
        mid = (lo + hi) / 2
        s = score_std(mid)
        if s < target_std:
            hi = mid
        else:
            lo = mid

    tau_opt = (lo + hi) / 2
    print(f"    τ calibré : {tau_opt:.2f}  (std visée={target_std}, obtenue={score_std(tau_opt):.2f})")
    return tau_opt


def softmin(values: np.ndarray, tau: float) -> np.ndarray:
    return shared_softmin(values, tau)
    """
    Softmin différentiable avec paramètre τ calibré.
    Ref : recommandation IA mathématiques (Mars 2026).
    τ→0 : converge vers min(x). τ→∞ : converge vers mean(x).
    Stabilisation numérique par décalage max.
    """
    v = np.array(values, dtype=float)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    v_s   = v - np.max(v, axis=1, keepdims=True)
    exp_n = np.exp(-v_s / max(tau, 1e-9))
    w     = exp_n / exp_n.sum(axis=1, keepdims=True)
    return (w * v).sum(axis=1)


def normalize(series, invert=False, pct_low=1.0, pct_high=99.0, stretch=False):
    return shared_normalize(series, invert, pct_low, pct_high, stretch)
    """
    Normalisation [0,1] robuste aux outliers.
    V3 : utilise les percentiles P1/P99 au lieu du min/max strict
    pour éviter qu'un pixel aberrant LiDAR comprime toute la distribution.
    V5 : paramètre stretch — transformation sigmoid pour étaler la distribution.
    """
    valid = series.dropna()
    mn = np.percentile(valid, pct_low)  if len(valid) > 0 else 0
    mx = np.percentile(valid, pct_high) if len(valid) > 0 else 1
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    n = (series.fillna(mn).clip(mn, mx) - mn) / (mx - mn)
    if stretch:
        # Applique une transformation sigmoid pour étaler la distribution
        # vers les extrêmes sans perdre l'ordre de ranking
        n = 1 / (1 + np.exp(-6 * (n - 0.5)))
        # Renormalise [0,1]
        n = (n - n.min()) / (n.max() - n.min() + 1e-9)
    return 1 - n if invert else n


def compute_flat_platform(slope_data: np.ndarray,
                           cellsize: float = 0.5,
                           theta_constructible: float = 7.0) -> tuple:
    return shared_compute_flat_platform(slope_data, cellsize, theta_constructible)


def compute_max_flat_area_per_parcel(flat_mask_path: Path,
                                      flat_labeled_path: Path,
                                      parcelles: gpd.GeoDataFrame,
                                      cellsize: float = 0.5) -> dict:
    return shared_compute_max_flat_area_per_parcel(
        flat_mask_path,
        flat_labeled_path,
        parcelles,
        cellsize,
    )


def compute_parcel_compactness(parcelles: gpd.GeoDataFrame) -> dict:
    return shared_compute_parcel_compactness(parcelles)
    """
    Calcul des métriques de forme (Polsby-Popper + allongement).
    Retourne compactness_ratio [0-1] et elongation_ratio.
    """
    compactness, elong = [], []
    for geom in parcelles.geometry:
        try:
            if geom is None or geom.is_empty:
                compactness.append(np.nan); elong.append(np.nan); continue
            area = geom.area
            perim = geom.length
            c = (4 * math.pi * area) / (perim ** 2) if perim > 0 and area > 0 else np.nan
            minx, miny, maxx, maxy = geom.bounds
            length = max(maxx - minx, maxy - miny)
            width  = min(maxx - minx, maxy - miny)
            e = (length / width) if width > 0 else np.nan
        except Exception:
            c, e = np.nan, np.nan
        compactness.append(c)
        elong.append(e)

    med_c = np.nanmedian(compactness) if compactness else np.nan
    n_prob = int(np.sum(np.array(compactness) < 0.20))
    pct_prob = n_prob / len(compactness) * 100 if compactness else 0.0
    print(f"    Compacité parcelles — médiane={med_c:.3f}  "
          f"ratio_problématiques (<0.20)={n_prob} ({pct_prob:.1f}%)")

    return {
        "compactness_ratio": compactness,
        "elongation_ratio":  elong
    }


def compute_ces_residuel(parcelles: gpd.GeoDataFrame,
                         bd_topo_path: Path,
                         plu_ces_max: float = 0.40) -> dict:
    return shared_compute_ces_residuel(parcelles, bd_topo_path, plu_ces_max=plu_ces_max)
    """
    Calcule le CES existant et résiduel par parcelle (BD TOPO bâtiments).
    Retourne ces_existant, ces_residuel, emprise_residuelle_m2.
    """
    n = len(parcelles)
    try:
        bati = gpd.read_file(bd_topo_path)
        if bati.empty:
            raise ValueError("BD TOPO vide")
        if bati.crs is None or str(bati.crs.to_epsg()) != "2154":
            bati = bati.to_crs(parcelles.crs)

        parc = parcelles.reset_index().rename(columns={"index": "parcel_idx"})
        inter = gpd.overlay(parc[["parcel_idx", "geometry"]],
                            bati[["geometry"]], how="intersection")
        inter["built_area"] = inter.geometry.area
        built_area = inter.groupby("parcel_idx")["built_area"].sum()
        built_series = built_area.reindex(range(n), fill_value=0.0)
        surface_parcelle = parcelles.geometry.area

        ces_existant = (built_series / surface_parcelle).clip(0, 1)
        ces_residuel = np.maximum(0, plu_ces_max - ces_existant)
        emprise_residuelle = ces_residuel * surface_parcelle

        med_res = float(np.nanmedian(ces_residuel)) if n else np.nan
        n_viable = int((emprise_residuelle >= 80).sum())
        n_sature = int((ces_residuel < 0.05).sum())
        pct_viable = n_viable / n * 100 if n else 0
        pct_sat = n_sature / n * 100 if n else 0

        print(f"    CES résiduel — médiane={med_res:.2f}")
        print(f"    Parcelles avec emprise résiduelle >= 80m²: {n_viable} ({pct_viable:.1f}%)")
        print(f"    Parcelles saturées (CES_résiduel < 0.05): {n_sature} ({pct_sat:.1f}%)")

        return {
            "ces_existant": ces_existant.values.tolist(),
            "ces_residuel": ces_residuel.values.tolist(),
            "emprise_residuelle_m2": emprise_residuelle.values.tolist()
        }
    except Exception as e:
        print(f"    ⚠ CES résiduel erreur : {e}")
        nan_list = [np.nan] * n
        return {
            "ces_existant": nan_list,
            "ces_residuel": nan_list,
            "emprise_residuelle_m2": nan_list
        }


def compute_cpi_v3(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    return shared_compute_cpi_v3(
        df,
        tau,
        group_weights=GROUP_WEIGHTS,
        group_weights_technique=GROUP_WEIGHTS_TECHNIQUE,
    )
    """
    Score CPI V3 — architecture deux niveaux avec τ calibré.

    Niveau 1 — Gates d'élimination
        has_thalweg_mean > 0.20  →  CPI ≤ 30
        slope_p90        > 25°   →  CPI ≤ 25

    Niveau 2 — Score continu (normalisation P1/P99 robuste)
        PENTE (35%)         : 0.40×f(p50) + 0.35×f(p90) + 0.25×f(std)
        HYDROLOGIE (30%)    : f(twi_mean)  (thalweg dans gate)
        MORPHOLOGIE (20%)   : 0.55×f(curv) + 0.45×f(tri)
        ENSOLEILLEMENT (15%): 0.40×f(aspect) + 0.35×f(hillshade) + 0.25×f(svf)

    Formule finale :
        CPI = gate × [0.70 × moy_pond + 0.30 × softmin(groupes, τ_calibré)]
    """
    def get(col, default=0.5):
        if col in df.columns:
            return df[col].fillna(
                df[col].median() if df[col].notna().any() else default)
        return pd.Series(default, index=df.index)

    # ── Scores par groupe [0,1]
    sp = (0.40 * normalize(get("slope_p50"), invert=True, stretch=True)
        + 0.35 * normalize(get("slope_p90"), invert=True, stretch=True)
        + 0.25 * normalize(get("slope_std"), invert=True))

    sh = normalize(get("twi_mean"), invert=True, stretch=True)
    if "has_thalweg_mean" in df.columns:
        thalw = df["has_thalweg_mean"].fillna(0).clip(0, 1)
        sh = sh * (1 - 0.4 * thalw)

    compactness_norm = normalize(get("compactness_ratio"), invert=False)
    sm = (0.44 * normalize(get("profile_curvature_mean"), invert=False)
        + 0.36 * normalize(get("tri_mean"), invert=True)
        + 0.20 * compactness_norm)

    # Ensoleillement V3 : 3 features (vs 2 en V2)
    has_svf = "svf_mean" in df.columns
    if has_svf:
        se = (0.40 * normalize(get("aspect_south_ratio_mean"))
            + 0.35 * normalize(get("hillshade_winter_mean"))
            + 0.25 * normalize(get("svf_mean")))
    else:
        se = (0.57 * normalize(get("aspect_south_ratio_mean"))
            + 0.43 * normalize(get("hillshade_winter_mean")))

    df["score_pente"]   = (sp * 100).round(1)
    df["score_hydro"]   = (sh * 100).round(1)
    df["score_morpho"]  = (sm * 100).round(1)
    df["score_soleil"]  = (se * 100).round(1)

    # ── Moyenne pondérée
    w = GROUP_WEIGHTS
    df["score_continu"] = (
        w["SLOPE"] * df["score_pente"] + w["HYDROLOGY"] * df["score_hydro"]
      + w["MORPHOLOGY"] * df["score_morpho"] + w["SUNLIGHT"] * df["score_soleil"]
    ).round(1)

    # ── Softmin avec τ calibré
    gscores = df[["score_pente","score_hydro","score_morpho","score_soleil"]].values
    df["score_softmin"] = softmin(gscores, tau).round(1)

    # ── Score brut
    df["cpi_brut"] = (0.70 * df["score_continu"] + 0.30 * df["score_softmin"]).round(1)

    # ── Niveau 1 : Gates
    df["gate_factor"] = 1.0
    df["gate_reason"] = ""

    if "has_thalweg_mean" in df.columns:
        m = df["has_thalweg_mean"].fillna(0) > 0.20
        df.loc[m, "gate_factor"] = np.minimum(
            df.loc[m, "gate_factor"],
            30.0 / df.loc[m, "cpi_brut"].clip(lower=1))
        df.loc[m, "gate_reason"] = "talweg_central"

    if "slope_p90" in df.columns:
        m = df["slope_p90"].fillna(0) > 25.0
        df.loc[m, "gate_factor"] = np.minimum(
            df.loc[m, "gate_factor"],
            25.0 / df.loc[m, "cpi_brut"].clip(lower=1))
        df.loc[m & (df["gate_reason"] == ""), "gate_reason"] = "pente_extreme"
        df.loc[m & (df["gate_reason"] != ""), "gate_reason"] = "talweg+pente"

    if "ces_existant" in df.columns:
        saturated = df["ces_existant"].fillna(0) > 0.80
        df.loc[saturated, "gate_factor"] = np.minimum(
            df.loc[saturated, "gate_factor"],
            40.0 / df.loc[saturated, "cpi_brut"].clip(lower=1)
        )
        df.loc[saturated & (df["gate_reason"] == ""), "gate_reason"] = "parcelle_saturée_CES"
        n_sat = int(saturated.sum())
        print(f"    Gate CES: {n_sat} parcelles saturées (>80% bâti) → CPI ≤ 40")

    df["CPI_v3"] = (df["cpi_brut"] * df["gate_factor"]).clip(0, 100).round(1)

    def interpret(s):
        if s < 20: return "Éliminatoire"
        if s < 40: return "Faible"
        if s < 65: return "Moyen"
        if s < 82: return "Bon"
        return "Excellent"

    df["CPI_v3_label"] = df["CPI_v3"].apply(interpret)
    df["shape_warning"] = ""
    if ("compactness_ratio" in df.columns) or ("elongation_ratio" in df.columns):
        comp = df.get("compactness_ratio", pd.Series(np.nan, index=df.index))
        elong = df.get("elongation_ratio", pd.Series(np.nan, index=df.index))
        df["shape_warning"] = (
            (comp < 0.20) | (elong > 3.0)
        ).map({True: "Parcelle allongée — emprise réelle limitée", False: ""})

    # ── CPI_technique (géotechnique) : SLOPE + HYDROLOGY + MORPHOLOGY
    wt = GROUP_WEIGHTS_TECHNIQUE
    score_cont_tech = (
        wt["SLOPE"] * df["score_pente"] + wt["HYDROLOGY"] * df["score_hydro"]
      + wt["MORPHOLOGY"] * df["score_morpho"]
    ).round(1)
    gscores_tech = df[["score_pente", "score_hydro", "score_morpho"]].values
    score_sm_tech = softmin(gscores_tech, tau).round(1)
    cpi_tech_brut = (0.70 * score_cont_tech + 0.30 * score_sm_tech).round(1)
    df["CPI_technique"] = (cpi_tech_brut * df["gate_factor"]).clip(0, 100).round(1)

    def interpret_technique(s):
        if s < 20: return "Éliminatoire"
        if s < 45: return "Contraint"
        if s < 65: return "Faisable"
        if s < 82: return "Favorable"
        return "Optimal"
    df["CPI_technique_label"] = df["CPI_technique"].apply(interpret_technique)

    # ── CPI_valeur (attractivité commerciale / RE2020) : SUNLIGHT uniquement
    if has_svf:
        cpi_valeur_raw = (
            0.40 * normalize(get("aspect_south_ratio_mean"))
          + 0.35 * normalize(get("hillshade_winter_mean"))
          + 0.25 * normalize(get("svf_mean"))
        )
    else:
        cpi_valeur_raw = (
            0.57 * normalize(get("aspect_south_ratio_mean"))
          + 0.43 * normalize(get("hillshade_winter_mean"))
        )
    df["CPI_valeur"] = (cpi_valeur_raw * 100).clip(0, 100).round(1)

    def interpret_valeur(s):
        if s < 33: return "Faible attractivité"
        if s < 67: return "Attractivité moyenne"
        return "Forte attractivité"
    df["CPI_valeur_label"] = df["CPI_valeur"].apply(interpret_valeur)

    df["ces_warning"] = ""
    if "emprise_residuelle_m2" in df.columns:
        low_emprise = df["emprise_residuelle_m2"].fillna(999) < 80
        df.loc[low_emprise, "ces_warning"] = (
            "Emprise résiduelle < 80m² — constructibilité très limitée"
        )

    n_valid = df["is_valid"].sum() if "is_valid" in df.columns else len(df)
    valid_cpi = df.loc[df.get("is_valid", pd.Series(True, index=df.index)), "CPI_v3"]
    print(f"    CPI_v3 (parcelles valides) — "
          f"moy={valid_cpi.mean():.1f}  std={valid_cpi.std():.1f}  "
          f"min={valid_cpi.min():.1f}  max={valid_cpi.max():.1f}")
    print(f"    Gates : {(df['gate_factor'] < 1).sum()} parcelles plafonnées")
    print(f"    Distribution :")
    for lbl in ["Éliminatoire","Faible","Moyen","Bon","Excellent"]:
        n = (df["CPI_v3_label"] == lbl).sum()
        print(f"      {lbl:<15}: {n:>5} ({n/len(df)*100:.1f}%)")

    return df


# ==============================================================================
# SECTION 7 — LABELS SNORKEL V3
# ==============================================================================

def prepare_dvf(dvf_path: Path) -> pd.DataFrame | None:
    return shared_prepare_dvf(dvf_path, commune_code=COMMUNE_CODE)




def load_cluster_scores(parcelles_index) -> pd.Series:
    return shared_load_cluster_scores(CLUSTER_SCORES_PATH, parcelles_index)


def create_snorkel_labels_v6(df: pd.DataFrame,
                              dvf: pd.DataFrame | None = None) -> pd.Series:
    cluster_scores = load_cluster_scores(df.index)
    return shared_create_snorkel_labels_v6(df, dvf=dvf, cluster_scores=cluster_scores)

def validate_labels_with_clustering(df: pd.DataFrame,
                                    labels: pd.Series,
                                    feat_cols: list) -> dict:
    return shared_validate_labels_with_clustering(df, labels, feat_cols)


# ==============================================================================
# SECTION 8 — SPATIAL CV + ANALYSE BLOCS
# ==============================================================================

def create_spatial_blocks(parcelles: gpd.GeoDataFrame,
                           grid_size: int = GRID_SIZE_M) -> pd.Series:
    return shared_create_spatial_blocks(parcelles, grid_size=grid_size)


def analyze_blocks(df: pd.DataFrame, labels: pd.Series) -> dict:
    return shared_analyze_blocks(df, labels, grid_size_m=GRID_SIZE_M)


# ==============================================================================
# SECTION 9 — COMPARAISON DES MODÈLES ML V3
# ==============================================================================

def compare_models_v3(df: pd.DataFrame,
                      labels: pd.Series,
                      parcelles: gpd.GeoDataFrame) -> dict:
    _ = parcelles
    return shared_compare_models(
        df,
        labels,
        all_features=ALL_FEATURES,
        grid_size_m=GRID_SIZE_M,
        n_folds=N_FOLDS,
    )


# ==============================================================================
# SECTION 10 — XGBOOST LAMBDAMART FINAL + SHAP V3
# ==============================================================================

def train_and_explain_v3(df: pd.DataFrame,
                          labels: pd.Series,
                          best_params: dict | None = None) -> None:
    return shared_train_and_explain(
        df,
        labels,
        all_features=ALL_FEATURES,
        feature_groups=FEATURE_GROUPS,
        group_weights=GROUP_WEIGHTS,
        data_dir=DATA_DIR,
        output_shap_parcelle=OUTPUT_SHAP_PARCELLE,
        shap_importance_path=DATA_DIR / "shap_importance_v3.csv",
        shap_group_path=DATA_DIR / "shap_importance_groupes_v3.csv",
        best_params=best_params,
    )




# ==============================================================================
# SECTION 10B — BOOTSTRAP CPI_ML + CONSENSUS SCORE
# ==============================================================================

def compute_cpi_bootstrap(df: pd.DataFrame,
                          feat_cols: list,
                          n_bootstrap: int = 100,
                          confidence: float = 0.95) -> pd.DataFrame:
    return shared_compute_cpi_bootstrap(
        df,
        feat_cols,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


def _normalize_parcelle_ids(series: pd.Series) -> pd.Series:
    return shared_normalize_parcelle_ids(series)


def compute_consensus_score(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Calcule le score de consensus entre les 4 approches de scoring.
    Retourne (DataFrame, meta) avec consensus_score, consensus_confidence, n_approaches_used.
    """
    return shared_compute_consensus_score(
        df,
        approach3_path=APPROACH3_PATH,
        approach4_path=APPROACH4_PATH,
        load_cluster_scores=load_cluster_scores,
    )

# ==============================================================================
# SECTION 11 — EXPORT
# ==============================================================================

def export_v3(df: pd.DataFrame, parcelles: gpd.GeoDataFrame,
              stats_report: dict) -> None:
    return shared_export_results(
        df,
        parcelles,
        stats_report,
        output_csv_v6=OUTPUT_CSV_V6,
        output_csv_ml_v6=OUTPUT_CSV_ML_V6,
        output_report=OUTPUT_REPORT,
        output_readme_ml_v6=OUTPUT_README_ML_V6,
        output_shap_parcelle=OUTPUT_SHAP_PARCELLE,
        all_features=ALL_FEATURES,
        feature_groups=FEATURE_GROUPS,
        grid_size_m=GRID_SIZE_M,
        commune_code=COMMUNE_CODE,
        tau_softmin=TAU_SOFTMIN,
    )
    """
    Export trois fichiers + rapport JSON.

    1. features_parcelles_v6.csv — Dataset complet
    2. ml_dataset_v6.csv         — Dataset propre coéquipier ML
    3. rapport_stats_v6.json     — Rapport machine-readable pour monitoring
    """
    score_cols = ["CPI_v3", "CPI_v3_label", "CPI_ML_v3",
                  "CPI_technique", "CPI_technique_label",
                  "score_pente", "score_hydro", "score_morpho", "score_soleil",
                  "score_continu", "score_softmin", "gate_factor", "gate_reason",
                  "zone_plu", "compactness_ratio", "elongation_ratio",
                  "shape_warning", "ces_existant", "ces_residuel",
                  "emprise_residuelle_m2", "ces_warning", "cluster_score",
                  "cpi_ml_ci_low", "cpi_ml_ci_high", "cpi_ml_std",
                  "consensus_score", "consensus_confidence", "n_approaches_used"]
    brgm_cols  = [c for c in df.columns if c.startswith("brgm_")]
    feat_cols  = [c for c in ALL_FEATURES if c in df.columns]
    meta_cols  = ["commune", "surface_m2", "nan_ratio", "is_valid",
                  "proxy_label", "block_id"]
    internal_val_cols = [c for c in ["CPI_valeur", "CPI_valeur_label"] if c in df.columns]
    all_cols   = [c for c in meta_cols + feat_cols + score_cols + brgm_cols + internal_val_cols if c in df.columns]

    df_export = df.copy()
    if "id" in parcelles.columns:
        df_export.insert(0, "id_parcelle", parcelles["id"].values)

    df_export[all_cols].to_csv(OUTPUT_CSV_V6, index=True)
    print(f"\n  ✓ Dataset complet : {OUTPUT_CSV_V6}")
    print(f"    {len(df_export):,} parcelles × {len(all_cols)} colonnes")

    # Dataset ML
    valid_labeled = (df["is_valid"] & (df["proxy_label"] != -1)) \
                    if "proxy_label" in df.columns and "is_valid" in df.columns \
                    else pd.Series(False, index=df.index)

    ml_cols = (["id_parcelle"] if "id_parcelle" in df_export.columns else []) \
              + ["block_id"] + feat_cols + ["proxy_label", "CPI_v3"]
    ml_cols = [c for c in ml_cols if c in df_export.columns]

    df_ml = df_export[valid_labeled][ml_cols].copy() if valid_labeled.any() \
            else pd.DataFrame(columns=ml_cols)

    df_ml.to_csv(OUTPUT_CSV_ML_V6, index=False)
    n_pos = (df_ml.get("proxy_label", pd.Series()) == 1).sum()
    n_neg = (df_ml.get("proxy_label", pd.Series()) == 0).sum()
    print(f"\n  ✓ Dataset ML : {OUTPUT_CSV_ML_V6}")
    print(f"    {len(df_ml):,} parcelles × {len(df_ml.columns)} colonnes")
    print(f"    Labels : {n_pos} pos / {n_neg} neg")

    # README coéquipier V6
    readme = build_ml_dataset_readme(
        output_csv_ml_v6=OUTPUT_CSV_ML_V6,
        feat_cols=feat_cols,
        feature_groups=FEATURE_GROUPS,
        grid_size_m=GRID_SIZE_M,
        shap_filename=OUTPUT_SHAP_PARCELLE.name,
        generated_from="python pipeline.py",
    )
    with open(OUTPUT_README_ML_V6, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"\n  ✓ README coéquipier : {OUTPUT_README_ML_V6}")

    # Rapport JSON
    valid = df[df.get("is_valid", pd.Series(True, index=df.index))]
    stats_report.update({
        "version": "6.0",
        "commune": COMMUNE_CODE,
        "n_total": int(len(df)),
        "n_valid": int(df.get("is_valid", pd.Series(True, index=df.index)).sum()),
        "n_labeled": int(len(df_ml)),
        "n_pos": int(n_pos), "n_neg": int(n_neg),
        "n_blocks": int(df.get("block_id", pd.Series()).nunique()),
        "cpi_v3_stats": {
            "mean": float(valid["CPI_v3"].mean()) if "CPI_v3" in valid else None,
            "std":  float(valid["CPI_v3"].std())  if "CPI_v3" in valid else None,
            "min":  float(valid["CPI_v3"].min())  if "CPI_v3" in valid else None,
            "max":  float(valid["CPI_v3"].max())  if "CPI_v3" in valid else None,
        },
        "cpi_technique_stats": {
            "mean": float(valid["CPI_technique"].mean()) if "CPI_technique" in valid.columns else None,
            "std":  float(valid["CPI_technique"].std())  if "CPI_technique" in valid.columns else None,
            "min":  float(valid["CPI_technique"].min())  if "CPI_technique" in valid.columns else None,
            "max":  float(valid["CPI_technique"].max())  if "CPI_technique" in valid.columns else None,
        },
        "brgm_flags": {c: int(df[c].sum()) for c in brgm_cols if c in df.columns},
        "features": feat_cols,
        "all_features": ALL_FEATURES,
        "grid_size_m": GRID_SIZE_M,
        "tau_softmin":  float(TAU_SOFTMIN) if TAU_SOFTMIN else None,
        "plu_distribution": (lambda s: {
            "U":  int((s.str.upper().str.startswith("U") & ~s.str.upper().str.startswith("AU")).sum()) if not s.empty else 0,
            "AU": int(s.str.upper().str.startswith("AU").sum()) if not s.empty else 0,
            "N":  int((s == "N").sum()) if not s.empty else 0,
            "A":  int((s == "A").sum()) if not s.empty else 0,
            "inconnu": int((s == "inconnu").sum()) if not s.empty else 0
        })(df["zone_plu"] if "zone_plu" in df.columns else pd.Series(dtype=str)),
        "shape_metrics": {
            "median_compactness": float(df["compactness_ratio"].median()) if "compactness_ratio" in df.columns else None,
            "n_elongated": int((df.get("compactness_ratio", pd.Series()) < 0.20).sum()) if "compactness_ratio" in df.columns else 0,
        },
        "ces_metrics": {
            "median_residuel": float(df["ces_residuel"].median()) if "ces_residuel" in df.columns else None,
            "n_viable_80m2": int((df.get("emprise_residuelle_m2", pd.Series()) >= 80).sum()) if "emprise_residuelle_m2" in df.columns else 0,
            "n_saturated": int((df.get("ces_existant", pd.Series()) > 0.80).sum()) if "ces_existant" in df.columns else 0,
        },
        "cluster_label_agreement": stats_report.get("cluster_label_agreement"),
    })
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  ✓ Rapport stats JSON : {OUTPUT_REPORT}")

    # Résumé console
    print(f"\n  ── Distribution CPI_v3 (parcelles valides) ──")
    for lbl in ["Éliminatoire","Faible","Moyen","Bon","Excellent"]:
        n = (df["CPI_v3_label"] == lbl).sum() if "CPI_v3_label" in df.columns else 0
        p = n / len(df) * 100
        print(f"    {lbl:<15} : {n:>6,} ({p:>5.1f}%)")

    if "CPI_technique_label" in df.columns:
        print(f"\n  ── Distribution CPI_technique ──")
        for lbl in ["Éliminatoire","Contraint","Faisable","Favorable","Optimal"]:
            n = (df["CPI_technique_label"] == lbl).sum()
            p = n / len(df) * 100
            print(f"    {lbl:<15} : {n:>6,} ({p:>5.1f}%)")

    print("\n  ── Analyse PLU ──")
    if "zone_plu" in df.columns:
        s = df["zone_plu"]
        counts = {
            "U":  (s.str.upper().str.startswith("U") & ~s.str.upper().str.startswith("AU")).sum(),
            "AU": s.str.upper().str.startswith("AU").sum(),
            "N":  (s == "N").sum(),
            "A":  (s == "A").sum(),
            "inconnu": (s == "inconnu").sum(),
        }
        for zone in ["U", "AU", "N", "A", "inconnu"]:
            print(f"    {zone:<10}: {counts[zone]:>6,} parcelles")
    else:
        print("    zone_plu indisponible")

    print("\n  ── Forme des parcelles ──")
    if "compactness_ratio" in df.columns:
        n_elong = (df["compactness_ratio"] < 0.20).sum()
        print(f"    Parcelles allongées (<0.20) : {n_elong:,} ({n_elong/len(df)*100:.1f}%)")
    else:
        print("    Compacité non calculée")

    print("\n  ── CES résiduel ──")
    if "emprise_residuelle_m2" in df.columns:
        n_viable = (df["emprise_residuelle_m2"] >= 80).sum()
        n_saturated = (df.get("ces_existant", pd.Series()))\
            .fillna(0).gt(0.80).sum() if "ces_existant" in df.columns else 0
        print(f"    Emprise >= 80m²  : {n_viable:,} parcelles")
        print(f"    Parcelles saturées (>80% bâti) : {n_saturated:,}")
    else:
        print("    CES résiduel non calculé")

    print("\n  ── Corrélations croisées des scores ──")
    score_cols_corr = ["CPI_v3", "CPI_technique",
                       "CPI_ML_v3", "cluster_score"]
    available = [c for c in score_cols_corr if c in df.columns]
    if len(available) >= 2:
        corr_matrix = df[available].corr(method="pearson").round(3)
        print(corr_matrix.to_string())
    else:
        print("    Corrélation non calculable (scores manquants)")

    if brgm_cols:
        print(f"\n  ── Alertes BRGM ──")
        for col in brgm_cols:
            if col in df.columns:
                n = df[col].sum()
                p = n / len(df) * 100
                print(f"    {col:<30} : {n:>6,} parcelles ({p:>5.1f}%)")

    if "CPI_ML_v3" in df.columns:
        print(f"\n  ── Top 5 parcelles CPI_ML_v3 ──")
        show_cols = [c for c in ["CPI_ML_v3","CPI_v3","slope_p50","slope_p90",
                                   "svf_mean","twi_mean","surface_m2"] if c in df.columns]
        valid_sc = df[df.get("is_valid", pd.Series(True, index=df.index))]
        print(valid_sc.nlargest(5, "CPI_ML_v3")[show_cols].round(2).to_string())


# ==============================================================================
# SECTION — BRGM GÉORISQUES
# Données BRGM/Georisques reproduites à titre informatif uniquement.
# Elles ne remplacent pas la consultation des documents PPR officiels
# auprès des services de l'État (DDT, préfecture).
# ==============================================================================

def load_brgm_local(brgm_dir: Path) -> dict:
    return shared_load_brgm_local(brgm_dir)
    """
    Charge les couches BRGM depuis des fichiers locaux (shapefile ou CSV).

    Structure attendue :
      data/lidar_chamberey/brgm/argiles/  → *.shp ou *.csv argiles
      data/lidar_chamberey/brgm/mvt/      → *.shp ou *.csv mouvements terrain

    Priorité : .shp > .csv
    Pour les CSV, les colonnes longitude/latitude (ou X/Y) sont converties
    en GeoDataFrame points (EPSG:4326).

    Ces fichiers sont téléchargeables sur georisques.gouv.fr
    Ils sont utilisés en lecture seule — aucun appel API.

    Disclaimer : données reproduites à titre informatif uniquement.
    Ne remplacent pas la consultation des documents PPR officiels.

    Retourne : dict {name: GeoDataFrame}
    """
    hazard_files = {}

    subdirs = {
        "argiles":     brgm_dir / "argiles",
        "mvt_terrain": brgm_dir / "mvt",
    }

    for name, folder in subdirs.items():
        if not folder.exists():
            print(f"  ℹ  BRGM {name} : dossier {folder} absent — ignoré")
            continue

        # 1. Cherche d'abord un shapefile
        shps = list(folder.glob("*.shp"))
        if shps:
            try:
                gdf = gpd.read_file(shps[0])
                print(f"  ✓ BRGM {name} : {shps[0].name} ({len(gdf)} entités)")
                hazard_files[name] = gdf
            except Exception as e:
                print(f"  ⚠  BRGM {name} : erreur lecture {shps[0].name} — {e}")
            continue

        # 2. Sinon, cherche un CSV
        csvs = list(folder.glob("*.csv"))
        if csvs:
            csv_path = csvs[0]
            df_csv = None
            try:
                # Stratégie 1 : lecture normale
                df_csv = pd.read_csv(csv_path, low_memory=False)
            except Exception:
                try:
                    # Stratégie 2 : skip les premières lignes de metadata (souvent 3-5 lignes)
                    df_csv = pd.read_csv(csv_path, skiprows=3, low_memory=False)
                except Exception:
                    try:
                        # Stratégie 3 : séparateur point-virgule (fréquent dans les CSV français)
                        df_csv = pd.read_csv(csv_path, sep=";", low_memory=False)
                    except Exception:
                        try:
                            df_csv = pd.read_csv(csv_path, sep=";", skiprows=3, low_memory=False)
                        except Exception as e:
                            print(f"  ⚠  BRGM {name} : impossible de lire le CSV — {e}")
                            continue

            # Après lecture réussie, afficher les colonnes pour debug
            print(f"    Colonnes CSV {name} : {list(df_csv.columns[:8])}")

            # Chercher colonnes géographiques — noms possibles dans les CSV BRGM
            lon_candidates = ["longitude", "lon", "x", "X", "LONGITUDE", "LON",
                              "coord_x", "coordx", "wgs84_x", "longitudeDoublePrec",
                              "xsaisi"]
            lat_candidates = ["latitude",  "lat", "y", "Y", "LATITUDE",  "LAT",
                              "coord_y", "coordy", "wgs84_y", "latitudeDoublePrec",
                              "ysaisi"]

            lon_col = next((c for c in lon_candidates if c in df_csv.columns), None)
            lat_col = next((c for c in lat_candidates if c in df_csv.columns), None)

            if lon_col and lat_col:
                gdf = gpd.GeoDataFrame(
                    df_csv,
                    geometry=gpd.points_from_xy(df_csv[lon_col], df_csv[lat_col]),
                    crs="EPSG:4326"
                )
                print(f"  ✓ BRGM {name} : {csv_path.name} "
                      f"({len(gdf)} points, colonnes {lon_col}/{lat_col})")
                hazard_files[name] = gdf
            else:
                print(f"  ⚠  BRGM {name} : colonnes géo introuvables dans CSV")
                print(f"       Colonnes disponibles : {list(df_csv.columns)}")
            continue

        print(f"  ℹ  BRGM {name} : aucun .shp ni .csv dans {folder} — ignoré")

    if not hazard_files:
        print("  ℹ  BRGM : aucune donnée locale — overlay ignoré ce run")
        print("            Pour activer : télécharger fichiers sur georisques.gouv.fr")
        print(f"            → {brgm_dir}/argiles/*.shp  (ou *.csv)")
        print(f"            → {brgm_dir}/mvt/*.shp      (ou *.csv)")

    return hazard_files


def join_brgm_to_parcelles(parcelles: gpd.GeoDataFrame,
                           hazard_files: dict) -> dict:
    return shared_join_brgm_to_parcelles(parcelles, hazard_files)
    """
    Joint les couches BRGM aux parcelles — flags booléens uniquement.

    Stratégie : flags séparés, JAMAIS fusionnés dans le CPI.
    Affichés comme warnings indépendants dans l'UI.

    Retourne un dict de colonnes booléennes :
    - brgm_argiles_flag : True si parcelle dans zone aléa retrait-gonflement moyen/fort
    - brgm_mvt_terrain_flag : True si parcelle dans zone mouvement de terrain recensé
    """
    result = {}

    for name, source in hazard_files.items():
        try:
            # source peut être un GeoDataFrame (chargé par load_brgm_local)
            # ou un Path/str (compatibilité ascendante)
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
                how="left", predicate="intersects"
            )
            flag = joined.groupby("index")["index_right"].count() > 0
            result[f"brgm_{name}_flag"] = flag.reindex(
                range(len(parcelles)), fill_value=False).values
            n_flagged = flag.sum()
            print(f"  BRGM {name} : {n_flagged} parcelles en zone à risque "
                  f"({n_flagged/len(parcelles)*100:.1f}%)")
        except Exception as e:
            print(f"  ⚠  BRGM join {name} : {e}")

    return result


STAGE3_STEP_COLUMNS = {
    "slope_percentiles": ["slope_p50", "slope_p90", "slope_std"],
    "flat_area_ratio": ["max_flat_area_m2", "flat_area_ratio"],
    "tri": ["tri_mean", "tri_max"],
    "twi": ["twi_mean", "twi_max"],
    "thalweg": ["has_thalweg_mean"],
    "svf": ["svf_mean", "svf_min"],
    "hillshade": ["hillshade_winter_mean"],
    "aspect_south": ["aspect_south_ratio_mean"],
    "curvature": ["profile_curvature_mean"],
    "height_obj": ["height_obj_max", "height_obj_mean"],
    "road_distance": ["dist_road_m"],
    "parcel_shape": ["compactness_ratio", "elongation_ratio"],
    "ces_residual": ["ces_existant", "ces_residuel", "emprise_residuelle_m2"],
    "zoning_context": ["commune", "id", "zone_plu", "block_id", "brgm_argiles_flag", "brgm_mvt_terrain_flag"],
}

STAGE3_STEP_DEPENDENCIES = {
    "slope_percentiles": ["mnt_path", "parcelles_path"],
    "flat_area_ratio": ["mnt_path", "parcelles_path"],
    "tri": ["mnt_path", "parcelles_path"],
    "twi": ["mnt_path", "parcelles_path"],
    "thalweg": ["mnt_path", "parcelles_path"],
    "svf": ["mnh_path", "parcelles_path"],
    "hillshade": ["mnt_path", "parcelles_path"],
    "aspect_south": ["mnt_path", "parcelles_path"],
    "curvature": ["mnt_path", "parcelles_path"],
    "height_obj": ["mnh_path", "parcelles_path"],
    "road_distance": ["parcelles_path", "osm_cache_path", "option:refresh_osm"],
    "parcel_shape": ["parcelles_path"],
    "ces_residual": ["parcelles_path", "bd_topo_path", "option:skip_bati"],
    "zoning_context": ["parcelles_path", "plu_path", "brgm_dir", "option:skip_plu"],
}

STAGE6_RESUME_DEPENDENCIES = ["dvf_path"]
STAGE9_RESUME_DEPENDENCIES = ["option:skip_bootstrap"]


def _restore_columns(target_df: pd.DataFrame, source_df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in source_df.columns:
            target_df[col] = source_df[col].reindex(target_df.index)


def _feature_availability_status(df: pd.DataFrame) -> dict:
    def _has_data(columns: list[str]) -> bool:
        return all(col in df.columns for col in columns) and any(df[col].notna().any() for col in columns)

    return {
        "slope_percentiles": _has_data(STAGE3_STEP_COLUMNS["slope_percentiles"]),
        "flat_area_ratio": _has_data(STAGE3_STEP_COLUMNS["flat_area_ratio"]),
        "tri": _has_data(STAGE3_STEP_COLUMNS["tri"]),
        "twi": _has_data(STAGE3_STEP_COLUMNS["twi"]),
        "thalweg": _has_data(STAGE3_STEP_COLUMNS["thalweg"]),
        "svf": _has_data(STAGE3_STEP_COLUMNS["svf"]),
        "hillshade": _has_data(STAGE3_STEP_COLUMNS["hillshade"]),
        "aspect_south": _has_data(STAGE3_STEP_COLUMNS["aspect_south"]),
        "curvature": _has_data(STAGE3_STEP_COLUMNS["curvature"]),
        "height_obj": _has_data(STAGE3_STEP_COLUMNS["height_obj"]),
        "road_distance": "dist_road_m" in df.columns and df["dist_road_m"].notna().any(),
        "parcel_shape": _has_data(STAGE3_STEP_COLUMNS["parcel_shape"]),
        "ces_residual": _has_data(STAGE3_STEP_COLUMNS["ces_residual"]),
        "zoning_context": _has_data(["zone_plu", "block_id"]),
    }


def _finalize_pipeline_run(
    df: pd.DataFrame,
    parcelles: gpd.GeoDataFrame,
    stats_report: dict,
    checkpoint: PipelineCheckpointManager,
    run_started_perf: float,
) -> pd.DataFrame:
    stats_report["feature_availability"] = _feature_availability_status(df)
    stats_report["checkpoint_status"] = checkpoint.checkpoint_status()
    stats_report.setdefault("runtime", {})
    stats_report["runtime"]["finished_at"] = pd.Timestamp.utcnow().isoformat()
    stats_report["runtime"]["total_seconds"] = round(time.perf_counter() - run_started_perf, 3)

    print("\n[ÉTAPE 11] Consensus score 4 approches")
    consensus_df, consensus_meta = compute_consensus_score(df)
    for col in ["consensus_score", "consensus_confidence", "n_approaches_used"]:
        if col not in df.columns:
            df[col] = np.nan
    if isinstance(consensus_df, pd.DataFrame) and not consensus_df.empty:
        df["consensus_score"] = consensus_df["consensus_score"]
        df["consensus_confidence"] = consensus_df["consensus_confidence"]
        df["n_approaches_used"] = consensus_df["n_approaches_used"]
    stats_report["consensus_meta"] = consensus_meta

    checkpoint.update_report_cache(stats_report)
    export_v3(df, parcelles, stats_report)
    checkpoint.mark_run_completed()

    print("\n" + "=" * 70)
    print("  ✓ PIPELINE V6 TERMINÉ")
    print(f"  → Dataset complet   : {OUTPUT_CSV_V6}")
    print(f"  → Dataset ML        : {OUTPUT_CSV_ML_V6}")
    print(f"  → SHAP par parcelle : {OUTPUT_SHAP_PARCELLE}")
    print(f"  → Intervalles de confiance : cpi_ml_ci_low / cpi_ml_ci_high")
    print(f"  → Consensus 4 approches   : consensus_score / consensus_confidence")
    print(f"  → Rapport JSON      : {OUTPUT_REPORT}")
    agg = stats_report.get("cluster_label_agreement", {}) if isinstance(stats_report, dict) else {}
    if isinstance(agg, dict) and agg.get("cluster_label_agreement") is not None:
        print(f"  → Accord labels/clusters : {agg.get('cluster_label_agreement'):.1f}%")
    cons = stats_report.get("consensus_meta", {}) if isinstance(stats_report, dict) else {}
    if cons:
        print("  ── Consensus 4 approches ──")
        print(f"  Approches disponibles : {cons.get('available_count', 0)}/4")
        if cons.get("median_confidence") is not None:
            print(f"  Confiance médiane     : {cons.get('median_confidence'):.2f}")
        if cons.get("high_conf") is not None and cons.get("n_total"):
            pct = cons["high_conf"] / max(cons["n_total"], 1) * 100
            print(f"  Haute confiance (>0.80) : {cons['high_conf']} parcelles ({pct:.1f}%)")
    print("=" * 70 + "\n")

    return df


# ==============================================================================
# PIPELINE PRINCIPAL V3
# ==============================================================================

def run_pipeline_v3():
    global TAU_SOFTMIN
    run_started_perf = time.perf_counter()
    run_started_at = pd.Timestamp.utcnow().isoformat()

    print("\n" + "=" * 70)
    print("  TERRA-IA — Pipeline complet v6.0 — Chambéry (73065)")
    print("  Couverture : 24 dalles IGN (4×6 km) | DVF Savoie | Blocs 300m")
    print("=" * 70)

    stats_report = {
        "runtime": {"started_at": run_started_at},
        "external_data_status": {},
    }
    SKIP_ZONAL = os.environ.get("SKIP_ZONAL", "False").lower() == "true"

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 0 — TÉLÉCHARGEMENT
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[ÉTAPE 0-A] Téléchargement {len(URLS_MNT)} dalles MNT + "
          f"{len(URLS_MNH)} dalles MNH (IGN LiDAR HD)")
    if SKIP_DOWNLOAD:
        print("  SKIP_DOWNLOAD=True")
        mnt_paths = sorted(DATA_DIR.glob("*MNT*.tif"))
        mnh_paths = sorted(DATA_DIR.glob("*MNH*.tif"))
        print(f"  MNT trouvés : {len(mnt_paths)}  |  MNH trouvés : {len(mnh_paths)}")
    else:
        mnt_paths = download_all_dalles(URLS_MNT, "MNT")
        mnh_paths = download_all_dalles(URLS_MNH, "MNH")

    print("\n[ÉTAPE 0-B] Parcelles cadastrales (Etalab 2.0)")
    if not download_parcelles(COMMUNE_CODE, PARCELLES_PATH):
        print("  ✗ Erreur"); sys.exit(1)

    print("\n[ÉTAPE 0-C] DVF Savoie (data.gouv.fr)")
    dvf_ok = download_dvf_v3(DVF_PATH)
    stats_report["dvf_available"] = dvf_ok

    print("\n[ÉTAPE 0-E] BRGM Géorisques (argiles, mouvements terrain)")
    brgm_hazard_files = load_brgm_local(BRGM_DIR)
    stats_report["brgm_available"] = list(brgm_hazard_files.keys())

    print("\n[ÉTAPE 0-F] PLU Chambéry (Géoportail Urbanisme)")
    if SKIP_PLU:
        print("  SKIP_PLU=True — jointure PLU sautée")
        plu_ok = PLU_PATH.exists()
    else:
        plu_ok = download_plu(PLU_PATH)
    stats_report["plu_available"] = bool(plu_ok)

    print("\n[ÉTAPE 0-G] Bâtiments existants (BD TOPO IGN)")
    if SKIP_BATI:
        print("  SKIP_BATI=True — BD TOPO sautée")
        bati_ok = BD_TOPO_PATH.exists()
    else:
        bati_ok = download_bd_topo_batiments(BD_TOPO_PATH)
    stats_report["bd_topo_available"] = bool(bati_ok)

    print("\n[ÉTAPE 0-D] Fusion des dalles")
    merge_dalles(mnt_paths, MNT_PATH, "MNT")
    merge_dalles(mnh_paths, MNH_PATH, "MNH")

    if not MNT_PATH.exists() or not MNH_PATH.exists():
        print("  ✗ Rasters manquants"); sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 — VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n[ÉTAPE 1] Validation des données")
    if not (validate_raster(MNT_PATH, "MNT") and validate_raster(MNH_PATH, "MNH")):
        sys.exit(1)
    parcelles = load_parcelles(PARCELLES_PATH)
    checkpoint = PipelineCheckpointManager(
        state_path=PIPELINE_STATE_PATH,
        stage3_features_path=STAGE3_FEATURES_CHECKPOINT,
        stage6_labels_path=STAGE6_LABELS_CHECKPOINT,
        stage9_scores_path=STAGE9_SCORES_CHECKPOINT,
        resume_requested=RESUME,
        options={
            "skip_download": SKIP_DOWNLOAD,
            "skip_features": SKIP_FEATURES,
            "skip_plu": SKIP_PLU,
            "skip_bati": SKIP_BATI,
            "skip_bootstrap": SKIP_BOOTSTRAP,
            "skip_zonal": SKIP_ZONAL,
            "refresh_osm": REFRESH_OSM,
        },
        input_paths={
            "mnt_path": MNT_PATH,
            "mnh_path": MNH_PATH,
            "parcelles_path": PARCELLES_PATH,
            "dvf_path": DVF_PATH,
            "plu_path": PLU_PATH,
            "bd_topo_path": BD_TOPO_PATH,
            "brgm_dir": BRGM_DIR,
            "osm_cache_path": OSM_ROADS_CACHE_PATH,
        },
    )
    stats_report.update(checkpoint.restore_report_cache())
    stats_report.setdefault("runtime", {})["started_at"] = run_started_at
    stats_report["external_data_status"] = {
        "dvf": {"available": bool(dvf_ok), "path": str(DVF_PATH)},
        "plu": {"available": bool(plu_ok), "path": str(PLU_PATH), "skip_requested": bool(SKIP_PLU)},
        "bd_topo": {"available": bool(bati_ok), "path": str(BD_TOPO_PATH), "skip_requested": bool(SKIP_BATI)},
        "brgm": {
            "available_layers": sorted(brgm_hazard_files.keys()),
            "path": str(BRGM_DIR),
        },
        "optional_modules": {
            "richdem": module_available("richdem"),
            "rvt": module_available("rvt"),
            "osmnx": module_available("osmnx"),
        },
    }
    checkpoint.update_report_cache(stats_report)

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 — CALCUL DES FEATURES RASTER
    # ══════════════════════════════════════════════════════════════════════
    print("\n[ÉTAPE 2] Calcul des features raster (9 features)")

    with rasterio.open(MNT_PATH) as src:
        mnt_data = src.read(1).astype(float)
        cellsize = src.res[0]
        nd       = src.nodata or -9999.0
    mnt_data[mnt_data == nd] = np.nan

    # Slope Horn (1981)
    slope_path = RASTER_DIR / "slope_v3.tif"
    if not slope_path.exists() or not SKIP_FEATURES:
        print("  ⚙  Slope Horn (1981) ...", end=" ", flush=True)
        slope = compute_slope_raster(mnt_data, cellsize)
        save_tif(slope, MNT_PATH, slope_path)
        print(f"OK  moy={np.nanmean(slope):.2f}°")
    else:
        print(f"  ✓ Slope existant : {slope_path.name}")
        slope = None  # déjà sur disque — sera relu par zonal_stats

    # Surface constructible (zones plates continues, θ=7°)
    flat_mask_path  = RASTER_DIR / "flat_mask_v4.tif"
    flat_label_path = RASTER_DIR / "flat_labeled_v4.tif"
    if not flat_mask_path.exists() or not SKIP_FEATURES:
        with rasterio.open(slope_path) as _s:
            _slope_data = _s.read(1)
        print("  ⚙  Surface constructible (θ=7°) ...", end=" ", flush=True)
        flat_mask_r, flat_label_r = compute_flat_platform(
            _slope_data, cellsize, THETA_CONSTRUCTIBLE)
        save_tif(flat_mask_r, MNT_PATH, flat_mask_path)
        save_tif(flat_label_r, MNT_PATH, flat_label_path)
        print(f"OK  ratio_plat={np.nanmean(flat_mask_r):.3f}")

    # TRI Riley (1999)
    tri_path = RASTER_DIR / "tri_v3.tif"
    if not tri_path.exists() or not SKIP_FEATURES:
        print("  ⚙  TRI Riley (1999) ...", end=" ", flush=True)
        tri = compute_tri_riley(mnt_data)
        save_tif(tri, MNT_PATH, tri_path)
        print(f"OK  moy={np.nanmean(tri):.4f}m")
    else:
        print(f"  ✓ TRI existant")

    # TWI + Thalweg (Beven & Kirkby 1979 / Barnes 2014) — un seul passage D8
    twi_path     = RASTER_DIR / "twi_v3.tif"
    thalweg_path = RASTER_DIR / "thalweg_v3.tif"
    if not twi_path.exists() or not SKIP_FEATURES:
        print("  ⚙  TWI_urban + Thalweg (Beven & Kirkby 1979) ...", end=" ", flush=True)
        twi_r, thalweg_r = compute_twi_and_thalweg(MNT_PATH, BETA_MIN_DEG)
        if twi_r is not None:
            save_tif(twi_r, MNT_PATH, twi_path)
            save_tif(thalweg_r, MNT_PATH, thalweg_path)
        else:
            twi_path = thalweg_path = None
    else:
        print(f"  ✓ TWI/Thalweg existants")
        if not twi_path.exists():     twi_path     = None
        if not thalweg_path.exists(): thalweg_path = None

    # SVF (Zakšek 2011)
    svf_path = RASTER_DIR / "svf_v3.tif"
    if not svf_path.exists() or not SKIP_FEATURES:
        print("  ⚙  Sky View Factor (Zakšek 2011) ...", end=" ", flush=True)
        svf_r = compute_svf(MNH_PATH)
        if svf_r is not None:
            save_tif(svf_r, MNH_PATH, svf_path)
        else:
            svf_path = None
    else:
        print(f"  ✓ SVF existant")

    # Hillshade hiver
    hs_path = RASTER_DIR / "hillshade_hiver_v3.tif"
    if not hs_path.exists() or not SKIP_FEATURES:
        print("  ⚙  Hillshade solstice hiver (21 déc. 12h) ...", end=" ", flush=True)
        hs_r = compute_hillshade_winter(mnt_data, cellsize, LAT_DEG)
        save_tif(hs_r, MNT_PATH, hs_path)
        print(f"OK  moy={np.nanmean(hs_r):.3f}")
    else:
        print(f"  ✓ Hillshade existant")

    # Aspect orientation sud
    asp_path = RASTER_DIR / "aspect_south_v3.tif"
    if not asp_path.exists() or not SKIP_FEATURES:
        print("  ⚙  Aspect orientation sud ...", end=" ", flush=True)
        asp_r = compute_aspect_south(mnt_data, cellsize)
        save_tif(asp_r, MNT_PATH, asp_path)
        print(f"OK  ratio_sud={np.nanmean(asp_r):.3f}")
    else:
        print(f"  ✓ Aspect existant")

    # Profile curvature (Zevenbergen & Thorne 1987)
    curv_path = RASTER_DIR / "curvature_v3.tif"
    if not curv_path.exists() or not SKIP_FEATURES:
        print("  ⚙  Profile curvature (Zevenbergen & Thorne 1987) ...", end=" ", flush=True)
        curv_r = compute_profile_curvature(mnt_data, cellsize)
        save_tif(curv_r, MNT_PATH, curv_path)
        print(f"OK  moy={np.nanmean(curv_r):.6f}")
    else:
        print(f"  ✓ Curvature existante")

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 — ZONAL STATS PAR PARCELLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n[ÉTAPE 3] Extraction features par parcelle (zonal stats)")
    resume_stage = 4
    stage3_resume_df = pd.DataFrame(index=parcelles.index)
    stage9_df = checkpoint.restore_stage_frame(
        "stage9",
        index=parcelles.index,
        dependencies=STAGE6_RESUME_DEPENDENCIES + STAGE9_RESUME_DEPENDENCIES,
    )
    if stage9_df is not None:
        df = stage9_df.copy()
        resume_stage = 10
        print("  Resume checkpoint stage9 -> export et consensus")
    else:
        stage6_df = checkpoint.restore_stage_frame(
            "stage6",
            index=parcelles.index,
            dependencies=STAGE6_RESUME_DEPENDENCIES,
        )
        if stage6_df is not None:
            df = stage6_df.copy()
            resume_stage = 7
            print("  Resume checkpoint stage6 -> analyse blocs et ML")
        else:
            stage3_resume_df = checkpoint.restore_stage3_frame(
                index=parcelles.index,
                step_columns=STAGE3_STEP_COLUMNS,
                step_dependencies=STAGE3_STEP_DEPENDENCIES,
            )
            if not stage3_resume_df.empty:
                print(f"  Resume checkpoint stage3 -> {len(stage3_resume_df.columns)} colonnes restaurees")
    features = {}

    df = stage3_resume_df.copy() if resume_stage == 4 else df
    if SKIP_ZONAL and resume_stage == 4 and df.empty:
        print("  SKIP_ZONAL=True — chargement features depuis features_parcelles_v6.csv")
        try:
            df_features_existing = pd.read_csv(OUTPUT_CSV_V6, index_col=0, low_memory=False)
            feat_cols_to_load = [c for c in df_features_existing.columns if c in ALL_FEATURES + FEATURES_CPI_ONLY + ["block_id","surface_m2","compactness_ratio","elongation_ratio","ces_existant","ces_residuel","emprise_residuelle_m2","cluster_score","zone_plu","brgm_argiles_flag","brgm_mvt_terrain_flag"]]
            for col in feat_cols_to_load:
                df[col] = df_features_existing[col].reindex(df.index)
            checkpoint.save_stage_frame("stage3", df)
            print(f"  {len(feat_cols_to_load)} colonnes chargées depuis CSV existant")
        except Exception as e:
            print(f"  ⚠ SKIP_ZONAL : échec chargement CSV ({e}) — retour au calcul complet")
            SKIP_ZONAL = False

    def _stage3_step_done(step_name: str) -> bool:
        return all(col in df.columns for col in STAGE3_STEP_COLUMNS[step_name])

    def _stage3_store(step_name: str, values: dict | None) -> None:
        if values:
            for col, vals in values.items():
                df[col] = vals
        checkpoint.save_stage_frame("stage3", df)
        checkpoint.complete_step(
            f"stage3.{step_name}",
            extra={"columns": STAGE3_STEP_COLUMNS[step_name]},
        )

    if not SKIP_ZONAL and resume_stage == 4:
        if _stage3_step_done("slope_percentiles"):
            print("  ✓ Slope P50/P90/std checkpoint")
        else:
            checkpoint.start_step("stage3.slope_percentiles")
            try:
                print("  ⚙  Slope P50/P90/std ...", end=" ", flush=True)
                slope_features = zonal_percentile(slope_path, parcelles, "slope", [50, 90])
                features.update(slope_features)
                _stage3_store("slope_percentiles", slope_features)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.slope_percentiles", exc)
                raise
    
        if _stage3_step_done("flat_area_ratio"):
            print("  ✓ Max plateforme plate + ratio checkpoint")
        else:
            checkpoint.start_step("stage3.flat_area_ratio")
            try:
                print("  ⚙  Max plateforme plate + ratio ...", end=" ", flush=True)
                if flat_mask_path and Path(flat_mask_path).exists():
                    flat_feats = compute_max_flat_area_per_parcel(
                        flat_mask_path, flat_label_path, parcelles, cellsize)
                else:
                    flat_feats = {
                        "max_flat_area_m2": [np.nan] * len(parcelles),
                        "flat_area_ratio": [np.nan] * len(parcelles),
                    }
                features.update(flat_feats)
                _stage3_store("flat_area_ratio", flat_feats)
            except Exception as exc:
                checkpoint.fail_step("stage3.flat_area_ratio", exc)
                raise
    
        if _stage3_step_done("tri"):
            print("  ✓ TRI checkpoint")
        else:
            checkpoint.start_step("stage3.tri")
            try:
                print("  ⚙  TRI ...", end=" ", flush=True)
                tri_features = zonal(tri_path, parcelles, "tri", ["mean", "max"])
                features.update(tri_features)
                _stage3_store("tri", tri_features)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.tri", exc)
                raise
    
        if _stage3_step_done("twi"):
            print("  ✓ TWI checkpoint")
        else:
            checkpoint.start_step("stage3.twi")
            try:
                print("  ⚙  TWI ...", end=" ", flush=True)
                twi_features = (
                    zonal(twi_path, parcelles, "twi", ["mean", "max"])
                    if twi_path and Path(twi_path).exists()
                    else {"twi_mean": [np.nan] * len(parcelles), "twi_max": [np.nan] * len(parcelles)}
                )
                features.update(twi_features)
                _stage3_store("twi", twi_features)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.twi", exc)
                raise
    
        if _stage3_step_done("thalweg"):
            print("  ✓ Thalweg ratio checkpoint")
        else:
            checkpoint.start_step("stage3.thalweg")
            try:
                print("  ⚙  Thalweg ratio ...", end=" ", flush=True)
                thalweg_features = (
                    zonal(thalweg_path, parcelles, "has_thalweg", ["mean"])
                    if thalweg_path and Path(thalweg_path).exists()
                    else {"has_thalweg_mean": [np.nan] * len(parcelles)}
                )
                features.update(thalweg_features)
                _stage3_store("thalweg", thalweg_features)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.thalweg", exc)
                raise
    
        if _stage3_step_done("svf"):
            print("  ✓ SVF checkpoint")
        else:
            checkpoint.start_step("stage3.svf")
            try:
                print("  ⚙  SVF ...", end=" ", flush=True)
                svf_features = (
                    zonal(svf_path, parcelles, "svf", ["mean", "min"])
                    if svf_path and Path(svf_path).exists()
                    else {"svf_mean": [np.nan] * len(parcelles), "svf_min": [np.nan] * len(parcelles)}
                )
                features.update(svf_features)
                _stage3_store("svf", svf_features)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.svf", exc)
                raise
    
        if _stage3_step_done("hillshade"):
            print("  ✓ Hillshade hiver checkpoint")
        else:
            checkpoint.start_step("stage3.hillshade")
            try:
                print("  ⚙  Hillshade hiver ...", end=" ", flush=True)
                hillshade_features = zonal(hs_path, parcelles, "hillshade_winter", ["mean"])
                features.update(hillshade_features)
                _stage3_store("hillshade", hillshade_features)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.hillshade", exc)
                raise
    
        if _stage3_step_done("aspect_south"):
            print("  ✓ Aspect sud checkpoint")
        else:
            checkpoint.start_step("stage3.aspect_south")
            try:
                print("  ⚙  Aspect sud ...", end=" ", flush=True)
                aspect_features = zonal(asp_path, parcelles, "aspect_south_ratio", ["mean"])
                features.update(aspect_features)
                _stage3_store("aspect_south", aspect_features)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.aspect_south", exc)
                raise
    
        if _stage3_step_done("curvature"):
            print("  ✓ Profile curvature checkpoint")
        else:
            checkpoint.start_step("stage3.curvature")
            try:
                print("  ⚙  Profile curvature ...", end=" ", flush=True)
                curvature_features = zonal(curv_path, parcelles, "profile_curvature", ["mean"])
                features.update(curvature_features)
                _stage3_store("curvature", curvature_features)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.curvature", exc)
                raise
    
        if _stage3_step_done("height_obj"):
            print("  ✓ Hauteur objets voisins checkpoint")
        else:
            checkpoint.start_step("stage3.height_obj")
            try:
                print("  ⚙  Hauteur objets voisins (MNH buffer 20m) ...", end=" ", flush=True)
                buf = parcelles.copy()
                buf["geometry"] = parcelles.geometry.buffer(20)
                buf_path = DATA_DIR / "buf20_v3.geojson"
                buf.to_file(buf_path, driver="GeoJSON")
                height_features = zonal(MNH_PATH, buf, "height_obj", ["max", "mean"])
                features.update(height_features)
                _stage3_store("height_obj", height_features)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.height_obj", exc)
                raise
    
        if _stage3_step_done("road_distance"):
            print("  ✓ Distance route checkpoint")
        else:
            checkpoint.start_step("stage3.road_distance")
            try:
                print("  ⚙  Distance route (OSM) ...", end=" ", flush=True)
                roads, osm_status = load_osm_roads_union(
                    cache_path=OSM_ROADS_CACHE_PATH,
                    refresh=REFRESH_OSM,
                    place_name="Chambéry, France",
                    target_crs="EPSG:2154",
                )
                stats_report["external_data_status"]["osm"] = osm_status
                if roads is None:
                    dist_features = {"dist_road_m": [np.nan] * len(parcelles)}
                    print("indisponible -> NaN")
                else:
                    dist_values = parcelles.geometry.centroid.apply(
                        lambda p: p.distance(roads)
                    ).values
                    dist_features = {"dist_road_m": dist_values}
                    print(f"OK  médiane={np.nanmedian(dist_values):.1f}m")
                features.update(dist_features)
                _stage3_store("road_distance", dist_features)
            except Exception as exc:
                checkpoint.fail_step("stage3.road_distance", exc)
                raise
    
        if _stage3_step_done("parcel_shape"):
            print("  ✓ Compacité et forme checkpoint")
        else:
            checkpoint.start_step("stage3.parcel_shape")
            try:
                print("  ⚙  Compacité et forme des parcelles ...", end=" ", flush=True)
                shape_metrics = compute_parcel_compactness(parcelles)
                features.update(shape_metrics)
                _stage3_store("parcel_shape", shape_metrics)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.parcel_shape", exc)
                raise
    
        if _stage3_step_done("ces_residual"):
            print("  ✓ CES résiduel checkpoint")
        else:
            checkpoint.start_step("stage3.ces_residual")
            try:
                print("  ⚙  CES résiduel (emprise bâtie existante) ...", end=" ", flush=True)
                if BD_TOPO_PATH.exists() and not SKIP_BATI:
                    ces_data = compute_ces_residuel(parcelles, BD_TOPO_PATH)
                else:
                    ces_data = {
                        "ces_existant": [np.nan] * len(parcelles),
                        "ces_residuel": [np.nan] * len(parcelles),
                        "emprise_residuelle_m2": [np.nan] * len(parcelles),
                    }
                features.update(ces_data)
                _stage3_store("ces_residual", ces_data)
                print("OK")
            except Exception as exc:
                checkpoint.fail_step("stage3.ces_residual", exc)
                raise
    
    
        # ── DataFrame principal
        if "commune" not in df.columns:
            df["commune"] = COMMUNE_CODE
        if "id" in parcelles.columns and "id" not in df.columns:
            df["id"] = parcelles["id"].values
        for col, vals in features.items():
            if vals is not None and len(vals) == len(df):
                df[col] = vals
    
        print("  âš™  Zone PLU ...", end=" ", flush=True)
        if SKIP_PLU:
            df["zone_plu"] = "inconnu"
            print("SKIP_PLU=True")
        else:
            df["zone_plu"] = join_plu_to_parcelles(parcelles, PLU_PATH).values
            print("OK")
    
        # Blocs spatiaux V3 (300m)
        df["block_id"] = create_spatial_blocks(parcelles, GRID_SIZE_M).values
        print(f"\n  Blocs spatiaux {GRID_SIZE_M}m : {df['block_id'].nunique()} blocs")
    
        # BRGM join (flags — hors features ML)
        if brgm_hazard_files:
            brgm_flags = join_brgm_to_parcelles(parcelles, brgm_hazard_files)
            for col, vals in brgm_flags.items():
                if vals is not None and len(vals) == len(df):
                    df[col] = vals
        else:
            df["brgm_argiles_flag"] = np.nan
            df["brgm_mvt_terrain_flag"] = np.nan

        checkpoint.save_stage_frame("stage3", df)
        stats_report["feature_availability"] = _feature_availability_status(df)
        checkpoint.update_report_cache(stats_report)

    # Compléments si SKIP_ZONAL (ou si certaines colonnes manquent)
    if "commune" not in df.columns:
        df.insert(0, "commune", COMMUNE_CODE)
    if "id" not in df.columns and "id" in parcelles.columns:
        df["id"] = parcelles["id"].values

    stats_report["feature_availability"] = _feature_availability_status(df)
    checkpoint.update_report_cache(stats_report)

    if resume_stage == 10:
        print("  Resume checkpoint stage9 confirme -> export final uniquement")
        print("\n[ETAPE 10] Export des resultats")
        return _finalize_pipeline_run(df, parcelles, stats_report, checkpoint, run_started_perf)
    
        # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 4 — FILTRAGE
    # ══════════════════════════════════════════════════════════════════════
    if resume_stage <= 6:
        print("\n[ÉTAPE 4] Filtrage et nettoyage")
        df = filter_parcelles(df, parcelles)
        stats_report["filtrage"] = {
            "n_total": int(len(df)),
            "n_valid": int(df["is_valid"].sum()),
            "n_nan":   int((df["nan_ratio"] > SEUIL_NAN).sum()),
        }

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 5 — SCORE CPI V3
    # ══════════════════════════════════════════════════════════════════════
    if resume_stage <= 6:
        print("\n[ÉTAPE 5] Score CPI V3 — calibration τ + score déterministe")

        valid_df = df[df["is_valid"]]

        if len(valid_df) > 50:
            gs = shared_precompute_group_scores_for_tau(valid_df)
            TAU_SOFTMIN = calibrate_tau(gs, target_std=18.0)
        else:
            TAU_SOFTMIN = 10.0
            print(f"    τ par défaut : {TAU_SOFTMIN}")

        df = compute_cpi_v3(df, TAU_SOFTMIN)

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 6 — LABELS SNORKEL V6
    # ══════════════════════════════════════════════════════════════════════
    if resume_stage <= 6:
        checkpoint.start_step("stage6.labels")
        try:
            print("\n[ÉTAPE 6] Labels Snorkel V6 (3 features stables + DVF + clustering)")
            dvf_data = prepare_dvf(DVF_PATH) if dvf_ok else None
            df["proxy_label"] = create_snorkel_labels_v6(df, dvf_data)

            print("\n[ÉTAPE 6B] Validation clustering (anti-circularity check)")
            feat_cols = [c for c in ALL_FEATURES if c in df.columns]
            cluster_results = validate_labels_with_clustering(
                df[df["is_valid"]],
                df.loc[df["is_valid"], "proxy_label"],
                feat_cols
            )
            df["cluster_score"] = np.nan
            cs = cluster_results.get("cluster_score")
            if isinstance(cs, pd.Series):
                df.loc[cs.index, "cluster_score"] = cs
            cluster_out = df[["cluster_score", "CPI_v3", "CPI_technique", "proxy_label"]].dropna(how="all")
            cluster_out.to_csv(OUTPUT_CLUSTER_SCORES, index=True)
            print(f"  Cluster scores: {OUTPUT_CLUSTER_SCORES} ({len(cluster_out)} lignes)")
            stats_report["cluster_label_agreement"] = {
                "cluster_label_agreement": cluster_results.get("cluster_label_agreement"),
                "pure_pos_in_best": cluster_results.get("pure_pos_in_best"),
                "pure_neg_in_worst": cluster_results.get("pure_neg_in_worst"),
            }
            checkpoint.save_stage_frame("stage6", df)
            checkpoint.complete_step(
                "stage6.labels",
                extra={"columns": ["proxy_label", "cluster_score", "CPI_v3", "CPI_technique"]},
            )
            checkpoint.update_report_cache(stats_report)
        except Exception as exc:
            checkpoint.fail_step("stage6.labels", exc)
            raise
    else:
        print("  Resume checkpoint stage6 -> saut des étapes 4 à 6")

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 7 — ANALYSE DES BLOCS
    # ══════════════════════════════════════════════════════════════════════
    checkpoint.start_step("stage7.block_analysis")
    try:
        print("\n[ÉTAPE 7] Analyse équilibre classes par bloc spatial")
        bloc_stats = analyze_blocks(df, df["proxy_label"])
        stats_report["bloc_analysis"] = bloc_stats
        checkpoint.complete_step("stage7.block_analysis")
        checkpoint.update_report_cache(stats_report)
    except Exception as exc:
        checkpoint.fail_step("stage7.block_analysis", exc)
        raise

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 8 — COMPARAISON MODÈLES
    # ══════════════════════════════════════════════════════════════════════
    checkpoint.start_step("stage8.model_compare")
    try:
        print("\n[ÉTAPE 8] Comparaison modèles ML (GroupKFold + GridSearch XGBoost)")
        ml_results = compare_models_v3(df, df["proxy_label"], parcelles)
        stats_report["ml_results"] = {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in ml_results.items() if isinstance(v, dict)
        }
        checkpoint.complete_step("stage8.model_compare")
        checkpoint.update_report_cache(stats_report)
    except Exception as exc:
        checkpoint.fail_step("stage8.model_compare", exc)
        raise

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 9 — LAMBDAMART FINAL + SHAP
    # ══════════════════════════════════════════════════════════════════════
    checkpoint.start_step("stage9.train_explain")
    try:
        print("\n[ÉTAPE 9] XGBoost LambdaMART final + SHAP V3")
        best_params = None
        for name, r in ml_results.items():
            if "rank:ndcg" in name or "XGBoost" in name:
                if isinstance(r, dict) and "_best_params" in r:
                    best_params = r["_best_params"]
        train_and_explain_v3(df, df["proxy_label"], best_params)
        checkpoint.complete_step("stage9.train_explain")
    except Exception as exc:
        checkpoint.fail_step("stage9.train_explain", exc)
        raise

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 10 — EXPORT
    # ══════════════════════════════════════════════════════════════════════
    # ──────────────────────────────────────────────────────────────────────────────
    # ÉTAPE 9B — BOOTSTRAP INCERTITUDE CPI_ML
    # ──────────────────────────────────────────────────────────────────────────────
    checkpoint.start_step("stage9.bootstrap")
    try:
        feat_cols = [c for c in ALL_FEATURES if c in df.columns]
        for c in ["cpi_ml_ci_low", "cpi_ml_ci_high", "cpi_ml_std"]:
            if c not in df.columns:
                df[c] = np.nan
        print("\n[ÉTAPE 9B] Bootstrap incertitude CPI_ML (100 runs)")
        if SKIP_BOOTSTRAP:
            print("  SKIP_BOOTSTRAP=True — bootstrap ignoré")
        else:
            bootstrap_results = compute_cpi_bootstrap(
                df[df["is_valid"]],
                feat_cols,
                n_bootstrap=100
            )
            if not bootstrap_results.empty:
                df.loc[df["is_valid"], "cpi_ml_ci_low"] = bootstrap_results["cpi_ml_ci_low"].values
                df.loc[df["is_valid"], "cpi_ml_ci_high"] = bootstrap_results["cpi_ml_ci_high"].values
                df.loc[df["is_valid"], "cpi_ml_std"] = bootstrap_results["cpi_ml_std"].values
            else:
                print("  Résultats bootstrap indisponibles")

        try:
            if Path(OUTPUT_SHAP_PARCELLE).exists():
                shap_df = pd.read_csv(OUTPUT_SHAP_PARCELLE)
                id_col_df = "id_parcelle" if "id_parcelle" in df.columns else "id"
                if "id_parcelle" in shap_df.columns and id_col_df in df.columns:
                    mapping = df.set_index(id_col_df)[["cpi_ml_ci_low", "cpi_ml_ci_high", "cpi_ml_std"]]
                    for col in ["cpi_ml_ci_low", "cpi_ml_ci_high", "cpi_ml_std"]:
                        if col in mapping.columns:
                            shap_df[col] = shap_df["id_parcelle"].map(mapping[col])
                    shap_df.to_csv(OUTPUT_SHAP_PARCELLE, index=False)
        except Exception as e:
            print(f"  ⚠  Mise à jour SHAP IC échouée : {e}")

        checkpoint.save_stage_frame("stage9", df)
        checkpoint.complete_step(
            "stage9.bootstrap",
            extra={
                "columns": ["cpi_ml_ci_low", "cpi_ml_ci_high", "cpi_ml_std"],
                "bootstrap_skipped": bool(SKIP_BOOTSTRAP),
            },
        )
        checkpoint.update_report_cache(stats_report)
    except Exception as exc:
        checkpoint.fail_step("stage9.bootstrap", exc)
        raise

    # ──────────────────────────────────────────────────────────────────────────────
    # ÉTAPE 10 — EXPORT
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n[ETAPE 10] Export des resultats")
    return _finalize_pipeline_run(df, parcelles, stats_report, checkpoint, run_started_perf)

    # ──────────────────────────────────────────────────────────────────────────────
    # ÉTAPE 11 — CONSENSUS SCORE 4 APPROCHES
    # ──────────────────────────────────────────────────────────────────────────────
    print("\n[ÉTAPE 11] Consensus score 4 approches")
    consensus_df, consensus_meta = compute_consensus_score(df)
    for col in ["consensus_score", "consensus_confidence", "n_approaches_used"]:
        if col not in df.columns:
            df[col] = np.nan
    if isinstance(consensus_df, pd.DataFrame) and not consensus_df.empty:
        df["consensus_score"] = consensus_df["consensus_score"]
        df["consensus_confidence"] = consensus_df["consensus_confidence"]
        df["n_approaches_used"] = consensus_df["n_approaches_used"]
    stats_report["consensus_meta"] = consensus_meta

    export_v3(df, parcelles, stats_report)

    print("\n" + "=" * 70)
    print("  ✓ PIPELINE V6 TERMINÉ")
    print(f"  → Dataset complet   : {OUTPUT_CSV_V6}")
    print(f"  → Dataset ML        : {OUTPUT_CSV_ML_V6}")
    print(f"  → SHAP par parcelle : {OUTPUT_SHAP_PARCELLE}")
    print(f"  → Intervalles de confiance : cpi_ml_ci_low / cpi_ml_ci_high")
    print(f"  → Consensus 4 approches   : consensus_score / consensus_confidence")
    print(f"  → Rapport JSON      : {OUTPUT_REPORT}")
    agg = stats_report.get("cluster_label_agreement", {}) if isinstance(stats_report, dict) else {}
    if isinstance(agg, dict) and agg.get("cluster_label_agreement") is not None:
        print(f"  → Accord labels/clusters : {agg.get('cluster_label_agreement'):.1f}%")
    cons = stats_report.get("consensus_meta", {}) if isinstance(stats_report, dict) else {}
    if cons:
        print("  ── Consensus 4 approches ──")
        print(f"  Approches disponibles : {cons.get('available_count', 0)}/4")
        if cons.get("median_confidence") is not None:
            print(f"  Confiance médiane     : {cons.get('median_confidence'):.2f}")
        if cons.get("high_conf") is not None and cons.get("n_total"):
            print(f"  Haute confiance (>0.80) : {cons['high_conf']} parcelles ({cons['high_conf']/max(cons['n_total'],1)*100:.1f}%)")
    print("=" * 70 + "\n")

    return df


def run_pipeline_v6():
    return run_pipeline_v3()


def main():
    return run_pipeline_v6()


if __name__ == "__main__":
    main()
