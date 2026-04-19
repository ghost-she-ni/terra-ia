from __future__ import annotations

import argparse
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from .project import (
    PROJECT_ROOT,
    ensure_directories,
    ensure_parent_directories,
    env_flag,
    env_path,
    normalize_path,
)

PIPELINE_OUTPUT_DEFAULTS = {
    "TERRA_IA_FEATURES_V3_PATH": "features_parcelles_v3.csv",
    "TERRA_IA_ML_V3_PATH": "ml_dataset_v3.csv",
    "TERRA_IA_REPORT_PATH": "rapport_stats_v6.json",
    "TERRA_IA_FEATURES_V4_PATH": "features_parcelles_v4.csv",
    "TERRA_IA_ML_V4_PATH": "ml_dataset_v4.csv",
    "TERRA_IA_FEATURES_PATH": "features_parcelles_v6.csv",
    "TERRA_IA_ML_PATH": "ml_dataset_v6.csv",
    "TERRA_IA_SHAP_PATH": "shap_par_parcelle_v6.csv",
    "TERRA_IA_CLUSTER_SCORES_PATH": "cluster_scores_v6.csv",
    "TERRA_IA_ML_README_PATH": "README_ML_dataset_v6.md",
}


@dataclass(frozen=True)
class PipelineRuntimeConfig:
    project_root: Path
    data_dir: Path
    raster_dir: Path
    output_dir: Path
    mnt_path: Path
    mnh_path: Path
    parcelles_path: Path
    dvf_path: Path
    plu_path: Path
    brgm_dir: Path
    sitadel_path: Path
    bd_topo_path: Path
    output_csv_v3: Path
    output_csv_ml_v3: Path
    output_report: Path
    output_csv_v4: Path
    output_csv_ml_v4: Path
    output_csv_v6: Path
    output_csv_ml_v6: Path
    output_shap_parcelle: Path
    output_cluster_scores: Path
    cluster_scores_path: Path
    approach3_path: Path
    approach4_path: Path
    output_readme_ml_v6: Path
    checkpoint_dir: Path
    pipeline_state_path: Path
    stage3_features_checkpoint: Path
    stage6_labels_checkpoint: Path
    stage9_scores_checkpoint: Path
    logs_dir: Path
    log_file: Path
    osm_roads_cache_path: Path
    skip_download: bool
    skip_features: bool
    skip_plu: bool
    skip_bati: bool
    skip_bootstrap: bool
    skip_zonal: bool
    resume: bool
    refresh_osm: bool


def output_env_defaults(output_dir: Path) -> dict[str, str]:
    return {name: str(output_dir / filename) for name, filename in PIPELINE_OUTPUT_DEFAULTS.items()}


def pipeline_env_updates_from_args(args: argparse.Namespace) -> dict[str, str]:
    env_updates: dict[str, str] = {}

    flag_names = {
        "SKIP_DOWNLOAD": args.skip_download,
        "SKIP_FEATURES": args.skip_features,
        "SKIP_PLU": args.skip_plu,
        "SKIP_BATI": args.skip_bati,
        "SKIP_BOOTSTRAP": args.skip_bootstrap,
        "SKIP_ZONAL": args.skip_zonal,
        "TERRA_IA_RESUME": args.resume,
        "TERRA_IA_REFRESH_OSM": args.refresh_osm,
    }
    for name, enabled in flag_names.items():
        if enabled:
            env_updates[name] = "True"

    if args.data_dir:
        env_updates["TERRA_IA_DATA_DIR"] = str(normalize_path(args.data_dir))

    if args.output_dir:
        output_dir = normalize_path(args.output_dir)
        env_updates["TERRA_IA_OUTPUT_DIR"] = str(output_dir)
        env_updates.update(output_env_defaults(output_dir))

    if args.cluster_scores_path:
        env_updates["TERRA_IA_CLUSTER_SCORES_PATH"] = str(normalize_path(args.cluster_scores_path))

    if args.log_file:
        env_updates["TERRA_IA_LOG_FILE"] = str(normalize_path(args.log_file))

    return env_updates


def build_pipeline_runtime_config() -> PipelineRuntimeConfig:
    data_dir = env_path("TERRA_IA_DATA_DIR", PROJECT_ROOT / "data" / "lidar_chamberey")
    raster_dir = env_path("TERRA_IA_RASTER_DIR", data_dir / "rasters_v3")
    output_dir = env_path("TERRA_IA_OUTPUT_DIR", PROJECT_ROOT)
    checkpoint_dir = env_path("TERRA_IA_CHECKPOINT_DIR", output_dir / "checkpoints")
    logs_dir = env_path("TERRA_IA_LOG_DIR", output_dir / "logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = env_path("TERRA_IA_LOG_FILE", logs_dir / f"pipeline_{timestamp}.log")

    config = PipelineRuntimeConfig(
        project_root=PROJECT_ROOT,
        data_dir=data_dir,
        raster_dir=raster_dir,
        output_dir=output_dir,
        mnt_path=data_dir / "mnt_chamberey_v3.tif",
        mnh_path=data_dir / "mnh_chamberey_v3.tif",
        parcelles_path=data_dir / "parcelles_73065.geojson",
        dvf_path=data_dir / "dvf_73_2023.csv",
        plu_path=data_dir / "plu_chamberey.geojson",
        brgm_dir=data_dir / "brgm",
        sitadel_path=data_dir / "sitadel_73065.csv",
        bd_topo_path=data_dir / "bd_topo_batiments_73065.geojson",
        output_csv_v3=env_path("TERRA_IA_FEATURES_V3_PATH", output_dir / "features_parcelles_v3.csv"),
        output_csv_ml_v3=env_path("TERRA_IA_ML_V3_PATH", output_dir / "ml_dataset_v3.csv"),
        output_report=env_path("TERRA_IA_REPORT_PATH", output_dir / "rapport_stats_v6.json"),
        output_csv_v4=env_path("TERRA_IA_FEATURES_V4_PATH", output_dir / "features_parcelles_v4.csv"),
        output_csv_ml_v4=env_path("TERRA_IA_ML_V4_PATH", output_dir / "ml_dataset_v4.csv"),
        output_csv_v6=env_path("TERRA_IA_FEATURES_PATH", output_dir / "features_parcelles_v6.csv"),
        output_csv_ml_v6=env_path("TERRA_IA_ML_PATH", output_dir / "ml_dataset_v6.csv"),
        output_shap_parcelle=env_path("TERRA_IA_SHAP_PATH", output_dir / "shap_par_parcelle_v6.csv"),
        output_cluster_scores=env_path("TERRA_IA_CLUSTER_SCORES_PATH", output_dir / "cluster_scores_v6.csv"),
        cluster_scores_path=env_path("TERRA_IA_CLUSTER_SCORES_PATH", output_dir / "cluster_scores_v6.csv"),
        approach3_path=env_path(
            "TERRA_IA_APPROACH3_PATH",
            output_dir / "approach3_outputs" / "approach3_all_parcels.csv",
        ),
        approach4_path=env_path(
            "TERRA_IA_APPROACH4_PATH",
            output_dir / "approach4_outputs" / "approach4_preference_scores.csv",
        ),
        output_readme_ml_v6=env_path("TERRA_IA_ML_README_PATH", output_dir / "README_ML_dataset_v6.md"),
        checkpoint_dir=checkpoint_dir,
        pipeline_state_path=env_path("TERRA_IA_PIPELINE_STATE_PATH", checkpoint_dir / "pipeline_state.json"),
        stage3_features_checkpoint=env_path(
            "TERRA_IA_STAGE3_CHECKPOINT_PATH",
            checkpoint_dir / "stage3_features.parquet",
        ),
        stage6_labels_checkpoint=env_path(
            "TERRA_IA_STAGE6_CHECKPOINT_PATH",
            checkpoint_dir / "stage6_labels.parquet",
        ),
        stage9_scores_checkpoint=env_path(
            "TERRA_IA_STAGE9_CHECKPOINT_PATH",
            checkpoint_dir / "stage9_scores.parquet",
        ),
        logs_dir=logs_dir,
        log_file=log_file,
        osm_roads_cache_path=env_path(
            "TERRA_IA_OSM_CACHE_PATH",
            data_dir / "osm_roads_drive_chambery.geojson",
        ),
        skip_download=env_flag("SKIP_DOWNLOAD"),
        skip_features=env_flag("SKIP_FEATURES"),
        skip_plu=env_flag("SKIP_PLU"),
        skip_bati=env_flag("SKIP_BATI"),
        skip_bootstrap=env_flag("SKIP_BOOTSTRAP"),
        skip_zonal=env_flag("SKIP_ZONAL"),
        resume=env_flag("TERRA_IA_RESUME"),
        refresh_osm=env_flag("TERRA_IA_REFRESH_OSM"),
    )

    ensure_directories(
        [
            config.data_dir,
            config.raster_dir,
            config.output_dir,
            config.checkpoint_dir,
            config.logs_dir,
        ]
    )
    ensure_parent_directories(
        [
            config.output_csv_v3,
            config.output_csv_ml_v3,
            config.output_report,
            config.output_csv_v4,
            config.output_csv_ml_v4,
            config.output_csv_v6,
            config.output_csv_ml_v6,
            config.output_shap_parcelle,
            config.output_cluster_scores,
            config.output_readme_ml_v6,
            config.approach3_path,
            config.approach4_path,
            config.pipeline_state_path,
            config.stage3_features_checkpoint,
            config.stage6_labels_checkpoint,
            config.stage9_scores_checkpoint,
            config.log_file,
            config.osm_roads_cache_path,
        ]
    )
    return config


def expected_generated_outputs() -> list[Path]:
    config = build_pipeline_runtime_config()
    return [
        config.output_csv_v6,
        config.output_csv_ml_v6,
        config.output_shap_parcelle,
        config.output_report,
        config.output_cluster_scores,
    ]
