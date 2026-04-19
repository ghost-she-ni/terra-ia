from __future__ import annotations

from pathlib import Path

from .pipeline_runtime import expected_generated_outputs
from .project import PROJECT_ROOT

CORE_FILES = [
    "pipeline.py",
    "demo.py",
    "preflight.py",
    "terra_ia_pipeline_v6.py",
    "terra_ia_demo_v6.py",
    "README.md",
    "environment.yml",
    ".streamlit/secrets.toml.example",
    "src/terra_ia/__init__.py",
    "src/terra_ia/project.py",
    "src/terra_ia/catalog.py",
    "src/terra_ia/downloads.py",
    "src/terra_ia/consensus.py",
    "src/terra_ia/exports.py",
    "src/terra_ia/hazards.py",
    "src/terra_ia/labeling.py",
    "src/terra_ia/ml.py",
    "src/terra_ia/raster_features.py",
    "src/terra_ia/reporting.py",
    "src/terra_ia/scoring.py",
    "src/terra_ia/urban_data.py",
    "src/terra_ia/spatial_data.py",
    "src/terra_ia/pipeline_runtime.py",
    "src/terra_ia/pipeline_resilience.py",
    "src/terra_ia/demo_runtime.py",
    "src/terra_ia/preflight_checks.py",
    "src/terra_ia/pipeline_cli.py",
    "src/terra_ia/demo_cli.py",
    "src/terra_ia/preflight_cli.py",
]

REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "geopandas",
    "scipy",
    "rasterio",
    "rasterstats",
    "requests",
    "sklearn",
    "streamlit",
    "plotly",
    "folium",
    "streamlit_folium",
]

OPTIONAL_MODULES = [
    "openai",
    "osmnx",
    "xgboost",
    "shap",
    "richdem",
    "rvt",
]


def core_file_paths() -> list[Path]:
    return [PROJECT_ROOT / rel_path for rel_path in CORE_FILES]


def generated_output_paths() -> list[Path]:
    return expected_generated_outputs()
