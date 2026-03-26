import sys
import json
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import terra_ia.pipeline as pipeline


def test_export_v3_writes_id_parcelle_readme_and_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(pipeline, "OUTPUT_CSV_V3", tmp_path / "features_parcelles_v3.csv")
    monkeypatch.setattr(pipeline, "OUTPUT_CSV_ML", tmp_path / "ml_dataset_v3.csv")
    monkeypatch.setattr(pipeline, "OUTPUT_REPORT", tmp_path / "rapport_stats_v3.json")
    monkeypatch.setattr(pipeline, "COMMUNE_CODE", "73065")
    monkeypatch.setattr(pipeline, "GRID_SIZE_M", 300)
    monkeypatch.setattr(pipeline, "TAU_SOFTMIN", 10.0)
    monkeypatch.setattr(pipeline, "ALL_FEATURES", ["f1", "f2"])
    monkeypatch.setattr(
        pipeline,
        "FEATURE_GROUPS",
        {"PENTE": ["f1"], "SOLEIL": ["f2"]}
    )

    df = pd.DataFrame({
        "commune": ["73065", "73065"],
        "surface_m2": [500, 800],
        "nan_ratio": [0.0, 0.1],
        "is_valid": [True, True],
        "proxy_label": [1, 0],
        "block_id": [1, 1],
        "f1": [0.2, 0.8],
        "f2": [0.6, 0.3],
        "CPI_v3": [75.0, 25.0],
        "CPI_v3_label": ["Bon", "Faible"],
        "CPI_ML_v3": [80.0, 20.0],
        "score_pente": [90, 10],
        "score_hydro": [80, 20],
        "score_morpho": [70, 30],
        "score_soleil": [60, 40],
        "score_continu": [75, 25],
        "score_softmin": [74, 24],
        "gate_factor": [1.0, 1.0],
        "gate_reason": ["", ""],
    })

    parcelles = gpd.GeoDataFrame(
        {
            "id": ["PARC_001", "PARC_002"],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    best_params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.03,
    }

    stats_report = {}

    pipeline.export_v3(df, parcelles, stats_report, best_params)

    csv_full = pd.read_csv(tmp_path / "features_parcelles_v3.csv")
    assert "id_parcelle" in csv_full.columns
    assert not any(col.startswith("Unnamed") for col in csv_full.columns)

    readme_text = (tmp_path / "README_ML_dataset_v3.md").read_text(encoding="utf-8")
    assert "n_estimators=100" in readme_text
    assert "max_depth=3" in readme_text
    assert "learning_rate=0.03" in readme_text

    report = json.loads((tmp_path / "rapport_stats_v3.json").read_text(encoding="utf-8"))
    assert report["xgb_best_params"] == best_params