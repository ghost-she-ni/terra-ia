import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import terra_ia.pipeline as pipeline


class FakeXGBRanker:
    last_init_kwargs = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeXGBRanker.last_init_kwargs = kwargs

    def fit(self, X, y, group=None):
        return self

    def predict(self, X):
        # Score artificiel pour forcer un meilleur choix sur 100 / 3 / 0.03
        score = 1000.0
        score -= abs(self.kwargs.get("n_estimators", 0) - 100) * 1.0
        score -= abs(self.kwargs.get("max_depth", 0) - 3) * 100.0
        score -= abs(self.kwargs.get("learning_rate", 0.0) - 0.03) * 10000.0
        return np.full(len(X), score, dtype=float)


class FakeTreeExplainer:
    def __init__(self, model):
        self.model = model

    def shap_values(self, X):
        return np.zeros((len(X), X.shape[1]), dtype=float)


def test_compare_models_v3_exposes_best_params(monkeypatch):
    fake_xgb_module = types.SimpleNamespace(XGBRanker=FakeXGBRanker)
    monkeypatch.setitem(sys.modules, "xgboost", fake_xgb_module)

    monkeypatch.setattr(pipeline, "ALL_FEATURES", ["f1", "f2"])
    monkeypatch.setattr(pipeline, "GRID_SIZE_M", 300)
    monkeypatch.setattr(pipeline, "N_FOLDS", 3)

    # On simplifie les métriques pour tester seulement la logique best_params
    monkeypatch.setattr(pipeline, "ndcg_at_k", lambda y_true, scores, k: float(np.mean(scores)))
    monkeypatch.setattr(pipeline, "precision_at_k", lambda y_true, scores, k: float(np.mean(scores)))

    n = 120
    df = pd.DataFrame({
        "f1": np.random.rand(n),
        "f2": np.random.rand(n),
        "is_valid": np.ones(n, dtype=bool),
        "block_id": np.repeat([0, 1, 2], repeats=n // 3 + 1)[:n],
        "CPI_v3": np.random.rand(n) * 100,
    })

    labels = pd.Series(([0, 1] * (n // 2))[:n])

    parcelles = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)] * n},
        geometry="geometry",
        crs="EPSG:4326",
    )

    results = pipeline.compare_models_v3(df, labels, parcelles)

    assert "XGBoost rank:ndcg" in results
    assert results["XGBoost rank:ndcg"]["best_params"] == {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.03,
    }


def test_train_and_explain_v3_uses_passed_best_params(tmp_path, monkeypatch):
    fake_xgb_module = types.SimpleNamespace(XGBRanker=FakeXGBRanker)
    fake_shap_module = types.SimpleNamespace(TreeExplainer=FakeTreeExplainer)

    monkeypatch.setitem(sys.modules, "xgboost", fake_xgb_module)
    monkeypatch.setitem(sys.modules, "shap", fake_shap_module)

    monkeypatch.setattr(pipeline, "DATA_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "ALL_FEATURES", ["f1", "f2"])
    monkeypatch.setattr(pipeline, "FEATURE_GROUPS", {
        "PENTE": ["f1"],
        "SOLEIL": ["f2"],
    })
    monkeypatch.setattr(pipeline, "GROUP_WEIGHTS", {
        "PENTE": 0.5,
        "SOLEIL": 0.5,
    })

    n = 60
    df = pd.DataFrame({
        "f1": np.random.rand(n),
        "f2": np.random.rand(n),
        "is_valid": np.ones(n, dtype=bool),
        "block_id": np.repeat([0, 1, 2], repeats=n // 3 + 1)[:n],
        "CPI_v3": np.random.rand(n) * 100,
    })

    labels = pd.Series(([0, 1] * (n // 2))[:n])

    expected_params = {
        "n_estimators": 111,
        "max_depth": 7,
        "learning_rate": 0.07,
    }

    pipeline.train_and_explain_v3(df, labels, expected_params)

    assert FakeXGBRanker.last_init_kwargs["n_estimators"] == 111
    assert FakeXGBRanker.last_init_kwargs["max_depth"] == 7
    assert FakeXGBRanker.last_init_kwargs["learning_rate"] == 0.07
    assert "CPI_ML_v3" in df.columns
    assert (tmp_path / "shap_importance_v3.csv").exists()