from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def run_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=True,
    )


def test_core_entrypoints_exist() -> None:
    for relative_path in [
        "pipeline.py",
        "demo.py",
        "preflight.py",
        "terra_ia_pipeline_v6.py",
        "terra_ia_demo_v6.py",
        "environment.yml",
        ".streamlit/secrets.toml.example",
    ]:
        assert (PROJECT_ROOT / relative_path).exists(), relative_path


def test_package_imports() -> None:
    package = importlib.import_module("terra_ia")
    assert package.PROJECT_ROOT == PROJECT_ROOT
    for module_name in [
        "terra_ia.catalog",
        "terra_ia.pipeline_runtime",
        "terra_ia.pipeline_resilience",
        "terra_ia.demo_runtime",
        "terra_ia.preflight_checks",
    ]:
        importlib.import_module(module_name)


def test_cli_dry_runs() -> None:
    pipeline = run_command(
        "pipeline.py",
        "--dry-run",
        "--resume",
        "--skip-download",
        "--skip-features",
    )
    assert "Terra-IA pipeline runtime configuration" in pipeline.stdout

    demo = run_command("demo.py", "--dry-run")
    assert "Terra-IA demo runtime configuration" in demo.stdout


def test_final_capstone_outputs_exist() -> None:
    for relative_path in [
        "features_parcelles_v6.csv",
        "ml_dataset_v6.csv",
        "cluster_scores_v6.csv",
        "shap_par_parcelle_v6.csv",
        "rapport_stats_v6.json",
        "README_ML_dataset_v6.md",
    ]:
        path = PROJECT_ROOT / relative_path
        assert path.exists(), relative_path
        assert path.stat().st_size > 0, relative_path
