from __future__ import annotations

import argparse
import importlib
import os
import sys

from .pipeline_resilience import live_log_tee
from .pipeline_runtime import build_pipeline_runtime_config, pipeline_env_updates_from_args
from .project import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stable capstone entrypoint for the Terra-IA training pipeline."
    )
    parser.add_argument("--skip-download", action="store_true", help="Reuse raw data already downloaded locally.")
    parser.add_argument("--skip-features", action="store_true", help="Reuse raster derivatives already computed.")
    parser.add_argument("--skip-plu", action="store_true", help="Skip PLU retrieval and zoning join.")
    parser.add_argument("--skip-bati", action="store_true", help="Skip BD TOPO building footprint overlay.")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap confidence intervals.")
    parser.add_argument("--skip-zonal", action="store_true", help="Reuse an existing parcel feature CSV when available.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest valid internal checkpoints.")
    parser.add_argument("--refresh-osm", action="store_true", help="Refresh the cached OSM roads layer.")
    parser.add_argument("--data-dir", type=str, help="Override the raw and intermediate data directory.")
    parser.add_argument("--output-dir", type=str, help="Override the directory used for CSV, JSON, and SHAP exports.")
    parser.add_argument("--cluster-scores-path", type=str, help="Override the clustering score CSV path.")
    parser.add_argument("--log-file", type=str, help="Write live pipeline logs to a dedicated file.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved runtime configuration and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    env_updates = pipeline_env_updates_from_args(args)

    if args.dry_run:
        print("Terra-IA pipeline runtime configuration")
        print(f"project_root={PROJECT_ROOT}")
        if not env_updates:
            print("env_updates=<none>")
        else:
            for key in sorted(env_updates):
                print(f"{key}={env_updates[key]}")
        return 0

    for key, value in env_updates.items():
        os.environ[key] = value

    runtime = build_pipeline_runtime_config()

    try:
        module = importlib.import_module("terra_ia_pipeline_v6")
    except ModuleNotFoundError as exc:
        print("Unable to import the Terra-IA pipeline.")
        print("Install the recommended environment first: conda env create -f environment.yml")
        print(f"Original error: {exc}")
        return 1

    runner = getattr(module, "run_pipeline_v6", None) or getattr(module, "run_pipeline_v3", None)
    if runner is None:
        print("No pipeline entrypoint found in terra_ia_pipeline_v6.py")
        return 1

    os.environ["PYTHONUNBUFFERED"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True, write_through=True)

    with live_log_tee(runtime.log_file):
        print(f"Live log file: {runtime.log_file}")
        runner()
        return 0
