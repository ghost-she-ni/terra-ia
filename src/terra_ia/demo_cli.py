from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .demo_runtime import demo_env_updates_from_args
from .project import PROJECT_ROOT

DEMO_SCRIPT = PROJECT_ROOT / "terra_ia_demo_v6.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stable capstone entrypoint for the Terra-IA Streamlit demo."
    )
    parser.add_argument("--features", type=str, help="Path to the exported parcel feature dataset.")
    parser.add_argument("--shap", type=str, help="Path to the SHAP export used for explainability panels.")
    parser.add_argument("--geo", type=str, help="Path to the parcel geometry file used by the map.")
    parser.add_argument("--output-dir", type=str, help="Directory containing the generated CSV and JSON outputs.")
    parser.add_argument("--port", type=int, default=8501, help="Streamlit port.")
    parser.add_argument("--host", default="localhost", help="Streamlit bind address.")
    parser.add_argument("--no-browser", action="store_true", help="Run Streamlit in headless mode.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved runtime configuration and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    env = os.environ.copy()
    env.update(demo_env_updates_from_args(args))

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DEMO_SCRIPT),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if args.no_browser:
        cmd.extend(["--server.headless", "true", "--browser.gatherUsageStats", "false"])

    if args.dry_run:
        print("Terra-IA demo runtime configuration")
        for key in ["TERRA_IA_FEATURES_PATH", "TERRA_IA_SHAP_PATH", "TERRA_IA_GEO_PATH"]:
            print(f"{key}={env.get(key, '')}")
        print("command=" + " ".join(cmd))
        return 0

    try:
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=False)
    except FileNotFoundError as exc:
        print("Unable to launch Streamlit.")
        print("Install the recommended environment first: conda env create -f environment.yml")
        print(f"Original error: {exc}")
        return 1
    return completed.returncode
