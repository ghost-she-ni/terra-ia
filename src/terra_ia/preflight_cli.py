from __future__ import annotations

import argparse
import importlib.util
import sys

from .preflight_checks import OPTIONAL_MODULES, REQUIRED_MODULES, core_file_paths, generated_output_paths
from .project import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick capstone readiness checker for Terra-IA.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when generated outputs are still missing.",
    )
    return parser


def _status(label: str, ok: bool, details: str = "") -> bool:
    prefix = "[OK]" if ok else "[MISS]"
    suffix = f" - {details}" if details else ""
    print(f"{prefix} {label}{suffix}")
    return ok


def _check_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print("Terra-IA preflight")
    print(f"project_root={PROJECT_ROOT}")
    print(f"python={sys.version.split()[0]}")

    ok = True
    if sys.version_info < (3, 10):
        ok = _status("Python >= 3.10", False, "current interpreter is too old") and ok
    else:
        _status("Python >= 3.10", True)

    print("\nCore files")
    for path in core_file_paths():
        rel_path = path.relative_to(PROJECT_ROOT)
        ok = _status(str(rel_path), path.exists()) and ok

    print("\nRequired modules")
    for module_name in REQUIRED_MODULES:
        ok = _status(module_name, _check_module(module_name)) and ok

    print("\nOptional modules")
    for module_name in OPTIONAL_MODULES:
        present = _check_module(module_name)
        prefix = "[OK]" if present else "[WARN]"
        print(f"{prefix} {module_name}")

    print("\nGenerated outputs")
    outputs_ready = True
    for path in generated_output_paths():
        rel_path = path.relative_to(PROJECT_ROOT) if path.is_relative_to(PROJECT_ROOT) else path
        outputs_ready = _status(str(rel_path), path.exists()) and outputs_ready

    if not outputs_ready:
        print("Run `python pipeline.py` to generate the demo-ready outputs.")

    if args.strict:
        return 0 if ok and outputs_ready else 1
    return 0 if ok else 1
