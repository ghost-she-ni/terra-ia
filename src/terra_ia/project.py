from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"


def normalize_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def env_path(name: str, default: Path | str) -> Path:
    raw = os.environ.get(name)
    return normalize_path(raw) if raw else normalize_path(default)


def env_flag(name: str, default: str = "False") -> bool:
    return os.environ.get(name, default).lower() == "true"


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def ensure_parent_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
