from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import pandas as pd


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


class TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    @property
    def encoding(self) -> str:
        return getattr(self.streams[0], "encoding", "utf-8")

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def live_log_tee(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)
    try:
        yield log_path
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def snapshot_path(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "exists": path.exists(), "mtime": None}
    if not path.exists():
        return info

    if path.is_dir():
        mtimes = [path.stat().st_mtime]
        for child in path.rglob("*"):
            try:
                mtimes.append(child.stat().st_mtime)
            except OSError:
                continue
        info["mtime"] = max(mtimes) if mtimes else path.stat().st_mtime
        return info

    try:
        info["mtime"] = path.stat().st_mtime
    except OSError:
        info["mtime"] = None
    return info


class PipelineCheckpointManager:
    def __init__(
        self,
        *,
        state_path: Path,
        stage3_features_path: Path,
        stage6_labels_path: Path,
        stage9_scores_path: Path,
        resume_requested: bool,
        options: dict[str, Any],
        input_paths: dict[str, Path],
    ) -> None:
        self.state_path = state_path
        self.stage_paths = {
            "stage3": stage3_features_path,
            "stage6": stage6_labels_path,
            "stage9": stage9_scores_path,
        }
        self.resume_requested = resume_requested
        self.options = {key: self._serialize(value) for key, value in options.items()}
        self.input_paths = input_paths
        self.input_snapshot = {key: snapshot_path(path) for key, path in input_paths.items()}
        self.previous_state = self._load_state()
        self.state = self.previous_state or self._default_state()
        self.changed_keys = self._changed_input_keys(self.previous_state.get("inputs", {}), self.input_snapshot)
        self.changed_keys.update(self._changed_option_keys(self.previous_state.get("options", {}), self.options))
        self.invalidated_stage3_steps: set[str] = set()
        self.restored_from: list[str] = []
        self.step_timers: dict[str, float] = {}

        self.state["version"] = 1
        self.state["options"] = self.options
        self.state["inputs"] = self.input_snapshot
        self.state["last_status"] = "running"
        self.state["started_at"] = self.state.get("started_at") or now_iso()
        self.state["last_updated_at"] = now_iso()
        self.state.setdefault("steps", {})
        self.state.setdefault("report_cache", {})
        self.state.setdefault("runtime", {})
        self.state.setdefault("resume", {})
        self.state["resume"].update(
            {
                "requested": self.resume_requested,
                "changed_keys": sorted(self.changed_keys),
                "restored_from": [],
                "invalidated_steps": [],
            }
        )
        self.save_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "options": {},
            "inputs": {},
            "steps": {},
            "report_cache": {},
            "runtime": {},
            "resume": {},
            "last_status": "idle",
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _changed_input_keys(self, previous: dict[str, Any], current: dict[str, Any]) -> set[str]:
        changed: set[str] = set()
        for key, current_info in current.items():
            previous_info = previous.get(key)
            if previous_info != current_info:
                changed.add(key)
        return changed

    def _changed_option_keys(self, previous: dict[str, Any], current: dict[str, Any]) -> set[str]:
        changed: set[str] = set()
        for key, current_value in current.items():
            if previous.get(key) != current_value:
                changed.add(f"option:{key}")
        return changed

    def save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2, ensure_ascii=False, default=str)

    def update_report_cache(self, report: dict[str, Any]) -> None:
        self.state["report_cache"] = report
        self.state["last_updated_at"] = now_iso()
        self.save_state()

    def restore_report_cache(self) -> dict[str, Any]:
        cached = self.state.get("report_cache", {})
        return cached.copy() if isinstance(cached, dict) else {}

    def start_step(self, step_name: str) -> None:
        self.step_timers[step_name] = time.perf_counter()
        self.state["steps"][step_name] = {
            "status": "running",
            "started_at": now_iso(),
        }
        self.state["last_stage"] = step_name
        self.state["last_updated_at"] = now_iso()
        self.save_state()

    def complete_step(self, step_name: str, *, extra: dict[str, Any] | None = None) -> None:
        duration_sec = time.perf_counter() - self.step_timers.pop(step_name, time.perf_counter())
        step = self.state["steps"].get(step_name, {})
        step.update(
            {
                "status": "completed",
                "completed_at": now_iso(),
                "duration_sec": round(duration_sec, 3),
            }
        )
        if extra:
            step.update(extra)
        self.state["steps"][step_name] = step
        self.state["last_updated_at"] = now_iso()
        self.save_state()
        print(f"    [checkpoint] {step_name} completed in {duration_sec:.1f}s")

    def fail_step(self, step_name: str, exc: Exception) -> None:
        duration_sec = time.perf_counter() - self.step_timers.pop(step_name, time.perf_counter())
        self.state["steps"][step_name] = {
            "status": "failed",
            "failed_at": now_iso(),
            "duration_sec": round(duration_sec, 3),
            "error": str(exc),
        }
        self.state["last_status"] = "failed"
        self.state["last_stage"] = step_name
        self.state["last_updated_at"] = now_iso()
        self.save_state()

    def mark_run_completed(self) -> None:
        runtime = self.state.setdefault("runtime", {})
        started_at = self.state.get("started_at")
        if started_at:
            try:
                started = datetime.fromisoformat(started_at)
                runtime["total_seconds"] = round(
                    (datetime.now().astimezone() - started).total_seconds(),
                    3,
                )
            except ValueError:
                pass
        self.state["last_status"] = "completed"
        self.state["last_success"] = now_iso()
        self.state["last_updated_at"] = now_iso()
        self.save_state()

    def invalidate_stage(self, stage_name: str) -> None:
        path = self.stage_paths[stage_name]
        if path.exists():
            path.unlink()

        prefix = f"{stage_name}."
        for key in list(self.state["steps"].keys()):
            if key.startswith(prefix):
                self.state["steps"].pop(key, None)

        if stage_name == "stage3":
            self.invalidated_stage3_steps = set()

        self.state["resume"]["restored_from"] = [
            entry for entry in self.state["resume"].get("restored_from", []) if entry != stage_name
        ]
        self.state["last_updated_at"] = now_iso()
        self.save_state()

    def save_stage_frame(self, stage_name: str, frame: pd.DataFrame) -> None:
        path = self.stage_paths[stage_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)

    def _load_stage_frame(self, stage_name: str, index: pd.Index) -> pd.DataFrame | None:
        path = self.stage_paths[stage_name]
        if not path.exists():
            return None
        try:
            frame = pd.read_parquet(path)
        except Exception:
            return None
        return frame.reindex(index)

    def restore_stage3_frame(
        self,
        *,
        index: pd.Index,
        step_columns: dict[str, list[str]],
        step_dependencies: dict[str, list[str]],
    ) -> pd.DataFrame:
        frame = self._load_stage_frame("stage3", index)
        if frame is None:
            return pd.DataFrame(index=index)

        if not self.resume_requested:
            return pd.DataFrame(index=index)

        invalidated = {
            step_name
            for step_name, dependencies in step_dependencies.items()
            if any(dependency in self.changed_keys for dependency in dependencies)
        }
        self.invalidated_stage3_steps = invalidated

        if invalidated:
            for step_name in invalidated:
                cols = step_columns.get(step_name, [])
                drop_cols = [col for col in cols if col in frame.columns]
                if drop_cols:
                    frame = frame.drop(columns=drop_cols)
                self.state["steps"].pop(f"stage3.{step_name}", None)

            self.invalidate_stage("stage6")
            self.invalidate_stage("stage9")

        if frame.empty and not frame.columns.tolist():
            return frame

        self.restored_from.append("stage3")
        self.state["resume"]["restored_from"] = self.restored_from.copy()
        self.state["resume"]["invalidated_steps"] = sorted(self.invalidated_stage3_steps)
        self.state["last_updated_at"] = now_iso()
        self.save_state()
        return frame

    def restore_stage_frame(
        self,
        stage_name: str,
        *,
        index: pd.Index,
        dependencies: Iterable[str] | None = None,
    ) -> pd.DataFrame | None:
        if not self.resume_requested:
            return None

        dependencies = list(dependencies or [])
        if any(dependency in self.changed_keys for dependency in dependencies):
            self.invalidate_stage(stage_name)
            if stage_name == "stage6":
                self.invalidate_stage("stage9")
            return None

        if stage_name in {"stage6", "stage9"} and self.invalidated_stage3_steps:
            self.invalidate_stage(stage_name)
            if stage_name == "stage6":
                self.invalidate_stage("stage9")
            return None

        frame = self._load_stage_frame(stage_name, index)
        if frame is None:
            return None

        self.restored_from.append(stage_name)
        self.state["resume"]["restored_from"] = self.restored_from.copy()
        self.state["last_updated_at"] = now_iso()
        self.save_state()
        return frame

    def checkpoint_status(self) -> dict[str, Any]:
        completed_steps = sorted(
            step_name
            for step_name, payload in self.state.get("steps", {}).items()
            if isinstance(payload, dict) and payload.get("status") == "completed"
        )
        return {
            "resume_requested": self.resume_requested,
            "state_path": str(self.state_path),
            "restored_from": self.state.get("resume", {}).get("restored_from", []),
            "changed_keys": self.state.get("resume", {}).get("changed_keys", []),
            "invalidated_steps": self.state.get("resume", {}).get("invalidated_steps", []),
            "stage3_checkpoint": str(self.stage_paths["stage3"]),
            "stage6_checkpoint": str(self.stage_paths["stage6"]),
            "stage9_checkpoint": str(self.stage_paths["stage9"]),
            "completed_steps": completed_steps,
        }


def load_osm_roads_union(
    *,
    cache_path: Path,
    refresh: bool,
    place_name: str,
    target_crs: str,
) -> tuple[Any | None, dict[str, Any]]:
    status = {
        "cache_path": str(cache_path),
        "cache_exists": cache_path.exists(),
        "refresh_requested": refresh,
        "source": None,
        "error": None,
    }

    def read_cached_union() -> Any | None:
        if not cache_path.exists():
            return None
        roads = gpd.read_file(cache_path)
        if roads.empty:
            return None
        if roads.crs is None or str(roads.crs) != target_crs:
            roads = roads.to_crs(target_crs)
        status["source"] = "cache"
        return roads.geometry.unary_union

    if cache_path.exists() and not refresh:
        cached = read_cached_union()
        if cached is not None:
            return cached, status

    if not module_available("osmnx"):
        status["error"] = "osmnx_missing"
        cached = read_cached_union()
        return cached, status

    try:
        import osmnx as ox

        graph = ox.graph_from_place(place_name, network_type="drive")
        roads = ox.graph_to_gdfs(graph, nodes=False).to_crs(target_crs)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        roads.to_file(cache_path, driver="GeoJSON")
        status["cache_exists"] = True
        status["source"] = "download"
        return roads.geometry.unary_union, status
    except Exception as exc:
        status["error"] = str(exc)
        cached = read_cached_union()
        if cached is not None:
            status["source"] = "cache_fallback"
            return cached, status
        return None, status


__all__ = [
    "PipelineCheckpointManager",
    "live_log_tee",
    "load_osm_roads_union",
    "module_available",
]
