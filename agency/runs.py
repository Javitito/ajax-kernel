"""Utilities for run folder management."""

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


_LATEST_STARTED = "_latest_started"


DEFAULT_RUN_ROOT = Path(os.getenv("LEANN_RUN_ROOT", "runs"))


@dataclass(frozen=True)
class RunPaths:
    """Resolve and prepare folders for a single broker run."""

    root: Path
    job_file: Path
    result_file: Path
    logs_dir: Path
    artifacts_dir: Path
    state_file: Path
    journal_file: Path
    metrics_file: Path

    @classmethod
    def create(cls, job_id: Optional[str] = None, root: Optional[Path] = None) -> "RunPaths":
        run_root = root or DEFAULT_RUN_ROOT
        run_root.mkdir(parents=True, exist_ok=True)

        identifier = job_id or cls._generate_id()
        base_dir = run_root / identifier

        logs_dir = base_dir / "logs"
        artifacts_dir = base_dir / "artifacts"

        for path in (base_dir, logs_dir, artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)

        _mark_latest_started(run_root, base_dir)

        return cls(
            root=base_dir,
            job_file=base_dir / "job.json",
            result_file=base_dir / "result.json",
            logs_dir=logs_dir,
            artifacts_dir=artifacts_dir,
            state_file=base_dir / "state.json",
            journal_file=base_dir / "journal.jsonl",
            metrics_file=base_dir / "metrics.jsonl",
        )

    @staticmethod
    def _generate_id() -> str:
        now = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return f"run-{now}-{uuid.uuid4().hex[:6]}"

    def write_job(self, payload: Dict[str, Any]) -> None:
        self.job_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_result(self, payload: Dict[str, Any]) -> None:
        self.result_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = ["RunPaths", "DEFAULT_RUN_ROOT"]


def _atomic_symlink(target: Path, link: Path) -> None:
    """Replace ``link`` with a symlink to ``target`` atomically."""

    target = target.resolve()
    tmp = link.with_name(f".{link.name}.tmp-{uuid.uuid4().hex}")
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(target, tmp, target_is_directory=target.is_dir())
        os.replace(tmp, link)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _mark_latest_started(run_root: Path, run_dir: Path) -> None:
    try:
        _atomic_symlink(run_dir, run_root / _LATEST_STARTED)
    except OSError:
        pass
