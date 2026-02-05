from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class MissionHistoryRecorder:
    """
    Registro estructurado de misiones para análisis posterior (Inspector/Científico).
    Es resiliente por diseño: un fallo de registro nunca debe romper la misión.
    """

    def __init__(self, root_dir: Path, history_dir: Optional[Path] = None, logger: Any = None) -> None:
        self.root_dir = root_dir
        self.history_dir = history_dir or (root_dir / "artifacts" / "history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def _iso(self, ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _safe_filename(self, mission_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", mission_id).strip("_")
        return safe or "mission"

    def _safe_obj(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, list):
            return [self._safe_obj(v) for v in obj]
        if isinstance(obj, dict):
            return {str(k): self._safe_obj(v) for k, v in obj.items()}
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    def log_mission(
        self,
        *,
        mission_id: str,
        intent_text: str,
        mode: str,
        timestamp_start: float,
        timestamp_end: float,
        providers_tried: Optional[List[Dict[str, Any]]] = None,
        final_status: str,
        final_error: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Guarda la ficha de misión en artifacts/history/mission-<id>.json.
        """
        try:
            entry = {
                "mission_id": mission_id,
                "timestamp_start": self._iso(timestamp_start),
                "timestamp_end": self._iso(timestamp_end),
                "duration_ms": int((timestamp_end - timestamp_start) * 1000),
                "intent_text": intent_text,
                "mode": mode,
                "providers_tried": providers_tried or [],
                "final_status": final_status,
                "final_error": final_error,
                "tags": tags or [],
                "metadata": self._safe_obj(metadata or {}),
            }
            fname = f"mission-{self._safe_filename(mission_id)}.json"
            fpath = self.history_dir / fname
            fpath.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            return fpath
        except Exception as exc:  # pragma: no cover - resiliencia
            if self.logger:
                try:
                    self.logger.warning("MissionHistoryRecorder: no se pudo escribir registro (%s)", exc)
                except Exception:
                    pass
            return None
