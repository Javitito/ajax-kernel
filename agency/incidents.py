from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class IncidentError(RuntimeError):
    pass


class IncidentReporter:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.incidents_dir = self.root_dir / "artifacts" / "incidents"

    def _dedupe_index_path(self) -> Path:
        return self.incidents_dir / "incident_dedupe.json"

    def _load_dedupe_index(self) -> Dict[str, str]:
        try:
            path = self._dedupe_index_path()
            if not path.exists():
                return {}
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_dedupe_index(self, index: Dict[str, str]) -> None:
        try:
            path = self._dedupe_index_path()
            path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            return

    def _next_path(self, kind: str) -> Path:
        ts = int(time.time())
        slug = kind.replace(" ", "_").replace("/", "_").lower()
        return self.incidents_dir / f"incident_{ts}_{slug}.json"

    def open_incident(
        self,
        *,
        kind: str,
        summary: str,
        context: Optional[Dict[str, Any]] = None,
        remediation: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        status: str = "open",
        dedupe_key: Optional[str] = None,
    ) -> str:
        self.incidents_dir.mkdir(parents=True, exist_ok=True)
        if dedupe_key:
            index = self._load_dedupe_index()
            existing = index.get(dedupe_key)
            if existing:
                existing_path = self.incidents_dir / f"{existing}.json"
                if existing_path.exists():
                    return existing
        ctx = context or {}
        payload = {
            "schema": "ajax.incident.v1",
            "incident_id": None,
            "kind": kind,
            "summary": summary,
            "status": status,
            "context": ctx,
            "remediation": remediation or [],
            "attachments": attachments or [],
            "opened_ts": time.time(),
        }
        path = self._next_path(kind)
        payload["incident_id"] = path.stem
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if dedupe_key:
            index = self._load_dedupe_index()
            index[dedupe_key] = payload["incident_id"]
            self._write_dedupe_index(index)
        if attachments:
            attach_path = self.incidents_dir / f"{path.stem}_attachments.json"
            attach_payload = {
                "incident_id": payload["incident_id"],
                "attachments": attachments,
                "generated_ts": payload["opened_ts"],
            }
            attach_path.write_text(json.dumps(attach_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return str(payload["incident_id"])
