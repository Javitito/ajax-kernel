from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class Inspector:
    """
    Inspector de misiones: lee artifacts/history y sintetiza patrones de fallo.
    No modifica configuración; solo describe.
    """

    def __init__(
        self,
        history_dir: Path,
        leann_client: Any = None,
        driver_status: Optional[Dict[str, Any]] = None,
        driver_policy: Optional[Dict[str, Any]] = None,
        include_driver: bool = False,
    ) -> None:
        self.history_dir = history_dir
        self.leann = leann_client
        self.driver_status = driver_status or {}
        self.driver_policy = driver_policy or {}
        self.include_driver = include_driver

    def _load_recent_missions(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.history_dir.exists():
            return []
        paths = sorted(self.history_dir.glob("mission-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        missions: List[Dict[str, Any]] = []
        for path in paths[:limit]:
            try:
                missions.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return missions

    def _aggregate_errors(self, missions: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "missions_analyzed": len(missions),
            "final_status": Counter(),
            "final_errors": Counter(),
            "tags": Counter(),
            "providers": defaultdict(lambda: {"total": 0, "failures": 0, "results": Counter(), "models": Counter()}),
        }
        for mission in missions:
            status = str(mission.get("final_status") or "unknown")
            summary["final_status"][status] += 1
            final_error = mission.get("final_error")
            if final_error:
                summary["final_errors"][str(final_error)] += 1
            for tag in mission.get("tags") or []:
                summary["tags"][str(tag)] += 1
            for attempt in mission.get("providers_tried") or []:
                provider_id = str(attempt.get("id") or "unknown")
                model = str(attempt.get("model") or "").strip()
                result = str(attempt.get("result") or "unknown")
                bucket = summary["providers"][provider_id]
                bucket["total"] += 1
                bucket["results"][result] += 1
                if model:
                    bucket["models"][model] += 1
                if result.lower() not in {"ok", "success"}:
                    bucket["failures"] += 1
        # normalizar counters a dict
        summary["final_status"] = dict(summary["final_status"])
        summary["final_errors"] = dict(summary["final_errors"])
        summary["tags"] = dict(summary["tags"])
        summary["providers"] = {k: {"total": v["total"], "failures": v["failures"], "results": dict(v["results"]), "models": dict(v["models"])} for k, v in summary["providers"].items()}
        return summary

    def _driver_summary(self, missions: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts = Counter()
        recent_errors: List[Dict[str, Any]] = []
        for mission in missions:
            err = str(mission.get("final_error") or "")
            if err.startswith("driver_"):
                counts[err] += 1
                recent_errors.append(
                    {
                        "mission_id": mission.get("mission_id"),
                        "error": err,
                        "timestamp_end": mission.get("timestamp_end"),
                    }
                )
        failures_snapshot = self.driver_status.get("failures") if isinstance(self.driver_status, dict) else None
        last_failure_reasons: List[Dict[str, Any]] = []
        snapshot_failure_count = 0
        if isinstance(failures_snapshot, list):
            snapshot_failure_count = len(failures_snapshot)
            for item in failures_snapshot[-5:]:
                if isinstance(item, dict):
                    last_failure_reasons.append({"ts": item.get("ts"), "reason": item.get("reason")})
        return {
            "status": self.driver_status.get("status", "unknown") if isinstance(self.driver_status, dict) else "unknown",
            "history_errors": dict(counts),
            "recent_errors": recent_errors[:5],
            "window_seconds": self.driver_policy.get("driver_failure_window_seconds"),
            "threshold": self.driver_policy.get("driver_failure_threshold"),
            "recovery_cooldown_seconds": self.driver_policy.get("driver_recovery_cooldown_seconds"),
            "snapshot": self.driver_status,
            "last_failure_reasons": last_failure_reasons,
            "snapshot_failure_count": snapshot_failure_count,
            "down_since": self.driver_status.get("down_since") if isinstance(self.driver_status, dict) else None,
        }

    def _enrich_with_leann(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        if not self.leann:
            return stats
        # Punto de extensión: buscar contexto en LEANN para los patrones principales.
        # Mantener stub para evitar dependencias duras en fase inicial.
        return stats

    def summarize_recent_failures(self, limit: int = 100) -> Dict[str, Any]:
        missions = self._load_recent_missions(limit=limit)
        stats = self._aggregate_errors(missions)
        if self.include_driver:
            stats["driver"] = self._driver_summary(missions)
        return self._enrich_with_leann(stats)
