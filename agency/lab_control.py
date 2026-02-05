from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


DEFAULT_PROBE_TTL_SECONDS = 900
DEFAULT_TRANSFER_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_ACK_TTL_SECONDS = 86400

_HIGH_RISK_JOB_KINDS = {
    "apply_recipe",
    "promote_recipe",
    "system_change",
    "os_change",
    "ui_invasive",
    "registry_change",
    "install_app",
}
_AUTO_CLOSE_JOB_KINDS = {
    "snap_lab",
    "snap_lab_silent",
    "probe_ui",
    "probe_notepad",
    "lab_notepad_smoke",
    "doctor_lab",
    "health_check",
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _to_float_ts(value: Any) -> Optional[float]:
    try:
        ts = float(value)
    except Exception:
        return None
    return ts if ts > 0 else None


class LabStateStore:
    """
    Persistencia mínima de LAB dual:
      - lab_control.json: estado de LAB_ORG y probes
      - lab_org_state.json: snapshot del loop autónomo (pausable)
      - probes/<probe_id>.json: ExperimentEnvelope derivado desde PROD (LAB_PROBE)
      - jobs/job_<ts>_<mission_id>.json: handoff de misiones a LAB
      - results/result_<ts>_<job_id>.json: outcome de LAB (PASS/FAIL/PARTIAL)
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.lab_dir = self.root_dir / "artifacts" / "lab"
        self.lab_dir.mkdir(parents=True, exist_ok=True)
        self.control_path = self.lab_dir / "lab_control.json"
        self.org_state_path = self.lab_dir / "lab_org_state.json"
        self.probes_dir = self.lab_dir / "probes"
        self.probes_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir = self.lab_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.lab_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_control()
        self.transfer_confidence_threshold = float(
            self.state.get("lab_probe", {}).get("transfer_confidence_threshold", DEFAULT_TRANSFER_CONFIDENCE_THRESHOLD)
        )

    @staticmethod
    def _normalize_priority(value: Any) -> int:
        if value is None:
            return 50
        if isinstance(value, str):
            label = value.strip().lower()
            if label in {"high", "h"}:
                return 90
            if label in {"med", "medium", "m"}:
                return 50
            if label in {"low", "l"}:
                return 30
            try:
                value = int(label)
            except Exception:
                return 50
        try:
            priority = int(value)
        except Exception:
            priority = 50
        return max(0, min(priority, 100))

    @staticmethod
    def _normalize_objective(value: Any) -> str:
        if value is None:
            return ""
        text = str(value)
        if not text:
            return ""
        try:
            text = unicodedata.normalize("NFKD", text)
            text = text.encode("ascii", "ignore").decode("ascii")
        except Exception:
            pass
        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = " ".join(text.split())
        return text.strip()

    @staticmethod
    def _intent_fingerprint(objective_norm: str) -> str:
        try:
            payload = (objective_norm or "").encode("utf-8", errors="ignore")
        except Exception:
            payload = b""
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _queued_since(job: Dict[str, Any], *, now_ts: Optional[float] = None) -> float:
        now = float(now_ts or time.time())
        queued_ts = _to_float_ts(job.get("queued_since_ts"))
        if queued_ts is not None:
            return queued_ts
        created_ts = _to_float_ts(job.get("created_ts"))
        if created_ts is not None:
            return created_ts
        started_ts = _to_float_ts(job.get("started_ts"))
        if started_ts is not None:
            return started_ts
        return now

    @staticmethod
    def _job_age_seconds(job: Dict[str, Any], *, now_ts: Optional[float] = None) -> float:
        now = float(now_ts or time.time())
        status = str(job.get("status") or "").upper()
        if status == "QUEUED":
            base = LabStateStore._queued_since(job, now_ts=now)
        else:
            base = _to_float_ts(job.get("started_ts")) or LabStateStore._queued_since(job, now_ts=now)
        return max(0.0, now - float(base))

    @staticmethod
    def _normalize_risk_level(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"high", "medium", "low"}:
            return raw
        return "medium"

    @staticmethod
    def _infer_ack_required(*, risk_level: str, job_kind: str, requires_ack: bool) -> bool:
        if requires_ack:
            return True
        if job_kind in _AUTO_CLOSE_JOB_KINDS:
            return False
        if job_kind in _HIGH_RISK_JOB_KINDS:
            return True
        return risk_level == "high"

    def _archive_unacked_result(self, payload: Dict[str, Any], path: Path, *, now_ts: float) -> None:
        payload["ack_status"] = "archived_unacked"
        payload["archived_ts"] = now_ts
        payload["archived_reason"] = "ttl_expired"
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            return

    def _effective_priority(self, job: Dict[str, Any], *, now_ts: Optional[float] = None) -> int:
        base = self._normalize_priority(job.get("priority"))
        status = str(job.get("status") or "").upper()
        if status != "QUEUED":
            return base
        now = float(now_ts or time.time())
        age = self._job_age_seconds(job, now_ts=now)
        try:
            starve_after = float(os.getenv("LAB_QUEUE_STARVE_SECONDS", "1800") or 1800)
        except Exception:
            starve_after = 1800.0
        try:
            boost_interval = float(os.getenv("LAB_QUEUE_BOOST_INTERVAL", "300") or 300)
        except Exception:
            boost_interval = 300.0
        try:
            boost_step = int(os.getenv("LAB_QUEUE_BOOST_STEP", "5") or 5)
        except Exception:
            boost_step = 5
        if age <= max(60.0, starve_after):
            return base
        interval = max(60.0, boost_interval)
        boosts = int(max(0.0, age - starve_after) // interval) + 1
        boosted = min(100, base + boosts * boost_step)
        return boosted

    def _queue_sort_key(self, job: Dict[str, Any], *, now_ts: Optional[float] = None, foreground_job_id: Optional[str] = None) -> tuple:
        now = float(now_ts or time.time())
        job_id = str(job.get("job_id") or "")
        is_foreground = bool(foreground_job_id and job_id and job_id == str(foreground_job_id))
        eff = self._effective_priority(job, now_ts=now)
        queued_ts = self._queued_since(job, now_ts=now)
        return (-int(is_foreground), -eff, queued_ts, job_id)

    def _load_waiting_payload(self) -> Optional[Dict[str, Any]]:
        path = self.root_dir / "artifacts" / "state" / "waiting_mission.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _foreground_job_id(self) -> Optional[str]:
        payload = self._load_waiting_payload()
        if not payload:
            return None
        mission_raw = payload.get("mission") if isinstance(payload.get("mission"), dict) else {}
        job_id = mission_raw.get("lab_job_id") or payload.get("lab_job_id")
        return str(job_id) if job_id else None

    def _ensure_lineage_fields(self, payload: Dict[str, Any]) -> bool:
        changed = False
        objective = payload.get("objective") or ""
        objective_norm = payload.get("objective_norm")
        if not isinstance(objective_norm, str) or not objective_norm.strip():
            objective_norm = self._normalize_objective(objective)
        if payload.get("objective_norm") != objective_norm:
            payload["objective_norm"] = objective_norm
            changed = True
        fingerprint = payload.get("intent_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            fingerprint = self._intent_fingerprint(objective_norm)
        if payload.get("intent_fingerprint") != fingerprint:
            payload["intent_fingerprint"] = fingerprint
            changed = True
        root_intent_id = payload.get("root_intent_id")
        if not isinstance(root_intent_id, str) or not root_intent_id.strip():
            payload["root_intent_id"] = fingerprint
            changed = True
        if payload.get("root_mission_id") is None:
            payload["root_mission_id"] = payload.get("mission_id")
            changed = True
        return changed

    def _sync_foreground_flag(self, payload: Dict[str, Any], path: Path, *, foreground_job_id: Optional[str]) -> None:
        job_id = str(payload.get("job_id") or "")
        is_foreground = bool(foreground_job_id and job_id and job_id == str(foreground_job_id))
        if payload.get("foreground") != is_foreground:
            payload["foreground"] = is_foreground
            try:
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass

    def sync_foreground_from_waiting(self) -> None:
        fg_id = self._foreground_job_id()
        if not fg_id:
            return
        for item in self.list_jobs(statuses={"QUEUED", "RUNNING"}):
            job = item.get("job") or {}
            path = item.get("path")
            if isinstance(path, Path):
                self._sync_foreground_flag(job, path, foreground_job_id=fg_id)

    def _job_timestamp(self, job: Dict[str, Any], path: Optional[Path], *, now_ts: Optional[float] = None) -> float:
        now = float(now_ts or time.time())
        candidates = [
            _to_float_ts(job.get("completed_ts")),
            _to_float_ts(job.get("last_heartbeat_ts")),
            _to_float_ts(job.get("started_ts")),
            _to_float_ts(job.get("created_ts")),
            _to_float_ts(job.get("queued_since_ts")),
        ]
        for ts in candidates:
            if ts is not None:
                return ts
        if path:
            try:
                return float(path.stat().st_mtime)
            except Exception:
                return now
        return now

    def _collect_reference_blob(self) -> str:
        roots = [
            self.root_dir / "artifacts" / "capability_gaps",
            self.root_dir / "artifacts" / "incidents",
        ]
        chunks: list[str] = []
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.json")):
                try:
                    chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    continue
        return "\n".join(chunks)

    def prune_terminal_jobs(
        self,
        *,
        older_than_s: Optional[float] = None,
        before_ts: Optional[float] = None,
        keep_per_fingerprint: Optional[int] = None,
        mode: str = "archive",
        dry_run: bool = False,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = float(now_ts or time.time())
        terminal_statuses = {"DONE", "FAILED", "CANCELLED"}
        terminal = self.list_jobs(statuses=terminal_statuses)
        active = self.list_jobs(statuses={"QUEUED", "RUNNING"})
        cutoff = None
        if before_ts is not None:
            cutoff = float(before_ts)
        elif older_than_s is not None:
            cutoff = now - float(older_than_s)

        by_fp: Dict[str, list[Dict[str, Any]]] = {}
        job_meta: Dict[str, Dict[str, Any]] = {}
        for item in terminal:
            job = item.get("job") or {}
            path = item.get("path")
            self._ensure_lineage_fields(job)
            fp = str(job.get("intent_fingerprint") or "unknown")
            ts = self._job_timestamp(job, path, now_ts=now)
            job_id = str(job.get("job_id") or (path.name if isinstance(path, Path) else "job"))
            entry = {"job": job, "path": path, "ts": ts, "job_id": job_id}
            by_fp.setdefault(fp, []).append(entry)
            job_meta[job_id] = {"fingerprint": fp, "ts": ts}

        keep_ids: set[str] = set()
        if keep_per_fingerprint is not None and keep_per_fingerprint > 0:
            for entries in by_fp.values():
                entries.sort(key=lambda e: (-float(e["ts"]), str(e["job_id"])))
                for entry in entries[:keep_per_fingerprint]:
                    keep_ids.add(str(entry["job_id"]))

        reference_blob = self._collect_reference_blob()
        actions: list[Dict[str, Any]] = []
        preserved: list[Dict[str, Any]] = []
        skipped_recent: list[str] = []
        skipped_keep: list[str] = []
        errors: list[str] = []
        moved = 0
        deleted = 0

        ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now))
        archive_root = self.root_dir / "artifacts" / "lab" / "archive" / ts_label
        jobs_archive = archive_root / "jobs"
        results_archive = archive_root / "results"

        for fp, entries in by_fp.items():
            for entry in entries:
                job = entry["job"]
                path = entry["path"]
                ts = float(entry["ts"])
                job_id = str(entry["job_id"])
                if job_id in keep_ids:
                    skipped_keep.append(job_id)
                    continue
                if cutoff is not None and ts > cutoff:
                    skipped_recent.append(job_id)
                    continue
                if not isinstance(path, Path):
                    continue
                result_paths = sorted(self.results_dir.glob(f"result_*_{job_id}.json"))
                tokens = {job_id, path.name, str(path)}
                try:
                    tokens.add(os.path.relpath(path, self.root_dir))
                except Exception:
                    pass
                for rpath in result_paths:
                    tokens.update({rpath.name, str(rpath)})
                    try:
                        tokens.add(os.path.relpath(rpath, self.root_dir))
                    except Exception:
                        pass
                referenced = any(tok and tok in reference_blob for tok in tokens)
                if referenced:
                    preserved.append({"job_id": job_id, "reason": "referenced_by_gap_or_incident"})
                    continue

                action_entry: Dict[str, Any] = {
                    "job_id": job_id,
                    "status": job.get("status"),
                    "intent_fingerprint": fp,
                    "job_path": str(path),
                    "result_paths": [str(p) for p in result_paths],
                    "action": mode,
                }
                if mode == "archive":
                    dest_job = str(jobs_archive / path.name)
                    dest_results = [str(results_archive / p.name) for p in result_paths]
                    action_entry["archive_job_path"] = dest_job
                    action_entry["archive_result_paths"] = dest_results
                    if not dry_run:
                        try:
                            jobs_archive.mkdir(parents=True, exist_ok=True)
                            results_archive.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(path), dest_job)
                            for src, dest in zip(result_paths, dest_results):
                                shutil.move(str(src), dest)
                            moved += 1
                        except Exception as exc:
                            errors.append(f"{job_id}:archive_failed:{exc}")
                    actions.append(action_entry)
                else:
                    if not dry_run:
                        try:
                            path.unlink(missing_ok=True)
                            for rpath in result_paths:
                                rpath.unlink(missing_ok=True)
                            deleted += 1
                        except Exception as exc:
                            errors.append(f"{job_id}:delete_failed:{exc}")
                    actions.append(action_entry)

        receipt = {
            "schema": "ajax.lab_prune.v1",
            "ts": now,
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            "mode": mode,
            "dry_run": dry_run,
            "filters": {
                "older_than_s": older_than_s,
                "before_ts": before_ts,
                "keep_per_fingerprint": keep_per_fingerprint,
            },
            "counts": {
                "terminal_jobs": len(terminal),
                "active_jobs": len(active),
                "kept_by_fingerprint": len(skipped_keep),
                "skipped_recent": len(skipped_recent),
                "preserved_referenced": len(preserved),
                "moved": moved,
                "deleted": deleted,
                "actions": len(actions),
            },
            "archive_path": str(archive_root) if mode == "archive" else None,
            "actions": actions,
            "preserved": preserved,
            "skipped_recent": skipped_recent,
            "skipped_keep": skipped_keep,
            "errors": errors,
        }
        doctor_dir = self.root_dir / "artifacts" / "doctor"
        doctor_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = doctor_dir / f"lab_prune_{ts_label}.json"
        try:
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        receipt["receipt_path"] = str(receipt_path)
        return receipt

    @staticmethod
    def _write_job_payload(payload: Dict[str, Any], path: Path) -> None:
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _load_control(self) -> Dict[str, Any]:
        if self.control_path.exists():
            try:
                data = json.loads(self.control_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        default = {
            "lab_org": {
                "status": "IDLE",
                "reason": "not_started",
                "updated_utc": _utc_now(),
                "state_path": str(self.org_state_path.relative_to(self.root_dir)) if self.org_state_path.exists() else "artifacts/lab/lab_org_state.json",
            },
            "lab_probe": {
                "active": None,
                "history": [],
                "last_probe": None,
                "transfer_confidence_threshold": DEFAULT_TRANSFER_CONFIDENCE_THRESHOLD,
            },
        }
        self._save_control(default)
        return default

    def _save_control(self, data: Optional[Dict[str, Any]] = None) -> None:
        payload = data or self.state
        try:
            self.control_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    # --- LAB_ORG helpers -------------------------------------------------
    def is_lab_org_running(self) -> bool:
        return (self.state.get("lab_org") or {}).get("status") == "RUNNING"

    def pause_lab_org(self, reason: str, metadata: Optional[Dict[str, Any]] = None, snapshot: Optional[Dict[str, Any]] = None) -> bool:
        lab_org = self.state.setdefault("lab_org", {})
        if lab_org.get("status") == "PAUSED" and lab_org.get("reason") == reason:
            return False
        updated = _utc_now()
        lab_org.update(
            {
                "status": "PAUSED",
                "reason": reason,
                "updated_utc": updated,
                "state_path": lab_org.get("state_path") or "artifacts/lab/lab_org_state.json",
                "metadata": metadata or {},
            }
        )
        if snapshot is not None:
            self._write_lab_org_state(snapshot, updated)
        self._save_control()
        return True

    def resume_lab_org(self, reason: str, metadata: Optional[Dict[str, Any]] = None, snapshot: Optional[Dict[str, Any]] = None) -> bool:
        lab_org = self.state.setdefault("lab_org", {})
        updated = _utc_now()
        changed = lab_org.get("status") != "RUNNING"
        lab_org.update(
            {
                "status": "RUNNING",
                "reason": reason,
                "updated_utc": updated,
                "state_path": lab_org.get("state_path") or "artifacts/lab/lab_org_state.json",
                "metadata": metadata or {},
            }
        )
        if snapshot is not None:
            self._write_lab_org_state(snapshot, updated)
        self._save_control()
        return changed

    def _write_lab_org_state(self, snapshot: Dict[str, Any], updated_utc: Optional[str] = None) -> None:
        payload = {
            "updated_utc": updated_utc or _utc_now(),
            "snapshot": snapshot,
        }
        try:
            self.org_state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    # --- LAB_PROBE helpers -----------------------------------------------
    def _resolve_probe_path(self, probe_id_or_path: str) -> Path:
        candidate = Path(probe_id_or_path)
        if candidate.exists():
            return candidate
        if not candidate.suffix:
            candidate = self.probes_dir / f"{probe_id_or_path}.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Probe '{probe_id_or_path}' not found (looked for {candidate}).")

    def load_probe(self, probe_id_or_path: str) -> Tuple[Dict[str, Any], Path]:
        path = self._resolve_probe_path(probe_id_or_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Probe payload in {path} is not a JSON object.")
        return data, path

    def _resolve_job_path(self, job_id_or_path: str) -> Path:
        candidate = Path(job_id_or_path)
        if candidate.exists():
            return candidate
        if not candidate.suffix:
            candidate = self.jobs_dir / f"{job_id_or_path}.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Job '{job_id_or_path}' not found (looked for {candidate}).")

    def load_job(self, job_id_or_path: str) -> Tuple[Dict[str, Any], Path]:
        path = self._resolve_job_path(job_id_or_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Job payload in {path} is not a JSON object.")
        return data, path

    def create_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(payload)
        mission_id = str(data.get("mission_id") or "").strip() or "mission"
        job_id = data.get("job_id") or f"job_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{mission_id}"
        data["job_id"] = job_id
        data.setdefault("mission_id", mission_id)
        data.setdefault("incident_id", None)
        data.setdefault("status", "QUEUED")
        started = float(data.get("started_ts") or time.time())
        data["started_ts"] = started
        data.setdefault("created_ts", started)
        data.setdefault("queued_since_ts", data.get("created_ts") or started)
        data.setdefault("last_heartbeat_ts", started)
        data["priority"] = self._normalize_priority(data.get("priority"))
        data.setdefault("priority_reason", "default")
        self._ensure_lineage_fields(data)
        data.setdefault("foreground", False)
        data.setdefault("stale", {"is_stale": False})
        data.setdefault("objective", "")
        data.setdefault("planned_steps", [])
        data.setdefault("evidence_expected", [])
        data.setdefault("output_paths", [])
        path = self.jobs_dir / f"{job_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"job_id": job_id, "job_path": str(path), "payload": data}

    def find_active_job_by_fingerprint(self, fingerprint: str) -> Optional[Tuple[Dict[str, Any], Path]]:
        if not fingerprint:
            return None
        candidates = self.list_jobs(statuses={"QUEUED", "RUNNING"})
        if not candidates:
            return None
        now = time.time()
        fg_id = self._foreground_job_id()
        def _key(item: Dict[str, Any]) -> tuple:
            return self._queue_sort_key(item["job"], now_ts=now, foreground_job_id=fg_id)
        candidates.sort(key=_key)
        for item in candidates:
            job = item["job"]
            changed = self._ensure_lineage_fields(job)
            if changed:
                self._write_job_payload(job, item["path"])
            if str(job.get("intent_fingerprint") or "") == str(fingerprint):
                return job, item["path"]
        return None

    def enqueue_job(
        self,
        payload: Dict[str, Any],
        *,
        coalesce_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        data = dict(payload)
        self._ensure_lineage_fields(data)
        fingerprint = data.get("intent_fingerprint")
        if not isinstance(fingerprint, str):
            fingerprint = ""
        mode = (coalesce_mode or os.getenv("AJAX_LAB_COALESCE_MODE") or "coalesce").strip().lower()
        existing = self.find_active_job_by_fingerprint(fingerprint)
        if existing:
            existing_job, path = existing
            fg_id = self._foreground_job_id()
            self._sync_foreground_flag(existing_job, path, foreground_job_id=fg_id)
            if mode == "replace":
                try:
                    self.cancel_job(existing_job.get("job_id") or str(path), reason="coalesce_replace")
                except Exception:
                    pass
            else:
                output_paths = data.get("output_paths") or []
                if output_paths:
                    try:
                        self.update_job_status(
                            existing_job.get("job_id") or str(path),
                            status=existing_job.get("status") or "QUEUED",
                            output_paths=output_paths,
                        )
                    except Exception:
                        pass
                try:
                    refreshed, _ = self.load_job(existing_job.get("job_id") or str(path))
                except Exception:
                    refreshed = existing_job
                return {
                    "job_id": refreshed.get("job_id"),
                    "job_path": str(path),
                    "payload": refreshed,
                    "coalesced": True,
                }
        record = self.create_job(data)
        try:
            fg_id = self._foreground_job_id()
            payload = record.get("payload") if isinstance(record, dict) else None
            job_path = record.get("job_path") if isinstance(record, dict) else None
            if isinstance(payload, dict) and isinstance(job_path, str):
                self._sync_foreground_flag(payload, Path(job_path), foreground_job_id=fg_id)
        except Exception:
            pass
        record["coalesced"] = False
        return record

    def update_job_status(
        self,
        job_id_or_path: str,
        *,
        status: str,
        output_paths: Optional[list[str]] = None,
    ) -> Tuple[Dict[str, Any], Path]:
        payload, path = self.load_job(job_id_or_path)
        payload["status"] = str(status).upper()
        now_ts = float(time.time())
        payload["last_heartbeat_ts"] = now_ts
        if payload.get("created_ts") is None:
            payload["created_ts"] = payload.get("started_ts") or now_ts
        if payload.get("started_ts") is None:
            payload["started_ts"] = now_ts
        if payload.get("queued_since_ts") is None:
            payload["queued_since_ts"] = payload.get("created_ts") or now_ts
        if output_paths:
            existing = payload.get("output_paths") or []
            if not isinstance(existing, list):
                existing = []
            for item in output_paths:
                if item and item not in existing:
                    existing.append(item)
            payload["output_paths"] = existing
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload, path

    def compute_staleness(
        self,
        job: Dict[str, Any],
        *,
        now_ts: Optional[float] = None,
        stale_minutes: Optional[float] = None,
        heartbeat_stale_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = float(now_ts or time.time())
        if stale_minutes is None:
            try:
                stale_minutes = float(os.getenv("LAB_STALE_MINUTES", "10") or 10)
            except Exception:
                stale_minutes = 10.0
        if heartbeat_stale_seconds is None:
            try:
                heartbeat_stale_seconds = float(os.getenv("LAB_HEARTBEAT_STALE_SECONDS", "90") or 90)
            except Exception:
                heartbeat_stale_seconds = 90.0
        stale_threshold = max(60.0, stale_minutes * 60.0)
        heartbeat_threshold = max(30.0, heartbeat_stale_seconds)

        status = str(job.get("status") or "").upper()
        created_ts = _to_float_ts(job.get("created_ts"))
        started_ts = _to_float_ts(job.get("started_ts"))
        queued_ts = _to_float_ts(job.get("queued_since_ts"))
        if status == "QUEUED":
            base_ts = queued_ts or created_ts or started_ts or now
        else:
            base_ts = started_ts or queued_ts or created_ts or now
        job_age_s = max(0.0, now - base_ts)

        heartbeat_ts = _to_float_ts(job.get("last_heartbeat_ts"))
        heartbeat_age_s = None
        if heartbeat_ts is not None:
            heartbeat_age_s = max(0.0, now - heartbeat_ts)

        is_stale = False
        reason = None
        if status == "QUEUED":
            if job_age_s >= stale_threshold:
                is_stale = True
                reason = "queued_too_long"
        elif status == "RUNNING":
            if heartbeat_age_s is None:
                if job_age_s >= stale_threshold:
                    is_stale = True
                    reason = "running_no_heartbeat"
            elif heartbeat_age_s >= heartbeat_threshold:
                is_stale = True
                reason = "running_no_heartbeat"

        return {
            "status": status,
            "job_age_s": job_age_s,
            "heartbeat_age_s": heartbeat_age_s,
            "is_stale": is_stale,
            "reason": reason,
            "stale_threshold_s": stale_threshold,
            "heartbeat_threshold_s": heartbeat_threshold,
        }

    def _stale_failure_summary(self, job: Dict[str, Any], info: Dict[str, Any]) -> str:
        reason = str(info.get("reason") or "stale")
        job_id = job.get("job_id") or "unknown"
        job_age = info.get("job_age_s")
        hb_age = info.get("heartbeat_age_s")
        job_age_txt = f"{int(job_age)}s" if isinstance(job_age, (int, float)) else "unknown"
        hb_age_txt = f"{int(hb_age)}s" if isinstance(hb_age, (int, float)) else "unknown"
        if reason == "queued_too_long":
            detail = f"queued too long (age={job_age_txt})"
        else:
            detail = f"heartbeat stale (age={hb_age_txt})"
        return (
            f"NoUserLeftBehind: LAB job {job_id} stalled ({detail}). "
            "Failing closed; requeue the job or open an incident."
        )

    def _emit_stale_gap(
        self,
        job: Dict[str, Any],
        info: Dict[str, Any],
        *,
        job_path: Optional[Path],
        summary: str,
    ) -> Optional[Path]:
        try:
            gap_dir = self.root_dir / "artifacts" / "capability_gaps"
            gap_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(time.time()))
            mission_id = job.get("mission_id") or "unknown"
            job_id = job.get("job_id") or "unknown"
            gap_id = f"{ts}_lab_job_stale_{mission_id}"
            payload = {
                "schema": "ajax.capability_gap.lab_job_stale.v1",
                "capability_gap_id": gap_id,
                "created_at": _utc_now(),
                "capability_family": "lab_job_stale",
                "mission_id": mission_id,
                "job_id": job_id,
                "status": job.get("status"),
                "reason": info.get("reason"),
                "job_age_s": info.get("job_age_s"),
                "heartbeat_age_s": info.get("heartbeat_age_s"),
                "stale_threshold_s": info.get("stale_threshold_s"),
                "heartbeat_threshold_s": info.get("heartbeat_threshold_s"),
                "job_path": str(job_path) if job_path else None,
                "user_explanation": summary,
                "fix_hint": "Requeue the LAB job or open an incident; verify LAB worker heartbeat.",
            }
            out_path = gap_dir / f"{gap_id}.json"
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return out_path
        except Exception:
            return None

    def _fail_stale_job(
        self,
        job: Dict[str, Any],
        *,
        job_path: Optional[Path],
        info: Dict[str, Any],
        now_ts: float,
    ) -> Optional[Path]:
        status = str(job.get("status") or "").upper()
        if status in {"DONE", "FAILED", "CANCELLED"}:
            return None
        stale = job.get("stale") if isinstance(job.get("stale"), dict) else {}
        if stale.get("fail_closed"):
            gap_path = stale.get("gap_path")
            return Path(gap_path) if isinstance(gap_path, str) else None

        summary = self._stale_failure_summary(job, info)
        gap_path = self._emit_stale_gap(job, info, job_path=job_path, summary=summary)
        job_id = job.get("job_id") or "unknown"
        mission_id = job.get("mission_id") or "unknown"
        failure_codes = [str(info.get("reason") or "stale"), "lab_job_stale"]
        evidence_refs = []
        if job_path:
            evidence_refs.append(str(job_path))
        if gap_path:
            evidence_refs.append(str(gap_path))
        result_path = None
        try:
            result = self.write_result(
                job_id=str(job_id),
                mission_id=str(mission_id),
                outcome="FAIL",
                efe_pass=False,
                failure_codes=failure_codes,
                evidence_refs=evidence_refs,
                next_action="requeue_or_open_incident",
                summary=summary,
                risk_level=str(job.get("risk_level") or ""),
                job_kind=str(job.get("job_kind") or job.get("lab_action") or job.get("action") or ""),
                requires_ack=bool(job.get("requires_ack")),
            )
            result_path = result.get("result_path") if isinstance(result, dict) else None
        except Exception:
            result_path = None

        job["status"] = "FAILED"
        job["failure_reason"] = info.get("reason") or "stale"
        job["failure_summary"] = summary
        job["failure_ts"] = now_ts
        stale["fail_closed"] = True
        stale["gap_path"] = str(gap_path) if gap_path else None
        stale["result_path"] = result_path
        job["stale"] = stale

        output_paths = job.get("output_paths") if isinstance(job.get("output_paths"), list) else []
        for item in (result_path, str(gap_path) if gap_path else None):
            if item and item not in output_paths:
                output_paths.append(item)
        job["output_paths"] = output_paths
        if job_path:
            try:
                job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass
        return gap_path

    def annotate_job_staleness(
        self,
        job: Dict[str, Any],
        path: Path,
        *,
        now_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = float(now_ts or time.time())
        info = self.compute_staleness(job, now_ts=now)
        job["job_age_s"] = info["job_age_s"]
        job["heartbeat_age_s"] = info["heartbeat_age_s"]
        if job.get("queued_since_ts") is None:
            job["queued_since_ts"] = self._queued_since(job, now_ts=now)
        stale = job.get("stale") if isinstance(job.get("stale"), dict) else {}
        if info["is_stale"]:
            prev_is_stale = bool(stale.get("is_stale"))
            prev_reason = stale.get("reason")
            if not prev_is_stale or prev_reason != info["reason"]:
                stale["since_ts"] = now
            stale["is_stale"] = True
            stale["reason"] = info["reason"]
            stale["detected_ts"] = now
            gap_path = self._fail_stale_job(job, job_path=path, info=info, now_ts=now)
            if gap_path:
                stale["gap_path"] = str(gap_path)
                stale["fail_closed"] = True
                info["fail_closed"] = True
                info["gap_path"] = str(gap_path)
                info["status"] = job.get("status")
        else:
            stale = {"is_stale": False}
        job["stale"] = stale
        incident_id = self._auto_open_stale_incident(job, info, job_path=path, now_ts=now)
        if incident_id:
            job["incident_id"] = incident_id
            stale["incident_id"] = incident_id
            stale["incident_opened_ts"] = now
        try:
            path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        return info

    def list_jobs(
        self,
        *,
        statuses: Optional[set[str]] = None,
    ) -> list[Dict[str, Any]]:
        records: list[Dict[str, Any]] = []
        for path in sorted(self.jobs_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status") or "").upper()
            if statuses and status not in statuses:
                continue
            records.append({"job": payload, "path": path})
        return records

    def list_queue(self, *, limit: int = 10) -> list[Dict[str, Any]]:
        now = time.time()
        queued = self.list_jobs(statuses={"QUEUED"})
        fg_id = self._foreground_job_id()
        def _key(item: Dict[str, Any]) -> tuple:
            return self._queue_sort_key(item["job"], now_ts=now, foreground_job_id=fg_id)
        queued.sort(key=_key)
        summaries: list[Dict[str, Any]] = []
        for item in queued[: max(0, limit)]:
            job = item["job"]
            changed = self._ensure_lineage_fields(job)
            if changed:
                self._write_job_payload(job, item["path"])
            self._sync_foreground_flag(job, item["path"], foreground_job_id=fg_id)
            job_id = job.get("job_id")
            age_s = self._job_age_seconds(job, now_ts=now)
            summaries.append(
                {
                    "job_id": job_id,
                    "status": job.get("status"),
                    "priority": self._normalize_priority(job.get("priority")),
                    "effective_priority": self._effective_priority(job, now_ts=now),
                    "priority_reason": job.get("priority_reason"),
                    "age_s": age_s,
                    "queued_since_ts": job.get("queued_since_ts"),
                    "objective": job.get("objective"),
                    "objective_norm": job.get("objective_norm"),
                    "intent_fingerprint": job.get("intent_fingerprint"),
                    "root_intent_id": job.get("root_intent_id"),
                    "root_mission_id": job.get("root_mission_id"),
                    "foreground": job.get("foreground"),
                }
            )
        return summaries

    def queue_position(self, job_id: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        queued = self.list_jobs(statuses={"QUEUED"})
        if not queued:
            return None
        fg_id = self._foreground_job_id()
        def _key(item: Dict[str, Any]) -> tuple:
            return self._queue_sort_key(item["job"], now_ts=now, foreground_job_id=fg_id)
        queued.sort(key=_key)
        for idx, item in enumerate(queued):
            if str(item["job"].get("job_id")) == str(job_id):
                return {"position": idx + 1, "depth": len(queued)}
        return None

    def pick_next_job(self) -> Optional[Dict[str, Any]]:
        queued = self.list_jobs(statuses={"QUEUED"})
        if not queued:
            return None
        now = time.time()
        fg_id = self._foreground_job_id()
        def _key(item: Dict[str, Any]) -> tuple:
            return self._queue_sort_key(item["job"], now_ts=now, foreground_job_id=fg_id)
        queued.sort(key=_key)
        if queued:
            item = queued[0]
            changed = self._ensure_lineage_fields(item["job"])
            if changed:
                self._write_job_payload(item["job"], item["path"])
            self._sync_foreground_flag(item["job"], item["path"], foreground_job_id=fg_id)
            return item
        return None

    def _auto_open_stale_incident(
        self,
        job: Dict[str, Any],
        info: Dict[str, Any],
        *,
        job_path: Optional[Path] = None,
        now_ts: Optional[float] = None,
    ) -> Optional[str]:
        try:
            enabled = (os.getenv("AJAX_LAB_STALE_AUTO_INCIDENT") or "").strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            enabled = False
        if not enabled or not info.get("is_stale"):
            return None
        stale = job.get("stale") if isinstance(job.get("stale"), dict) else {}
        if job.get("incident_id") or stale.get("incident_id"):
            return None
        stale_threshold = float(info.get("stale_threshold_s") or 0.0)
        if stale_threshold <= 0:
            return None
        now = float(now_ts or time.time())
        since_ts = _to_float_ts(stale.get("since_ts")) or now
        if now - since_ts < (2.0 * stale_threshold):
            return None
        try:
            from agency.incidents import IncidentReporter
        except Exception:
            return None
        reporter = IncidentReporter(self.root_dir)
        mission_id = job.get("mission_id") or "unknown"
        job_id = job.get("job_id") or "unknown"
        reason = info.get("reason") or "stale"
        summary = f"LAB job stale ({reason}) for mission {mission_id}"
        context = {
            "mission_id": mission_id,
            "job_id": job_id,
            "status": job.get("status"),
            "stale_reason": reason,
            "job_age_s": info.get("job_age_s"),
            "heartbeat_age_s": info.get("heartbeat_age_s"),
            "job_path": str(job_path) if job_path else None,
        }
        attachments = [str(job_path)] if job_path else None
        try:
            return reporter.open_incident(
                kind="lab_stale_job",
                summary=summary,
                context=context,
                remediation=["cancel_job", "requeue_job", "review_lab_runner"],
                attachments=attachments,
            )
        except Exception:
            return None

    def cancel_job(self, job_id_or_path: str, *, reason: Optional[str] = None) -> Tuple[Dict[str, Any], Path]:
        payload, path = self.load_job(job_id_or_path)
        payload["status"] = "CANCELLED"
        payload["last_heartbeat_ts"] = float(time.time())
        payload["cancel_reason"] = reason or "user_cancel"
        payload["cancelled_ts"] = float(time.time())
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return payload, path

    def requeue_job(self, job_id_or_path: str, *, reason: Optional[str] = None) -> Dict[str, Any]:
        payload, path = self.load_job(job_id_or_path)
        original_id = payload.get("job_id") or str(job_id_or_path)
        mission_id = str(payload.get("mission_id") or "mission").strip() or "mission"
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        nonce = int(time.time_ns() % 1_000_000)
        job_id = f"job_{ts}_{mission_id}_requeue_{nonce:06d}"
        priority = self._normalize_priority(payload.get("priority"))
        requeue_payload = {
            "job_id": job_id,
            "mission_id": payload.get("mission_id"),
            "incident_id": payload.get("incident_id"),
            "objective": payload.get("objective"),
            "objective_norm": payload.get("objective_norm"),
            "intent_fingerprint": payload.get("intent_fingerprint"),
            "root_intent_id": payload.get("root_intent_id"),
            "root_mission_id": payload.get("root_mission_id") or payload.get("mission_id"),
            "planned_steps": payload.get("planned_steps"),
            "evidence_expected": payload.get("evidence_expected"),
            "output_paths": payload.get("output_paths"),
            "priority": priority,
            "priority_reason": payload.get("priority_reason") or "requeue",
            "status": "QUEUED",
            "requeue_of": original_id,
            "requeue_reason": reason or "requeue_requested",
        }
        record = self.create_job(requeue_payload)
        payload["status"] = "CANCELLED"
        payload["requeued_to"] = record.get("job_id")
        payload["requeued_ts"] = float(time.time())
        payload["last_heartbeat_ts"] = float(time.time())
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return record

    def _load_result_path(self, result_path: Path) -> Tuple[Dict[str, Any], Path]:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Result payload in {result_path} is not a JSON object.")
        return data, result_path

    def find_result_for_job(self, job_id: str) -> Optional[Tuple[Dict[str, Any], Path]]:
        pattern = f"result_*_{job_id}.json"
        candidates = sorted(self.results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        try:
            return self._load_result_path(candidates[0])
        except Exception:
            return None

    def write_result(
        self,
        *,
        job_id: str,
        mission_id: str,
        outcome: str,
        efe_pass: bool,
        failure_codes: list[str],
        evidence_refs: list[str],
        next_action: str,
        summary: str,
        risk_level: Optional[str] = None,
        job_kind: Optional[str] = None,
        requires_ack: Optional[bool] = None,
        episode_fields: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        outcome_norm = str(outcome).upper()
        risk_n = self._normalize_risk_level(risk_level)
        kind_n = str(job_kind or "").strip().lower()
        ack_required = self._infer_ack_required(
            risk_level=risk_n,
            job_kind=kind_n,
            requires_ack=bool(requires_ack),
        )
        now_ts = float(time.time())
        payload = {
            "job_id": job_id,
            "mission_id": mission_id,
            "outcome": outcome_norm,
            "efe_pass": bool(efe_pass),
            "failure_codes": failure_codes or [],
            "evidence_refs": evidence_refs or [],
            "next_action": next_action,
            "summary": summary,
            "risk_level": risk_n,
            "job_kind": kind_n,
            "ack_required": bool(ack_required),
            "ack_status": "pending" if ack_required else "auto_closed",
            "acknowledged": False if ack_required else True,
            "acknowledged_ts": None if ack_required else now_ts,
            "created_ts": now_ts,
        }
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out_path = self.results_dir / f"result_{ts}_{job_id}.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # LAB episode (canonical, deterministic; written on every result).
        episodes_dir = self.lab_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        episode_path = episodes_dir / f"episode_{ts}_{job_id}.json"
        episode: Dict[str, Any] = {
            "schema": "lab.episode.v1",
            "ts_utc": _utc_now(),
            "job_id": job_id,
            "job_kind": kind_n,
            "explore_state": None,
            "human_active": None,
            "ok": True if outcome_norm in {"PASS"} else False if outcome_norm in {"FAIL", "BLOCKED", "CANCELLED"} else None,
            "failure_codes": failure_codes or [],
            "evidence_paths": {
                "result": str(out_path),
                "job": None,
                "evidence_refs": evidence_refs or [],
                "logs": [],
            },
            "hypothesis": "",
            "delta": summary or "",
            "conclusion": summary or "",
            "tags": [],
        }
        if isinstance(episode_fields, dict):
            for k, v in episode_fields.items():
                if k in {"schema"}:
                    continue
                episode[k] = v
        try:
            episode_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
        status = "DONE" if outcome_norm in {"PASS", "PARTIAL"} else "FAILED"
        try:
            job, job_path = self.load_job(job_id)
            if str(job.get("status") or "").upper() == "CANCELLED":
                # Preserve CANCELLED while still attaching output paths.
                existing = job.get("output_paths") or []
                if not isinstance(existing, list):
                    existing = []
                if str(out_path) not in existing:
                    existing.append(str(out_path))
                job["output_paths"] = existing
                job["last_heartbeat_ts"] = float(time.time())
                job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            # Backfill episode.evidence_paths.job when possible.
            try:
                ep = json.loads(episode_path.read_text(encoding="utf-8"))
                if isinstance(ep, dict):
                    ep_paths = ep.get("evidence_paths") if isinstance(ep.get("evidence_paths"), dict) else {}
                    ep_paths["job"] = str(job_path)
                    ep["evidence_paths"] = ep_paths
                    episode_path.write_text(json.dumps(ep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass
            else:
                self.update_job_status(job_id, status=status, output_paths=[str(out_path)])
        except Exception:
            pass
        return {"result_path": str(out_path), "payload": payload, "episode_path": str(episode_path)}

    def acknowledge_result(self, job_id: str) -> Optional[Path]:
        found = self.find_result_for_job(job_id)
        if not found:
            return None
        payload, path = found
        payload["acknowledged"] = True
        payload["acknowledged_ts"] = float(time.time())
        payload["ack_status"] = "acknowledged"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def list_unacknowledged_results(self, *, limit: int = 5) -> list[Tuple[Dict[str, Any], Path]]:
        results: list[Tuple[Dict[str, Any], Path]] = []
        candidates = sorted(self.results_dir.glob("result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        now_ts = float(time.time())
        try:
            ttl_s = int(os.getenv("AJAX_LAB_ACK_TTL_SECONDS", str(DEFAULT_ACK_TTL_SECONDS)) or DEFAULT_ACK_TTL_SECONDS)
        except Exception:
            ttl_s = DEFAULT_ACK_TTL_SECONDS
        for path in candidates:
            if len(results) >= limit:
                break
            try:
                payload, _ = self._load_result_path(path)
            except Exception:
                continue
            if payload.get("ack_required") is not True:
                continue
            if payload.get("acknowledged") is True:
                continue
            created_ts = _to_float_ts(payload.get("created_ts"))
            if created_ts is None:
                try:
                    created_ts = float(path.stat().st_mtime)
                except Exception:
                    created_ts = now_ts
            if ttl_s > 0 and (now_ts - created_ts) >= ttl_s:
                self._archive_unacked_result(payload, path, now_ts=now_ts)
                continue
            if str(payload.get("ack_status") or "").lower() == "archived_unacked":
                continue
            results.append((payload, path))
        return results

    def list_inbox_results(self, *, limit: int = 10) -> list[Tuple[Dict[str, Any], Path]]:
        results: list[Tuple[Dict[str, Any], Path]] = []
        candidates = sorted(self.results_dir.glob("result_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        now_ts = float(time.time())
        try:
            ttl_s = int(os.getenv("AJAX_LAB_ACK_TTL_SECONDS", str(DEFAULT_ACK_TTL_SECONDS)) or DEFAULT_ACK_TTL_SECONDS)
        except Exception:
            ttl_s = DEFAULT_ACK_TTL_SECONDS
        for path in candidates:
            if len(results) >= limit:
                break
            try:
                payload, _ = self._load_result_path(path)
            except Exception:
                continue
            if payload.get("ack_required") is not True:
                continue
            if payload.get("acknowledged") is True:
                continue
            created_ts = _to_float_ts(payload.get("created_ts"))
            if created_ts is None:
                try:
                    created_ts = float(path.stat().st_mtime)
                except Exception:
                    created_ts = now_ts
            if ttl_s > 0 and (now_ts - created_ts) >= ttl_s:
                self._archive_unacked_result(payload, path, now_ts=now_ts)
                continue
            results.append((payload, path))
        return results

    def create_probe(self, payload: Dict[str, Any], ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        data = dict(payload)
        probe_id = data.get("probe_id") or f"probe_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        data["probe_id"] = probe_id
        data.setdefault("created_utc", _utc_now())
        data["ttl_seconds"] = ttl_seconds or data.get("ttl_seconds") or DEFAULT_PROBE_TTL_SECONDS
        data.setdefault("status", "pending")
        outputs = data.setdefault("outputs", {})
        outputs.setdefault("recipe", None)
        outputs.setdefault("evidence_pack", None)
        outputs.setdefault("transfer_confidence", None)
        outputs.setdefault("status", "awaiting_lab")
        outputs.setdefault("updated_utc", None)
        path = self.probes_dir / f"{probe_id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lab_probe = self.state.setdefault("lab_probe", {})
        lab_probe["active"] = probe_id
        history = lab_probe.get("history") or []
        history.append({"probe_id": probe_id, "created_utc": data["created_utc"], "reason": (data.get("origin") or {}).get("reason")})
        lab_probe["history"] = history[-20:]
        lab_probe["last_probe"] = {"probe_id": probe_id, "path": str(path)}
        self._save_control()
        return {"probe_id": probe_id, "probe_path": str(path), "payload": data}

    def update_probe_outputs(
        self,
        probe_id_or_path: str,
        *,
        recipe: Optional[Dict[str, Any]],
        evidence_pack: Optional[Dict[str, Any]],
        transfer_confidence: float,
        notes: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Path]:
        payload, path = self.load_probe(probe_id_or_path)
        outputs = payload.setdefault("outputs", {})
        outputs["recipe"] = recipe
        outputs["evidence_pack"] = evidence_pack
        outputs["transfer_confidence"] = transfer_confidence
        outputs["updated_utc"] = _utc_now()
        outputs["status"] = "ready_for_review"
        status = "pending_review"
        approved = False
        if transfer_confidence is not None and transfer_confidence >= self.transfer_confidence_threshold and evidence_pack:
            outputs["status"] = "ready_for_reintegration"
            status = "ready_for_reintegration"
            approved = True
        payload["status"] = status
        review = payload.setdefault("review", {})
        review.update(
            {
                "updated_utc": outputs["updated_utc"],
                "approved_for_prod": approved,
                "threshold": self.transfer_confidence_threshold,
                "notes": notes or {},
            }
        )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            origin = payload.get("origin") if isinstance(payload.get("origin"), dict) else {}
            job_id = str(origin.get("lab_job_id") or "").strip()
            mission_id = str(origin.get("mission_id") or "").strip()
            if job_id and mission_id:
                outcome = "FAIL"
                failure_codes: list[str] = []
                efe_pass = False
                if evidence_pack:
                    if transfer_confidence is not None and transfer_confidence >= self.transfer_confidence_threshold:
                        outcome = "PASS"
                        efe_pass = True
                    else:
                        outcome = "PARTIAL"
                        failure_codes.append("transfer_confidence_below_threshold")
                        efe_pass = False
                else:
                    failure_codes.append("missing_evidence_pack")
                evidence_refs: list[str] = []
                if isinstance(evidence_pack, dict):
                    for key in ("evidence_refs", "refs", "paths", "artifacts"):
                        val = evidence_pack.get(key)
                        if isinstance(val, list):
                            evidence_refs.extend([str(v) for v in val if v])
                next_action = "review_for_reintegration" if outcome == "PASS" else "collect_more_evidence"
                if outcome == "FAIL":
                    next_action = "replan_or_cancel"
                summary = (
                    "LAB probe listo para reintegracion." if outcome == "PASS" else
                    "LAB probe parcial; falta evidencia o confianza." if outcome == "PARTIAL" else
                    "LAB probe fallido; requiere replanteo."
                )
                self.write_result(
                    job_id=job_id,
                    mission_id=mission_id,
                    outcome=outcome,
                    efe_pass=efe_pass,
                    failure_codes=failure_codes,
                    evidence_refs=evidence_refs,
                    next_action=next_action,
                    summary=summary,
                )
        except Exception:
            pass
        return payload, path

    def probe_ready_for_prod(self, payload: Dict[str, Any]) -> bool:
        outputs = payload.get("outputs") or {}
        return bool(
            outputs.get("recipe")
            and outputs.get("evidence_pack")
            and isinstance(outputs.get("transfer_confidence"), (int, float))
            and float(outputs["transfer_confidence"]) >= self.transfer_confidence_threshold
        )
