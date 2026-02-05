from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore

from agency.p0_leann import build_p0_context


@dataclass
class RunSummary:
    run_id: str
    run_dir: Path
    operational_status: str
    gate_decision: Dict[str, Any]
    published_record: Optional[Path]
    capability_gap: Optional[Path]


class ExperimentRunError(RuntimeError):
    pass


_MISSING = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _resolve_path(base: Path, target: str) -> Path:
    if len(target) > 1 and target[1] == ":":
        return Path(target)
    path = Path(target)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _derive_objective(envelope: Dict[str, Any], envelope_dir: Path) -> str:
    objective = envelope.get("objective")
    if isinstance(objective, str) and objective.strip():
        return objective.strip()
    title = envelope.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    efe_path = envelope.get("efe_path")
    if isinstance(efe_path, str) and efe_path.strip():
        try:
            efe = _read_json(_resolve_path(envelope_dir, efe_path))
        except Exception:
            efe = {}
        desc = efe.get("description")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()
        invariants = efe.get("invariants")
        if isinstance(invariants, list) and invariants:
            parts = [str(item).strip() for item in invariants if str(item).strip()]
            if parts:
                return "; ".join(parts)
        assertions = efe.get("assertions")
        if isinstance(assertions, list) and assertions:
            parts = []
            for item in assertions:
                if isinstance(item, dict):
                    ref = item.get("id") or item.get("path") or ""
                else:
                    ref = str(item)
                ref = str(ref).strip()
                if ref:
                    parts.append(ref)
            if parts:
                return "; ".join(parts)
    fallback = envelope.get("id") or ""
    return str(fallback)

def _get_by_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if not part:
            continue
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _compare(op: str, actual: Any, expected: Any) -> bool:
    if op == "exists":
        return actual is not _MISSING
    if actual is _MISSING:
        return False
    if op == "eq":
        return actual == expected
    if op == "neq":
        return actual != expected
    if op == "gte":
        try:
            return actual >= expected
        except Exception:
            return False
    if op == "lte":
        try:
            return actual <= expected
        except Exception:
            return False
    if op == "contains":
        try:
            return expected in actual
        except Exception:
            return False
    return False


def _safe_signal_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


def _derive_learning_signals(verification: Dict[str, Any]) -> list[str]:
    signals = []
    seen = set()

    def _add(value: str) -> None:
        if value and value not in seen:
            signals.append(value)
            seen.add(value)

    ok = verification.get("ok", False)
    failures = verification.get("failures") or []
    if ok:
        _add("efe_pass")
    else:
        _add("verify_efe_strict")

    for failure in failures:
        if failure in {"efe_missing", "efe_empty"}:
            _add(failure)
            continue
        if failure.startswith("assertion_failed:"):
            raw_id = failure.split(":", 1)[1]
            cleaned = _safe_signal_id(raw_id)
            if cleaned == "probe_ok":
                _add("probe_failed")
            elif cleaned == "latency_ok":
                _add("latency_too_high")
            elif cleaned:
                _add(f"efe_failed_{cleaned}")
            else:
                _add("efe_failed")
            continue
        if failure.startswith("missing_source:"):
            _add("missing_source")
            continue
        _add("efe_failed")

    return signals


def verify_efe(efe: Dict[str, Any], sources: Dict[str, Any], missing: bool) -> Dict[str, Any]:
    failures = []
    if missing:
        return {"ok": False, "failures": ["efe_missing"], "checked": 0}
    assertions = efe.get("assertions", [])
    if not assertions:
        return {"ok": False, "failures": ["efe_empty"], "checked": 0}
    checked = 0
    for assertion in assertions:
        checked += 1
        source_name = assertion.get("source")
        source_obj = sources.get(source_name)
        if source_obj is None:
            failures.append(f"missing_source:{source_name}")
            continue
        path = assertion.get("path", "")
        actual = _get_by_path(source_obj, path)
        op = assertion.get("op", "eq")
        expected = assertion.get("expected")
        if not _compare(op, actual, expected):
            failures.append(f"assertion_failed:{assertion.get('id', path)}")
    return {"ok": len(failures) == 0, "failures": failures, "checked": checked}


def _parse_utc(value: str) -> Optional[datetime]:
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except Exception:
        return None


def _break_glass_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"ok": False, "expires_utc": None}
    raw = path.read_text(encoding="utf-8").strip()
    expires = _parse_utc(raw)
    if not expires:
        return {"ok": False, "expires_utc": None}
    now = datetime.now(timezone.utc)
    return {"ok": now < expires, "expires_utc": expires.strftime("%Y-%m-%dT%H:%M:%SZ")}


class ScientificMethodRunner:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.experiments_dir = root_dir / "experiments"
        self.runs_dir = self.experiments_dir / "runs"
        self.published_dir = self.experiments_dir / "published"
        self.capability_gaps_dir = root_dir / "artifacts" / "capability_gaps"
        self.open_gaps_dir = self.capability_gaps_dir / "open"

    def _rel_path(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.root_dir)
        except ValueError:
            rel = path
        return str(rel).replace("\\", "/")

    def run(
        self,
        envelope_path: Path,
        promote_task: Optional[str] = None,
        allow_promotion: bool = False,
    ) -> RunSummary:
        envelope = _read_json(envelope_path)
        envelope_id = envelope.get("id")
        rail = envelope.get("rail")
        kind = envelope.get("kind")
        efe_path_raw = envelope.get("efe_path")
        if not envelope_id or not rail or not kind or not efe_path_raw:
            raise ExperimentRunError("envelope missing required fields")

        run_id = f"{envelope_id}_{_utc_compact()}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.published_dir.mkdir(parents=True, exist_ok=True)
        self.open_gaps_dir.mkdir(parents=True, exist_ok=True)

        phase_index = {
            "schema_version": "0.1",
            "run_id": run_id,
            "envelope_id": envelope_id,
            "created_utc": _utc_now(),
            "phases": {
                "A": {"name": "p0_context", "status": "pending"},
                "B": {"name": "efe", "status": "pending"},
                "C": {"name": "gate", "status": "pending"},
                "D": {"name": "execute", "status": "pending"},
                "E": {"name": "verify", "status": "pending"},
                "SPL": {"name": "epistemic_feedback", "status": "pending"},
                "F": {"name": "evolve", "status": "pending"}
            }
        }
        phase_index_path = run_dir / "phase_index.json"
        _write_json(phase_index_path, phase_index)

        def _update_phase(phase: str, status: str) -> None:
            phase_index["phases"][phase]["status"] = status
            _write_json(phase_index_path, phase_index)

        # Phase A
        envelope_copy = run_dir / "experiment_envelope.json"
        _write_json(envelope_copy, envelope)
        objective_query = _derive_objective(envelope, envelope_path.parent)
        try:
            p0_context = build_p0_context(
                self.root_dir,
                envelope_id=envelope_id,
                objective_query=objective_query,
            )
        except Exception as exc:
                p0_context = {
                    "schema_version": "0.1",
                    "envelope_id": envelope_id,
                    "created_utc": _utc_now(),
                    "scoring_method": "token_overlap_v0",
                    "score_threshold": 0.2,
                    "objective_query": objective_query,
                    "source_resolution": [
                        {
                            "kind": "history",
                            "resolved_from": "root_dir",
                            "path": str(self.root_dir / "pre" / "ajax_history_v1.ndjson"),
                        },
                        {
                            "kind": "pepitas",
                            "resolved_from": "root_dir",
                            "path": str(self.root_dir / "Pepitas_Index.json"),
                        },
                    ],
                    "sources_checked": [],
                    "refs_used": [],
                    "gaps_detected": [
                        {
                            "kind": "unknown",
                            "note": f"p0_error: {exc}",
                            "signals": [],
                            "ref_ids": [],
                        }
                    ],
                    "novelty": True,
                    "summary": "P0 failed; no sources read.",
            }
        _write_json(run_dir / "p0_context.json", p0_context)
        _update_phase("A", "done")

        # Phase B
        efe_missing = False
        try:
            efe_path = _resolve_path(envelope_path.parent, str(efe_path_raw))
            efe = _read_json(efe_path)
            _write_json(run_dir / "efe.json", efe)
            _update_phase("B", "done")
        except Exception:
            efe_missing = True
            efe = {
                "schema_version": "0.1",
                "envelope_id": envelope_id,
                "created_utc": _utc_now(),
                "assertions": [],
                "error": "efe_missing"
            }
            _write_json(run_dir / "efe.json", efe)
            _update_phase("B", "failed")

        # Phase C
        break_glass_path = envelope.get("break_glass_path") or str(self.experiments_dir / "break_glass.flag")
        break_status = _break_glass_status(Path(break_glass_path))
        break_required = bool(rail == "prod" and kind == "experiment")
        gate_decision = {
            "schema_version": "0.1",
            "envelope_id": envelope_id,
            "created_utc": _utc_now(),
            "rail": rail,
            "decision": "allow",
            "reasons": [],
            "break_glass_required": break_required,
            "break_glass_ok": break_status.get("ok", False)
        }
        if efe_missing:
            gate_decision["decision"] = "block"
            gate_decision["reasons"].append("efe_missing")
        elif break_required and not break_status.get("ok", False):
            gate_decision["decision"] = "block"
            gate_decision["reasons"].append("break_glass_required")
        _write_json(run_dir / "gate_decision.json", gate_decision)
        _update_phase("C", "done")

        # Phase D
        execution_trace: Dict[str, Any]
        if gate_decision["decision"] == "block":
            execution_trace = {
                "schema_version": "0.1",
                "envelope_id": envelope_id,
                "created_utc": _utc_now(),
                "status": "blocked",
                "observations": {},
                "notes": "blocked by gate"
            }
            _update_phase("D", "skipped")
        else:
            fixture_path = envelope.get("execution_fixture_path")
            if fixture_path:
                fixture = _read_json(_resolve_path(envelope_path.parent, fixture_path))
            else:
                fixture = {}
            execution_trace = {
                "schema_version": "0.1",
                "envelope_id": envelope_id,
                "created_utc": _utc_now(),
                "status": fixture.get("status", "fixture"),
                "observations": fixture.get("observations", {}),
                "steps": fixture.get("steps", []),
                "notes": fixture.get("notes", "fixture")
            }
            _update_phase("D", "done")
        _write_json(run_dir / "execution_trace.json", execution_trace)

        # Phase E
        verification = verify_efe(
            efe,
            sources={
                "execution_trace": execution_trace,
                "p0_context": p0_context
            },
            missing=efe_missing
        )
        state_delta = {
            "schema_version": "0.1",
            "envelope_id": envelope_id,
            "created_utc": _utc_now(),
            "verification": verification,
            "observations": execution_trace.get("observations", {}),
            "notes": "verify_efe only"
        }
        _write_json(run_dir / "state_delta.json", state_delta)
        operational_status = "PASS" if verification.get("ok") else "FAIL"
        _write_json(run_dir / "operational_status.json", {
            "schema_version": "0.1",
            "envelope_id": envelope_id,
            "created_utc": _utc_now(),
            "status": operational_status,
            "checked": verification.get("checked", 0),
            "failures": verification.get("failures", [])
        })
        _update_phase("E", "done")

        evidence_refs = [
            self._rel_path(run_dir / "state_delta.json"),
            self._rel_path(run_dir / "execution_trace.json"),
        ]
        learning_signals = _derive_learning_signals(verification)

        # Phase SPL
        spl = envelope.get("spl") or "LOW"
        epistemic_feedback = {
            "schema_version": "0.1",
            "envelope_id": envelope_id,
            "created_utc": _utc_now(),
            "spl": spl,
            "learning_signals": learning_signals,
            "evidence_refs": evidence_refs,
            "notes": "meta only"
        }
        _write_json(run_dir / "epistemic_feedback.json", epistemic_feedback)
        _update_phase("SPL", "done")

        # Phase F
        publication_record = None
        capability_gap = None
        if operational_status == "PASS":
            promoted = False
            promotion_task = None
            promotion_note = "promotion skipped"
            if promote_task and allow_promotion:
                promoted = self._promote_task(promote_task)
                promotion_task = promote_task
                promotion_note = "promotion applied" if promoted else "promotion failed"
            publication_record = run_dir / "publication_record.json"
            record = {
                "schema_version": "0.1",
                "envelope_id": envelope_id,
                "created_utc": _utc_now(),
                "outcome": "pass",
                "promoted": promoted,
                "promotion_task": promotion_task,
                "notes": promotion_note,
                "evidence_refs": evidence_refs,
                "artifacts": [
                    str((run_dir / "state_delta.json").name),
                    str((run_dir / "execution_trace.json").name)
                ]
            }
            _write_json(publication_record, record)
            published_path = self.published_dir / f"{run_id}_publication_record.json"
            _write_json(published_path, record)
        else:
            capability_gap = run_dir / "capability_gap.json"
            summary = "EFE failed"
            failures = verification.get("failures") or []
            if failures:
                summary = "EFE failed: " + ", ".join(failures)
            fingerprint = hashlib.sha256(
                (envelope_id + "|" + summary + "|" + "|".join(failures)).encode("utf-8")
            ).hexdigest()
            gap_payload = {
                "schema_version": "0.1",
                "envelope_id": envelope_id,
                "created_utc": _utc_now(),
                "gap_id": f"gap_{run_id}",
                "summary": summary,
                "fingerprint": fingerprint,
                "evidence": ["state_delta.json", "execution_trace.json"],
                "evidence_refs": evidence_refs
            }
            _write_json(capability_gap, gap_payload)
            dedup_path, deduped = self._dedup_gap(fingerprint, gap_payload)
            if deduped and dedup_path:
                gap_payload["deduped_to"] = str(dedup_path)
                _write_json(capability_gap, gap_payload)
        _update_phase("F", "done")

        return RunSummary(
            run_id=run_id,
            run_dir=run_dir,
            operational_status=operational_status,
            gate_decision=gate_decision,
            published_record=publication_record,
            capability_gap=capability_gap,
        )

    def _dedup_gap(self, fingerprint: str, payload: Dict[str, Any]) -> tuple[Optional[Path], bool]:
        for gap_file in self.open_gaps_dir.glob("*.json"):
            try:
                existing = _read_json(gap_file)
            except Exception:
                continue
            if existing.get("fingerprint") == fingerprint:
                return gap_file, True
        out_path = self.open_gaps_dir / f"{payload['gap_id']}.json"
        _write_json(out_path, payload)
        return out_path, False

    def _promote_task(self, task_name: str) -> bool:
        registry_path = self.root_dir / "tasks_registry.yaml"
        if yaml is None:
            return False
        if registry_path.exists():
            data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        else:
            data = {}
        entry = data.get(task_name, {})
        entry["maturity"] = "stable"
        entry["last_verified"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        data[task_name] = entry
        registry_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=False),
            encoding="utf-8"
        )
        return True
