from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agency.experiment_cancellation import get_experiment_record, is_experiment_cancelled
from agency.explore_policy import dummy_display_ok, evaluate_explore_state, load_explore_policy, state_rules
from agency.human_permission import read_human_permission_status
from agency.lab_control import LabStateStore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "config" / "lab_org_manifest.yaml"
DEFAULT_EXPLORE_POLICY = ROOT / "config" / "explore_policy.yaml"
MAINTENANCE_ALLOWLIST = (
    "providers_probe",
    "capabilities_refresh",
    "doctor_*",
    "health_*",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
    return payload if isinstance(payload, dict) else None


def _load_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema": "ajax.lab_org_manifest.v1", "micro_challenges": []}
    if yaml:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {"micro_challenges": []}
        except Exception:
            pass
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"micro_challenges": []}
    except Exception:
        return {"micro_challenges": []}


def _is_destructive(kind: str, tags: List[str], policy: Dict[str, Any]) -> bool:
    pol = policy.get("policy") if isinstance(policy.get("policy"), dict) else {}
    destructive = pol.get("destructive_job_kinds")
    destructive_set = {str(x).strip().lower() for x in destructive} if isinstance(destructive, list) else set()
    if str(kind or "").strip().lower() in destructive_set:
        return True
    tag_set = {str(t).strip().lower() for t in tags if str(t).strip()}
    return "destructive" in tag_set or "high_risk" in tag_set


def _away_ui_allowed(kind: str, policy: Dict[str, Any]) -> bool:
    pol = policy.get("policy") if isinstance(policy.get("policy"), dict) else {}
    allowed = pol.get("away_ui_allowed_job_kinds")
    allowed_set = {str(x).strip().lower() for x in allowed} if isinstance(allowed, list) else set()
    if not allowed_set:
        return False
    return str(kind or "").strip().lower() in allowed_set


def _preempt_lab_org_ui_jobs(root: Path, store: LabStateStore, *, reason: str) -> List[str]:
    cancelled: List[str] = []
    for item in store.list_jobs(statuses={"QUEUED", "RUNNING"}):
        job = item.get("job") or {}
        job_id = str(job.get("job_id") or "").strip()
        if not job_id:
            continue
        if str(job.get("mission_id") or "") != "lab_org":
            continue
        params = job.get("params") if isinstance(job.get("params"), dict) else {}
        ui_intrusive = bool(params.get("ui_intrusive")) if "ui_intrusive" in params else False
        if not ui_intrusive:
            continue
        try:
            store.cancel_job(job_id, reason=reason)
            cancelled.append(job_id)
        except Exception:
            continue
    return cancelled


def _load_org_runtime(store: LabStateStore) -> Dict[str, Any]:
    data = _read_json(store.org_state_path)
    snap = data.get("snapshot") if isinstance(data, dict) else None
    return snap if isinstance(snap, dict) else {}


def _write_org_runtime(store: LabStateStore, runtime: Dict[str, Any]) -> None:
    store._write_lab_org_state(runtime, _utc_now())  # type: ignore[attr-defined]


def _normalize_challenges(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip()
        kind = str(item.get("job_kind") or "").strip()
        if not cid or not kind:
            continue
        enabled = bool(item.get("enabled", True))
        ui_intrusive = bool(item.get("ui_intrusive", False))
        try:
            cadence_s = int(item.get("cadence_s") or 0)
        except Exception:
            cadence_s = 0
        try:
            budget_s = int(item.get("budget_s") or 0)
        except Exception:
            budget_s = 0
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        out.append(
            {
                "id": cid,
                "job_kind": kind,
                "enabled": enabled,
                "ui_intrusive": ui_intrusive,
                "cadence_s": max(0, cadence_s),
                "budget_s": max(0, budget_s),
                "tags": [str(t) for t in tags if str(t)],
            }
        )
    return out


def _due_challenges(challenges: List[Dict[str, Any]], last_job_ts: Dict[str, Any], *, now: float) -> List[Dict[str, Any]]:
    due: List[Dict[str, Any]] = []
    for item in challenges:
        if not item.get("enabled", True):
            continue
        cid = str(item.get("id") or "")
        cadence_s = int(item.get("cadence_s") or 0)
        prev = last_job_ts.get(cid)
        try:
            prev_ts = float(prev) if prev is not None else None
        except Exception:
            prev_ts = None
        if cadence_s <= 0:
            due.append(item)
            continue
        if prev_ts is None or now - prev_ts >= float(cadence_s):
            due.append(item)
    return due


def _pick_next(due: List[Dict[str, Any]], last_job_ts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not due:
        return None
    # pick least recently run (or never run)
    def key(item: Dict[str, Any]) -> Tuple[float, str]:
        cid = str(item.get("id") or "")
        prev = last_job_ts.get(cid)
        try:
            prev_ts = float(prev) if prev is not None else 0.0
        except Exception:
            prev_ts = 0.0
        return (prev_ts, cid)

    return sorted(due, key=key)[0]


def _is_maintenance_job_kind(kind: str) -> bool:
    normalized = str(kind or "").strip().lower()
    if not normalized:
        return False
    if normalized in {"providers_probe", "capabilities_refresh"}:
        return True
    for prefix in ("doctor", "health"):
        if normalized == prefix:
            return True
        for sep in ("_", "-", ":", "/", "."):
            if normalized.startswith(f"{prefix}{sep}"):
                return True
    return False


def lab_org_tick(
    root_dir: Path,
    *,
    manifest_path: Optional[Path] = None,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    LAB_ORG tick: selects at most one micro-challenge (declarative manifest) and enqueues it as a LAB job.
    Also emits a scheduler receipt under artifacts/lab_org/<ts>/receipt.json.
    """
    root = Path(root_dir)
    store = LabStateStore(root)
    manifest_file = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST

    now = time.time()
    ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
    receipt_dir = (Path(out_base) if out_base else (root / "artifacts" / "lab_org" / ts_label))
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / "receipt.json"

    runtime = _load_org_runtime(store)
    last_job_ts = runtime.get("last_job_ts")
    if not isinstance(last_job_ts, dict):
        last_job_ts = {}
    prev_state = None
    explore_state_prev = runtime.get("explore_state")
    if isinstance(explore_state_prev, dict):
        prev_state = explore_state_prev.get("state")
    elif isinstance(explore_state_prev, str):
        prev_state = explore_state_prev

    receipt: Dict[str, Any] = {
        "schema": "ajax.lab_org.receipt.v1",
        "ts_utc": _utc_now(),
        "manifest_path": str(manifest_file),
        "explore_policy_path": str(DEFAULT_EXPLORE_POLICY),
        "lab_org_status": (store.state.get("lab_org") or {}).get("status"),
        "explore_state": None,
        "state": None,
        "trigger": None,
        "human_active": None,
        "mode": "NORMAL",
        "allowlist_used": [],
        "selected_job": None,
        "reason": None,
        "skipped_reason": None,
        "experiment_id": None,
        "cancelled_experiments_skipped": [],
        "actionable_hint": None,
        "forced_non_ui_due": False,
        "last_job_ts": dict(last_job_ts),
        "enqueued": False,
        "enqueued_job_id": None,
        "job_kind": None,
        "preempted_job": None,
        "preemption_path": None,
        "receipt_path": str(receipt_path),
    }

    try:
        if not store.is_lab_org_running():
            receipt["skipped_reason"] = "lab_org_not_running"
            receipt["actionable_hint"] = "Run: python bin/ajaxctl lab start"
            return receipt

        policy = load_explore_policy(root)
        explore_eval = evaluate_explore_state(root, policy=policy, prev_state=str(prev_state) if prev_state else None, now_ts=now)
        receipt["explore_state"] = explore_eval
        receipt["state"] = explore_eval.get("state")
        receipt["trigger"] = explore_eval.get("trigger")
        receipt["human_active"] = bool(explore_eval.get("human_active"))
        if receipt["human_active"]:
            receipt["mode"] = "MAINTENANCE_ONLY"
            receipt["allowlist_used"] = list(MAINTENANCE_ALLOWLIST)

        if receipt["trigger"] == "AWAY->HUMAN_DETECTED":
            cancelled = _preempt_lab_org_ui_jobs(root, store, reason="preempt_human_detected")
            if cancelled:
                receipt["preempted_job"] = cancelled if len(cancelled) != 1 else cancelled[0]
            preempt_ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now))
            preempt_path = root / "artifacts" / "lab" / f"preemption_{preempt_ts}.json"
            preempt_payload = {
                "schema": "ajax.lab.preemption.v1",
                "ts_utc": _utc_now(),
                "trigger": receipt["trigger"],
                "prev_state": "AWAY",
                "state": "HUMAN_DETECTED",
                "cancelled_job_ids": cancelled,
                "reason": "human_detected",
            }
            try:
                preempt_path.parent.mkdir(parents=True, exist_ok=True)
                preempt_path.write_text(json.dumps(preempt_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                receipt["preemption_path"] = str(preempt_path)
            except Exception:
                pass

        manifest = _load_manifest(manifest_file)
        challenges = _normalize_challenges(manifest.get("micro_challenges"))
        due = _due_challenges(challenges, last_job_ts, now=now)

        # Canonical safety invariant: no destructive ops in exploration unless explicitly allowed.
        pol = policy.get("policy") if isinstance(policy.get("policy"), dict) else {}
        allow_destructive = bool(pol.get("allow_destructive_ops_when_away", False))
        if not allow_destructive:
            due = [c for c in due if not _is_destructive(str(c.get("job_kind") or ""), list(c.get("tags") or []), policy)]

        state = str(receipt.get("state") or "HUMAN_DETECTED")
        if state == "HUMAN_DETECTED":
            if receipt.get("human_active"):
                due = [c for c in due if _is_maintenance_job_kind(str(c.get("job_kind") or ""))]
                if not due:
                    receipt["skipped_reason"] = "maintenance_only_no_due"
                    receipt["reason"] = "maintenance_only_no_due"
                    return receipt
            # Block UI intrusive by default while human detected.
            if not due:
                receipt["skipped_reason"] = "no_due_jobs"
                receipt["reason"] = "no_due_jobs"
                return receipt
            allowed = [c for c in due if not bool(c.get("ui_intrusive"))]
            # Optional override: allow ui_intrusive only if lease present + config allows it.
            human_cfg = state_rules(policy, "HUMAN_DETECTED")
            allow_ui_with_lease = bool(human_cfg.get("allow_ui_intrusive_with_lease", False))
            if allow_ui_with_lease:
                perm = read_human_permission_status(root)
                if perm.get("ok"):
                    allowed = due
            if not allowed:
                receipt["skipped_reason"] = "blocked_human_detected_only_ui_due"
                receipt["reason"] = "human_detected"
                return receipt
            due = allowed
        else:
            # AWAY: allow ui_intrusive only if dummy display is verified ok AND job_kind is allowlisted.
            away_cfg = state_rules(policy, "AWAY")
            force_non_ui = bool(away_cfg.get("force_non_ui_due", False))
            non_ui_enabled = [c for c in challenges if c.get("enabled", True) and not bool(c.get("ui_intrusive"))]
            if not due and force_non_ui and non_ui_enabled:
                fallback = _pick_next(non_ui_enabled, last_job_ts)
                if fallback:
                    due = [fallback]
                    receipt["forced_non_ui_due"] = True
            if not due:
                receipt["skipped_reason"] = "no_due_jobs"
                receipt["reason"] = "no_due_jobs"
                return receipt
            require_dummy = bool(away_cfg.get("require_dummy_display_ok", True))
            dummy_ok = dummy_display_ok(root)
            filtered: List[Dict[str, Any]] = []
            for c in due:
                if not bool(c.get("ui_intrusive")):
                    filtered.append(c)
                    continue
                kind = str(c.get("job_kind") or "")
                if not _away_ui_allowed(kind, policy):
                    continue
                if require_dummy and not dummy_ok:
                    continue
                filtered.append(c)
            if not filtered and force_non_ui and non_ui_enabled:
                fallback = _pick_next(non_ui_enabled, last_job_ts)
                if fallback:
                    filtered = [fallback]
                    receipt["forced_non_ui_due"] = True
            if not filtered:
                receipt["skipped_reason"] = "blocked_away_no_allowed_jobs"
                receipt["reason"] = "away_policy_block"
                if require_dummy and not dummy_ok and any(bool(c.get("ui_intrusive")) for c in due):
                    receipt["skipped_reason"] = "blocked_away_dummy_display_required"
                    receipt["reason"] = "dummy_display_required"
                return receipt
            due = filtered

        non_cancelled_due: List[Dict[str, Any]] = []
        cancelled_due: List[str] = []
        for item in due:
            experiment_id = str(item.get("id") or "").strip().upper()
            if experiment_id and is_experiment_cancelled(root, experiment_id):
                cancelled_due.append(experiment_id)
                continue
            non_cancelled_due.append(item)
        if cancelled_due:
            receipt["cancelled_experiments_skipped"] = sorted(set(cancelled_due))
        due = non_cancelled_due
        if not due:
            if cancelled_due:
                experiment_id = cancelled_due[0]
                row = get_experiment_record(root, experiment_id) or {}
                receipt["skipped_reason"] = "experiment_cancelled"
                receipt["reason"] = "experiment_cancelled"
                receipt["experiment_id"] = experiment_id
                if row.get("reason"):
                    receipt["actionable_hint"] = str(row.get("reason"))
                return receipt
            receipt["skipped_reason"] = "no_due_jobs"
            receipt["reason"] = "no_due_jobs"
            return receipt

        picked = _pick_next(due, last_job_ts)
        if not picked:
            receipt["skipped_reason"] = "no_due_jobs"
            return receipt

        cid = str(picked["id"])
        kind = str(picked["job_kind"])
        receipt["selected_job"] = cid
        receipt["job_kind"] = kind
        receipt["reason"] = "scheduled"

        payload = {
            "mission_id": "lab_org",
            "objective": f"lab_org:{cid}",
            "job_kind": kind,
            "risk_level": "low",
            "requires_ack": False,
            "priority": 5,
            "priority_reason": "lab_org",
            "params": {
                "lab_org_id": cid,
                "ui_intrusive": bool(picked.get("ui_intrusive")),
                "budget_s": int(picked.get("budget_s") or 0),
                "tags": list(picked.get("tags") or []),
                "display_target": "dummy" if bool(picked.get("ui_intrusive")) else "any",
            },
        }
        created = store.enqueue_job(payload)
        receipt["enqueued"] = True
        receipt["enqueued_job_id"] = created.get("job_id")

        last_job_ts[cid] = float(now)
        runtime["last_job_ts"] = last_job_ts
        runtime["last_job_id"] = created.get("job_id")
        runtime["last_job_utc"] = _utc_now()
        runtime["manifest_path"] = str(manifest_file)
        runtime["explore_policy_path"] = str(DEFAULT_EXPLORE_POLICY)
        runtime["explore_state"] = explore_eval
        _write_org_runtime(store, runtime)
        receipt["last_job_ts"] = dict(last_job_ts)
        return receipt
    finally:
        try:
            receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(prog="lab_org", description="LAB_ORG scheduler tick (declarative).")
    parser.add_argument("--root", default=".", help="Root del repo (AJAX_HOME).")
    parser.add_argument("--manifest", default=None, help="Override manifest path.")
    parser.add_argument("--out-base", default=None, help="Override output base directory.")
    args = parser.parse_args(argv)

    root = Path(args.root)
    manifest = Path(args.manifest) if args.manifest else None
    out_base = Path(args.out_base) if args.out_base else None
    receipt = lab_org_tick(root, manifest_path=manifest, out_base=out_base)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt.get("enqueued") or receipt.get("skipped_reason") else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
