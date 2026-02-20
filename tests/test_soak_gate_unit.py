from __future__ import annotations

import json
import time
from pathlib import Path

from agency import soak_gate


def _label(ts: float) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts))


def _make_root(tmp_path: Path, *, name: str = "ajax-kernel") -> Path:
    root = tmp_path / name
    (root / "agency").mkdir(parents=True, exist_ok=True)
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "health").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "lab").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "lab_org").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "lab" / "results").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "receipts").mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("# test\n", encoding="utf-8")
    (root / "bin" / "ajaxctl").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    return root


def _write_provider_status(root: Path, updated_ts: float) -> None:
    payload = {
        "schema": "ajax.providers_status.v1",
        "updated_ts": float(updated_ts),
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(updated_ts)),
    }
    path = root / "artifacts" / "health" / "providers_status.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_worker(root: Path, *, ts: float, pid: int = 1234) -> None:
    (root / "artifacts" / "lab" / "worker.pid").write_text(str(pid), encoding="utf-8")
    hb = {
        "ts": float(ts),
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "status": "READY",
        "worker_id": "test_worker",
    }
    (root / "artifacts" / "lab" / "heartbeat.json").write_text(
        json.dumps(hb, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_lab_org_receipt(root: Path, *, ts: float, enqueued: bool, status: str = "RUNNING") -> Path:
    folder = root / "artifacts" / "lab_org" / time.strftime("%Y%m%d-%H%M%S", time.gmtime(ts))
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "ajax.lab_org.receipt.v1",
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "lab_org_status": status,
        "enqueued": bool(enqueued),
        "enqueued_job_id": "job_test" if enqueued else None,
    }
    path = folder / "receipt.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_result(root: Path, *, ts: float) -> Path:
    path = root / "artifacts" / "lab" / "results" / f"result_{_label(ts)}_job_test.json"
    payload = {
        "job_id": "job_test",
        "created_ts": float(ts),
        "completed_ts": float(ts),
        "outcome": "PASS",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_manifest(root: Path, *, work_declared: bool) -> None:
    if work_declared:
        content = (
            "schema: ajax.lab_org_manifest.v1\n"
            "micro_challenges:\n"
            "  - id: capabilities_refresh\n"
            "    job_kind: capabilities_refresh\n"
            "    enabled: true\n"
        )
    else:
        content = "schema: ajax.lab_org_manifest.v1\nmicro_challenges: []\n"
    (root / "config" / "lab_org_manifest.yaml").write_text(content, encoding="utf-8")


def _write_microfilm(
    root: Path,
    *,
    ts: float,
    overall_ok: bool,
    checks: list[dict],
    actionable_hint: str = "",
) -> Path:
    path = root / "artifacts" / "receipts" / f"microfilm_check_{_label(ts)}.json"
    payload = {
        "schema": "ajax.microfilm_compliance.v1",
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "overall_ok": bool(overall_ok),
        "actionable_hint": actionable_hint,
        "checks": checks,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_anchor(root: Path, *, ts: float, mismatches: list[dict]) -> Path:
    path = root / "artifacts" / "receipts" / f"anchor_preflight_{_label(ts)}.json"
    payload = {
        "schema": "ajax.anchor_preflight.v1",
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
        "mismatches": mismatches,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _setup_alive_safe_baseline(tmp_path: Path, monkeypatch) -> tuple[Path, float]:
    now = 1_800_000_000.0
    root = _make_root(tmp_path)
    _write_provider_status(root, now - 10)
    _write_worker(root, ts=now - 5, pid=4321)
    _write_lab_org_receipt(root, ts=now - 5, enqueued=False, status="RUNNING")
    _write_manifest(root, work_declared=False)
    _write_microfilm(
        root,
        ts=now - 4,
        overall_ok=True,
        checks=[{"name": "doctor_anchor", "ok": True, "code": "ANCHOR_OK", "actionable_hint": ""}],
    )
    monkeypatch.setattr(soak_gate, "pid_running", lambda _pid: True)
    return root, now


def test_soak_pass_no_work_declared(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    out = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now)
    assert out["ok"] is True
    assert out["signals"]["effective"]["code"] == "NO_WORK_DECLARED"
    assert Path(out["receipt_path"]).exists()
    assert Path(out["report_path"]).exists()


def test_soak_alive_fails_when_provider_stale(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    _write_provider_status(root, now - 9999)
    out = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now)
    assert out["ok"] is False
    assert out["signals"]["alive"]["code"] == "ALIVE_PROVIDER_STALE"
    alive_detail = out["signals"]["alive"]["detail"]
    assert alive_detail.get("refresh_owner") == "lab_worker.providers_probe"
    assert alive_detail.get("last_refresh_age_s") == alive_detail["provider_ttl"].get("age_seconds")
    report = Path(out["report_path"]).read_text(encoding="utf-8")
    assert "last_refresh_age_s" in report
    assert "refresh_owner" in report


def test_soak_alive_fails_when_worker_down(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    monkeypatch.setattr(soak_gate, "pid_running", lambda _pid: False)
    out = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now)
    assert out["ok"] is False
    assert out["signals"]["alive"]["code"] == "ALIVE_WORKER_NOT_RUNNING"


def test_soak_effective_fails_without_activity_when_work_declared(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    _write_manifest(root, work_declared=True)
    out = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now)
    assert out["ok"] is False
    assert out["signals"]["effective"]["code"] == "EFFECTIVE_NO_ACTIVITY"


def test_soak_effective_passes_with_enqueued_activity(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    _write_manifest(root, work_declared=True)
    _write_lab_org_receipt(root, ts=now - 30, enqueued=True, status="RUNNING")
    out = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now)
    assert out["signals"]["effective"]["ok"] is True
    assert out["signals"]["effective"]["code"] == "EFFECTIVE_OK"


def test_soak_effective_passes_with_completed_activity(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    _write_manifest(root, work_declared=True)
    _write_result(root, ts=now - 20)
    out = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now)
    assert out["signals"]["effective"]["ok"] is True
    assert out["signals"]["effective"]["code"] == "EFFECTIVE_OK"


def test_soak_safe_fails_on_microfilm_blocked(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    _write_microfilm(
        root,
        ts=now - 1,
        overall_ok=False,
        actionable_hint="Run verification",
        checks=[
            {
                "name": "verify_before_done",
                "ok": False,
                "code": "BLOCKED_VERIFY_REQUIRED",
                "actionable_hint": "Run verification",
            }
        ],
    )
    out = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now)
    assert out["ok"] is False
    assert out["signals"]["safe"]["code"] == "BLOCKED_VERIFY_REQUIRED"
    assert "Run verification" in out["signals"]["safe"]["actionable_hint"]


def test_soak_safe_classifies_anchor_mismatch_as_blocked_env(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    _write_anchor(root, ts=now - 1, mismatches=[{"code": "display_target_missing"}])
    _write_microfilm(
        root,
        ts=now - 1,
        overall_ok=False,
        actionable_hint="Fix anchor",
        checks=[
            {
                "name": "doctor_anchor",
                "ok": False,
                "code": "BLOCKED_RAIL_MISMATCH",
                "actionable_hint": "Fix anchor",
            }
        ],
    )
    out = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now)
    assert out["ok"] is False
    assert out["signals"]["safe"]["classification"] == "BLOCKED_ENV"


def test_soak_root_mismatch_returns_blocked_env(tmp_path: Path) -> None:
    root = _make_root(tmp_path, name="legacy-root")
    out = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=1_800_000_000.0)
    assert out["ok"] is False
    assert out["outcome_code"] == "BLOCKED_ENV"
    assert Path(out["receipt_path"]).exists()


def test_soak_forced_blocked_env_reason(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    out = soak_gate.run_soak_check(
        root,
        rail="lab",
        window_min=60,
        now_ts=now,
        forced_blocked_env_reason="repo_root no detectado",
        requested_root="/bad/path",
    )
    assert out["ok"] is False
    assert out["outcome_code"] == "BLOCKED_ENV"
    assert out["requested_root"] == "/bad/path"


def test_soak_state_latest_tracks_latest_timestamps(monkeypatch, tmp_path: Path) -> None:
    root, now = _setup_alive_safe_baseline(tmp_path, monkeypatch)
    _write_manifest(root, work_declared=True)
    _write_lab_org_receipt(root, ts=now - 120, enqueued=True, status="RUNNING")
    _write_result(root, ts=now - 100)
    first = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now)
    first_state = first["state_latest"]
    assert first_state["latest_enqueued_ts"] is not None
    assert first_state["latest_completed_ts"] is not None

    _write_result(root, ts=now + 10)
    second = soak_gate.run_soak_check(root, rail="lab", window_min=60, now_ts=now + 20)
    second_state = second["state_latest"]
    assert float(second_state["latest_completed_ts"]) >= float(first_state["latest_completed_ts"])
