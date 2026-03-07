from __future__ import annotations

import json
from pathlib import Path

from agency import boot_watchdog


def _write_manifest(root: Path, tasks: list[dict]) -> Path:
    cfg = root / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    path = cfg / "expected_tasks.json"
    path.write_text(
        json.dumps(
            {
                "schema": "ajax.expected_tasks.v1",
                "default_user": "Javi",
                "default_delay": "PT45S",
                "tasks": tasks,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_expected_tasks_normalizes_minimum(tmp_path: Path) -> None:
    _write_manifest(
        tmp_path,
        [
            {
                "task_name": r"\AJAX\AJAX_Lab_Bootstrap",
                "action": {"script": r"scripts\ops\ajax_lab_bootstrap.ps1"},
            }
        ],
    )
    out = boot_watchdog.load_expected_tasks(tmp_path)
    assert out["schema"] == "ajax.expected_tasks.v1"
    assert len(out["tasks"]) == 1
    task = out["tasks"][0]
    assert task["trigger"]["user"] == "Javi"
    assert task["trigger"]["delay"] == "PT45S"
    assert task["settings"]["restart_count"] == 3
    assert task["action"]["script"] == "scripts/ops/ajax_lab_bootstrap.ps1"


def test_load_expected_tasks_requires_tasks(tmp_path: Path) -> None:
    cfg = tmp_path / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "expected_tasks.json").write_text(json.dumps({"schema": "ajax.expected_tasks.v1"}), encoding="utf-8")
    try:
        boot_watchdog.load_expected_tasks(tmp_path)
    except ValueError as exc:
        assert "tasks_missing" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing tasks")


def test_expected_task_names_extracts_all() -> None:
    payload = {
        "tasks": [
            {"task_name": r"\AJAX\AJAX_Start_AJAX"},
            {"task_name": r"\AJAX\AJAX_Lab_Bootstrap"},
        ]
    }
    names = boot_watchdog.expected_task_names(payload)
    assert names == [r"\AJAX\AJAX_Start_AJAX", r"\AJAX\AJAX_Lab_Bootstrap"]


def test_watchdog_tick_decision_port_down() -> None:
    out = boot_watchdog.watchdog_tick_decision(
        rail="lab",
        listener=False,
        health_ok=False,
        displays_ok=False,
    )
    assert out["would_start"] is True
    assert "listener_down" in out["reasons"]
    assert "health_not_ok" in out["reasons"]


def test_watchdog_tick_decision_lab_requires_displays() -> None:
    out = boot_watchdog.watchdog_tick_decision(
        rail="lab",
        listener=True,
        health_ok=True,
        displays_ok=False,
    )
    assert out["would_start"] is True
    assert "display_catalog_unavailable" in out["reasons"]


def test_watchdog_tick_decision_prod_ignores_displays() -> None:
    out = boot_watchdog.watchdog_tick_decision(
        rail="prod",
        listener=True,
        health_ok=True,
        displays_ok=False,
    )
    assert out["would_start"] is False
    assert out["reasons"] == []


def test_format_doctor_boot_summary_contains_key_fields() -> None:
    payload = {
        "status": "BLOCKED",
        "rail": "lab",
        "tasks": {"ok": True, "missing": [r"\AJAX\AJAX_DriverWatchdog"], "drifted": []},
        "ports": {
            "5010": {"listener": True, "health_ok": True, "displays_ok": None},
            "5012": {"listener": False, "health_ok": False, "displays_ok": False},
        },
        "worker": {"status": "down", "fresh": False, "pid": None},
        "providers_status_ttl": {"stale": True, "reason": "providers_status_stale"},
        "probable_causes": [{"code": "expected_port_missing", "detail": "listener_down:5012"}],
        "next_hint": ["python bin/ajaxctl ops tasks ensure --apply"],
    }
    summary = boot_watchdog.format_doctor_boot_summary(payload)
    assert "AJAX Doctor boot" in summary
    assert "status: BLOCKED" in summary
    assert "port 5012: listener=False" in summary
    assert "expected_port_missing" in summary


def test_unsupported_platform_payload_schema() -> None:
    out = boot_watchdog.unsupported_platform_payload(command="ops tasks ensure")
    assert out["schema"] == "ajax.ops.unsupported_platform.v1"
    assert out["status"] == "UNSUPPORTED_PLATFORM"
    assert out["ok"] is False


def test_worker_heartbeat_status_fresh(tmp_path: Path, monkeypatch) -> None:
    lab_dir = tmp_path / "artifacts" / "lab"
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "worker.pid").write_text("123", encoding="utf-8")
    now = 1_700_000_000.0
    (lab_dir / "heartbeat.json").write_text(json.dumps({"ts": now - 5}), encoding="utf-8")
    monkeypatch.setattr(boot_watchdog, "pid_running", lambda pid: pid == 123)
    monkeypatch.setattr(boot_watchdog.time, "time", lambda: now)
    out = boot_watchdog.worker_heartbeat_status(tmp_path)
    assert out["fresh"] is True
    assert out["status"] == "fresh"
    assert out["pid"] == 123


def test_worker_heartbeat_status_down_when_pid_not_running(tmp_path: Path, monkeypatch) -> None:
    lab_dir = tmp_path / "artifacts" / "lab"
    lab_dir.mkdir(parents=True, exist_ok=True)
    (lab_dir / "worker.pid").write_text("456", encoding="utf-8")
    (lab_dir / "heartbeat.json").write_text(json.dumps({"ts": 1_700_000_000.0}), encoding="utf-8")
    monkeypatch.setattr(boot_watchdog, "pid_running", lambda _pid: False)
    out = boot_watchdog.worker_heartbeat_status(tmp_path)
    assert out["fresh"] is False
    assert out["status"] == "down"


def test_run_doctor_boot_writes_artifacts_with_injected_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        boot_watchdog,
        "run_tasks_audit",
        lambda _root: {
            "ok": True,
            "missing": [r"\AJAX\AJAX_DriverWatchdog"],
            "drifted": [],
            "tasks": [],
        },
    )
    monkeypatch.setattr(
        boot_watchdog,
        "probe_port_state",
        lambda **kwargs: {
            "port": kwargs["port"],
            "listener": False if kwargs["port"] == 5012 else True,
            "health_ok": False if kwargs["port"] == 5012 else True,
            "displays_ok": False if kwargs.get("include_displays") else None,
            "displays_error": "display_catalog_unavailable",
        },
    )
    monkeypatch.setattr(
        boot_watchdog,
        "worker_heartbeat_status",
        lambda _root: {"fresh": False, "status": "down", "pid": None},
    )
    monkeypatch.setattr(
        boot_watchdog,
        "provider_status_ttl",
        lambda _root, ttl_seconds=900: {"stale": True, "reason": "providers_status_stale", "ttl_seconds": ttl_seconds},
    )
    monkeypatch.setattr(
        boot_watchdog,
        "run_anchor_preflight",
        lambda **_kwargs: {"ok": False, "mismatches": [{"code": "display_catalog_unavailable", "detail": "x"}]},
    )
    out = boot_watchdog.run_doctor_boot(tmp_path, rail="lab")
    assert out["ok"] is False
    assert out["status"] == "BLOCKED"
    assert Path(out["artifact_path"]).exists()
    assert Path(out["receipt_path"]).exists()
    codes = {row["code"] for row in out["probable_causes"] if isinstance(row, dict)}
    assert "expected_tasks_missing" in codes
    assert "expected_port_missing" in codes


def test_run_tasks_audit_returns_unsupported_when_not_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(boot_watchdog, "windows_supported", lambda: False)
    out = boot_watchdog.run_tasks_audit(tmp_path)
    assert out["status"] == "UNSUPPORTED_PLATFORM"
    assert out["ok"] is False
