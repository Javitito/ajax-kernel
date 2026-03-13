from __future__ import annotations

import json
import types
from pathlib import Path

import agency.ajax_core as ajax_core_mod
from agency.ajax_core import AjaxConfig, AjaxCore
from agency import driver_revive


def _make_core(tmp_path: Path) -> AjaxCore:
    core = AjaxCore.__new__(AjaxCore)
    state_dir = tmp_path / "artifacts" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    core.config = AjaxConfig(root_dir=tmp_path, state_dir=state_dir)
    core.log = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None)
    core.system_executor = types.SimpleNamespace(run=lambda *a, **k: None)
    core._driver_cb = {"status": "up", "failures": [], "down_since": None}
    return core


def _touch(path: Path, text: str = "# stub\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _receipt_events(result: dict) -> list[str]:
    events: list[str] = []
    for raw in result.get("receipt_paths") or []:
        payload = json.loads(Path(str(raw)).read_text(encoding="utf-8"))
        events.append(str(payload.get("event")))
    return events


def test_resolve_driver_endpoint_lab_defaults_to_5012(monkeypatch) -> None:
    monkeypatch.delenv("OS_DRIVER_URL_LAB", raising=False)
    monkeypatch.delenv("OS_DRIVER_HOST_LAB", raising=False)
    monkeypatch.delenv("OS_DRIVER_PORT_LAB", raising=False)
    endpoint = driver_revive.resolve_driver_endpoint("lab")
    assert endpoint.rail == "lab"
    assert endpoint.host == "127.0.0.1"
    assert endpoint.port == 5012
    assert endpoint.source == "default:lab"


def test_resolve_driver_endpoint_prod_defaults_to_5010(monkeypatch) -> None:
    monkeypatch.delenv("OS_DRIVER_URL", raising=False)
    monkeypatch.delenv("OS_DRIVER_HOST", raising=False)
    monkeypatch.delenv("OS_DRIVER_PORT", raising=False)
    endpoint = driver_revive.resolve_driver_endpoint("prod")
    assert endpoint.rail == "prod"
    assert endpoint.host == "127.0.0.1"
    assert endpoint.port == 5010
    assert endpoint.source == "default:prod"


def test_run_driver_revive_skips_when_expected_driver_is_already_healthy(tmp_path: Path, monkeypatch) -> None:
    _touch(tmp_path / "agency" / "lab_dummy_driver.py")
    monkeypatch.setattr(
        driver_revive,
        "check_driver_health",
        lambda endpoint, timeout_s=1.5: {
            "healthy": True,
            "reachable": True,
            "http_status": 200,
            "detail": "health_ok",
            "url": endpoint.url + "/health",
            "payload": {"ok": True, "driver": "lab_dummy"},
            "simulated": True,
        },
        raising=True,
    )

    result = driver_revive.run_driver_revive(root_dir=tmp_path, rail="lab")

    assert result["ok"] is True
    assert result["skipped"] is True
    assert result["launch_attempted"] is False
    assert _receipt_events(result) == [
        "driver_health_checked",
        "driver_revive_skipped_healthy",
    ]


def test_run_driver_revive_missing_target_is_explicit_and_does_not_spawn(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        driver_revive,
        "check_driver_health",
        lambda endpoint, timeout_s=1.5: {
            "healthy": False,
            "reachable": False,
            "http_status": None,
            "detail": "connection_refused",
            "url": endpoint.url + "/health",
            "payload": {},
            "simulated": False,
        },
        raising=True,
    )

    def _unexpected_launch(*args, **kwargs):
        raise AssertionError("launch should not run when target is unavailable")

    monkeypatch.setattr(driver_revive, "_launch_target", _unexpected_launch, raising=True)

    result = driver_revive.run_driver_revive(root_dir=tmp_path, rail="prod")

    assert result["ok"] is False
    assert result["launch_attempted"] is False
    assert result["failure_reason"] == "missing_entrypoint"
    assert result["target"]["target_exists"] is False
    assert _receipt_events(result) == [
        "driver_health_checked",
        "driver_revive_target_missing",
    ]


def test_run_driver_revive_attempts_launch_for_unhealthy_lab_target(tmp_path: Path, monkeypatch) -> None:
    _touch(tmp_path / "agency" / "lab_dummy_driver.py")
    monkeypatch.setattr(
        driver_revive,
        "check_driver_health",
        lambda endpoint, timeout_s=1.5: {
            "healthy": False,
            "reachable": False,
            "http_status": None,
            "detail": "connection_refused",
            "url": endpoint.url + "/health",
            "payload": {},
            "simulated": False,
        },
        raising=True,
    )
    seen: dict[str, object] = {}

    def _fake_launch(root_dir, target, endpoint, *, system_executor, timeout_s):
        seen["root_dir"] = root_dir
        seen["target"] = target.resolved_target
        seen["port"] = endpoint.port
        return {
            "ok": True,
            "returncode": 0,
            "timed_out": False,
            "detail": {"started": True},
            "reason": None,
        }

    monkeypatch.setattr(driver_revive, "_launch_target", _fake_launch, raising=True)
    monkeypatch.setattr(
        driver_revive,
        "_wait_for_postcheck",
        lambda endpoint, timeout_s, poll_interval_s: {
            "healthy": True,
            "reachable": True,
            "http_status": 200,
            "detail": "health_ok",
            "url": endpoint.url + "/health",
            "payload": {"ok": True},
            "simulated": True,
        },
        raising=True,
    )

    result = driver_revive.run_driver_revive(root_dir=tmp_path, rail="lab")

    assert result["ok"] is True
    assert result["launch_attempted"] is True
    assert seen["port"] == 5012
    assert str(seen["target"]).endswith("agency\\lab_dummy_driver.py") or str(seen["target"]).endswith("agency/lab_dummy_driver.py")
    assert _receipt_events(result) == [
        "driver_health_checked",
        "driver_revive_launch_attempted",
        "driver_revive_postcheck_success",
    ]


def test_run_driver_revive_reports_postcheck_failure_honestly(tmp_path: Path, monkeypatch) -> None:
    _touch(tmp_path / "agency" / "lab_dummy_driver.py")
    monkeypatch.setattr(
        driver_revive,
        "check_driver_health",
        lambda endpoint, timeout_s=1.5: {
            "healthy": False,
            "reachable": False,
            "http_status": None,
            "detail": "connection_refused",
            "url": endpoint.url + "/health",
            "payload": {},
            "simulated": False,
        },
        raising=True,
    )
    monkeypatch.setattr(
        driver_revive,
        "_launch_target",
        lambda root_dir, target, endpoint, *, system_executor, timeout_s: {
            "ok": False,
            "returncode": 2,
            "timed_out": False,
            "detail": {"stderr": "missing entrypoint"},
            "reason": "launch_returncode_2",
        },
        raising=True,
    )
    monkeypatch.setattr(
        driver_revive,
        "_wait_for_postcheck",
        lambda endpoint, timeout_s, poll_interval_s: {
            "healthy": False,
            "reachable": False,
            "http_status": None,
            "detail": "still_down",
            "url": endpoint.url + "/health",
            "payload": {},
            "simulated": False,
        },
        raising=True,
    )

    result = driver_revive.run_driver_revive(root_dir=tmp_path, rail="lab")

    assert result["ok"] is False
    assert result["launch_attempted"] is True
    assert result["failure_reason"] == "launch_returncode_2"
    assert _receipt_events(result) == [
        "driver_health_checked",
        "driver_revive_launch_attempted",
        "driver_revive_launch_timeout_or_failed",
        "driver_revive_postcheck_failed",
    ]


def test_ajax_core_driver_health_snapshot_uses_rail_aware_endpoint(tmp_path: Path, monkeypatch) -> None:
    core = _make_core(tmp_path)
    monkeypatch.setenv("AJAX_RAIL", "lab")
    monkeypatch.delenv("OS_DRIVER_PORT_LAB", raising=False)
    seen: dict[str, object] = {}

    def _fake_health(endpoint):
        seen["rail"] = endpoint.rail
        seen["port"] = endpoint.port
        return {
            "healthy": True,
            "simulated": True,
            "detail": "health_ok",
            "http_status": 200,
            "url": endpoint.url + "/health",
            "payload": {"ok": True},
        }

    monkeypatch.setattr(ajax_core_mod, "check_driver_health", _fake_health, raising=True)

    snap = core._driver_health_snapshot()

    assert seen == {"rail": "lab", "port": 5012}
    assert snap["ok"] is True
    assert snap["port"] == 5012
    assert snap["rail"] == "lab"


def test_ajax_core_start_driver_delegates_to_real_revive_module(tmp_path: Path, monkeypatch) -> None:
    core = _make_core(tmp_path)
    monkeypatch.setenv("AJAX_RAIL", "lab")
    monkeypatch.setattr(
        ajax_core_mod,
        "run_driver_revive",
        lambda **kwargs: {
            "ok": True,
            "receipt_paths": [],
            "endpoint": {"rail": "lab", "port": 5012},
        },
        raising=True,
    )
    monkeypatch.setattr(ajax_core_mod.AjaxCore, "_init_driver", lambda self: "driver-client", raising=True)

    ok = core._start_driver()

    assert ok is True
    assert core.driver == "driver-client"
