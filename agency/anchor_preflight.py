from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agency.display_targets import fetch_driver_displays, load_display_map

try:
    from agency.ops_ports_sessions import run_services_doctor
except Exception:  # pragma: no cover
    run_services_doctor = None  # type: ignore


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ts_label() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _normalize_rail(raw: Optional[str]) -> str:
    value = str(raw or "").strip().lower()
    if value in {"prod", "production", "live"}:
        return "prod"
    return "lab"


def _expected_anchor(rail: str) -> Dict[str, Any]:
    rail_n = _normalize_rail(rail)
    if rail_n == "prod":
        return {"rail": "prod", "expected_user": "Javi", "expected_port": 5010}
    return {"rail": "lab", "expected_user": "AJAX", "expected_port": 5012}


def _extract_port_entry(ports_map: Dict[str, Any], port: int) -> Dict[str, Any]:
    if not isinstance(ports_map, dict):
        return {}
    if port in ports_map and isinstance(ports_map[port], dict):
        return ports_map[port]
    port_s = str(port)
    if port_s in ports_map and isinstance(ports_map[port_s], dict):
        return ports_map[port_s]
    return {}


def _parse_session_id(payload: Any) -> Optional[int]:
    if isinstance(payload, int):
        return payload
    if isinstance(payload, float):
        try:
            return int(payload)
        except Exception:
            return None
    if isinstance(payload, str) and payload.strip():
        try:
            return int(payload.strip())
        except Exception:
            return None
    if isinstance(payload, dict):
        for key in ("SessionId", "session_id", "sessionId"):
            if key in payload:
                return _parse_session_id(payload.get(key))
    return None


def _resolve_display_target_id(display_map: Dict[str, Any], rail: str) -> Optional[int]:
    if not isinstance(display_map, dict):
        return None
    targets = display_map.get("display_targets") if isinstance(display_map.get("display_targets"), dict) else {}
    item = targets.get(_normalize_rail(rail))
    if isinstance(item, int):
        return item
    if isinstance(item, str) and item.strip().isdigit():
        return int(item.strip())
    if isinstance(item, dict):
        for key in ("display_id", "id"):
            raw = item.get(key)
            parsed = _parse_session_id(raw)
            if parsed is not None:
                return parsed
    return None


def _driver_url_for_rail(rail: str) -> str:
    rail_n = _normalize_rail(rail)
    if rail_n == "prod":
        host = (os.getenv("OS_DRIVER_HOST") or "127.0.0.1").strip()
        port = (os.getenv("OS_DRIVER_PORT") or "5010").strip() or "5010"
        return f"http://{host}:{port}"
    host = (os.getenv("OS_DRIVER_HOST_LAB") or "127.0.0.1").strip()
    port = (os.getenv("OS_DRIVER_PORT_LAB") or "5012").strip() or "5012"
    return f"http://{host}:{port}"


def _display_catalog_ids(display_catalog: Dict[str, Any]) -> List[int]:
    displays = display_catalog.get("displays") if isinstance(display_catalog, dict) else None
    if not isinstance(displays, list):
        return []
    ids: List[int] = []
    for entry in displays:
        if not isinstance(entry, dict):
            continue
        parsed = _parse_session_id(entry.get("id"))
        if parsed is None:
            continue
        if parsed not in ids:
            ids.append(parsed)
    return ids


def evaluate_anchor_snapshot(
    *,
    rail: str,
    services_report: Dict[str, Any],
    display_target_id: Optional[int],
    display_catalog: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    expected = _expected_anchor(rail)
    expected_port = int(expected["expected_port"])
    expected_user = str(expected["expected_user"])

    mismatches: List[Dict[str, str]] = []
    sessions = services_report.get("sessions") if isinstance(services_report, dict) else {}
    ports = services_report.get("ports") if isinstance(services_report, dict) else {}
    health = services_report.get("health") if isinstance(services_report, dict) else {}

    expected_session = None
    if isinstance(sessions, dict):
        expected_session = _parse_session_id(sessions.get(expected_user))
    if expected_session is None:
        mismatches.append(
            {
                "code": "expected_session_missing",
                "detail": f"Session for expected user {expected_user} not found",
            }
        )

    port_entry = _extract_port_entry(ports if isinstance(ports, dict) else {}, expected_port)
    actual_session = _parse_session_id(port_entry)
    if not port_entry:
        mismatches.append(
            {
                "code": "expected_port_missing",
                "detail": f"No listener entry found for port {expected_port}",
            }
        )
    elif expected_session is not None and actual_session != expected_session:
        mismatches.append(
            {
                "code": "port_session_mismatch",
                "detail": f"Port {expected_port} session {actual_session} != expected {expected_session}",
            }
        )

    port_health = None
    if isinstance(health, dict):
        port_health = health.get(expected_port)
        if port_health is None:
            port_health = health.get(str(expected_port))
    if bool(port_health) is not True:
        mismatches.append(
            {
                "code": "port_health_not_ok",
                "detail": f"Health for expected port {expected_port} is not OK",
            }
        )

    if display_target_id is None:
        mismatches.append(
            {
                "code": "display_target_missing",
                "detail": f"display_targets.{_normalize_rail(rail)} is missing in config/display_map.json",
            }
        )

    display_ids: List[int] = []
    if display_catalog is None:
        mismatches.append(
            {
                "code": "display_catalog_unavailable",
                "detail": "Could not fetch /displays from target rail driver",
            }
        )
    else:
        display_ids = _display_catalog_ids(display_catalog)
        if display_target_id is not None and display_ids and display_target_id not in display_ids:
            mismatches.append(
                {
                    "code": "display_target_not_found",
                    "detail": f"Configured display id {display_target_id} not present in driver displays",
                }
            )

    ok = not mismatches
    return {
        "schema": "ajax.anchor_preflight.v1",
        "ts_utc": _utc_now(),
        "ok": ok,
        "status": "READY" if ok else "BLOCKED",
        "rail": _normalize_rail(rail),
        "expected": expected,
        "observed": {
            "expected_session": expected_session,
            "actual_session": actual_session,
            "port_health": bool(port_health),
            "display_target_id": display_target_id,
            "display_catalog_ids": display_ids,
        },
        "mismatches": mismatches,
    }


def run_anchor_preflight(
    *,
    root_dir: Path,
    rail: str,
    write_receipt: bool = True,
    services_report: Optional[Dict[str, Any]] = None,
    display_map: Optional[Dict[str, Any]] = None,
    display_catalog: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(root_dir)
    rail_n = _normalize_rail(rail)

    services = services_report
    if services is None:
        if run_services_doctor is None:
            services = {"ok": False, "error": "services_doctor_unavailable"}
        else:
            try:
                services = run_services_doctor(root)
            except Exception as exc:
                services = {"ok": False, "error": f"services_doctor_failed:{str(exc)[:120]}"}

    disp_map = display_map if isinstance(display_map, dict) else load_display_map(root)
    display_target_id = _resolve_display_target_id(disp_map, rail_n)

    catalog = display_catalog
    if catalog is None:
        try:
            catalog = fetch_driver_displays(_driver_url_for_rail(rail_n))
        except Exception:
            catalog = None

    payload = evaluate_anchor_snapshot(
        rail=rail_n,
        services_report=services if isinstance(services, dict) else {},
        display_target_id=display_target_id,
        display_catalog=catalog,
    )
    payload["services_ok"] = bool(isinstance(services, dict) and services.get("ok", False))
    payload["driver_url"] = _driver_url_for_rail(rail_n)

    receipt_path = None
    if write_receipt:
        try:
            receipt_dir = root / "artifacts" / "receipts"
            receipt_dir.mkdir(parents=True, exist_ok=True)
            receipt = receipt_dir / f"anchor_preflight_{_ts_label()}.json"
            receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            receipt_path = str(receipt)
        except Exception:
            receipt_path = None

    payload["receipt_path"] = receipt_path
    return payload
