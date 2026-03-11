"""
Expected state helpers for missions: describe the desired final state (EFE) and
compute a delta when reality does not match. Version 1 focuses on windows and
files with a simple polling loop.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from agency.process_utils import pid_running
from agency.receipt_validator import validate_receipt


class WindowExpectation(TypedDict, total=False):
    title_contains: Optional[str]
    process_equals: Optional[str]
    must_exist: bool  # True = debe existir; False = no debe existir


class FileExpectation(TypedDict, total=False):
    path: str
    must_exist: bool  # True = debe existir; False = no debe existir


class ExpectedState(TypedDict, total=False):
    windows: List[WindowExpectation]
    files: List[FileExpectation]
    checks: List[Dict[str, Any]]
    meta: Dict[str, Any]


class MismatchKind:
    MISSING = "missing"
    UNEXPECTED = "unexpected"
    MISMATCH = "mismatched"
    TIMEOUT = "timeout"
    OTHER = "other"


class StateDelta(TypedDict, total=False):
    expected: Dict[str, Any]
    actual: Dict[str, Any]
    diff: str
    kind: str


def _window_matches(win: Dict[str, Any], exp: WindowExpectation) -> bool:
    if not win:
        return False
    title = str((win or {}).get("title") or "").lower()
    proc = str((win or {}).get("process") or "").lower()
    title_tok = (exp.get("title_contains") or "").lower()
    proc_eq = (exp.get("process_equals") or "").lower()
    if title_tok and title_tok not in title:
        return False
    if proc_eq and proc != proc_eq:
        return False
    return True


def _collect_windows(driver: Any) -> List[Dict[str, Any]]:
    """
    Colección tolerante: intenta listar ventanas; si no existe, usa solo la activa.
    """
    windows: List[Dict[str, Any]] = []
    if not driver:
        return windows
    # Mejor esfuerzo: list_windows si existe
    if hasattr(driver, "list_windows"):
        try:
            wins = driver.list_windows()  # type: ignore[attr-defined]
            if isinstance(wins, list):
                windows.extend([w for w in wins if isinstance(w, dict)])
                return windows
        except Exception:
            pass
    # Fallback: solo ventana activa
    if hasattr(driver, "get_active_window"):
        try:
            fg = driver.get_active_window()
            if isinstance(fg, dict) and fg:
                windows.append(fg)
        except Exception:
            pass
    return windows


def _collect_file_state(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def _safe_stat(path: str) -> Optional[os.stat_result]:
    try:
        return os.stat(path)
    except Exception:
        return None


def _sha256_file(path: str) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None


def _process_running_by_name(name: str) -> bool:
    wanted = str(name or "").strip().lower()
    if not wanted:
        return False
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None  # type: ignore
    if psutil is not None:
        try:
            for proc in psutil.process_iter(["name"]):
                pname = str((proc.info or {}).get("name") or "").strip().lower()
                if pname == wanted:
                    return True
        except Exception:
            pass
    if os.name == "nt":
        try:
            proc = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            for line in (proc.stdout or "").splitlines():
                low = line.lower()
                if wanted and wanted in low:
                    return True
        except Exception:
            return False
    return False


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except Exception:
        return False


def _verify_single_check(check: Dict[str, Any], *, root_dir: Path) -> Optional[StateDelta]:
    kind = str(check.get("kind") or "").strip().lower()

    if kind == "fs":
        path = str(check.get("path") or "").strip()
        should_exist = bool(check.get("exists", True))
        exists = _collect_file_state(path)
        if should_exist and not exists:
            return StateDelta(
                expected={"check": check},
                actual={"exists": False},
                diff=f"missing file '{path}'",
                kind=MismatchKind.MISSING,
            )
        if (not should_exist) and exists:
            return StateDelta(
                expected={"check": check},
                actual={"exists": True},
                diff=f"unexpected file '{path}'",
                kind=MismatchKind.UNEXPECTED,
            )
        if not exists:
            return None
        stat_result = _safe_stat(path)
        if check.get("mtime", {}).get("required") and stat_result is None:
            return StateDelta(
                expected={"check": check},
                actual={"stat": None},
                diff=f"mtime_unavailable:'{path}'",
                kind=MismatchKind.OTHER,
            )
        if check.get("size", {}).get("required") and stat_result is None:
            return StateDelta(
                expected={"check": check},
                actual={"stat": None},
                diff=f"size_unavailable:'{path}'",
                kind=MismatchKind.OTHER,
            )
        if check.get("sha256", {}).get("required"):
            digest = _sha256_file(path)
            if not digest:
                return StateDelta(
                    expected={"check": check},
                    actual={"sha256": None},
                    diff=f"sha256_unavailable:'{path}'",
                    kind=MismatchKind.OTHER,
                )
        return None

    if kind == "process":
        pid = check.get("pid")
        name = str(check.get("name") or "").strip()
        should_run = bool(check.get("running", True))
        running = False
        if isinstance(pid, int):
            running = pid_running(pid)
        elif name:
            running = _process_running_by_name(name)
        if should_run and not running:
            return StateDelta(
                expected={"check": check},
                actual={"running": False},
                diff="process_not_running",
                kind=MismatchKind.MISSING,
            )
        if (not should_run) and running:
            return StateDelta(
                expected={"check": check},
                actual={"running": True},
                diff="process_still_running",
                kind=MismatchKind.UNEXPECTED,
            )
        return None

    if kind == "port":
        host = str(check.get("host") or "127.0.0.1").strip() or "127.0.0.1"
        port = int(check.get("port") or 0)
        should_open = bool(check.get("open", True))
        is_open = _port_is_open(host, port)
        if should_open and not is_open:
            return StateDelta(
                expected={"check": check},
                actual={"open": False},
                diff=f"port_closed:{host}:{port}",
                kind=MismatchKind.MISSING,
            )
        if (not should_open) and is_open:
            return StateDelta(
                expected={"check": check},
                actual={"open": True},
                diff=f"port_open:{host}:{port}",
                kind=MismatchKind.UNEXPECTED,
            )
        return None

    if kind == "receipt_schema":
        path = str(check.get("path") or "").strip()
        expected_schema = str(check.get("schema") or "").strip()
        if not path:
            return StateDelta(
                expected={"check": check},
                actual={"path": None},
                diff="receipt_path_missing",
                kind=MismatchKind.MISMATCH,
            )
        report = validate_receipt(root_dir, Path(path))
        if not bool(report.get("ok")):
            return StateDelta(
                expected={"check": check},
                actual={"report": report},
                diff="receipt_schema_invalid",
                kind=MismatchKind.MISMATCH,
            )
        if expected_schema and str(report.get("schema_in_receipt") or "").strip() != expected_schema:
            return StateDelta(
                expected={"check": check},
                actual={"schema_in_receipt": report.get("schema_in_receipt")},
                diff="receipt_schema_mismatch",
                kind=MismatchKind.MISMATCH,
            )
        return None

    if kind == "structured_output":
        path = str(check.get("path") or "").strip()
        if not path or not os.path.exists(path):
            return StateDelta(
                expected={"check": check},
                actual={"exists": False},
                diff="structured_output_missing",
                kind=MismatchKind.MISSING,
            )
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return StateDelta(
                expected={"check": check},
                actual={"parse_ok": False},
                diff="structured_output_parse_error",
                kind=MismatchKind.MISMATCH,
            )
        root_type = str(check.get("root_type") or "").strip().lower()
        if root_type == "object" and not isinstance(payload, dict):
            return StateDelta(
                expected={"check": check},
                actual={"payload_type": type(payload).__name__},
                diff="structured_output_not_object",
                kind=MismatchKind.MISMATCH,
            )
        if root_type == "array" and not isinstance(payload, list):
            return StateDelta(
                expected={"check": check},
                actual={"payload_type": type(payload).__name__},
                diff="structured_output_not_array",
                kind=MismatchKind.MISMATCH,
            )
        required_keys = check.get("required_keys") if isinstance(check.get("required_keys"), list) else []
        if required_keys and isinstance(payload, dict):
            missing = [str(key) for key in required_keys if str(key) not in payload]
            if missing:
                return StateDelta(
                    expected={"check": check},
                    actual={"keys": sorted(payload.keys())},
                    diff=f"structured_output_missing_keys:{','.join(missing)}",
                    kind=MismatchKind.MISSING,
                )
        return None

    return StateDelta(
        expected={"check": check},
        actual={"supported_kinds": ["fs", "process", "port", "receipt_schema", "structured_output"]},
        diff=f"unsupported_check_kind:{kind or 'unknown'}",
        kind=MismatchKind.MISMATCH,
    )


def verify_efe(
    expected_state: ExpectedState,
    *,
    driver: Any = None,
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.3,
    root_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[StateDelta]]:
    """
    Evalúa el ExpectedState contra el estado real. Si todos los checks pasan
    dentro del timeout, devuelve (True, None). Si no, devuelve (False, delta).
    """
    start = time.time()
    windows_exp: List[WindowExpectation] = list(expected_state.get("windows", []) or [])
    files_exp: List[FileExpectation] = list(expected_state.get("files", []) or [])
    checks_exp: List[Dict[str, Any]] = list(expected_state.get("checks", []) or [])
    verify_root = Path(root_dir or Path.cwd()).resolve()

    last_delta: Optional[StateDelta] = None
    while time.time() - start <= timeout_s:
        windows = _collect_windows(driver)
        active_window: Optional[Dict[str, Any]] = None
        try:
            if driver and hasattr(driver, "get_active_window"):
                active_window = driver.get_active_window()
        except Exception:
            active_window = None
        files_state: Dict[str, bool] = {}
        for fexp in files_exp:
            path = fexp.get("path") or ""
            if not path:
                continue
            files_state[path] = _collect_file_state(path)

        # Meta checks (p.ej. foco)
        meta = expected_state.get("meta") or {}
        if isinstance(meta, dict) and meta.get("must_be_active"):
            if not windows_exp:
                return False, StateDelta(
                    expected={"meta": {"must_be_active": True}},
                    actual={"active_window": active_window},
                    diff="active_window_required_but_no_window_expectations",
                    kind=MismatchKind.MISMATCH,
                )
            if not active_window or not isinstance(active_window, dict):
                return False, StateDelta(
                    expected={"meta": {"must_be_active": True}, "windows": windows_exp},
                    actual={"active_window": active_window},
                    diff="active_window_missing",
                    kind=MismatchKind.MISSING,
                )
            must_match = [w for w in windows_exp if bool(w.get("must_exist", True))]
            if must_match and not any(_window_matches(active_window, wexp) for wexp in must_match):
                return False, StateDelta(
                    expected={"meta": {"must_be_active": True}, "active_window_should_match": must_match},
                    actual={"active_window": active_window},
                    diff="active_window_mismatch",
                    kind=MismatchKind.MISMATCH,
                )

        # Check windows expectations
        win_failure: Optional[StateDelta] = None
        for wexp in windows_exp:
            must_exist = bool(wexp.get("must_exist", True))
            match = any(_window_matches(win, wexp) for win in windows)
            if must_exist and not match:
                win_failure = StateDelta(
                    expected={"window": wexp},
                    actual={"windows": windows},
                    diff=f"missing window title_contains='{wexp.get('title_contains')}' process='{wexp.get('process_equals')}'",
                    kind=MismatchKind.MISSING,
                )
                break
            if not must_exist and match:
                win_failure = StateDelta(
                    expected={"window": wexp},
                    actual={"windows": windows},
                    diff=f"unexpected window title_contains='{wexp.get('title_contains')}' process='{wexp.get('process_equals')}'",
                    kind=MismatchKind.UNEXPECTED,
                )
                break

        file_failure: Optional[StateDelta] = None
        check_failure: Optional[StateDelta] = None
        if not win_failure:
            for fexp in files_exp:
                path = fexp.get("path") or ""
                if not path:
                    continue
                must_exist = bool(fexp.get("must_exist", True))
                exists = files_state.get(path, False)
                if must_exist and not exists:
                    file_failure = StateDelta(
                        expected={"file": fexp},
                        actual={"exists": exists},
                        diff=f"missing file '{path}'",
                        kind=MismatchKind.MISSING,
                    )
                    break
                if (not must_exist) and exists:
                    file_failure = StateDelta(
                        expected={"file": fexp},
                        actual={"exists": exists},
                        diff=f"unexpected file '{path}'",
                        kind=MismatchKind.UNEXPECTED,
                    )
                    break

        if not win_failure and not file_failure:
            for check in checks_exp:
                if not isinstance(check, dict):
                    check_failure = StateDelta(
                        expected={"check": check},
                        actual={"type": type(check).__name__},
                        diff="invalid_check_shape",
                        kind=MismatchKind.MISMATCH,
                    )
                    break
                delta = _verify_single_check(check, root_dir=verify_root)
                if delta is not None:
                    check_failure = delta
                    break

        if not win_failure and not file_failure and not check_failure:
            return True, None

        last_delta = win_failure or file_failure or check_failure
        time.sleep(poll_interval_s)

    if last_delta:
        # Etiqueta de timeout si venimos reintentando
        last_delta.setdefault("kind", MismatchKind.TIMEOUT)
        return False, last_delta

    return False, StateDelta(
        expected=dict(expected_state),
        actual={},
        diff="expected_state_timeout",
        kind=MismatchKind.TIMEOUT,
    )
