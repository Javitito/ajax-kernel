"""
Expected state helpers for missions: describe the desired final state (EFE) and
compute a delta when reality does not match. Version 1 focuses on windows and
files with a simple polling loop.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict


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


def verify_efe(
    expected_state: ExpectedState,
    *,
    driver: Any = None,
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.3,
) -> Tuple[bool, Optional[StateDelta]]:
    """
    Evalúa el ExpectedState contra el estado real. Si todos los checks pasan
    dentro del timeout, devuelve (True, None). Si no, devuelve (False, delta).
    """
    start = time.time()
    windows_exp: List[WindowExpectation] = list(expected_state.get("windows", []) or [])
    files_exp: List[FileExpectation] = list(expected_state.get("files", []) or [])

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
            return True, None

        last_delta = win_failure or file_failure
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
