"""
Cierre seguro para editores simples (Notepad / WordPad).

No lanza nuevas instancias; opera únicamente sobre la ventana activa y maneja el
diálogo de guardado si aparece tras Alt+F4.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional

from agency.windows_driver_client import WindowsDriverClient, WindowsDriverError, DriverTimeout, DriverConnectionError

log = logging.getLogger(__name__)

SAVE_DIALOG_TOKENS = (
    "guardar como",
    "guardar cambios",
    "save as",
    "do you want to save",
    "save changes",
)


def _normalize_windows_path(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().strip('"')
    if not text:
        return None
    if text.startswith("/mnt/") and len(text) > 6:
        drive = text[5]
        remainder = text[6:]
        if drive.isalpha():
            return f"{drive.upper()}:\\{remainder.replace('/', '\\')}"
    if text.startswith("~/"):
        try:
            home = Path(os.path.expanduser("~"))
            text = str(home / text[2:])
        except Exception:
            pass
    if "://" not in text:
        text = text.replace("/", "\\")
    return text


def _resolve_userprofile() -> Optional[str]:
    env = os.getenv("USERPROFILE")
    if env:
        return _normalize_windows_path(env)
    hd = os.getenv("HOMEDRIVE")
    hp = os.getenv("HOMEPATH")
    if hd and hp:
        return _normalize_windows_path(f"{hd}{hp}")
    uname = os.getenv("USERNAME")
    if uname:
        return f"C:\\Users\\{uname}"
    try:
        # Derivar desde la ruta del repo (WSL -> Windows)
        repo_root = Path(__file__).resolve().parents[1]
        win_root = _normalize_windows_path(str(repo_root))
        if win_root and ":" in win_root:
            return str(Path(PureWindowsPath(win_root)).parent)
    except Exception:
        pass
    return None


def _detect_editor(fg: Dict[str, Any]) -> str:
    proc = str((fg or {}).get("process") or "").lower()
    title = str((fg or {}).get("title") or "").lower()
    if "notepad.exe" in proc or "bloc de notas" in title or "notepad" in title:
        return "notepad"
    if "wordpad.exe" in proc or "wordpad" in title:
        return "wordpad"
    return "unknown"


def _is_save_dialog(fg: Dict[str, Any]) -> bool:
    title = str((fg or {}).get("title") or "").lower()
    if not title:
        return False
    return any(tok in title for tok in SAVE_DIALOG_TOKENS)


def close_editor_safe(
    driver: WindowsDriverClient,
    *,
    action: str = "dont_save",
    file_path: Optional[str] = None,
    timeout_s: float = 6.0,
    dialog_wait_s: float = 1.5,
) -> Dict[str, Any]:
    """
    Cierra un editor activo (Notepad/WordPad) gestionando el diálogo de guardado si aparece.
    No abre nuevas instancias.
    """
    start = time.time()
    errors = []
    try:
        fg = driver.get_active_window()
    except Exception as exc:
        return {
            "ok": False,
            "error": "cannot_read_active_window",
            "errors": [str(exc)],
            "editor": "unknown",
            "dialog_present": False,
            "action_taken": "none",
        }

    editor = _detect_editor(fg)
    active_proc = str((fg or {}).get("process") or "").lower()
    if editor == "unknown":
        return {
            "ok": False,
            "error": "wrong_active_window",
            "errors": [f"active_process={active_proc or 'unknown'}"],
            "editor": editor,
            "dialog_present": False,
            "action_taken": "none",
        }

    log.info("close_editor_safe: start editor=%s action=%s fg=%s", editor, action, fg)

    title_before = str((fg or {}).get("title") or "")
    expected_unsaved = title_before.strip().startswith("*")

    try:
        driver.keyboard_hotkey("alt", "f4", ignore_safety=True)
    except Exception as exc:
        errors.append(f"alt_f4_failed:{exc}")

    dialog_present = False
    dialog_fg: Dict[str, Any] = {}
    end_dialog_wait = time.time() + dialog_wait_s
    while time.time() < end_dialog_wait:
        try:
            dialog_fg = driver.get_active_window()
        except Exception:
            dialog_fg = {}
        if _is_save_dialog(dialog_fg):
            dialog_present = True
            break
        time.sleep(0.1)

    action_taken = "no_dialog"

    def _is_same_window(win: Dict[str, Any]) -> bool:
        if not win:
            return False
        title = str((win or {}).get("title") or "").lower()
        proc = str((win or {}).get("process") or "").lower()
        title_before_l = title_before.lower()
        return (proc == active_proc) and (title == title_before_l or title_before_l in title or title in title_before_l)

    def _wait_editor_closed(deadline: float) -> bool:
        while time.time() < deadline:
            try:
                current = driver.get_active_window()
                if not _is_same_window(current):
                    return True
            except Exception:
                return True
            time.sleep(0.2)
        return False

    if dialog_present:
        action_taken = action if action in ("save", "dont_save") else "dont_save"
        if action_taken == "save":
            if not file_path:
                return {
                    "ok": False,
                    "error": "file_path_required",
                    "errors": ["file_path missing for save"],
                    "editor": editor,
                    "dialog_present": True,
                    "action_taken": action_taken,
                }
            expanded = os.path.expandvars(str(file_path))
            if "%USERPROFILE%" in str(file_path) and "%USERPROFILE%" in expanded:
                up = _resolve_userprofile()
                if up:
                    expanded = str(file_path).replace("%USERPROFILE%", up)
            norm_path = _normalize_windows_path(expanded) or expanded
            try:
                driver.keyboard_paste(norm_path, submit=False)
                time.sleep(0.2)
                driver.keyboard_hotkey("enter", ignore_safety=True)
            except Exception as exc:
                errors.append(f"save_dialog_input_failed:{exc}")
        else:
            try:
                driver.keyboard_hotkey("alt", "n", ignore_safety=True)
            except Exception as exc:
                errors.append(f"dont_save_hotkey_failed:{exc}")
                try:
                    driver.keyboard_hotkey("tab", "enter", ignore_safety=True)
                except Exception:
                    pass
        closed = _wait_editor_closed(time.time() + timeout_s)
        if not closed:
            # Asegurar foco en la ventana original antes de reintentar
            try:
                driver.window_focus(title_contains=title_before)
            except Exception:
                pass
            try:
                driver.keyboard_hotkey("alt", "f4", ignore_safety=True)
            except Exception as exc:
                errors.append(f"alt_f4_retry_failed:{exc}")
            closed = _wait_editor_closed(time.time() + min(1.5, timeout_s))
        if not closed:
            errors.append("editor_window_still_open")
    else:
        # Caso sin diálogo de guardado
        closed = _wait_editor_closed(time.time() + timeout_s)
        if expected_unsaved and not dialog_present and not closed:
            log.warning("close_editor_safe: expected save dialog but none appeared (editor=%s)", editor)
        if not closed:
            try:
                driver.window_focus(title_contains=title_before)
            except Exception:
                pass
            try:
                driver.keyboard_hotkey("alt", "f4", ignore_safety=True)
            except Exception as exc:
                errors.append(f"alt_f4_retry_failed:{exc}")
            closed = _wait_editor_closed(time.time() + min(1.5, timeout_s))
        if not closed:
            errors.append("editor_window_still_open")

    ok = not errors
    result: Dict[str, Any] = {
        "ok": ok,
        "editor": editor,
        "dialog_present": dialog_present,
        "action_taken": action_taken,
    }
    if errors:
        result["errors"] = errors
        result["error"] = errors[0]
    log.info(
        "close_editor_safe: end ok=%s editor=%s dialog=%s action=%s errors=%s",
        ok,
        editor,
        dialog_present,
        action_taken,
        errors,
    )
    return result
