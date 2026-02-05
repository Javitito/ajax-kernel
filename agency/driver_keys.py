from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
    env_root = os.environ.get("AJAX_HOME")
    if env_root:
        try:
            return Path(env_root)
        except Exception:
            pass
    return Path(__file__).resolve().parents[1]


def _public_key_paths() -> List[Path]:
    public_root = os.environ.get("PUBLIC") or r"C:\Users\Public"
    paths = [
        Path(public_root) / ".leann" / "os_api_key",
        Path(public_root) / ".ajax" / "ajax_api_key.txt",
    ]
    if os.name != "nt":
        paths.extend(
            [
                Path("/mnt/c/Users/Public/.leann/os_api_key"),
                Path("/mnt/c/Users/Public/.ajax/ajax_api_key.txt"),
            ]
        )
    return paths


def ajax_driver_key_paths(root: Optional[Path] = None) -> List[Path]:
    base = Path(root) if root else _repo_root()
    return [
        base / ".secret" / "ajax_api_key.txt",
        base / ".secrets" / "ajax_api_key.txt",
    ]


def load_ajax_driver_api_key() -> Optional[str]:
    env_val = os.environ.get("AJAX_API_KEY")
    if env_val:
        return env_val.strip()
    legacy_val = os.environ.get("OS_DRIVER_API_KEY")
    if legacy_val:
        return legacy_val.strip()
    for path in _public_key_paths():
        try:
            if path.is_file():
                data = path.read_text(encoding="utf-8").strip()
                if data:
                    return data
        except Exception:
            continue
    for path in ajax_driver_key_paths():
        try:
            if path.is_file():
                data = path.read_text(encoding="utf-8").strip()
                if data:
                    return data
        except Exception:
            continue
    return None


def require_ajax_driver_api_key() -> str:
    key = load_ajax_driver_api_key()
    if key:
        return key
    path = ajax_driver_key_paths()[0]
    raise SystemExit(
        f"Missing AJAX_API_KEY. Set env AJAX_API_KEY or OS_DRIVER_API_KEY or create {path}"
    )


def os_driver_key_paths() -> List[Path]:
    return ajax_driver_key_paths()


def load_os_driver_api_key() -> Optional[str]:
    return load_ajax_driver_api_key()
