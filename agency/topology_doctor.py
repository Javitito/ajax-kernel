from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _ts_label(ts: Optional[float] = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts or time.time()))


def _utc_now(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _git_toplevel(path: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    top = (proc.stdout or "").strip()
    return top or None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def inspect_topology(*, kernel_root: Path, cwd: Optional[Path] = None) -> Dict[str, Any]:
    now = time.time()
    cwd_path = (cwd or Path.cwd()).resolve()
    kernel = Path(kernel_root).resolve()
    ajax_home_candidate = kernel.parent if kernel.name == "ajax-kernel" else kernel
    cwd_top = _git_toplevel(cwd_path)
    kernel_top = _git_toplevel(kernel)
    home_top = _git_toplevel(ajax_home_candidate) if ajax_home_candidate != kernel else None
    warnings: list[str] = []

    in_kernel_tree = _is_relative_to(cwd_path, kernel)
    in_home_tree = _is_relative_to(cwd_path, ajax_home_candidate)
    location = "outside"
    if in_kernel_tree:
        location = "ajax-kernel"
    elif in_home_tree:
        location = "ajax-home-root"

    mismatch = False
    reason = "topology_ok"
    if location == "ajax-home-root":
        mismatch = True
        reason = "running_from_root_not_kernel"
        warnings.append(
            "Estás en ROOT/AJAX_HOME; este comando pertenece al repo ajax-kernel."
        )
    elif location == "outside":
        mismatch = True
        reason = "cwd_outside_ajax_topology"
        warnings.append("El directorio actual está fuera de AJAX_HOME/ajax-kernel.")
    elif cwd_top and kernel_top and cwd_top != kernel_top:
        mismatch = True
        reason = "git_toplevel_mismatch"
        warnings.append("El git toplevel actual no coincide con el git toplevel de ajax-kernel.")

    recommended_cd = f"cd {kernel}"
    recommended_cmd = "python bin/ajaxctl doctor topology"
    payload: Dict[str, Any] = {
        "schema": "ajax.topology_doctor.v1",
        "ts_utc": _utc_now(now),
        "ok": not mismatch,
        "reason": reason,
        "cwd": str(cwd_path),
        "git_toplevel_current": cwd_top,
        "git_toplevel_kernel": kernel_top,
        "git_toplevel_ajax_home": home_top,
        "paths": {
            "ajax_home_candidate": str(ajax_home_candidate),
            "ajax_kernel": str(kernel),
        },
        "location": location,
        "warnings": warnings,
        "recommended": {
            "cd": recommended_cd,
            "command": recommended_cmd,
            "full": f"{recommended_cd} && {recommended_cmd}",
        },
    }
    return payload


def write_topology_receipt(root_dir: Path, payload: Dict[str, Any]) -> Path:
    receipt_dir = Path(root_dir) / "artifacts" / "receipts"
    receipt_path = receipt_dir / f"topology_doctor_{_ts_label()}.json"
    _safe_write_json(receipt_path, payload)
    return receipt_path


def run_topology_doctor(*, kernel_root: Path, cwd: Optional[Path] = None) -> Tuple[Dict[str, Any], Path]:
    payload = inspect_topology(kernel_root=kernel_root, cwd=cwd)
    receipt_path = write_topology_receipt(kernel_root, payload)
    payload["receipt_path"] = str(receipt_path)
    _safe_write_json(receipt_path, payload)
    return payload, receipt_path

