from __future__ import annotations

import calendar
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]


def _is_truthy(val: Optional[str]) -> bool:
    return str(val or "").strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _wsl_to_windows_path(path: str) -> str:
    if path.startswith("/mnt/"):
        parts = path.split("/")
        if len(parts) >= 4:
            drive = parts[2]
            rest = "\\".join(parts[3:])
            return f"{drive.upper()}:\\{rest}"
    return path


def _parse_shot_ts(name: str) -> Optional[float]:
    if not name.startswith("shot_"):
        return None
    rest = name[len("shot_"):]
    ts_part, _, _ = rest.partition("_")
    if not ts_part:
        return None
    try:
        dt = time.strptime(ts_part, "%Y%m%dT%H%M%SZ")
    except Exception:
        return None
    return float(calendar.timegm(dt))


def _write_dummy_png(path: Path) -> tuple[int, int]:
    # 1x1 PNG, no external deps.
    payload = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00"
        b"\x00\x02\x00\x01\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    path.write_bytes(payload)
    return 1, 1


def _parse_json_from_stdout(stdout: str) -> Dict[str, Any]:
    if not stdout:
        return {}
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    return payload
            except Exception:
                continue
    return {}


def _capture_with_powershell(
    output_path: Path, *, active_window: bool, display_id: Optional[int] = None
) -> Dict[str, Any]:
    ps = shutil.which("powershell.exe") or shutil.which("powershell")
    if not ps:
        raise RuntimeError("powershell_not_found")
    script = ROOT / "scripts" / "desktop_snap.ps1"
    if not script.exists():
        raise RuntimeError(f"missing_script:{script}")
    out_win = _wsl_to_windows_path(str(output_path))
    script_win = _wsl_to_windows_path(str(script))
    cmd = [
        ps,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        script_win,
        "-OutputPath",
        out_win,
    ]
    if display_id is not None:
        cmd.extend(["-DisplayId", str(int(display_id))])
    if active_window:
        cmd.append("-ActiveWindow")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        raise RuntimeError(f"powershell_failed:{res.returncode}:{stderr[:200]}")
    payload = _parse_json_from_stdout(res.stdout)
    if not payload:
        raise RuntimeError("powershell_no_json")
    return payload


def _list_displays_with_powershell() -> Dict[str, Any]:
    ps = shutil.which("powershell.exe") or shutil.which("powershell")
    if not ps:
        raise RuntimeError("powershell_not_found")
    script = ROOT / "scripts" / "desktop_list_displays.ps1"
    if not script.exists():
        raise RuntimeError(f"missing_script:{script}")
    script_win = _wsl_to_windows_path(str(script))
    cmd = [
        ps,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        script_win,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        raise RuntimeError(f"powershell_failed:{res.returncode}:{stderr[:200]}")
    payload = _parse_json_from_stdout(res.stdout)
    if not payload:
        raise RuntimeError("powershell_no_json")
    return payload


def list_local_displays() -> Dict[str, Any]:
    return _list_displays_with_powershell()


def _enforce_retention(out_dir: Path) -> None:
    try:
        max_per_day = int(os.getenv("AJAX_SNAP_MAX_PER_DAY", "50") or 50)
    except Exception:
        max_per_day = 50
    try:
        ttl_days = float(os.getenv("AJAX_SNAP_TTL_DAYS", "7") or 7)
    except Exception:
        ttl_days = 7.0
    if max_per_day <= 0 and ttl_days <= 0:
        return
    now = time.time()
    entries: list[tuple[str, float]] = []
    for path in out_dir.glob("shot_*.png"):
        ts = _parse_shot_ts(path.name)
        if ts is None:
            continue
        entries.append((path.stem, ts))
    # TTL purge
    if ttl_days > 0:
        ttl_seconds = ttl_days * 86400.0
        for stem, ts in entries:
            if now - ts > ttl_seconds:
                for ext in (".png", ".json"):
                    target = out_dir / f"{stem}{ext}"
                    try:
                        if target.exists():
                            target.unlink()
                    except Exception:
                        pass
    # Per-day cap
    if max_per_day > 0:
        buckets: dict[str, list[tuple[str, float]]] = {}
        for stem, ts in entries:
            day = time.strftime("%Y%m%d", time.gmtime(ts))
            buckets.setdefault(day, []).append((stem, ts))
        for items in buckets.values():
            items.sort(key=lambda x: x[1], reverse=True)
            for stem, _ts in items[max_per_day:]:
                for ext in (".png", ".json"):
                    target = out_dir / f"{stem}{ext}"
                    try:
                        if target.exists():
                            target.unlink()
                    except Exception:
                        pass


def capture_desktop_snapshot(
    *,
    root_dir: Optional[Path] = None,
    active_window: bool = False,
    mission_id: Optional[str] = None,
    job_id: Optional[str] = None,
    context: Optional[str] = None,
    mock: Optional[bool] = None,
    display_id: Optional[int] = None,
    selection: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(root_dir) if root_dir else ROOT
    out_dir = root / "artifacts" / "observability" / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    ctx = str(context or mission_id or job_id or "manual").strip() or "manual"
    stem = f"shot_{ts}_{ctx}"
    png_path = out_dir / f"{stem}.png"
    json_path = out_dir / f"{stem}.json"

    use_mock = _is_truthy(os.getenv("AJAX_SNAP_MOCK")) if mock is None else bool(mock)
    capture_info: Dict[str, Any] = {}
    width = None
    height = None
    if use_mock:
        width, height = _write_dummy_png(png_path)
        capture_info = {"active_window_title": "mock_window"}
    else:
        capture_info = _capture_with_powershell(
            png_path, active_window=active_window, display_id=display_id
        )
        width = capture_info.get("width")
        height = capture_info.get("height")
        if not png_path.exists():
            raise RuntimeError(f"screenshot_missing:{png_path}")

    selected = selection or {}
    selected_display_id = selected.get("display_id")
    if selected_display_id is None:
        selected_display_id = capture_info.get("display_id")
    if selected_display_id is None:
        selected_display_id = display_id
    display_entry = selected.get("display")
    if display_entry is None and capture_info.get("display_bounds") is not None:
        display_entry = {
            "id": capture_info.get("display_id"),
            "name": capture_info.get("display_name"),
            "bounds": capture_info.get("display_bounds"),
            "is_primary": capture_info.get("display_primary"),
        }

    meta = {
        "ts": time.time(),
        "ts_utc": _utc_now(),
        "host": socket.gethostname(),
        "resolution": {"width": width, "height": height},
        "active_window": bool(active_window),
        "active_window_title": capture_info.get("active_window_title") or None,
        "mission_id": mission_id,
        "job_id": job_id,
        "context": ctx,
        "screenshot_path": str(png_path),
        "selected_display_id": selected_display_id,
        "selected_display_label": selected.get("display_label"),
        "selected_by": selected.get("selected_by"),
        "display": display_entry,
        "display_warnings": selected.get("warnings") or capture_info.get("warnings") or [],
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    _enforce_retention(out_dir)
    return {
        "ok": True,
        "png_path": str(png_path),
        "json_path": str(json_path),
        "meta": meta,
    }
