"""Heartkeeper: supervisor de latido AJAX.

Decisiones deterministas (no IA):
- GREEN: registrar y dormir
- YELLOW: registrar
- RED: ejecutar jobs/fix_infra_basic.json vía broker y revaluar
- MISSING/UNKNOWN: generar latido (ajax_heartbeat.py) y continuar

Escalada: tras N ciclos consecutivos en rojo, genera capability_gap infra_marcapasos_failed_<session>.json.
Persistencia: artifacts/health/marcapasos_history.jsonl
Control: requiere flag ~/.ajax_enable_marcapasos
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
HEARTBEAT_PATH = ROOT / "artifacts" / "health" / "ajax_heartbeat.json"
HEARTBEAT_HISTORY = ROOT / "artifacts" / "health" / "ajax_heartbeat_history.jsonl"
MARCAPASOS_HISTORY = ROOT / "artifacts" / "health" / "marcapasos_history.jsonl"
GAP_DIR = ROOT / "artifacts" / "capability_gaps"
FLAG_PATH = Path("~/.ajax_enable_marcapasos").expanduser()
PIDFILE = ROOT / "artifacts" / "health" / "heartkeeper.pid"
HEARTBEAT_SCRIPT = ROOT / "agency" / "ajax_heartbeat.py"
FIX_INFRA_JOB = ROOT / "jobs" / "fix_infra_basic.json"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_history(entry: Dict[str, Any]) -> None:
    MARCAPASOS_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with MARCAPASOS_HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _run_cmd(cmd: list[str], timeout: float = 120.0) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        ok = proc.returncode == 0
        out = (proc.stdout or "") + (proc.stderr or "")
        return ok, out.strip()
    except Exception as exc:
        return False, f"exception:{exc}"


def _run_heartbeat() -> Dict[str, Any]:
    _run_cmd(["python", str(HEARTBEAT_SCRIPT)], timeout=60)
    return _read_json(HEARTBEAT_PATH)


def _run_fix_infra() -> Dict[str, Any]:
    cmd = ["python", "-m", "agency.broker", str(FIX_INFRA_JOB), "--print-result"]
    ok, out = _run_cmd(cmd, timeout=180)
    return {"ok": ok, "detail": out}


def _emit_gap(session: str, hb: Dict[str, Any], reason: str) -> Path:
    GAP_DIR.mkdir(parents=True, exist_ok=True)
    gap_id = f"infra_marcapasos_failed_{session}"
    payload = {
        "gap_id": gap_id,
        "capability_family": "infra.self_heal",
        "reason": reason,
        "heartbeat_status": hb,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_paths": [
            str(HEARTBEAT_PATH),
            str(HEARTBEAT_HISTORY),
            str(MARCAPASOS_HISTORY),
        ],
        "next_actions": [
            "Director: registrar en research_backlog.yaml",
            "Scout: buscar patrones de supervisión/auto-restart para web/driver",
        ],
    }
    gap_path = GAP_DIR / f"{gap_id}.json"
    gap_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return gap_path


def _status_label(hb: Dict[str, Any]) -> str:
    return str(hb.get("status", "unknown")).lower()


def _check_flag() -> bool:
    return FLAG_PATH.exists()


def _already_running() -> bool:
    if not PIDFILE.exists():
        return False
    try:
        pid = int(PIDFILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True
    except Exception:
        PIDFILE.unlink(missing_ok=True)
        return False


def run_loop(interval: float, max_cycles: Optional[int], escalate_after: int, session: str) -> None:
    if not _check_flag():
        print("Marcapasos deshabilitado (flag ~/.ajax_enable_marcapasos no existe).")
        return
    red_streak = 0
    cycles = 0
    while True:
        cycles += 1
        if not _check_flag():
            print("Flag desactivado; saliendo.")
            break
        hb = _read_json(HEARTBEAT_PATH)
        status = _status_label(hb)
        action = "noop"
        detail = ""

        if status in {"missing", "unknown"} or not hb:
            hb = _run_heartbeat()
            status = _status_label(hb)
            action = "heartbeat_probe"
            detail = "probe_generated"
        elif status == "green":
            red_streak = 0
            action = "sleep"
        elif status == "yellow":
            red_streak = 0
            action = "note"
        elif status == "red":
            fix_res = _run_fix_infra()
            hb = _run_heartbeat()
            status = _status_label(hb)
            action = "fix_infra"
            detail = fix_res.get("detail", "")
            red_streak = red_streak + 1 if status == "red" else 0
        else:
            action = "note"

        entry = {
            "ts": time.time(),
            "status": status,
            "action": action,
            "detail": detail,
            "session": session,
            "red_streak": red_streak,
        }
        _write_history(entry)

        if status == "red" and red_streak >= escalate_after:
            gap_path = _emit_gap(session, hb, reason=f"red_streak>={escalate_after}")
            entry["escalation"] = str(gap_path)
            _write_history(entry)

        if max_cycles and cycles >= max_cycles:
            break
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Heartkeeper: supervisor de latido AJAX.")
    parser.add_argument("--interval", type=float, default=float(os.getenv("HEARTKEEPER_INTERVAL", "30")), help="Intervalo entre ciclos (s)")
    parser.add_argument("--max-cycles", type=int, default=None, help="Máximo de ciclos (para test).")
    parser.add_argument("--escalate-after", type=int, default=int(os.getenv("HEARTKEEPER_ESCALATE_AFTER", "3")), help="Ciclos rojos consecutivos antes de escalar.")
    parser.add_argument("--session", type=str, default=None, help="ID de sesión (opcional).")
    parser.add_argument("--pidfile", type=Path, default=PIDFILE, help="Ruta de pidfile para modo daemon.")
    parser.add_argument("--daemon", action="store_true", help="Ejecutar en background (pidfile).")
    args = parser.parse_args(argv)

    session = args.session or f"heartkeeper_{int(time.time())}"

    if args.daemon:
        if _already_running():
            print("Heartkeeper ya está en ejecución.")
            return 0
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--interval", str(args.interval),
                                 "--max-cycles", str(args.max_cycles) if args.max_cycles else "0",
                                 "--escalate-after", str(args.escalate_after),
                                 "--session", session, "--pidfile", str(args.pidfile)],
                                cwd=ROOT, start_new_session=True)
        args.pidfile.parent.mkdir(parents=True, exist_ok=True)
        args.pidfile.write_text(str(proc.pid), encoding="utf-8")
        print(f"Heartkeeper iniciado (pid={proc.pid}).")
        return 0

    # foreground
    if args.pidfile:
        try:
            args.pidfile.parent.mkdir(parents=True, exist_ok=True)
            args.pidfile.write_text(str(os.getpid()), encoding="utf-8")
        except Exception:
            pass
    try:
        run_loop(args.interval, args.max_cycles if args.max_cycles and args.max_cycles > 0 else None, args.escalate_after, session)
    finally:
        try:
            args.pidfile.unlink(missing_ok=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
