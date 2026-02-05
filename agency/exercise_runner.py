from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agency.tool_inventory import append_tool_use_note  # noqa: E402
from agency.exercise_schema import (
    normalize_verification,
    make_gap,
    extract_tools_used_from_steps,
)  # noqa: E402
try:
    from agency.motivo_vital import compute_motivo_vital, persist_mv  # noqa: E402
except Exception:  # pragma: no cover - opcional
    compute_motivo_vital = None  # type: ignore
    persist_mv = None  # type: ignore
DEFAULT_REGISTRY = ROOT / "config" / "exercises_registry.yaml"
DEFAULT_OUT = ROOT / "artifacts" / "exercises" / "exercise_results.json"


def _load_registry(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    data: Dict[str, Any] = {}
    if yaml:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}
    if not data and path.suffix.lower() in {".json", ".ndjson"}:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    exercises = data.get("exercises") if isinstance(data, dict) else None
    if not isinstance(exercises, list):
        raise ValueError("Registry malformed: expected 'exercises' list")
    normalized: List[Dict[str, Any]] = []
    for item in exercises:
        if not isinstance(item, dict):
            continue
        if not item.get("id") or not item.get("command"):
            continue
        normalized.append(item)
    return normalized


def _verify_artifact(path: Optional[str], expect_min_bytes: Optional[int]) -> bool:
    if not path:
        return True
    target = ROOT / path
    if not target.exists():
        return False
    try:
        if expect_min_bytes is not None and target.stat().st_size < expect_min_bytes:
            return False
    except Exception:
        return False
    return True


def _run_command(cmd: str, timeout: int) -> Dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
        )
        duration_ms = int((time.time() - started) * 1000)
        return {
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:4000],
            "stderr": (proc.stderr or "")[:4000],
            "duration_ms": duration_ms,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout": (exc.stdout or "")[:4000] if hasattr(exc, "stdout") else "",
            "stderr": f"timeout:{exc}",
            "duration_ms": int((time.time() - started) * 1000),
        }
    except Exception as exc:  # pragma: no cover - resiliencia
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"error:{exc}",
            "duration_ms": int((time.time() - started) * 1000),
        }


def _collect_tools_from_artifact(path: Optional[str]) -> List[str]:
    """
    Extrae acciones/herramientas de artefactos JSON de ejercicios (best-effort).
    """
    if not path:
        return []
    target = ROOT / path
    if not target.exists():
        return []
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return []
    tools: List[str] = []
    if isinstance(data, dict):
        tools.extend([str(t) for t in data.get("tools_used") or [] if str(t)])
        for proc in data.get("procedures") or []:
            if not isinstance(proc, dict):
                continue
            for step in proc.get("steps") or []:
                if not isinstance(step, dict):
                    continue
                action = str(step.get("action") or "").strip()
                if action:
                    tools.append(action)
    return sorted(set(t for t in tools if t))


def _load_artifact(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    target = ROOT / path
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _evaluate_gates(artifact: Optional[Dict[str, Any]], gates: List[str]) -> bool:
    if not gates:
        return True
    if not artifact:
        return False
    verification = artifact.get("verification") if isinstance(artifact.get("verification"), dict) else {}
    outcome = str(verification.get("outcome") or "").lower()
    ok = bool(verification.get("ok"))
    for gate in gates:
        if gate == "heartbeat_green":
            if not (ok and outcome == "success"):
                return False
        # futuros gates aquí
    return True


def _append_note(tool: str, exercise_id: str, outcome: str, meta: Dict[str, Any]) -> None:
    if not tool:
        return
    note = f"exercise:{exercise_id} outcome:{outcome}"
    try:
        append_tool_use_note(tool, note=note, outcome=outcome, meta=meta)
    except Exception:
        return


def run_exercises(exercises: List[Dict[str, Any]], only: Optional[List[str]] = None) -> Dict[str, Any]:
    selected_ids = set([i.strip() for i in (only or []) if i.strip()])
    results: List[Dict[str, Any]] = []
    aggregated_tools: List[str] = []
    for item in exercises:
        ex_id = str(item.get("id"))
        if selected_ids and ex_id not in selected_ids:
            continue
        cmd = str(item.get("command"))
        tool = str(item.get("tool") or "")
        timeout = int(item.get("timeout") or 90)
        verify_path = item.get("verify_path")
        expect_min_bytes = item.get("expect_min_bytes")
        required = bool(item.get("required"))
        gates = item.get("gates") or []
        started = time.time()
        cmd_result = _run_command(cmd, timeout=timeout)
        verify_ok = _verify_artifact(verify_path, expect_min_bytes)
        artifact_data = _load_artifact(verify_path) if verify_ok else None
        gates_ok = _evaluate_gates(artifact_data, gates)
        success = cmd_result["returncode"] == 0 and verify_ok and gates_ok
        outcome = "success" if success else "fail"
        artifact_tools = extract_tools_used_from_steps(
            (artifact_data or {}).get("procedures", [{}])[0].get("steps") if artifact_data else []
        )
        if not artifact_tools and verify_ok:
            artifact_tools = _collect_tools_from_artifact(verify_path)
        meta = {
            "cmd": cmd,
            "returncode": cmd_result["returncode"],
            "verify_ok": verify_ok,
            "gates_ok": gates_ok,
            "duration_ms": cmd_result["duration_ms"],
            "tools_used": artifact_tools,
        }
        seen_tools = set()
        _append_note(tool, ex_id, outcome, meta)
        if tool:
            seen_tools.add(tool)
            aggregated_tools.append(tool)
        for t in artifact_tools:
            if not t or t in seen_tools:
                continue
            note = f"exercise:{ex_id} artifact_tool:{t} outcome:{outcome}"
            try:
                append_tool_use_note(t, note=note, outcome=outcome, meta=meta)
            except Exception:
                continue
            aggregated_tools.append(t)
        results.append(
            {
                "id": ex_id,
                "description": item.get("description"),
                "tool": tool,
                "command": cmd,
                "returncode": cmd_result["returncode"],
                "stdout": cmd_result["stdout"],
                "stderr": cmd_result["stderr"],
                "verify_ok": verify_ok,
                "status": "success" if success else "fail",
                "required": required,
                "duration_ms": cmd_result["duration_ms"],
                "started_at": started,
            }
        )
    all_ok = all((not r.get("required")) or r.get("status") == "success" for r in results) if results else False
    any_ran = bool(results)
    outcome = "success" if all_ok else ("neutral" if any_ran else "neutral")
    verification = normalize_verification(
        {
            "outcome": outcome,
            "ok": all_ok and any_ran,
            "notes": [r["id"] for r in results if r.get("status") != "success"],
            "metrics": {"exercises_run": len(results)},
        },
        default_outcome=outcome,
    )
    gaps: List[Dict[str, Any]] = []
    for r in results:
        if r.get("status") != "success":
            gaps.append(make_gap("exercise", r.get("id", "unknown"), severity="high"))
        if r.get("required") and r.get("status") != "success":
            gaps.append(make_gap("exercise_required", r.get("id", "unknown"), severity="critical"))
    return {
        "timestamp": time.time(),
        "results": results,
        "verification": verification,
        "gaps": gaps,
        "observability_gaps": [g["code"] for g in gaps] if gaps else [],
        "tool_gaps": [],
        "tools_used": sorted(set(aggregated_tools)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Runner de ejercicios de mantenimiento (Gym).")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="Ruta al registry YAML/JSON.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Archivo JSON de resultados.")
    ap.add_argument("--only", type=str, default="", help="Comma-separated ids a ejecutar.")
    args = ap.parse_args()

    try:
        registry = _load_registry(args.registry)
    except Exception as exc:
        print(json.dumps({"error": f"registry_load_failed:{exc}"}))
        return 1

    selected = [s for s in args.only.split(",") if s.strip()] if args.only else []
    summary = run_exercises(registry, only=selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Recalcula MV post-Gym
    if compute_motivo_vital and persist_mv:
        try:
            mv = compute_motivo_vital()
            persist_mv(mv)
            try:
                summary["motivo_vital"] = {
                    "score": mv.score,
                    "dimensions": mv.dimensions,
                    "deltas": mv.deltas,
                }
                args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        except Exception:
            pass

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
