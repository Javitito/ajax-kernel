"""
Módulo de Fricción para AJAX.
Calcula un score [0,1] basado en señales de resistencia/bloqueo del sistema con decaimiento temporal.
"""
from __future__ import annotations

import json
import time
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

@dataclass
class FrictionSignals:
    waiting_count: float = 0.0
    recent_failures: float = 0.0
    recent_retries: float = 0.0
    unavailable_providers: float = 0.0
    budget_exhausted: bool = False
    
@dataclass
class FrictionResult:
    score: float
    components: Dict[str, Any]
    
def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))

def compute_friction(signals: FrictionSignals) -> FrictionResult:
    # Weights refinados (v0.1.2) con soporte para floats (decay)
    # waiting: 0.2 por misión, máximo 0.5
    comp_waiting = min(0.5, signals.waiting_count * 0.2)
    
    # failures: 0.15 por peso acumulado (con decay), máximo 0.4
    comp_failures = min(0.4, signals.recent_failures * 0.15)
    
    # unavailable: 0.05 por proveedor (con decay), máximo 0.3
    comp_unavailable = min(0.3, signals.unavailable_providers * 0.05)
    
    # budget: 0.4 fijo si está agotado
    comp_budget = 0.4 if signals.budget_exhausted else 0.0
    
    # retries: 0.05 por reintento (con decay), máximo 0.2
    comp_retries = min(0.2, signals.recent_retries * 0.05)
    
    total = comp_waiting + comp_failures + comp_unavailable + comp_retries + comp_budget
    score = _clamp01(total)
    
    # Determinar causa principal para UX (mostramos solo señales materiales > 0.1)
    reasons = []
    if signals.waiting_count >= 1: reasons.append(f"{int(signals.waiting_count)} WAITING")
    if signals.budget_exhausted: reasons.append("BUDGET EXHAUSTED")
    if signals.recent_failures > 0.5: reasons.append("RECENT FAILURES")
    if signals.unavailable_providers > 0.5: reasons.append("PROVIDERS DOWN")
    
    primary_reason = ", ".join(reasons) if reasons else "clean"
    
    return FrictionResult(
        score=score,
        components={
            "waiting": comp_waiting,
            "failures": comp_failures,
            "unavailable": comp_unavailable,
            "retries": comp_retries,
            "budget": comp_budget,
            "primary_reason": primary_reason
        }
    )

def collect_friction_signals(root_dir: Path) -> FrictionSignals:
    signals = FrictionSignals()
    now = time.time()
    window_24h = 24 * 3600
    
    # 1. Waiting missions (sin decay, si están, están)
    try:
        waiting_dir = root_dir / "artifacts" / "waiting_for_user"
        if waiting_dir.exists():
            signals.waiting_count = float(len(list(waiting_dir.glob("*.json"))))
    except Exception:
        pass
        
    # 2. Receipts (last 10 execs, last 24h decay)
    try:
        receipts_dir = root_dir / "artifacts" / "receipts"
        if receipts_dir.exists():
            execs = sorted(receipts_dir.glob("exec_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]
            for p in execs:
                mtime = p.stat().st_mtime
                age = now - mtime
                if age > window_24h:
                    continue
                
                # Decay lineal: 1.0 ahora, 0.0 en 24h
                decay = max(0.0, 1.0 - (age / window_24h))
                
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if data.get("verify_ok") is False:
                        signals.recent_failures += decay
                    
                    err = str(data.get("error") or "")
                    if "budget" in err.lower() or "exhausted" in err.lower():
                        if decay > 0.5: # Solo si es "reciente" (12h)
                            signals.budget_exhausted = True
                except Exception:
                    pass
    except Exception:
        pass

    # 3. Ledger (decay más agresivo: 1h)
    try:
        ledger_path = root_dir / "artifacts" / "provider_ledger" / "latest.json"
        if ledger_path.exists():
            mtime = ledger_path.stat().st_mtime
            age = now - mtime
            # Si el ledger tiene más de 1h, su impacto cae linealmente hasta las 4h
            ledger_decay = max(0.0, 1.0 - (age / (4 * 3600)))
            
            if ledger_decay > 0:
                data = json.loads(ledger_path.read_text(encoding="utf-8"))
                rows = data.get("rows", [])
                for row in rows:
                    if row.get("status") != "ok":
                        signals.unavailable_providers += ledger_decay
    except Exception:
        pass
        
    return signals


def _iso_utc(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _utc_stamp(ts: float | None = None) -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(ts or time.time()))


def _is_local_provider_name(name: str) -> bool:
    n = str(name or "").strip().lower()
    return "lmstudio" in n or "ollama" in n or n.startswith("local_")


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _archive_waiting_for_user(
    *,
    waiting_dir: Path,
    apply: bool,
    older_than_seconds: float,
    now_ts: float,
) -> Dict[str, Any]:
    archived: List[str] = []
    candidates: List[str] = []
    skipped_recent = 0
    archive_dir = waiting_dir / "_archived" / time.strftime("%Y-%m-%d", time.gmtime(now_ts))

    if not waiting_dir.exists():
        return {
            "waiting_dir": str(waiting_dir),
            "archive_dir": str(archive_dir),
            "candidates": [],
            "archived": [],
            "skipped_recent": 0,
        }

    for path in sorted(waiting_dir.glob("*.json")):
        try:
            age = max(0.0, now_ts - float(path.stat().st_mtime))
        except Exception:
            age = 0.0
        if age < float(older_than_seconds):
            skipped_recent += 1
            continue
        candidates.append(str(path))
        if apply:
            archive_dir.mkdir(parents=True, exist_ok=True)
            target = _next_available_path(archive_dir / path.name)
            shutil.move(str(path), str(target))
            archived.append(str(target))

    return {
        "waiting_dir": str(waiting_dir),
        "archive_dir": str(archive_dir),
        "candidates": candidates,
        "archived": archived,
        "skipped_recent": skipped_recent,
    }


def _reset_provider_ledger_minimum_budget(
    *,
    ledger_path: Path,
    apply: bool,
    now_ts: float,
) -> Dict[str, Any]:
    snapshots_dir = ledger_path.parent / "_snapshots"
    snapshot_path = snapshots_dir / f"latest_{_utc_stamp(now_ts)}.json"
    ledger_exists = ledger_path.exists()
    payload: Dict[str, Any] = {
        "ledger_path": str(ledger_path),
        "ledger_exists": ledger_exists,
        "snapshot_path": str(snapshot_path),
        "local_rows_kept_ok": 0,
        "cloud_rows_soft_failed": 0,
        "applied": False,
    }

    if not ledger_exists:
        return payload

    doc = _safe_read_json(ledger_path)
    rows = doc.get("rows")
    if not isinstance(rows, list):
        rows = []
        doc["rows"] = rows

    if apply:
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ledger_path, snapshot_path)

        for row in rows:
            if not isinstance(row, dict):
                continue
            provider = str(row.get("provider") or "").strip()
            if _is_local_provider_name(provider):
                payload["local_rows_kept_ok"] += 1
                continue
            payload["cloud_rows_soft_failed"] += 1
            row["status"] = "soft_fail"
            row["reason"] = "minimum_budget_mode"
            row["cooldown_until_ts"] = now_ts + 3600.0
            row["cooldown_until"] = _iso_utc(now_ts + 3600.0)
            details = row.get("details")
            if not isinstance(details, dict):
                details = {}
                row["details"] = details
            details["source"] = "friction_gc"
            details["detail"] = "minimum_budget_mode"

        doc["updated_ts"] = now_ts
        doc["updated_utc"] = _iso_utc(now_ts)
        doc["mode"] = "minimum_budget"
        doc["gc_policy"] = "friction_gc_safe.v0"
        _safe_write_json(ledger_path, doc)
        payload["applied"] = True

    return payload


def run_friction_gc(
    *,
    root_dir: Path,
    apply: bool,
    older_than_hours: float = 24.0,
) -> Dict[str, Any]:
    """
    SAFE Friction GC:
    - Archiva waiting_for_user antiguos (sin borrar irreversible)
    - Snapshot + reset de provider_ledger a minimum_budget_mode (si existe)
    - Escribe receipt artifacts/receipts/friction_gc_<ts>.json
    """
    now_ts = time.time()
    root = Path(root_dir)
    older_than_seconds = max(0.0, float(older_than_hours) * 3600.0)

    waiting_dir = root / "artifacts" / "waiting_for_user"
    ledger_path = root / "artifacts" / "provider_ledger" / "latest.json"
    receipts_dir = root / "artifacts" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)

    waiting_summary = _archive_waiting_for_user(
        waiting_dir=waiting_dir,
        apply=apply,
        older_than_seconds=older_than_seconds,
        now_ts=now_ts,
    )
    ledger_summary = _reset_provider_ledger_minimum_budget(
        ledger_path=ledger_path,
        apply=apply,
        now_ts=now_ts,
    )

    summary = {
        "schema": "ajax.ops.friction_gc.v0",
        "created_at": _iso_utc(now_ts),
        "mode": "apply" if apply else "dry_run",
        "older_than_hours": float(older_than_hours),
        "waiting_for_user": waiting_summary,
        "provider_ledger": ledger_summary,
    }

    receipt_path = receipts_dir / f"friction_gc_{_utc_stamp(now_ts)}.json"
    _safe_write_json(receipt_path, summary)
    summary["receipt_path"] = str(receipt_path)
    return summary
