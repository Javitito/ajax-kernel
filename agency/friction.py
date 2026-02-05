"""
Módulo de Fricción para AJAX.
Calcula un score [0,1] basado en señales de resistencia/bloqueo del sistema con decaimiento temporal.
"""
from __future__ import annotations

import json
import time
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