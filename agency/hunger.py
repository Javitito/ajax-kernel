"""
Módulo de Hambre v0.1 para AJAX.
Calcula la disonancia entre ambición y capacidad para impulsar la exploración.
"""
from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

from agency.friction import collect_friction_signals, compute_friction


@dataclass
class HungerInputs:
    user_ambition: float
    system_capacity: float
    context_novelty: float
    capabilities_signature: str
    friction_score: float = 0.0
    cap_sig_schema: str = "v1.2" # Versionado de formato de firma


@dataclass
class HungerComponents:
    mismatch: float
    novelty_score: float
    friction_impact: float = 0.0


@dataclass
class HungerDecision:
    explore_budget: float
    intensity: str
    gap_threshold: int


@dataclass
class HungerState:
    hunger: float
    components: HungerComponents
    decision: HungerDecision
    inputs: HungerInputs
    policy_version: str
    timestamp: float
    evidence_min: str = "computed_v0.1"


DEFAULT_POLICY = {
    "version": "0.1",
    "weights": {"mismatch": 0.7, "novelty": 0.3, "friction": 0.5},
    "bands": [
        {"max": 0.3, "explore_budget": 0.1, "intensity": "low", "gap_threshold": 80},
        {"max": 0.7, "explore_budget": 0.4, "intensity": "med", "gap_threshold": 60},
        {"max": 1.0, "explore_budget": 0.8, "intensity": "high", "gap_threshold": 40},
    ],
    "defaults": {
        "user_ambition": 0.5,
        "system_capacity": 0.5,
        "context_novelty": 0.0,
    },
}


def load_policy(root_dir: Path) -> Dict[str, Any]:
    path = root_dir / "config" / "hunger_policy.yaml"
    if path.exists() and yaml:
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or DEFAULT_POLICY
        except Exception:
            pass
    return DEFAULT_POLICY


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def estimate_capacity(root_dir: Path) -> float:
    """Calcula capacidad real basada en slots de modelos operativos."""
    slots_path = root_dir / "config" / "model_slots.json"
    inv_path = root_dir / "config" / "model_inventory_cloud.json"
    if not slots_path.exists() or not inv_path.exists():
        return 0.5
    try:
        slots = json.loads(slots_path.read_text(encoding="utf-8"))
        inventory = json.loads(inv_path.read_text(encoding="utf-8"))
        inv_list = inventory.get("providers") or []
        present = {(str(item.get("provider")), str(item.get("id"))) for item in inv_list if item}
        
        total = 0
        ok_count = 0
        for slot, costs in slots.items():
            if not isinstance(costs, dict): continue
            for cost_mode, val in costs.items():
                if not isinstance(val, str) or ":" not in val: continue
                total += 1
                provider, model_id = val.split(":", 1)
                if (provider, model_id) in present:
                    ok_count += 1
        if total == 0: return 1.0
        return ok_count / total
    except Exception:
        return 0.5


def get_capabilities_signatures(root_dir: Path) -> Dict[str, str]:
    """Genera firmas separadas para cambios materiales y auxiliares."""
    core_files = [
        root_dir / "config" / "model_slots.json",
        root_dir / "config" / "model_providers.yaml",
    ]
    # Eliminamos model_providers.json por riesgo de no-determinismo (ruido cosmético)
    aux_files = [
        root_dir / "config" / "model_utility.yaml",
    ]
    
    def _hash_files(files):
        hasher = hashlib.md5()
        for f in files:
            if f.exists():
                hasher.update(f.read_bytes())
        return hasher.hexdigest()[:12]

    return {
        "core": _hash_files(core_files),
        "aux": _hash_files(aux_files)
    }


def compute_hunger(
    inputs: HungerInputs, prev_state: Optional[HungerState], policy: Dict[str, Any]
) -> HungerState:
    weights = policy.get("weights", DEFAULT_POLICY["weights"])
    w_mismatch = float(weights.get("mismatch", 0.7))
    w_novelty = float(weights.get("novelty", 0.3))

    mismatch = max(0.0, inputs.user_ambition - inputs.system_capacity)
    
    novelty_val = inputs.context_novelty
    if prev_state:
        # 1. Detectar cambio de formato (schema)
        schema_changed = inputs.cap_sig_schema != getattr(prev_state.inputs, "cap_sig_schema", "v1.1")
        
        # 2. Extraer firmas
        prev_sig = prev_state.inputs.capabilities_signature
        current_sigs = json.loads(inputs.capabilities_signature) if inputs.capabilities_signature.startswith("{") else {"core": inputs.capabilities_signature, "aux": ""}
        prev_sigs = json.loads(prev_sig) if prev_sig.startswith("{") else {"core": prev_sig, "aux": ""}

        if schema_changed:
            # Si cambia el schema, ignoramos el pico de novedad o lo ponemos muy bajo
            novelty_val = 0.2
        elif current_sigs.get("core") != prev_sigs.get("core"):
            novelty_val = 1.0 # Cambio material (slots/providers)
        elif current_sigs.get("aux") != prev_sigs.get("aux"):
            novelty_val = max(novelty_val, 0.2) # Cambio menor

    weighted_score = (w_mismatch * mismatch) + (w_novelty * novelty_val)
    hunger_score = _clamp01(weighted_score)

    bands = policy.get("bands", DEFAULT_POLICY["bands"])
    selected_band = bands[-1]
    for band in bands:
        if hunger_score <= float(band.get("max", 1.0)):
            selected_band = band
            break

    decision = HungerDecision(
        explore_budget=float(selected_band.get("explore_budget", 0.1)),
        intensity=str(selected_band.get("intensity", "low")),
        gap_threshold=int(selected_band.get("gap_threshold", 80)),
    )

    components = HungerComponents(
        mismatch=mismatch, 
        novelty_score=novelty_val,
        friction_impact=inputs.friction_score
    )

    return HungerState(
        hunger=hunger_score,
        components=components,
        decision=decision,
        inputs=inputs,
        policy_version=str(policy.get("version", "0.1")),
        timestamp=time.time(),
    )


def persist_state(root_dir: Path, state: HungerState) -> None:
    out_dir = root_dir / "artifacts" / "hunger"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "state_latest.json"
    data = asdict(state)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_state(root_dir: Path) -> Optional[HungerState]:
    path = root_dir / "artifacts" / "hunger" / "state_latest.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Reconstruct dataclasses safely
        inputs_data = data["inputs"]
        # Handle cases where cap_sig_schema might be missing in old state
        if "cap_sig_schema" not in inputs_data:
            inputs_data["cap_sig_schema"] = "v1.1"
            
        inputs = HungerInputs(**inputs_data)
        components = HungerComponents(**data["components"])
        decision = HungerDecision(**data["decision"])
        del data["inputs"]
        del data["components"]
        del data["decision"]
        return HungerState(
            inputs=inputs,
            components=components,
            decision=decision,
            **data
        )
    except Exception:
        return None


def get_current_hunger(root_dir: Path) -> HungerState:
    """Inyecta señales reales de capacidad, firma y fricción."""
    policy = load_policy(root_dir)
    prev = load_state(root_dir)
    defaults = policy.get("defaults", DEFAULT_POLICY["defaults"])
    weights = policy.get("weights", DEFAULT_POLICY["weights"])
    
    capacity = estimate_capacity(root_dir)
    signatures = get_capabilities_signatures(root_dir)
    
    friction_signals = collect_friction_signals(root_dir)
    friction_res = compute_friction(friction_signals)
    
    base_ambition = float(defaults.get("user_ambition", 0.5))
    w_friction = float(weights.get("friction", 0.5))
    ambition = _clamp01(base_ambition + (w_friction * friction_res.score))
    
    inputs = HungerInputs(
        user_ambition=ambition,
        system_capacity=capacity,
        context_novelty=0.0,
        capabilities_signature=json.dumps(signatures),
        friction_score=friction_res.score,
        cap_sig_schema="v1.2" # Actualizado
    )
    
    state = compute_hunger(inputs, prev, policy)
    reason = friction_res.components.get("primary_reason", "clean")
    state.evidence_min = f"friction_by:{reason}"
    
    persist_state(root_dir, state)
    return state
