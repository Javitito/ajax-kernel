from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "motivo_vital.yaml"
MV_DIR = ROOT / "artifacts" / "motivo_vital"
MV_LATEST = MV_DIR / "latest.json"
MV_HISTORY = MV_DIR / "history.jsonl"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists() or not yaml:
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _list_capability_gaps() -> List[Path]:
    gaps_dir = ROOT / "artifacts" / "capability_gaps"
    if not gaps_dir.exists():
        return []
    return sorted([p for p in gaps_dir.glob("*.json") if p.is_file()])


def _breaker_tripped() -> bool:
    gov_dir = ROOT / "artifacts" / "governance"
    mission_state = _load_json(gov_dir / "mission_breaker_state.json") or {}
    infra_state = _load_json(gov_dir / "infra_breaker_state.json") or {}
    if mission_state.get("mission_should_block") or mission_state.get("last_blocked"):
        return True
    if infra_state.get("infra_should_block") or infra_state.get("last_blocked"):
        return True
    return False


def _load_tool_notes() -> Dict[str, List[Dict[str, Any]]]:
    notes_path = ROOT / "artifacts" / "tools" / "tool_use_notes.json"
    data = _load_json(notes_path)
    return data if isinstance(data, dict) else {}


def _load_gym_results() -> Dict[str, Any]:
    gym_path = ROOT / "artifacts" / "exercises" / "gym_daily.json"
    data = _load_json(gym_path)
    return data if isinstance(data, dict) else {}


def _latest_procedure_confidence() -> Optional[float]:
    exercises_dir = ROOT / "artifacts" / "exercises"
    candidates = sorted(exercises_dir.glob("procedure_induction*.json"))
    for path in reversed(candidates):
        data = _load_json(path)
        if not data:
            continue
        try:
            proc = (data.get("procedures") or [None])[0] or {}
            conf = float(proc.get("confidence"))
            return max(0.0, min(1.0, conf))
        except Exception:
            continue
    return None


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.5


@dataclass
class MVResult:
    score: float
    dimensions: Dict[str, float]
    deltas: Dict[str, float]
    signals: Dict[str, Dict[str, float]]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "score": round(self.score, 4),
            "dimensions": {k: round(v, 4) for k, v in self.dimensions.items()},
            "deltas": {k: round(v, 4) for k, v in self.deltas.items()},
            "signals": self.signals,
        }


def _compute_signals() -> Dict[str, float]:
    signals: Dict[str, float] = {}

    gym = _load_gym_results()
    results = gym.get("results") or []
    success_count = sum(1 for r in results if r.get("status") == "success" and r.get("verify_ok"))
    total_runs = len(results)
    fraction_success = success_count / total_runs if total_runs else 0.0
    verify_ok_fraction = (
        sum(1 for r in results if r.get("verify_ok")) / total_runs if total_runs else 0.0
    )
    tools_used = set()
    for r in results:
        for t in r.get("tools_used") or []:
            tools_used.add(str(t))

    tool_notes = _load_tool_notes()
    total_tools = len(tool_notes)
    reused_tools = sum(1 for entries in tool_notes.values() if len(entries) > 1)
    single_use_tools = sum(1 for entries in tool_notes.values() if len(entries) == 1)

    gaps_count = len(_list_capability_gaps())
    proc_conf = _latest_procedure_confidence()
    breaker_flag = _breaker_tripped()

    # Robustness
    signals["breaker_tripped"] = 0.0 if breaker_flag else 1.0
    signals["infra_recovery_success"] = 1.0 if not breaker_flag else 0.5
    signals["verification_strict_success"] = verify_ok_fraction

    # Autonomy
    signals["tasks_completed_without_manual_fix"] = fraction_success
    signals["self_heal_success"] = 0.7 if gaps_count == 0 else 0.4
    signals["gap_closed_without_human"] = max(0.0, 1.0 - min(1.0, gaps_count / 5.0))

    # Leverage
    reuse_ratio = reused_tools / total_tools if total_tools else 0.0
    signals["tool_reuse"] = reuse_ratio if total_tools else 0.5
    signals["procedure_induction_confidence"] = proc_conf if proc_conf is not None else 0.5
    signals["habit_crystallized"] = 0.7 if reuse_ratio >= 0.2 and total_tools >= 2 else 0.5

    # Curiosity
    signals["new_exercise_attempted"] = min(1.0, total_runs / 3.0) if total_runs else 0.3
    signals["new_tool_proposed"] = 0.7 if single_use_tools > 0 else (0.5 if total_tools else 0.3)
    signals["unexplored_tool_used"] = 0.6 if single_use_tools > 0 else 0.5

    return signals


def compute_motivo_vital(config_path: Path = CONFIG_PATH) -> MVResult:
    cfg = _load_yaml(config_path)
    dim_cfg = cfg.get("dimensions") if isinstance(cfg, dict) else {}
    weights: Dict[str, float] = {}
    for name, data in (dim_cfg or {}).items():
        try:
            weights[name] = float(data.get("weight", 0.0))
        except Exception:
            continue
    total_weight = sum(weights.values()) or 1.0
    signals = _compute_signals()

    dim_scores: Dict[str, float] = {}
    dim_signals: Dict[str, Dict[str, float]] = {}
    for name, data in (dim_cfg or {}).items():
        sig_list: List[str] = data.get("signals") or []
        sig_values: Dict[str, float] = {}
        for sig in sig_list:
            val = float(signals.get(sig, 0.5))
            sig_values[sig] = round(val, 4)
        dim_signals[name] = sig_values
        dim_scores[name] = _avg(list(sig_values.values())) if sig_values else 0.5

    overall = sum((weights.get(name, 0.0) / total_weight) * dim_scores.get(name, 0.5) for name in weights)

    # Penalizar día malo del gym (no subir sin verificación ok)
    gym = _load_gym_results()
    gym_ver = gym.get("verification") if isinstance(gym.get("verification"), dict) else {}
    gym_ok = bool(gym_ver.get("ok"))
    if not gym_ok:
        overall *= 0.98
        dim_scores = {k: v * 0.98 for k, v in dim_scores.items()}

    # deltas from history
    last_score = None
    if MV_HISTORY.exists():
        try:
            with MV_HISTORY.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "score" in obj:
                        last_score = float(obj["score"])
        except Exception:
            last_score = None
    deltas = {}
    if last_score is not None:
        deltas["score"] = overall - last_score
    # per-dimension deltas comparing to latest if present
    try:
        last = json.loads(MV_LATEST.read_text(encoding="utf-8")) if MV_LATEST.exists() else {}
        if isinstance(last, dict):
            last_dims = last.get("dimensions") or {}
            for name, val in dim_scores.items():
                try:
                    if name in last_dims:
                        deltas[name] = val - float(last_dims.get(name, 0.0))
                except Exception:
                    continue
    except Exception:
        pass

    ts = _iso_now()
    return MVResult(score=overall, dimensions=dim_scores, deltas=deltas, signals=dim_signals, timestamp=ts)


def persist_mv(result: MVResult) -> None:
    MV_DIR.mkdir(parents=True, exist_ok=True)
    latest = result.to_dict()
    MV_LATEST.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        with MV_HISTORY.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(latest, ensure_ascii=False) + "\n")
    except Exception:
        pass


def main() -> int:
    res = compute_motivo_vital()
    persist_mv(res)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
