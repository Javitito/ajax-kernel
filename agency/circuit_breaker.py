from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agency.voice.vibevoice_client import VoiceIO
except Exception:  # pragma: no cover - voz opcional
    VoiceIO = None  # type: ignore


ROOT_DIR = Path(__file__).resolve().parents[1]
GOV_DIR = ROOT_DIR / "artifacts" / "governance"
CAP_GAP_DIR = ROOT_DIR / "artifacts" / "capability_gaps"
HB_PATH = ROOT_DIR / "artifacts" / "health" / "ajax_heartbeat.json"


def _now() -> float:
    return time.time()


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _prune_timestamps(values: List[float], now: float, window: float) -> List[float]:
    pruned: List[float] = []
    for ts in values:
        try:
            ts_f = float(ts)
        except Exception:
            continue
        if now - ts_f <= window:
            pruned.append(ts_f)
    return pruned


def _hash_intent(text: str, extra: Optional[str] = None) -> str:
    raw = (text or "").strip()
    if extra:
        raw += f"::{extra}"
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()
    return digest[:16]


def speak_instability_alert(kind: str) -> None:
    """
    Emite un aviso por voz (best-effort) cuando se dispara un breaker.
    No bloquea si la voz falla o no está configurada.
    """
    messages = {
        "mission": "Sistemas inestables. Abortando misión.",
        "infra": "Sistemas inestables. Entrando en modo seguro de infraestructura.",
    }
    msg = messages.get(kind, "Sistemas inestables.")
    try:
        if VoiceIO:
            VoiceIO().speak(msg)
        else:
            print(f"🔊 {msg}")
    except Exception:
        pass


def _load_heartbeat_snapshot() -> Dict[str, Any]:
    if not HB_PATH.exists():
        return {"status": "unknown", "problems": ["missing_heartbeat"]}
    return _safe_read_json(HB_PATH) or {"status": "unknown"}


@dataclass
class MissionBreaker:
    base_dir: Path = ROOT_DIR
    threshold: int = 3
    window_seconds: float = 60.0
    state_path: Optional[Path] = None
    gap_dir: Optional[Path] = None
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        base = Path(self.base_dir)
        self.state_path = self.state_path or base / "artifacts" / "governance" / "mission_breaker_state.json"
        self.gap_dir = self.gap_dir or base / "artifacts" / "capability_gaps"
        self.state = _safe_read_json(self.state_path) or {}
        self.state.setdefault("failures", {})
        self.state.setdefault("window_seconds", self.window_seconds)
        self.state.setdefault("threshold", self.threshold)

    @staticmethod
    def hash_intention(text: str, extra: Optional[str] = None) -> str:
        return _hash_intent(text, extra)

    def _get_entry(self, intent_hash: str) -> Dict[str, Any]:
        failures = self.state.setdefault("failures", {})
        entry = failures.get(intent_hash) or {}
        entry.setdefault("ts", [])
        failures[intent_hash] = entry
        return entry

    def _emit_gap(self, intent_hash: str, entry: Dict[str, Any], now: float) -> Optional[Path]:
        try:
            ts_list = entry.get("ts", [])
            payload = {
                "capability_gap_id": f"mission_repeated_failure_{int(now)}",
                "intent_hash": intent_hash,
                "intent": entry.get("intent"),
                "failure_timestamps": ts_list,
                "threshold": self.threshold,
                "window_seconds": self.window_seconds,
                "last_error": entry.get("last_error"),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            }
            gap_dir = Path(self.gap_dir or CAP_GAP_DIR)
            gap_dir.mkdir(parents=True, exist_ok=True)
            gap_path = gap_dir / f"{payload['capability_gap_id']}.json"
            gap_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return gap_path
        except Exception:
            return None

    def _evaluate(self, intent_hash: str, entry: Dict[str, Any], now: float, *, intent_text: Optional[str], last_error: Optional[str], emit_gap: bool) -> bool:
        entry["ts"] = _prune_timestamps(entry.get("ts", []), now, self.window_seconds)
        if intent_text:
            entry["intent"] = intent_text
        if last_error:
            entry["last_error"] = last_error
        blocked = len(entry["ts"]) >= self.threshold
        if blocked and emit_gap:
            last_gap_ts = float(entry.get("last_gap_ts") or 0)
            if now - last_gap_ts >= max(1.0, self.window_seconds * 0.25):
                gap_path = self._emit_gap(intent_hash, entry, now)
                entry["last_gap_ts"] = now
                if gap_path:
                    entry["last_gap_path"] = str(gap_path)
                speak_instability_alert("mission")
        return blocked

    def mission_should_block(self, intent_hash: str, now: Optional[float] = None, intent_text: Optional[str] = None,
                              last_error: Optional[str] = None) -> bool:
        now = now or _now()
        entry = self._get_entry(intent_hash)
        blocked = self._evaluate(intent_hash, entry, now, intent_text=intent_text, last_error=last_error, emit_gap=True)
        _safe_write_json(Path(self.state_path), self.state)
        return blocked

    def mission_register_failure(self, intent_hash: str, now: Optional[float] = None, intent_text: Optional[str] = None,
                                 last_error: Optional[str] = None) -> None:
        now = now or _now()
        entry = self._get_entry(intent_hash)
        entry["ts"] = _prune_timestamps(entry.get("ts", []), now, self.window_seconds)
        entry["ts"].append(now)
        blocked = self._evaluate(intent_hash, entry, now, intent_text=intent_text, last_error=last_error, emit_gap=True)
        self.state.setdefault("last_update", now)
        self.state["last_blocked"] = blocked
        _safe_write_json(Path(self.state_path), self.state)


@dataclass
class InfraBreaker:
    base_dir: Path = ROOT_DIR
    threshold: int = 5
    window_seconds: float = 3600.0
    state_path: Optional[Path] = None
    gap_dir: Optional[Path] = None
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        base = Path(self.base_dir)
        self.state_path = self.state_path or base / "artifacts" / "governance" / "infra_breaker_state.json"
        self.gap_dir = self.gap_dir or base / "artifacts" / "capability_gaps"
        self.state = _safe_read_json(self.state_path) or {}
        self.state.setdefault("failures", [])
        self.state.setdefault("window_seconds", self.window_seconds)
        self.state.setdefault("threshold", self.threshold)

    def _prune_failures(self, now: float) -> None:
        failures = self.state.get("failures") or []
        pruned: List[Dict[str, Any]] = []
        for item in failures:
            ts = item.get("ts")
            try:
                ts_f = float(ts)
            except Exception:
                continue
            if now - ts_f <= self.window_seconds:
                pruned.append({"ts": ts_f, "kind": item.get("kind")})
        self.state["failures"] = pruned

    def _emit_gap(self, now: float) -> Optional[Path]:
        try:
            failures = self.state.get("failures") or []
            heartbeat = _load_heartbeat_snapshot()
            payload = {
                "capability_gap_id": f"infra_meltdown_{int(now)}",
                "failures": failures,
                "threshold": self.threshold,
                "window_seconds": self.window_seconds,
                "heartbeat_snapshot": heartbeat,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
            }
            gap_dir = Path(self.gap_dir or CAP_GAP_DIR)
            gap_dir.mkdir(parents=True, exist_ok=True)
            gap_path = gap_dir / f"{payload['capability_gap_id']}.json"
            gap_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            return gap_path
        except Exception:
            return None

    def infra_should_block(self, now: Optional[float] = None) -> bool:
        now = now or _now()
        self._prune_failures(now)
        blocked = len(self.state.get("failures") or []) >= self.threshold
        if blocked:
            last_gap_ts = float(self.state.get("last_gap_ts") or 0)
            if now - last_gap_ts >= max(1.0, self.window_seconds * 0.25):
                gap_path = self._emit_gap(now)
                self.state["last_gap_ts"] = now
                if gap_path:
                    self.state["last_gap_path"] = str(gap_path)
                speak_instability_alert("infra")
        _safe_write_json(Path(self.state_path), self.state)
        return blocked

    def infra_register_failure(self, kind: str, now: Optional[float] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        now = now or _now()
        self._prune_failures(now)
        entry = {"ts": now, "kind": str(kind or "unknown")}
        if meta:
            entry["meta"] = meta
        failures = self.state.get("failures") or []
        failures.append(entry)
        self.state["failures"] = failures
        blocked = self.infra_should_block(now)
        self.state.setdefault("last_update", now)
        self.state["last_blocked"] = blocked
        _safe_write_json(Path(self.state_path), self.state)


__all__ = ["MissionBreaker", "InfraBreaker", "speak_instability_alert"]
