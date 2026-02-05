from __future__ import annotations

import json
import re
import time
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class CrystallizationError(RuntimeError):
    """Errores de pipeline de cristalización."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - surfaced as CrystallizationError
        raise CrystallizationError(f"No existe {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - surfaced as CrystallizationError
        raise CrystallizationError(f"JSON inválido en {path}: {exc}") from exc


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-") or "generic"


def _utc_iso(ts: Optional[float] = None) -> str:
    dt = _dt.datetime.utcfromtimestamp(ts or time.time()).replace(microsecond=0)
    return dt.isoformat() + "Z"


def _stringify_success_spec(spec: Any) -> List[str]:
    if spec is None:
        return []
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, list):
        return [str(item) for item in spec if item is not None]
    if isinstance(spec, dict):
        return [f"{k}:{spec[k]}" for k in sorted(spec)]
    return [str(spec)]


_BANNED_ARG_KEYS = {"provider", "providers", "model", "models", "tier", "llm"}
DEFAULT_SIGNAL_REGISTRY = {
    "canonical": {
        "execution_result.success",
        "driver.screenshot",
        "driver.active_window",
        "driver.ui_inspect",
        "uia.active_window_process_in",
        "vision.visual_audit",
        "runner.ok",
    },
    "aliases": {
        "success": "execution_result.success",
        "runner_success": "runner.ok",
        "uia.active_window": "uia.active_window_process_in",
        "driver.activewindow": "driver.active_window",
        "driver.ui": "driver.ui_inspect",
        "vision.audit": "vision.visual_audit",
        "active_window_process": "uia.active_window_process_in",
    },
}


def _sanitize_args(args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(args, dict):
        return {}
    clean: Dict[str, Any] = {}
    for key, value in args.items():
        if key.lower() in _BANNED_ARG_KEYS:
            continue
        clean[key] = value
    return clean


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


class CrystallizationEngine:
    """
    Pipeline Episode -> Recipe -> Validation -> Habit.

    La implementación es explícitamente best-effort: nunca bloquea misiones,
    pero sí falla con CrystallizationError cuando el CLI espera evidencia concreta.
    """

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir)
        self.artifacts_dir = self.root_dir / "artifacts"
        self.episodes_dir = self.artifacts_dir / "episodes"
        self.recipes_dir = self.artifacts_dir / "recipes"
        self.candidate_dir = self.recipes_dir / "candidates"
        self.validation_dir = self.recipes_dir / "validated"
        self.habits_dir = self.artifacts_dir / "habits"
        self.index_dir = self.artifacts_dir / "indexes"
        self.signal_registry = self._load_signal_registry()

    # ------------------------------------------------------------------ helpers
    def _mission_attempt_path(self, mission_id: str) -> Path:
        missions_dir = self.artifacts_dir / "missions"
        candidates = sorted(missions_dir.glob(f"{mission_id}_attempt*.json"))
        if not candidates:
            raise CrystallizationError(f"No existe artifacts/missions/{mission_id}_attemptN.json")
        # attempts usan sufijo _attempt<idx>; usamos el mayor índice
        def _attempt_num(path: Path) -> int:
            match = re.search(r"_attempt(\d+)\.json$", path.name)
            return int(match.group(1)) if match else 0

        return max(candidates, key=_attempt_num)

    def _history_path(self, mission_id: str) -> Optional[Path]:
        path = self.artifacts_dir / "history" / f"mission-{mission_id}.json"
        return path if path.exists() else None

    def _snapshot_path(self, mission_id: str) -> Optional[Path]:
        snap_dir = self.artifacts_dir / "missions" / "snapshots"
        path = snap_dir / f"{mission_id}_snapshot0.json"
        return path if path.exists() else None

    def _doctor_path(self) -> Optional[Path]:
        directory = self.artifacts_dir / "health" / "providers"
        return _latest_file(directory, "doctor_*.json")

    def _env_snapshot_path(self) -> Optional[Path]:
        directory = self.artifacts_dir / "health" / "env"
        return _latest_file(directory, "doctor_env_*.txt")

    def _timeline_from_attempt(self, attempt: Dict[str, Any], attempt_path: Path) -> List[Dict[str, Any]]:
        timeline: List[Dict[str, Any]] = []
        plan_ts = attempt.get("plan", {}).get("timestamp")
        if plan_ts:
            timeline.append(
                {
                    "t": _utc_iso(plan_ts),
                    "event": "plan",
                    "ref": str(attempt_path),
                    "notes": attempt.get("plan", {}).get("summary"),
                }
            )
        for entry in attempt.get("execution_log") or []:
            ts = entry.get("ts")
            event = {
                "t": _utc_iso(ts) if ts else _utc_iso(),
                "event": "actuate",
                "ref": str(attempt_path),
                "notes": entry.get("action"),
            }
            timeline.append(event)
            evaluation = entry.get("detail", {}).get("evaluation")
            if isinstance(evaluation, dict):
                timeline.append(
                    {
                        "t": _utc_iso(ts),
                        "event": "verify",
                        "ref": str(attempt_path),
                        "notes": evaluation.get("reason"),
                    }
                )
        return timeline

    def _waiting_payload_path(self, history: Optional[Dict[str, Any]]) -> Optional[str]:
        if not history:
            return None
        metadata = history.get("metadata") or {}
        detail = metadata.get("result_detail") or {}
        path = detail.get("waiting_mission_path")
        return str(path) if path else None

    def _intent_class(self, attempt: Dict[str, Any], mission_id: str) -> str:
        meta = attempt.get("plan", {}).get("metadata") or {}
        candidates = [
            meta.get("intent_class"),
            meta.get("habit_id"),
            attempt.get("plan", {}).get("id"),
            attempt.get("intent"),
            mission_id,
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return _slugify(candidate)
        return "generic"

    def _load_index_payload(self) -> Dict[str, Any]:
        index_path = self.index_dir / "crystallization_index.json"
        default_payload: Dict[str, Any] = {
            "schema": "ajax.crystallization.index.v2",
            "episodes": [],
            "recipes": [],
            "updated_ts": time.time(),
        }
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default_payload
        except json.JSONDecodeError:
            return default_payload
        if not isinstance(payload, dict):
            return default_payload
        payload.setdefault("episodes", [])
        payload.setdefault("recipes", [])
        return payload

    def _save_index_payload(self, payload: Dict[str, Any]) -> None:
        index_path = self.index_dir / "crystallization_index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_ts"] = time.time()
        index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _record_episode_index(
        self,
        *,
        episode_id: str,
        mission_id: str,
        intent_class: str,
        ts: int,
        passed: bool,
        signals: List[str],
        episode_path: str,
        recipe_path: str,
    ) -> None:
        payload = self._load_index_payload()
        episodes = payload.get("episodes")
        if isinstance(episodes, list):
            episodes.append(
                {
                    "episode_id": episode_id,
                    "mission_id": mission_id,
                    "intent_class": intent_class,
                    "ts": ts,
                    "pass": passed,
                    "signals": signals,
                    "paths": {"episode": episode_path, "recipe_candidate": recipe_path},
                }
            )
        self._save_index_payload(payload)

    def _record_recipe_candidate_index(
        self, *, recipe_id: str, intent_class: str, candidate_path: str, ts: int
    ) -> None:
        payload = self._load_index_payload()
        recipes = payload.get("recipes")
        if not isinstance(recipes, list):
            recipes = []
            payload["recipes"] = recipes
        entry = None
        for item in recipes:
            if isinstance(item, dict) and item.get("recipe_id") == recipe_id:
                entry = item
                break
        if entry is None:
            entry = {
                "recipe_id": recipe_id,
                "intent_class": intent_class,
                "ts": ts,
                "paths": {
                    "candidate": candidate_path,
                    "latest_validation": None,
                    "latest_habit": None,
                },
            }
            recipes.append(entry)
        else:
            entry["intent_class"] = intent_class
            entry["ts"] = ts
            paths = entry.setdefault("paths", {})
            paths["candidate"] = candidate_path
        self._save_index_payload(payload)

    def _record_recipe_validation_index(
        self, recipe_id: str, validation_path: str, reason: Optional[str], eligible: bool
    ) -> None:
        payload = self._load_index_payload()
        recipes = payload.get("recipes")
        if isinstance(recipes, list):
            for item in recipes:
                if isinstance(item, dict) and item.get("recipe_id") == recipe_id:
                    paths = item.setdefault("paths", {})
                    paths["latest_validation"] = validation_path
                    item["last_validation_reason"] = reason
                    item["last_validation_ts"] = time.time()
                    item["last_validation_eligible"] = eligible
                    break
        self._save_index_payload(payload)

    def _record_recipe_habit_index(self, recipe_id: str, habit_path: str) -> None:
        payload = self._load_index_payload()
        recipes = payload.get("recipes")
        if isinstance(recipes, list):
            for item in recipes:
                if isinstance(item, dict) and item.get("recipe_id") == recipe_id:
                    paths = item.setdefault("paths", {})
                    paths["latest_habit"] = habit_path
                    item["last_habit_ts"] = time.time()
                    break
        self._save_index_payload(payload)

    def promote_eligible_batch(
        self,
        *,
        since_ts: Optional[float] = None,
        min_signals: Optional[int] = None,
        runs: int = 2,
        rail: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload = self._load_index_payload()
        recipes = payload.get("recipes") or []
        candidates: List[Dict[str, Any]] = [
            item for item in recipes if isinstance(item, dict) and item.get("paths", {}).get("candidate")
        ]
        candidates.sort(key=lambda item: item.get("ts", 0), reverse=True)
        summary = {
            "promoted": [],
            "blocked_unknown_signal": [],
            "blocked_no_evidence": [],
            "blocked_other": [],
        }
        processed = 0
        for entry in candidates:
            if limit and processed >= limit:
                break
            entry_ts = entry.get("ts", 0)
            if since_ts and entry_ts < since_ts:
                continue
            paths = entry.get("paths") or {}
            candidate_path = paths.get("candidate")
            latest_habit = paths.get("latest_habit")
            if latest_habit:
                continue
            recipe_id = entry.get("recipe_id")
            try:
                validation_result = self.validate_recipe(
                    str(candidate_path),
                    runs=runs,
                    source="episodes",
                    since_ts=since_ts,
                    min_signals=min_signals,
                    rail=rail,
                )
            except CrystallizationError as exc:
                summary["blocked_other"].append({"recipe_id": recipe_id, "reason": str(exc)})
                processed += 1
                continue
            reason = validation_result.get("reason")
            if validation_result.get("eligible"):
                try:
                    promote_res = self.promote_recipe(str(candidate_path))
                    summary["promoted"].append(promote_res["habit_id"])
                except CrystallizationError as exc:
                    summary["blocked_other"].append({"recipe_id": recipe_id, "reason": str(exc)})
            else:
                bucket = "blocked_no_evidence"
                if reason == "unknown_signal":
                    bucket = "blocked_unknown_signal"
                elif reason and reason not in {"no_observed_evidence", "insufficient_passes"}:
                    bucket = "blocked_other"
                summary[bucket].append({"recipe_id": recipe_id, "reason": reason or "insufficient_data"})
            processed += 1
        summary["processed_recipes"] = processed
        return summary

    # ---------------------------------------------------------------- episodes/recipes
    def crystallize_mission(self, mission_id: str) -> Dict[str, Any]:
        attempt_path = self._mission_attempt_path(mission_id)
        attempt = _read_json(attempt_path)
        history_path = self._history_path(mission_id)
        history_doc = _read_json(history_path) if history_path else None
        snapshot_path = self._snapshot_path(mission_id)
        snapshot_doc = _read_json(snapshot_path) if snapshot_path else None
        waiting_payload_path = self._waiting_payload_path(history_doc)
        intent_class = self._intent_class(attempt, mission_id)
        ts = int(time.time())
        mission_slug = _slugify(mission_id)
        episode_id = f"episode_{ts}_{mission_id}"
        user_goal = (history_doc or {}).get("intent_text") or attempt.get("intent") or ""
        slots = attempt.get("plan", {}).get("metadata", {}).get("slots") or {}
        rail = (snapshot_doc or {}).get("rail")
        mode = (history_doc or {}).get("mode")
        doctor_path = self._doctor_path()
        env_path = self._env_snapshot_path()
        status = "FAILED"
        execution_res = attempt.get("execution_result") or {}
        if waiting_payload_path:
            status = "WAITING_FOR_USER"
        elif execution_res.get("success"):
            status = "SUCCESS"
        elif execution_res:
            status = "FAILED"
        success_spec = _stringify_success_spec(attempt.get("plan", {}).get("success_spec"))
        failure_codes: List[str] = []
        history_error = (history_doc or {}).get("final_error")
        if history_error:
            failure_codes.append(str(history_error))
        mission_error = attempt.get("mission_error") or {}
        if mission_error.get("reason"):
            failure_codes.append(str(mission_error.get("reason")))
        if mission_error.get("kind"):
            failure_codes.append(str(mission_error.get("kind")))
        if not failure_codes and status != "SUCCESS":
            failure_codes.append("unknown_failure")
        evidence_refs = [str(attempt_path)]
        if snapshot_path:
            evidence_refs.append(str(snapshot_path))
        if waiting_payload_path:
            evidence_refs.append(waiting_payload_path)
        timeline = self._timeline_from_attempt(attempt, attempt_path)
        if waiting_payload_path:
            timeline.append(
                {
                    "t": _utc_iso(),
                    "event": "ask_user",
                    "ref": waiting_payload_path,
                    "notes": (history_doc or {}).get("metadata", {}).get("ask_user_request", {}).get("question"),
                }
            )
        efe_signals = [
            {"name": "execution_result.success", "value": execution_res.get("success"), "ref": str(attempt_path)}
        ]
        if waiting_payload_path:
            efe_signals.append({"name": "waiting_payload", "value": True, "ref": waiting_payload_path})
        episode = {
            "episode_id": episode_id,
            "mission_id": mission_id,
            "intent": {
                "intent_class": intent_class,
                "user_goal_text": user_goal,
                "slots": slots,
            },
            "context": {
                "rail": rail,
                "mode": mode,
                "env_fingerprint_ref": str(env_path) if env_path else None,
                "provider_doctor_ref": str(doctor_path) if doctor_path else None,
            },
            "timeline": timeline,
            "outcome": {
                "status": status,
                "failure_codes": failure_codes,
                "evidence_refs": evidence_refs,
            },
            "efe": {
                "success_spec": success_spec,
                "observed_signals": efe_signals,
                "pass": True if status == "SUCCESS" else False if status == "FAILED" else None,
            },
        }
        observed_signals = [
            self._normalize_signal_name(sig.get("name"))
            for sig in efe_signals
            if isinstance(sig, dict) and sig.get("name")
        ]
        recipe = self._build_candidate_recipe(intent_class, attempt, success_spec, episode_id)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
        episode_path = self.episodes_dir / f"{episode_id}.json"
        episode_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")
        observed_signals = [
            self._normalize_signal_name(sig.get("name"))
            for sig in (episode.get("efe", {}).get("observed_signals") or [])
            if isinstance(sig, dict) and sig.get("name")
        ]
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        recipe_path = self.candidate_dir / f"recipe_{ts}_{intent_class}_{mission_slug}.json"
        recipe["recipe_id"] = recipe_path.stem
        recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
        self._record_episode_index(
            episode_id=episode_id,
            mission_id=mission_id,
            intent_class=intent_class,
            ts=ts,
            passed=status == "SUCCESS",
            signals=[sig for sig in observed_signals if sig],
            episode_path=str(episode_path),
            recipe_path=str(recipe_path),
        )
        self._record_recipe_candidate_index(
            recipe_id=recipe["recipe_id"],
            intent_class=intent_class,
            candidate_path=str(recipe_path),
            ts=ts,
        )
        return {
            "ok": True,
            "episode_path": str(episode_path),
            "recipe_path": str(recipe_path),
            "episode_id": episode_id,
            "recipe_id": recipe["recipe_id"],
        }

    def _build_candidate_recipe(
        self, intent_class: str, attempt: Dict[str, Any], success_spec: List[str], episode_id: str
    ) -> Dict[str, Any]:
        plan = attempt.get("plan") or {}
        steps_raw = plan.get("steps") or []
        recipe_steps: List[Dict[str, Any]] = []
        for step in steps_raw:
            action = step.get("action")
            if not action:
                continue
            recipe_steps.append(
                {
                    "action": action,
                    "args": _sanitize_args(step.get("params") or step.get("args") or {}),
                    "evidence_required": step.get("evidence_required") or [f"{action}.ok"],
                    "on_fail": "ASK_USER",
                }
            )
        if not recipe_steps:
            recipe_steps.append(
                {
                    "action": "noop.wait_for_operator",
                    "args": {},
                    "evidence_required": ["operator_ack"],
                    "on_fail": "ASK_USER",
                }
            )
        risk = (attempt.get("governance") or {}).get("risk_level") or "medium"
        safety = {
            "risk_level": risk,
            "requires_confirmation": risk not in {"low", "LOW"},
        }
        normalized_signals = []
        canonical_set = self.signal_registry.get("canonical", set())
        for sig in (success_spec or []):
            norm = self._normalize_signal_name(sig)
            if norm in canonical_set:
                normalized_signals.append(norm)
        if not normalized_signals:
            normalized_signals = ["execution_result.success"]
        recipe = {
            "schema": "ajax.recipe.candidate.v1",
            "recipe_id": "",
            "intent_class": intent_class,
            "version": 1,
            "preconditions": [
                {"check": "driver_available", "expect": "true", "on_fail": "ASK_USER"},
                {"check": "provider_health", "expect": "ok_or_degraded", "on_fail": "INCIDENT"},
            ],
            "steps": recipe_steps,
            "success_spec": normalized_signals,
            "efe": {
                "required_signals": normalized_signals,
                "validation_n": 3,
                "promotion_threshold": {"passes_min": 2, "max_critical_failures": 0},
            },
            "safety": safety,
            "provenance": {
                "source_episode_id": episode_id,
                "notes": "auto-crystallized from mission attempt",
            },
        }
        return recipe

    # ------------------------------------------------------------------ validation / promotion
    def validate_recipe(
        self,
        recipe_ref: str,
        runs: int,
        *,
        source: str = "episodes",
        since_ts: Optional[float] = None,
        min_signals: Optional[int] = None,
        rail: Optional[str] = None,
    ) -> Dict[str, Any]:
        recipe_path = self._resolve_recipe_path(recipe_ref)
        recipe = _read_json(recipe_path)
        intent_class = recipe.get("intent_class") or "generic"
        requested_runs = max(1, int(runs))
        required_signals = [self._normalize_signal_name(sig) for sig in recipe.get("efe", {}).get("required_signals", []) if sig]
        canonical_set = self.signal_registry.get("canonical", set())
        normalized_required = []
        unknown_signals: List[str] = []
        for sig in required_signals:
            if sig in canonical_set:
                normalized_required.append(sig)
            else:
                unknown_signals.append(sig)
        if unknown_signals:
            runs_payload: List[Dict[str, Any]] = []
        elif source == "episodes":
            runs_payload = self._build_runs_from_episodes(
                recipe,
                intent_class=intent_class,
                limit=requested_runs,
                required_signals=normalized_required,
                since_ts=since_ts,
                min_signals=min_signals,
                rail=rail,
            )
        elif source == "replay":
            raise CrystallizationError("Replay validation aún no implementada; usa --source episodes.")
        else:
            raise CrystallizationError(f"Fuente de validación desconocida: {source}")
        passes = sum(1 for r in runs_payload if r.get("pass"))
        critical_fails = sum(1 for r in runs_payload if "critical_failure" in (r.get("failure_codes") or []))
        fails = len(runs_payload) - passes
        threshold = recipe.get("efe", {}).get("promotion_threshold") or {"passes_min": requested_runs, "max_critical_failures": 0}
        eligible = passes >= int(threshold.get("passes_min", requested_runs)) and critical_fails <= int(
            threshold.get("max_critical_failures", 0)
        )
        reason = None
        if unknown_signals:
            reason = "unknown_signal"
        if not runs_payload and reason is None:
            reason = "no_observed_evidence"
        if not eligible and reason is None:
            if passes < int(threshold.get("passes_min", requested_runs)):
                reason = "insufficient_passes"
            elif critical_fails > int(threshold.get("max_critical_failures", 0)):
                reason = "too_many_critical_failures"
        ts = int(time.time())
        report = {
            "schema": "ajax.recipe.validation.v1",
            "recipe_id": recipe.get("recipe_id") or recipe_path.stem,
            "runs": runs_payload,
            "summary": {"passes": passes, "fails": fails, "critical_fails": critical_fails},
            "promotion_decision": {"eligible": eligible, "reason": reason or ("ok" if eligible else "insufficient_data")},
            "source": source,
        }
        if unknown_signals:
            report["unknown_signals"] = unknown_signals
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        file_slug = _slugify(report["recipe_id"])
        validation_path = self.validation_dir / f"validation_{ts}_{file_slug}.json"
        validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        recipe_identifier = recipe.get("recipe_id") or recipe_path.stem
        self._record_recipe_validation_index(recipe_identifier, str(validation_path), reason, eligible)
        return {
            "ok": True,
            "validation_path": str(validation_path),
            "eligible": eligible,
            "summary": report["summary"],
            "reason": reason,
        }

    def promote_recipe(self, recipe_ref: str) -> Dict[str, Any]:
        recipe_path = self._resolve_recipe_path(recipe_ref)
        recipe = _read_json(recipe_path)
        latest_validation = self._latest_validation(recipe.get("recipe_id") or recipe_path.stem)
        if not latest_validation:
            raise CrystallizationError("No existe validación para este recipe. Ejecuta `ajaxctl validate recipe ...` primero.")
        if not latest_validation.get("promotion_decision", {}).get("eligible"):
            raise CrystallizationError("La última validación no es elegible para promoción.")
        intent_class = recipe.get("intent_class") or "generic"
        habit_version = self._next_habit_version(intent_class)
        habit_payload = {
            "schema": "ajax.habit.v1",
            "habit_id": f"habit_{intent_class}_v{habit_version}",
            "intent_class": intent_class,
            "version": habit_version,
            "recipe_ref": str(recipe_path),
            "validation_ref": latest_validation.get("_path"),
            "enabled": True,
            "effective_date": _utc_iso(),
        }
        self.habits_dir.mkdir(parents=True, exist_ok=True)
        habit_path = self.habits_dir / f"habit_{intent_class}_v{habit_version}.json"
        habit_path.write_text(json.dumps(habit_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._record_recipe_habit_index(recipe.get("recipe_id") or recipe_path.stem, str(habit_path))
        return {"ok": True, "habit_path": str(habit_path), "habit_id": habit_payload["habit_id"]}

    # ------------------------------------------------------------------ private validations
    def _episodes_for_intent(
        self, intent_class: str, *, since_ts: Optional[float] = None, rail: Optional[str] = None
    ) -> Iterable[Tuple[Path, Dict[str, Any]]]:
        if not self.episodes_dir.exists():
            return []

        def _iter() -> Iterable[Tuple[Path, Dict[str, Any]]]:
            for path in sorted(self.episodes_dir.glob("episode_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                doc = _read_json(path)
                if doc.get("intent", {}).get("intent_class") != intent_class:
                    continue
                if rail:
                    doc_rail = (doc.get("context") or {}).get("rail")
                    if doc_rail and str(doc_rail).lower() != str(rail).lower():
                        continue
                if since_ts:
                    ts = self._episode_ts(doc)
                    if ts and ts < since_ts:
                        continue
                doc["_path"] = str(path)
                yield path, doc

        return _iter()

    def _episode_ts(self, episode: Dict[str, Any]) -> Optional[float]:
        timeline = episode.get("timeline") or []
        if timeline and isinstance(timeline, list):
            for event in timeline:
                if isinstance(event, dict) and event.get("t"):
                    try:
                        return _dt.datetime.fromisoformat(str(event["t"]).replace("Z", "+00:00")).timestamp()
                    except Exception:
                        continue
        episode_id = episode.get("episode_id") or ""
        match = re.search(r"episode_(\d+)_", episode_id)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return None

    def _build_runs_from_episodes(
        self,
        recipe: Dict[str, Any],
        *,
        intent_class: str,
        limit: int,
        required_signals: List[str],
        since_ts: Optional[float],
        min_signals: Optional[int],
        rail: Optional[str],
    ) -> List[Dict[str, Any]]:
        episodes = list(self._episodes_for_intent(intent_class, since_ts=since_ts, rail=rail))
        runs_payload: List[Dict[str, Any]] = []
        req_set = {self._normalize_signal_name(sig) for sig in required_signals if sig}
        signal_threshold = max(0, int(min_signals or 0))
        for path, episode in episodes:
            if len(runs_payload) >= limit:
                break
            efe = episode.get("efe") or {}
            if efe.get("pass") is not True:
                continue
            observed_signals = {
                self._normalize_signal_name(str(sig.get("name")))
                for sig in (efe.get("observed_signals") or [])
                if isinstance(sig, dict) and sig.get("name")
            }
            if req_set:
                if req_set.issubset(observed_signals):
                    pass
                elif signal_threshold and len(req_set & observed_signals) >= signal_threshold:
                    pass
                else:
                    continue
            elif signal_threshold and len(observed_signals) < signal_threshold:
                continue
            run_id = f"{recipe.get('recipe_id') or recipe.get('intent_class') or 'recipe'}:{episode.get('episode_id')}"
            evidence_refs = episode.get("outcome", {}).get("evidence_refs") or [str(path)]
            runs_payload.append(
                {
                    "run_id": run_id,
                    "ts": episode.get("timeline", [{}])[0].get("t") or _utc_iso(),
                    "pass": True,
                    "evidence_refs": evidence_refs,
                    "failure_codes": [],
                    "evidence_kind": "observed",
                }
            )
        return runs_payload

    def _load_signal_registry(self) -> Dict[str, Any]:
        registry = {
            "canonical": set(DEFAULT_SIGNAL_REGISTRY["canonical"]),
            "aliases": dict(DEFAULT_SIGNAL_REGISTRY["aliases"]),
        }
        registry_path = self.root_dir / "config" / "signal_registry.json"
        if not registry_path.exists():
            return registry
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            file_canonical = {str(name) for name in data.get("canonical", []) if name}
            file_aliases = {str(k): str(v) for k, v in (data.get("aliases") or {}).items()}
            registry["canonical"].update(file_canonical)
            registry["aliases"].update(file_aliases)
            return registry
        except Exception:
            return registry

    def _normalize_signal_name(self, name: Optional[str], strict: bool = False) -> Optional[str]:
        if not name:
            return None
        name = str(name).strip()
        canonical = self.signal_registry.get("canonical", set())
        if name in canonical:
            return name
        aliases = self.signal_registry.get("aliases") or {}
        mapped = aliases.get(name)
        if mapped:
            return mapped if mapped in canonical else mapped
        return name if not strict else None

    def _resolve_recipe_path(self, recipe_ref: str) -> Path:
        candidate = Path(recipe_ref)
        if candidate.exists():
            return candidate
        slug = _slugify(recipe_ref)
        if self.candidate_dir.exists():
            for path in self.candidate_dir.glob("recipe_*.json"):
                if path.stem == recipe_ref or _slugify(path.stem) == slug:
                    return path
            # fallback: search by recipe_id inside files
            for path in self.candidate_dir.glob("recipe_*.json"):
                try:
                    doc = _read_json(path)
                except CrystallizationError:
                    continue
                rid = doc.get("recipe_id")
                if rid == recipe_ref or _slugify(str(rid)) == slug:
                    return path
        raise CrystallizationError(f"No se encontró recipe '{recipe_ref}'.")

    def _latest_validation(self, recipe_id: str) -> Optional[Dict[str, Any]]:
        if not self.validation_dir.exists():
            return None
        docs: List[Tuple[float, Dict[str, Any]]] = []
        for path in self.validation_dir.glob("validation_*.json"):
            try:
                doc = _read_json(path)
            except CrystallizationError:
                continue
            if doc.get("recipe_id") == recipe_id or _slugify(doc.get("recipe_id", "")) == _slugify(recipe_id):
                doc["_path"] = str(path)
                docs.append((path.stat().st_mtime, doc))
        return max(docs, key=lambda item: item[0])[1] if docs else None

    def _next_habit_version(self, intent_class: str) -> int:
        if not self.habits_dir.exists():
            return 1
        prefix = f"habit_{intent_class}_v"
        version = 0
        for path in self.habits_dir.glob(f"{prefix}*.json"):
            match = re.search(r"_v(\d+)\.json$", path.name)
            if match:
                version = max(version, int(match.group(1)))
        return version + 1


__all__ = ["CrystallizationEngine", "CrystallizationError"]
