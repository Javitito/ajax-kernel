from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agency.efe_template_catalog import (
    MISSION_FAMILY_ANALYSIS,
    MISSION_FAMILY_ARTIFACT,
    MISSION_FAMILY_DESKTOP,
    MISSION_FAMILY_OBSERVATION,
    MISSION_FAMILY_REPO_PATCH,
    classify_mission_family,
    materialize_template_expected_state,
    resolve_template,
)
from agency.verify.efe_autogen import autogen_efe_candidate, extract_action_descriptor, generate_expected_state


@dataclass
class RepairResult:
    success: bool
    plan: Optional[Dict[str, Any]] = None
    attempts: int = 0
    gap_logged: bool = False
    receipt_path: Optional[str] = None
    reason: Optional[str] = None
    terminal: str = "DONE"
    repair_path: Optional[str] = None
    template_id: Optional[str] = None
    candidate_path: Optional[str] = None
    waiting_prompt: Optional[str] = None
    boundary: Optional[Dict[str, Any]] = None


@dataclass
class GuardDecision:
    allowed: bool
    rail: str
    task_tier: str
    mission_family: str
    rollback_assumption: Dict[str, Any]
    refusal_reasons: List[str] = field(default_factory=list)


@dataclass
class StepRepairDecision:
    step_id: str
    step_index: int
    repair_path: str
    template_id: Optional[str]
    mission_family: str
    expected_state: Optional[Dict[str, Any]]
    auto_materialized: bool
    auto_materialization_allowed: bool
    refusal_reasons: List[str]
    verify_evidence_expected: List[Dict[str, Any]]
    rollback_assumption: Dict[str, Any]
    reason: Optional[str] = None
    candidate_path: Optional[str] = None
    candidate_receipt_path: Optional[str] = None
    explain: List[str] = field(default_factory=list)


class EFERepairLoop:
    RECEIPT_SCHEMA = "ajax.receipt.efe_repair.v1"

    def __init__(self, receipts_dir: Optional[str] = None):
        if receipts_dir:
            base_dir = Path(receipts_dir)
        else:
            override = os.getenv("AJAX_ARTIFACTS_DIR")
            base_dir = Path(override) / "receipts" if override else Path("artifacts/receipts")
        self.receipts_dir = base_dir
        self.receipts_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir = self._derive_root_dir(base_dir)
        self.candidates_dir = self.root_dir / "artifacts" / "efe_candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def repair_plan(
        self, plan: Dict[str, Any], drafter_fn: Optional[Callable] = None
    ) -> RepairResult:
        if self._has_expected_state(plan):
            return RepairResult(success=True, plan=plan, attempts=0, terminal="DONE")

        started = time.time()
        original = self._deep_copy(plan)
        repaired = self._deep_copy(plan)
        steps = repaired.get("steps")
        if not isinstance(steps, list) or not steps:
            return self._finalize_waiting(
                original=original,
                started=started,
                reason="missing_steps_for_efe_repair",
                boundary={
                    "kind": "efe_boundary_completion",
                    "missing": ["steps[].success_spec.expected_state"],
                    "exact_boundary": "Provide at least one actionable step with falsifiable success_spec.expected_state.",
                },
                repair_path="waiting",
                decisions=[],
                drafter_available=bool(drafter_fn),
            )

        decisions: List[StepRepairDecision] = []
        waiting_boundary: Optional[Dict[str, Any]] = None

        for idx, step in enumerate(steps):
            if not isinstance(step, dict):
                return self._log_gap(
                    original=original,
                    started=started,
                    reason=f"invalid_step_type_at_{idx + 1}",
                    repair_path="waiting",
                    decisions=decisions,
                    drafter_available=bool(drafter_fn),
                )
            action = str(step.get("action") or "").strip()
            if action == "await_user_input":
                continue
            success_spec = step.get("success_spec") if isinstance(step.get("success_spec"), dict) else {}
            if self._expected_state_has_checks(success_spec.get("expected_state")):
                continue

            decision = self._resolve_step_expected_state(
                original_plan=original,
                step=step,
                step_index=idx,
            )
            decisions.append(decision)
            if not decision.auto_materialized or not isinstance(decision.expected_state, dict):
                waiting_boundary = {
                    "kind": "efe_boundary_completion",
                    "step_id": decision.step_id,
                    "repair_path": decision.repair_path,
                    "template_id": decision.template_id,
                    "candidate_path": decision.candidate_path,
                    "refusal_reasons": list(decision.refusal_reasons),
                    "exact_boundary": "Confirm or complete the missing success contract so it stays bounded and world-checkable.",
                }
                break

            step_success_spec = step.get("success_spec") if isinstance(step.get("success_spec"), dict) else {}
            step_success_spec["expected_state"] = decision.expected_state
            step["success_spec"] = step_success_spec

        if waiting_boundary is not None:
            return self._finalize_waiting(
                original=original,
                started=started,
                reason="guarded_auto_materialization_refused",
                boundary=waiting_boundary,
                repair_path=self._overall_repair_path(decisions, waiting=True),
                decisions=decisions,
                drafter_available=bool(drafter_fn),
            )

        receipt_path = self._write_receipt(
            original=original,
            repaired=repaired,
            started=started,
            terminal="DONE",
            success=True,
            repair_path=self._overall_repair_path(decisions, waiting=False),
            template_id=self._overall_template_id(decisions),
            reason="expected_state_materialized_without_llm",
            decisions=decisions,
            boundary=None,
            candidate_path=self._latest_candidate_path(decisions),
            drafter_available=bool(drafter_fn),
        )
        return RepairResult(
            success=True,
            plan=repaired,
            attempts=1,
            receipt_path=receipt_path,
            terminal="DONE",
            repair_path=self._overall_repair_path(decisions, waiting=False),
            template_id=self._overall_template_id(decisions),
            candidate_path=self._latest_candidate_path(decisions),
        )

    def _resolve_step_expected_state(
        self,
        *,
        original_plan: Dict[str, Any],
        step: Dict[str, Any],
        step_index: int,
    ) -> StepRepairDecision:
        source_doc = self._build_step_source_doc(original_plan=original_plan, step=step)
        step_id = str(step.get("id") or f"step-{step_index + 1}")

        template_resolution = resolve_template(source_doc)
        if template_resolution is not None:
            try:
                expected_state = materialize_template_expected_state(
                    template_resolution.template_id,
                    template_resolution.fields,
                )
            except Exception as exc:
                template_explain = list(template_resolution.explain) + [f"template_materialize_error:{type(exc).__name__}"]
            else:
                guard = self._evaluate_guard(
                    source_doc=source_doc,
                    mission_family=template_resolution.mission_family,
                    expected_state=expected_state,
                    template_id=template_resolution.template_id,
                )
                if guard.allowed:
                    return self._build_step_decision(
                        step_id=step_id,
                        step_index=step_index,
                        repair_path="template",
                        template_id=template_resolution.template_id,
                        mission_family=template_resolution.mission_family,
                        expected_state=expected_state,
                        guard=guard,
                        explain=template_resolution.explain,
                    )
                return self._candidate_or_waiting(
                    source_doc=source_doc,
                    step_id=step_id,
                    step_index=step_index,
                    guard=guard,
                    mission_family=template_resolution.mission_family,
                    explain=list(template_resolution.explain) + ["template_guard_refused"],
                    template_id=template_resolution.template_id,
                )

        expected_state, explain, unsupported_kind, unsupported_hint = self._derive_expected_state(source_doc)
        if isinstance(expected_state, dict) and self._expected_state_has_checks(expected_state):
            mission_family = classify_mission_family(source_doc)
            guard = self._evaluate_guard(
                source_doc=source_doc,
                mission_family=mission_family,
                expected_state=expected_state,
                template_id=None,
            )
            if guard.allowed:
                return self._build_step_decision(
                    step_id=step_id,
                    step_index=step_index,
                    repair_path="derivation",
                    template_id=None,
                    mission_family=mission_family,
                    expected_state=expected_state,
                    guard=guard,
                    explain=explain,
                )
            return self._candidate_or_waiting(
                source_doc=source_doc,
                step_id=step_id,
                step_index=step_index,
                guard=guard,
                mission_family=mission_family,
                explain=list(explain) + ["derivation_guard_refused"],
                template_id=None,
            )

        extra_explain = list(explain)
        if unsupported_kind:
            extra_explain.append(str(unsupported_kind))
        if unsupported_hint:
            extra_explain.append(str(unsupported_hint))
        return self._candidate_or_waiting(
            source_doc=source_doc,
            step_id=step_id,
            step_index=step_index,
            guard=None,
            mission_family=classify_mission_family(source_doc),
            explain=extra_explain,
            template_id=None,
        )

    def _candidate_or_waiting(
        self,
        *,
        source_doc: Dict[str, Any],
        step_id: str,
        step_index: int,
        guard: Optional[GuardDecision],
        mission_family: str,
        explain: List[str],
        template_id: Optional[str],
    ) -> StepRepairDecision:
        candidate = self._generate_candidate(source_doc=source_doc, step_id=step_id)
        candidate_path = candidate.get("efe_candidate_path")
        candidate_receipt_path = candidate.get("receipt_path")
        candidate_expected = candidate.get("expected_state")
        candidate_ok = bool(candidate.get("ok"))

        if candidate_ok and isinstance(candidate_expected, dict) and self._expected_state_has_checks(candidate_expected):
            candidate_guard = self._evaluate_guard(
                source_doc=source_doc,
                mission_family=mission_family,
                expected_state=candidate_expected,
                template_id=template_id,
            )
            if candidate_guard.allowed:
                return self._build_step_decision(
                    step_id=step_id,
                    step_index=step_index,
                    repair_path="candidate",
                    template_id=template_id,
                    mission_family=mission_family,
                    expected_state=candidate_expected,
                    guard=candidate_guard,
                    explain=list(explain) + ["candidate_autogen_ok"],
                    candidate_path=str(candidate_path) if candidate_path else None,
                    candidate_receipt_path=str(candidate_receipt_path) if candidate_receipt_path else None,
                )
            refusal_reasons = list(candidate_guard.refusal_reasons)
            rollback_assumption = candidate_guard.rollback_assumption
        else:
            refusal_reasons = list(guard.refusal_reasons) if guard is not None else []
            rollback_assumption = (
                guard.rollback_assumption
                if guard is not None
                else self._rollback_assumption(mission_family=mission_family, template_id=template_id)
            )
            unsupported = candidate.get("unsupported_action_kind")
            hint = candidate.get("hint")
            if unsupported:
                refusal_reasons.append(str(unsupported))
            if hint:
                refusal_reasons.append(str(hint))
            if not candidate_path:
                refusal_reasons.append("candidate_path_missing")

        return StepRepairDecision(
            step_id=step_id,
            step_index=step_index,
            repair_path="candidate" if candidate_path else "waiting",
            template_id=template_id,
            mission_family=mission_family,
            expected_state=candidate_expected if isinstance(candidate_expected, dict) else None,
            auto_materialized=False,
            auto_materialization_allowed=False,
            refusal_reasons=refusal_reasons or ["guarded_auto_materialization_refused"],
            verify_evidence_expected=self._summarize_verify_evidence(
                candidate_expected if isinstance(candidate_expected, dict) else None
            ),
            rollback_assumption=rollback_assumption,
            reason="guarded_auto_materialization_refused",
            candidate_path=str(candidate_path) if candidate_path else None,
            candidate_receipt_path=str(candidate_receipt_path) if candidate_receipt_path else None,
            explain=list(explain) + ["candidate_autogen_attempted"],
        )

    def _derive_expected_state(
        self, source_doc: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], List[str], Optional[str], Optional[str]]:
        descriptor = extract_action_descriptor(source_doc)
        if not isinstance(descriptor, dict):
            return None, ["descriptor_missing"], "unsupported_action_kind", "No deterministic descriptor found."
        expected_state, explain, unsupported_kind, unsupported_hint = generate_expected_state(descriptor)
        return expected_state, list(explain), unsupported_kind, unsupported_hint

    def _generate_candidate(self, *, source_doc: Dict[str, Any], step_id: str) -> Dict[str, Any]:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        safe_step = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in step_id).strip("_") or "step"
        out_path = self.candidates_dir / f"efe_candidate_{stamp}_{safe_step}.json"
        try:
            return autogen_efe_candidate(
                source_doc=source_doc,
                out_path=out_path,
                source_path=None,
                receipts_dir=self.receipts_dir,
            )
        except Exception as exc:
            return {
                "ok": False,
                "efe_candidate_path": None,
                "receipt_path": None,
                "unsupported_action_kind": "autogen_exception",
                "hint": f"autogen_exception:{type(exc).__name__}",
                "expected_state": None,
            }

    def _evaluate_guard(
        self,
        *,
        source_doc: Dict[str, Any],
        mission_family: str,
        expected_state: Dict[str, Any],
        template_id: Optional[str],
    ) -> GuardDecision:
        rail = self._current_rail(source_doc)
        task_tier = self._infer_task_tier(source_doc=source_doc, mission_family=mission_family)
        rollback = self._rollback_assumption(mission_family=mission_family, template_id=template_id)
        reasons: List[str] = []

        if rail != "lab":
            reasons.append("rail_not_lab")
        if not self._expected_state_has_checks(expected_state):
            reasons.append("expected_state_missing_checks")
        if not self._expected_state_is_bounded(expected_state):
            reasons.append("verify_not_bounded")
        if task_tier not in {"T0", "T1"} and not (
            task_tier == "T2" and rollback.get("kind") in {"not_applicable", "trivial_cleanup", "reversible"}
        ):
            reasons.append(f"task_tier_not_allowed:{task_tier}")
        if mission_family in {MISSION_FAMILY_DESKTOP, MISSION_FAMILY_REPO_PATCH}:
            reasons.append(f"mission_family_requires_boundary:{mission_family}")

        return GuardDecision(
            allowed=len(reasons) == 0,
            rail=rail,
            task_tier=task_tier,
            mission_family=mission_family,
            rollback_assumption=rollback,
            refusal_reasons=reasons,
        )

    def _build_step_decision(
        self,
        *,
        step_id: str,
        step_index: int,
        repair_path: str,
        template_id: Optional[str],
        mission_family: str,
        expected_state: Dict[str, Any],
        guard: GuardDecision,
        explain: List[str],
        candidate_path: Optional[str] = None,
        candidate_receipt_path: Optional[str] = None,
    ) -> StepRepairDecision:
        return StepRepairDecision(
            step_id=step_id,
            step_index=step_index,
            repair_path=repair_path,
            template_id=template_id,
            mission_family=mission_family,
            expected_state=expected_state,
            auto_materialized=True,
            auto_materialization_allowed=True,
            refusal_reasons=[],
            verify_evidence_expected=self._summarize_verify_evidence(expected_state),
            rollback_assumption=guard.rollback_assumption,
            candidate_path=candidate_path,
            candidate_receipt_path=candidate_receipt_path,
            explain=explain,
        )

    def _finalize_waiting(
        self,
        *,
        original: Dict[str, Any],
        started: float,
        reason: str,
        boundary: Dict[str, Any],
        repair_path: str,
        decisions: List[StepRepairDecision],
        drafter_available: bool,
    ) -> RepairResult:
        prompt = self._build_waiting_prompt(boundary)
        receipt_path = self._write_receipt(
            original=original,
            repaired=None,
            started=started,
            terminal="WAITING_FOR_USER",
            success=False,
            repair_path=repair_path,
            template_id=self._overall_template_id(decisions),
            reason=reason,
            decisions=decisions,
            boundary=boundary,
            candidate_path=self._latest_candidate_path(decisions),
            drafter_available=drafter_available,
        )
        return RepairResult(
            success=False,
            plan=None,
            attempts=1,
            receipt_path=receipt_path,
            reason=reason,
            terminal="WAITING_FOR_USER",
            repair_path=repair_path,
            template_id=self._overall_template_id(decisions),
            candidate_path=self._latest_candidate_path(decisions),
            waiting_prompt=prompt,
            boundary=boundary,
        )

    def _log_gap(
        self,
        *,
        original: Dict[str, Any],
        started: float,
        reason: str,
        repair_path: str,
        decisions: List[StepRepairDecision],
        drafter_available: bool,
    ) -> RepairResult:
        receipt_path = self._write_receipt(
            original=original,
            repaired=None,
            started=started,
            terminal="GAP_LOGGED",
            success=False,
            repair_path=repair_path,
            template_id=self._overall_template_id(decisions),
            reason=reason,
            decisions=decisions,
            boundary=None,
            candidate_path=self._latest_candidate_path(decisions),
            drafter_available=drafter_available,
        )
        return RepairResult(
            success=False,
            attempts=1,
            gap_logged=True,
            receipt_path=receipt_path,
            reason=reason,
            terminal="GAP_LOGGED",
            repair_path=repair_path,
            template_id=self._overall_template_id(decisions),
            candidate_path=self._latest_candidate_path(decisions),
        )

    def _write_receipt(
        self,
        *,
        original: Dict[str, Any],
        repaired: Optional[Dict[str, Any]],
        started: float,
        terminal: str,
        success: bool,
        repair_path: Optional[str],
        template_id: Optional[str],
        reason: Optional[str],
        decisions: List[StepRepairDecision],
        boundary: Optional[Dict[str, Any]],
        candidate_path: Optional[str],
        drafter_available: bool,
    ) -> str:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(started))
        receipt_path = self.receipts_dir / f"efe_repair_{stamp}.json"
        payload = {
            "schema": self.RECEIPT_SCHEMA,
            "version": "v1",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
            "success": bool(success),
            "terminal": terminal,
            "repair_path": repair_path,
            "template_id": template_id,
            "candidate_path": candidate_path,
            "reason": reason,
            "decision_order": ["template", "derivation", "candidate", "waiting"],
            "auto_materialized": bool(success),
            "auto_materialization_allowed": bool(success),
            "auto_materialization_refused_reasons": self._collect_refusal_reasons(decisions),
            "verify_evidence_expected": self._collect_verify_evidence(decisions),
            "rollback_assumption": self._overall_rollback(decisions),
            "exact_boundary": boundary,
            "original_plan_summary": self._summarize_plan(original),
            "repaired_plan_summary": self._summarize_plan(repaired) if isinstance(repaired, dict) else None,
            "step_decisions": [asdict(decision) for decision in decisions],
            "legacy_drafter_available": bool(drafter_available),
            "legacy_drafter_invoked": False,
        }
        receipt_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(receipt_path)

    def _build_waiting_prompt(self, boundary: Dict[str, Any]) -> str:
        step_id = str(boundary.get("step_id") or "step").strip()
        reasons = boundary.get("refusal_reasons")
        reason_txt = ", ".join(str(item) for item in reasons if isinstance(item, str)) if isinstance(reasons, list) else ""
        candidate_path = str(boundary.get("candidate_path") or "").strip()
        parts = [f"Necesito completar el contrato EFE para {step_id}."]
        if reason_txt:
            parts.append(f"Motivo fail-closed: {reason_txt}.")
        if candidate_path:
            parts.append(f"Revisa o completa el candidato en: {candidate_path}.")
        return " ".join(parts).strip()

    def _build_step_source_doc(self, *, original_plan: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        meta = original_plan.get("metadata") if isinstance(original_plan.get("metadata"), dict) else {}
        source_doc = {
            "goal": original_plan.get("goal") or original_plan.get("description"),
            "metadata": dict(meta),
            "action": step.get("action"),
            "args": dict(step.get("args") or {}),
            "step_id": step.get("id"),
        }
        if "receipt_path" in step:
            source_doc["receipt_path"] = step.get("receipt_path")
        if "output_path" in step:
            source_doc["output_path"] = step.get("output_path")
        return source_doc

    def _has_expected_state(self, plan: Dict[str, Any]) -> bool:
        if not isinstance(plan, dict):
            return False
        expected = plan.get("expected_state")
        if self._expected_state_has_checks(expected):
            return True
        steps = plan.get("steps")
        if not isinstance(steps, list) or not steps:
            return False
        for step in steps:
            if not isinstance(step, dict):
                return False
            if str(step.get("action") or "").strip() == "await_user_input":
                continue
            succ = step.get("success_spec")
            if not isinstance(succ, dict) or not self._expected_state_has_checks(succ.get("expected_state")):
                return False
        return True

    def _expected_state_has_checks(self, expected: Any) -> bool:
        if not isinstance(expected, dict):
            return False
        if expected.get("windows"):
            return True
        if expected.get("files"):
            return True
        if isinstance(expected.get("checks"), list) and expected.get("checks"):
            return True
        meta = expected.get("meta") or {}
        return bool(isinstance(meta, dict) and meta.get("must_be_active"))

    def _expected_state_is_bounded(self, expected: Dict[str, Any]) -> bool:
        allowed_keys = {"windows", "files", "checks", "meta"}
        if not isinstance(expected, dict):
            return False
        if any(key not in allowed_keys for key in expected.keys()):
            return False
        checks = expected.get("checks")
        if isinstance(checks, list):
            supported = {"fs", "process", "port", "receipt_schema", "structured_output"}
            for check in checks:
                if not isinstance(check, dict):
                    return False
                if str(check.get("kind") or "").strip().lower() not in supported:
                    return False
        return self._expected_state_has_checks(expected)

    def _infer_task_tier(self, *, source_doc: Dict[str, Any], mission_family: str) -> str:
        action = str(source_doc.get("action") or "").strip().lower()
        if mission_family in {MISSION_FAMILY_ANALYSIS, MISSION_FAMILY_OBSERVATION}:
            return "T1"
        if mission_family in {MISSION_FAMILY_ARTIFACT, MISSION_FAMILY_REPO_PATCH, MISSION_FAMILY_DESKTOP}:
            return "T2"
        if any(token in action for token in ("delete", "remove", "kill", "shutdown")):
            return "T2"
        return "T1"

    def _rollback_assumption(self, *, mission_family: str, template_id: Optional[str]) -> Dict[str, Any]:
        if mission_family in {MISSION_FAMILY_ANALYSIS, MISSION_FAMILY_OBSERVATION}:
            return {
                "kind": "not_applicable",
                "reason": "EFE materialization is metadata-only and mission family is read-only/observational.",
            }
        if template_id == "efe.fs_path_materialized.v0" and mission_family == MISSION_FAMILY_ARTIFACT:
            return {
                "kind": "trivial_cleanup",
                "reason": "Generated artifact paths can be deleted if later execution fails.",
            }
        return {
            "kind": "unknown",
            "reason": "Rollback semantics require boundary completion before auto-materializing this EFE.",
        }

    def _current_rail(self, source_doc: Dict[str, Any]) -> str:
        meta = source_doc.get("metadata")
        if isinstance(meta, dict):
            rail = str(meta.get("rail") or "").strip().lower()
            if rail in {"lab", "prod"}:
                return rail
        rail = str(os.getenv("AJAX_RAIL") or "lab").strip().lower()
        return rail if rail in {"lab", "prod"} else "lab"

    def _summarize_verify_evidence(self, expected_state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(expected_state, dict):
            return []
        out: List[Dict[str, Any]] = []
        for file_exp in expected_state.get("files") or []:
            if isinstance(file_exp, dict):
                out.append(
                    {
                        "kind": "file",
                        "path": file_exp.get("path"),
                        "must_exist": file_exp.get("must_exist", True),
                    }
                )
        for check in expected_state.get("checks") or []:
            if isinstance(check, dict):
                summary = {"kind": check.get("kind")}
                for key in ("path", "port", "schema", "root_type"):
                    if key in check:
                        summary[key] = check.get(key)
                out.append(summary)
        return out

    def _collect_verify_evidence(self, decisions: List[StepRepairDecision]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for decision in decisions:
            out.extend(decision.verify_evidence_expected)
        return out

    def _collect_refusal_reasons(self, decisions: List[StepRepairDecision]) -> List[str]:
        out: List[str] = []
        seen = set()
        for decision in decisions:
            for reason in decision.refusal_reasons:
                if reason in seen:
                    continue
                seen.add(reason)
                out.append(reason)
        return out

    def _overall_rollback(self, decisions: List[StepRepairDecision]) -> Dict[str, Any]:
        if not decisions:
            return {"kind": "unknown", "reason": "no_decisions"}
        first = decisions[0].rollback_assumption
        if all(decision.rollback_assumption == first for decision in decisions):
            return first
        return {"kind": "mixed", "reason": "Multiple rollback assumptions were required across repaired steps."}

    def _overall_repair_path(self, decisions: List[StepRepairDecision], *, waiting: bool) -> Optional[str]:
        if not decisions:
            return "waiting" if waiting else None
        unique = {decision.repair_path for decision in decisions if decision.repair_path}
        if waiting and unique == {"candidate"}:
            return "candidate"
        if waiting and "waiting" in unique:
            return "waiting"
        if len(unique) == 1:
            return next(iter(unique))
        return "mixed"

    def _overall_template_id(self, decisions: List[StepRepairDecision]) -> Optional[str]:
        template_ids = [decision.template_id for decision in decisions if decision.template_id]
        if not template_ids:
            return None
        if len(set(template_ids)) == 1:
            return template_ids[0]
        return "mixed"

    def _latest_candidate_path(self, decisions: List[StepRepairDecision]) -> Optional[str]:
        for decision in reversed(decisions):
            if decision.candidate_path:
                return decision.candidate_path
        return None

    def _summarize_plan(self, plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(plan, dict):
            return None
        steps = plan.get("steps")
        return {
            "steps_count": len(steps) if isinstance(steps, list) else 0,
            "has_expected_state": self._has_expected_state(plan),
            "goal": str(plan.get("goal") or plan.get("description") or "")[:160],
        }

    def _deep_copy(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(payload, ensure_ascii=False))

    def _derive_root_dir(self, receipts_dir: Path) -> Path:
        base = receipts_dir.resolve()
        if base.name.lower() == "receipts":
            parent = base.parent
            if parent.name.lower() == "artifacts":
                return parent.parent
            return parent
        return base.parent


_default_repair_loop: Optional[EFERepairLoop] = None


def get_repair_loop(receipts_dir: Optional[str] = None) -> EFERepairLoop:
    global _default_repair_loop
    if _default_repair_loop is None:
        _default_repair_loop = EFERepairLoop(receipts_dir=receipts_dir)
    elif receipts_dir:
        try:
            current = Path(str(_default_repair_loop.receipts_dir)).resolve()
            requested = Path(receipts_dir).resolve()
        except Exception:
            current = None
            requested = None
        if current != requested:
            _default_repair_loop = EFERepairLoop(receipts_dir=receipts_dir)
    return _default_repair_loop


def repair_plan_if_needed(
    plan: Dict[str, Any],
    drafter_fn: Optional[Callable] = None,
    receipts_dir: Optional[str] = None,
) -> RepairResult:
    loop = get_repair_loop(receipts_dir)
    return loop.repair_plan(plan, drafter_fn)


def validate_plan_has_efe(plan: Dict[str, Any]) -> bool:
    loop = get_repair_loop()
    return loop._has_expected_state(plan)
