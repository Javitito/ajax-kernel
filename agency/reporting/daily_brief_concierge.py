from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
LEANN_INDEX_REL = Path(".leann/indexes/antigravity_skills_safe/documents.leann")


def _utc_now(now_ts: Optional[float] = None) -> str:
    ts = float(now_ts if now_ts is not None else time.time())
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(now_ts: Optional[float] = None) -> str:
    ts = float(now_ts if now_ts is not None else time.time())
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def _safe_load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _parse_iso_to_ts(raw: Any) -> Optional[float]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return float(datetime.fromisoformat(text).timestamp())
    except Exception:
        return None


def _to_rel_path(root: Path, raw: Any) -> str:
    path = Path(str(raw))
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _evidence_ref(kind: str, path: Any) -> Dict[str, str]:
    return {"kind": str(kind), "path": str(path)}


def _collect_recent_gaps(root: Path, limit: int = 3) -> List[Dict[str, Any]]:
    gaps_dir = root / "artifacts" / "capability_gaps"
    if not gaps_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in gaps_dir.rglob("*.json"):
        parts = {part.lower() for part in path.parts}
        if "cancelled" in parts:
            continue
        payload = _safe_load_json(path)
        created_ts = _parse_iso_to_ts(payload.get("created_utc"))
        if created_ts is None:
            try:
                created_ts = float(path.stat().st_mtime)
            except Exception:
                created_ts = 0.0
        summary = (
            payload.get("summary")
            or payload.get("title")
            or payload.get("reason")
            or payload.get("code")
            or path.stem
        )
        rows.append(
            {
                "gap_id": payload.get("gap_id") or path.stem,
                "summary": str(summary),
                "created_ts": float(created_ts),
                "created_utc": payload.get("created_utc") or _utc_now(created_ts),
                "path": _to_rel_path(root, path),
            }
        )
    rows.sort(key=lambda item: (-float(item.get("created_ts") or 0.0), str(item.get("gap_id") or "")))
    return rows[: max(0, int(limit))]


def _build_gap_query(gaps: Sequence[Mapping[str, Any]]) -> str:
    if not gaps:
        return "capability gap mitigation patterns for ajax kernel"
    bits = []
    for gap in list(gaps)[:3]:
        gid = str(gap.get("gap_id") or "").strip()
        summary = str(gap.get("summary") or "").strip()
        bits.append(f"{gid} {summary}".strip())
    query = " ; ".join([b for b in bits if b]).strip()
    if not query:
        query = "capability gap mitigation patterns for ajax kernel"
    return query[:420]


def _run_providers_audit(
    root: Path,
    *,
    run_named_audit_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        payload = run_named_audit_fn(name="providers", root_dir=root, run_id=None, last=1)
    except Exception as exc:
        return {
            "status": "error",
            "ok": False,
            "error": f"providers_audit_failed:{exc}",
            "evidence_refs": [],
            "payload": {},
        }
    artifact = payload.get("artifact_path")
    receipt = payload.get("receipt_path")
    evidence = []
    if artifact:
        evidence.append(_evidence_ref("audit_artifact", _to_rel_path(root, artifact)))
    if receipt:
        evidence.append(_evidence_ref("audit_receipt", _to_rel_path(root, receipt)))
    return {
        "status": "ok",
        "ok": bool(payload.get("ok")),
        "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
        "findings_count": len(payload.get("findings") or []) if isinstance(payload.get("findings"), list) else 0,
        "payload": payload,
        "evidence_refs": evidence,
    }


def _run_gates_check(
    root: Path,
    *,
    now_ts: float,
    run_soak_check_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        payload = run_soak_check_fn(root, rail="lab", window_min=60, now_ts=now_ts)
    except Exception as exc:
        return {
            "status": "error",
            "ok": False,
            "error": f"gates_check_failed:{exc}",
            "evidence_refs": [],
            "payload": {},
        }
    evidence = []
    receipt_path = payload.get("receipt_path")
    report_path = payload.get("report_path")
    if receipt_path:
        evidence.append(_evidence_ref("gate_receipt", _to_rel_path(root, receipt_path)))
    if report_path:
        evidence.append(_evidence_ref("gate_report", _to_rel_path(root, report_path)))
    return {
        "status": "ok",
        "ok": bool(payload.get("ok")),
        "outcome_code": str(payload.get("outcome_code") or ""),
        "summary_paragraph": str(payload.get("summary_paragraph") or ""),
        "payload": payload,
        "evidence_refs": evidence,
    }


def _run_optional_eki_audit(
    root: Path,
    *,
    list_registered_audits_fn: Callable[[], Mapping[str, Any]],
    run_named_audit_fn: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        registry = list_registered_audits_fn() or {}
    except Exception as exc:
        return {
            "status": "not_available",
            "ok": None,
            "reason": f"registry_unavailable:{exc}",
            "evidence_refs": [],
            "payload": {},
        }
    names = sorted(str(name) for name in registry.keys())
    candidate = next((name for name in names if name.lower() == "eki"), None)
    if candidate is None:
        candidate = next((name for name in names if "eki" in name.lower()), None)
    if not candidate:
        return {
            "status": "not_available",
            "ok": None,
            "reason": "eki_audit_not_registered",
            "available_audits": names,
            "evidence_refs": [],
            "payload": {},
        }
    try:
        payload = run_named_audit_fn(name=candidate, root_dir=root, run_id=None, last=1)
    except Exception as exc:
        return {
            "status": "error",
            "ok": False,
            "reason": f"eki_audit_failed:{exc}",
            "audit_name": candidate,
            "evidence_refs": [],
            "payload": {},
        }
    evidence = []
    artifact = payload.get("artifact_path")
    receipt = payload.get("receipt_path")
    if artifact:
        evidence.append(_evidence_ref("audit_artifact", _to_rel_path(root, artifact)))
    if receipt:
        evidence.append(_evidence_ref("audit_receipt", _to_rel_path(root, receipt)))
    return {
        "status": "ok",
        "ok": bool(payload.get("ok")),
        "audit_name": candidate,
        "summary": payload.get("summary") if isinstance(payload.get("summary"), dict) else {},
        "payload": payload,
        "evidence_refs": evidence,
    }


def _run_leann_search(
    root: Path,
    *,
    gaps: Sequence[Mapping[str, Any]],
    query_leann_fn: Callable[..., List[Dict[str, Any]]],
) -> Dict[str, Any]:
    index_base = root / LEANN_INDEX_REL
    meta_path = Path(f"{index_base}.meta.json")
    passages_path = Path(f"{index_base}.passages.jsonl")
    query_text = _build_gap_query(gaps)
    evidence = [
        _evidence_ref("leann_index_meta", _to_rel_path(root, meta_path)),
        _evidence_ref("leann_index_passages", _to_rel_path(root, passages_path)),
    ]
    if not meta_path.exists() or not passages_path.exists():
        return {
            "status": "capability_missing",
            "reason": "missing_leann_index",
            "index_base": _to_rel_path(root, index_base),
            "query": query_text,
            "results": [],
            "evidence_refs": evidence,
        }
    try:
        results_raw = query_leann_fn(str(index_base), query_text, top_k=5, fallback_grep=True)
    except Exception as exc:
        return {
            "status": "capability_missing",
            "reason": f"leann_query_failed:{exc}",
            "index_base": _to_rel_path(root, index_base),
            "query": query_text,
            "results": [],
            "evidence_refs": evidence,
        }
    results: List[Dict[str, Any]] = []
    for item in results_raw or []:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "text": str(item.get("text") or "")[:300],
                "score": item.get("score"),
                "id": item.get("id"),
                "source_mode": item.get("source_mode"),
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            }
        )
    return {
        "status": "ok",
        "reason": "query_ok",
        "index_base": _to_rel_path(root, index_base),
        "query": query_text,
        "results": results,
        "evidence_refs": evidence,
    }


def _build_proposals(
    *,
    providers: Mapping[str, Any],
    gates: Mapping[str, Any],
    leann: Mapping[str, Any],
    gaps: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    if gaps:
        top_gap = gaps[0]
        evidence = [_evidence_ref("gap", top_gap.get("path"))]
        evidence.extend(list(providers.get("evidence_refs") or [])[:1])
        proposals.append(
            {
                "id": "proposal_gap_focus",
                "mode": "proposal_only",
                "summary": f"Priorizar hipótesis para {top_gap.get('gap_id')} usando señales recientes y criterios de verificación explícitos.",
                "why": str(top_gap.get("summary") or ""),
                "evidence_refs": evidence,
            }
        )
    if str(leann.get("status")) == "ok":
        first = (leann.get("results") or [{}])[0] if (leann.get("results") or []) else {}
        evidence = list(leann.get("evidence_refs") or [])
        if isinstance(first, dict) and first.get("id"):
            evidence = evidence + [_evidence_ref("leann_hit", str(first.get("id")))]
        proposals.append(
            {
                "id": "proposal_skill_mapping",
                "mode": "proposal_only",
                "summary": "Mapear los 3 gaps recientes contra skills de LEANN para redactar un plan de mitigación sin ejecutar acciones.",
                "why": str(leann.get("query") or ""),
                "evidence_refs": evidence[:3],
            }
        )
    if not bool(providers.get("ok")) or not bool(gates.get("ok")):
        evidence = list(providers.get("evidence_refs") or []) + list(gates.get("evidence_refs") or [])
        proposals.append(
            {
                "id": "proposal_health_backlog",
                "mode": "proposal_only",
                "summary": "Crear backlog de remediación priorizado por riesgo (providers/gates) y validarlo en LAB antes de cualquier cambio.",
                "why": "Audits/gates no están completamente verdes.",
                "evidence_refs": evidence[:4],
            }
        )
    return proposals[:3]


def _unique_evidence_refs(groups: Sequence[Sequence[Mapping[str, Any]]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for group in groups:
        for item in group:
            kind = str(item.get("kind") or "").strip()
            path = str(item.get("path") or "").strip()
            if not kind or not path:
                continue
            key = (kind, path)
            if key in seen:
                continue
            seen.add(key)
            out.append({"kind": kind, "path": path})
    return out


def _render_md(payload: Mapping[str, Any]) -> str:
    audits = payload.get("audits") if isinstance(payload.get("audits"), dict) else {}
    gaps = payload.get("gaps") if isinstance(payload.get("gaps"), dict) else {}
    leann = payload.get("leann") if isinstance(payload.get("leann"), dict) else {}
    proposals = payload.get("proposals") if isinstance(payload.get("proposals"), list) else []

    providers = audits.get("providers") if isinstance(audits.get("providers"), dict) else {}
    gates = audits.get("gates") if isinstance(audits.get("gates"), dict) else {}
    eki = audits.get("eki") if isinstance(audits.get("eki"), dict) else {}
    recent = gaps.get("recent") if isinstance(gaps.get("recent"), list) else []

    lines = [
        "# Concierge Daily Brief",
        "",
        f"- created_utc: {payload.get('created_utc')}",
        f"- status: {payload.get('status')}",
        "- mode: read-only (proposal_only)",
        "",
        "## Audits",
        f"- providers: status={providers.get('status')} ok={providers.get('ok')}",
        f"- gates: status={gates.get('status')} ok={gates.get('ok')} code={gates.get('outcome_code')}",
        f"- eki: status={eki.get('status')} ok={eki.get('ok')}",
        "",
        "## Recent Gaps",
    ]
    if recent:
        for idx, gap in enumerate(recent, start=1):
            lines.append(f"{idx}. {gap.get('gap_id')}: {gap.get('summary')}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## LEANN",
            f"- status: {leann.get('status')}",
            f"- query: {leann.get('query')}",
            f"- hits: {len(leann.get('results') or [])}",
            "",
            "## Proposals",
        ]
    )
    if proposals:
        for idx, item in enumerate(proposals, start=1):
            lines.append(f"{idx}. {item.get('summary')}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def generate_daily_brief(
    root: Path = ROOT,
    *,
    now_ts: Optional[float] = None,
    run_named_audit_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    list_registered_audits_fn: Optional[Callable[[], Mapping[str, Any]]] = None,
    run_soak_check_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    query_leann_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    now = float(now_ts if now_ts is not None else time.time())
    stamp = _stamp(now)
    root_dir = Path(root)

    if run_named_audit_fn is None or list_registered_audits_fn is None:
        from agency.audits.runner import list_registered_audits, run_named_audit

        run_named_audit_fn = run_named_audit
        list_registered_audits_fn = list_registered_audits
    if run_soak_check_fn is None:
        from agency.soak_gate import run_soak_check

        run_soak_check_fn = run_soak_check
    if query_leann_fn is None:
        from agency.leann_query_client import query_leann

        query_leann_fn = query_leann

    providers_audit = _run_providers_audit(root_dir, run_named_audit_fn=run_named_audit_fn)
    gates_check = _run_gates_check(root_dir, now_ts=now, run_soak_check_fn=run_soak_check_fn)
    eki_audit = _run_optional_eki_audit(
        root_dir,
        list_registered_audits_fn=list_registered_audits_fn,
        run_named_audit_fn=run_named_audit_fn,
    )
    recent_gaps = _collect_recent_gaps(root_dir, limit=3)
    leann = _run_leann_search(root_dir, gaps=recent_gaps, query_leann_fn=query_leann_fn)

    proposals = _build_proposals(
        providers=providers_audit,
        gates=gates_check,
        leann=leann,
        gaps=recent_gaps,
    )

    concierge_dir = root_dir / "artifacts" / "concierge"
    receipt_dir = root_dir / "artifacts" / "receipts"
    concierge_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    json_path = concierge_dir / f"daily_brief_{stamp}.json"
    md_path = concierge_dir / f"daily_brief_{stamp}.md"
    receipt_path = receipt_dir / f"daily_brief_{stamp}.json"

    json_rel = _to_rel_path(root_dir, json_path)
    md_rel = _to_rel_path(root_dir, md_path)
    receipt_rel = _to_rel_path(root_dir, receipt_path)

    status = "ok" if str(leann.get("status")) == "ok" else "capability_missing"
    terminal = "DONE" if status == "ok" else "GAP_LOGGED"

    claims = [
        {
            "type": "diagnosed",
            "statement": "Audits read-only de providers y gates ejecutados para el brief diario.",
            "evidence_refs": _unique_evidence_refs(
                [providers_audit.get("evidence_refs") or [], gates_check.get("evidence_refs") or []]
            ),
        },
        {
            "type": "verified",
            "statement": "Daily brief generado en artefactos JSON/MD + receipt.",
            "evidence_refs": [
                _evidence_ref("brief_json", json_rel),
                _evidence_ref("brief_md", md_rel),
                _evidence_ref("brief_receipt", receipt_rel),
            ],
        },
    ]

    payload: Dict[str, Any] = {
        "schema": "ajax.concierge.daily_brief.v1",
        "created_utc": _utc_now(now),
        "status": status,
        "terminal": terminal,
        "read_only": True,
        "mode": "proposal_only",
        "expected_state": {
            "artifacts_written": [json_rel, md_rel, receipt_rel],
            "no_actions_executed": True,
            "single_leann_query": True,
        },
        "audits": {
            "providers": {
                "status": providers_audit.get("status"),
                "ok": providers_audit.get("ok"),
                "summary": providers_audit.get("summary"),
                "findings_count": providers_audit.get("findings_count"),
                "evidence_refs": providers_audit.get("evidence_refs") or [],
            },
            "gates": {
                "status": gates_check.get("status"),
                "ok": gates_check.get("ok"),
                "outcome_code": gates_check.get("outcome_code"),
                "summary_paragraph": gates_check.get("summary_paragraph"),
                "evidence_refs": gates_check.get("evidence_refs") or [],
            },
            "eki": {
                "status": eki_audit.get("status"),
                "ok": eki_audit.get("ok"),
                "audit_name": eki_audit.get("audit_name"),
                "reason": eki_audit.get("reason"),
                "evidence_refs": eki_audit.get("evidence_refs") or [],
            },
        },
        "gaps": {
            "recent": recent_gaps,
            "count": len(recent_gaps),
        },
        "leann": {
            "status": leann.get("status"),
            "reason": leann.get("reason"),
            "index_base": leann.get("index_base"),
            "query": leann.get("query"),
            "results": leann.get("results") or [],
            "evidence_refs": leann.get("evidence_refs") or [],
        },
        "proposals": proposals,
        "claims": claims,
        "evidence_refs": _unique_evidence_refs(
            [
                providers_audit.get("evidence_refs") or [],
                gates_check.get("evidence_refs") or [],
                eki_audit.get("evidence_refs") or [],
                leann.get("evidence_refs") or [],
                claims[1]["evidence_refs"],
            ]
        ),
        "artifacts": {
            "md_path": md_rel,
            "json_path": json_rel,
            "receipt_path": receipt_rel,
        },
    }

    if status != "ok":
        payload["hypothesis"] = (
            "LEANN antigravity_skills_safe no está disponible para completar el brief con recuperación de skills."
        )
        payload["verification_commands"] = [
            "python bin/ajaxctl doctor leann",
            "leann build antigravity_skills_safe --docs .leann/sources/skills/antigravity --include-hidden --file-types .md --force",
            "python bin/ajaxctl report daily-brief",
        ]

    md_content = _render_md(payload)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(md_content, encoding="utf-8")

    receipt = {
        "schema": "ajax.concierge.daily_brief.receipt.v1",
        "created_utc": _utc_now(now),
        "status": status,
        "read_only": True,
        "mode": "proposal_only",
        "artifacts": payload.get("artifacts"),
        "evidence_refs": payload.get("evidence_refs"),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "ok": status == "ok",
        "status": status,
        "payload": payload,
        "json_path": str(json_path),
        "md_path": str(md_path),
        "receipt_path": str(receipt_path),
        "exit_code": 0 if status == "ok" else 2,
    }
