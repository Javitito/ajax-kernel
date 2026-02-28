from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from agency.audits.runner import list_registered_audits, run_named_audit
from agency.leann_query_client import query_leann
from agency.soak_gate import run_soak_check

ROOT = Path(__file__).resolve().parents[1]
COOLDOWN_SECONDS = 6 * 3600
DEFAULT_GAP_LOOKBACK_HOURS = 72
LEANN_INDEX_REL = Path(".leann/indexes/antigravity_skills_safe/documents.leann")


def _utc_now(now_ts: Optional[float] = None) -> str:
    ts = float(now_ts if now_ts is not None else time.time())
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(now_ts: Optional[float] = None) -> str:
    ts = float(now_ts if now_ts is not None else time.time())
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")


def _safe_json_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _to_rel(root: Path, value: Any) -> str:
    path = Path(str(value))
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _evidence(kind: str, path: Any) -> Dict[str, str]:
    return {"kind": str(kind), "path": str(path)}


def _dedupe_evidence(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for item in items:
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


def _extract_ts(payload: Mapping[str, Any], *, default: float) -> float:
    for key in ("ts", "created_ts", "completed_ts", "updated_ts"):
        raw = payload.get(key)
        if isinstance(raw, (int, float)):
            try:
                return float(raw)
            except Exception:
                continue
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                continue
            try:
                return float(text)
            except Exception:
                try:
                    if text.endswith("Z"):
                        text = text[:-1] + "+00:00"
                    return float(datetime.fromisoformat(text).timestamp())
                except Exception:
                    continue
    ts_utc = payload.get("ts_utc")
    if isinstance(ts_utc, str) and ts_utc.strip():
        text = ts_utc.strip()
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return float(datetime.fromisoformat(text).timestamp())
        except Exception:
            return default
    return default


def _normalize_gap_code(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.:-]", "", text)
    if not text:
        return None
    if len(text) > 80:
        text = text[:80]
    return text.lower()


def _extract_gap_codes(payload: Mapping[str, Any]) -> List[str]:
    codes: List[str] = []

    def _push(val: Any) -> None:
        code = _normalize_gap_code(val)
        if code:
            codes.append(code)

    for key in ("gap_code", "code", "outcome_code"):
        _push(payload.get(key))

    for key in ("failure_codes", "codes"):
        raw = payload.get(key)
        if isinstance(raw, list):
            for item in raw:
                _push(item)

    findings = payload.get("findings")
    if isinstance(findings, list):
        for item in findings:
            if isinstance(item, dict):
                _push(item.get("code"))

    gaps = payload.get("gaps")
    if isinstance(gaps, list):
        for item in gaps:
            if isinstance(item, dict):
                _push(item.get("code"))

    return codes


def _collect_top_gap_codes_from_receipts(
    root: Path,
    *,
    now_ts: float,
    lookback_hours: int = DEFAULT_GAP_LOOKBACK_HOURS,
) -> List[Dict[str, Any]]:
    receipt_dir = root / "artifacts" / "receipts"
    if not receipt_dir.exists():
        return []
    lookback_s = max(24, int(lookback_hours)) * 3600
    min_ts = now_ts - float(lookback_s)

    counts: Counter[str] = Counter()
    evidence: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for path in sorted(receipt_dir.glob("*.json")):
        payload = _safe_json_load(path)
        if not payload:
            continue
        ts = _extract_ts(payload, default=float(path.stat().st_mtime))
        if ts < min_ts:
            continue
        rel = _to_rel(root, path)
        for code in _extract_gap_codes(payload):
            counts[code] += 1
            evidence[code].append(_evidence("receipt", rel))

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top = ranked[:3]
    out: List[Dict[str, Any]] = []
    for code, count in top:
        out.append(
            {
                "gap_code": code,
                "count": int(count),
                "evidence_refs": _dedupe_evidence(evidence.get(code, []))[:3],
            }
        )
    return out


def _run_system_audits(
    root: Path,
    *,
    now_ts: float,
    run_named_audit_fn: Callable[..., Dict[str, Any]] = run_named_audit,
    list_registered_audits_fn: Callable[[], Mapping[str, Any]] = list_registered_audits,
    run_soak_check_fn: Callable[..., Dict[str, Any]] = run_soak_check,
) -> Dict[str, Any]:
    audits: Dict[str, Any] = {}

    providers_payload = run_named_audit_fn(name="providers", root_dir=root, run_id=None, last=1)
    providers_summary = providers_payload.get("summary") if isinstance(providers_payload.get("summary"), dict) else {}
    providers_evidence: List[Dict[str, str]] = []
    if providers_payload.get("artifact_path"):
        providers_evidence.append(_evidence("audit_artifact", _to_rel(root, providers_payload.get("artifact_path"))))
    if providers_payload.get("receipt_path"):
        providers_evidence.append(_evidence("audit_receipt", _to_rel(root, providers_payload.get("receipt_path"))))
    audits["providers"] = {
        "ok": bool(providers_payload.get("ok")),
        "counts": {
            "critical": int(providers_summary.get("critical") or 0),
            "error": int(providers_summary.get("error") or 0),
            "warn": int(providers_summary.get("warn") or 0),
        },
        "evidence_refs": _dedupe_evidence(providers_evidence),
    }

    gates_payload = run_soak_check_fn(root, rail="lab", window_min=60, now_ts=now_ts)
    signals = gates_payload.get("signals") if isinstance(gates_payload.get("signals"), dict) else {}
    pass_count = 0
    fail_count = 0
    for sig in signals.values():
        if isinstance(sig, dict) and bool(sig.get("ok")):
            pass_count += 1
        elif isinstance(sig, dict):
            fail_count += 1
    gates_evidence: List[Dict[str, str]] = []
    if gates_payload.get("receipt_path"):
        gates_evidence.append(_evidence("gate_receipt", _to_rel(root, gates_payload.get("receipt_path"))))
    if gates_payload.get("report_path"):
        gates_evidence.append(_evidence("gate_report", _to_rel(root, gates_payload.get("report_path"))))
    audits["gates"] = {
        "ok": bool(gates_payload.get("ok")),
        "counts": {"pass": pass_count, "fail": fail_count},
        "outcome_code": str(gates_payload.get("outcome_code") or ""),
        "evidence_refs": _dedupe_evidence(gates_evidence),
    }

    registry = list_registered_audits_fn() or {}
    eki_name = next((str(k) for k in registry.keys() if str(k).lower() == "eki"), None)
    if eki_name:
        eki_payload = run_named_audit_fn(name=eki_name, root_dir=root, run_id=None, last=1)
        eki_summary = eki_payload.get("summary") if isinstance(eki_payload.get("summary"), dict) else {}
        eki_evidence: List[Dict[str, str]] = []
        if eki_payload.get("artifact_path"):
            eki_evidence.append(_evidence("audit_artifact", _to_rel(root, eki_payload.get("artifact_path"))))
        if eki_payload.get("receipt_path"):
            eki_evidence.append(_evidence("audit_receipt", _to_rel(root, eki_payload.get("receipt_path"))))
        audits["eki"] = {
            "status": "ok",
            "ok": bool(eki_payload.get("ok")),
            "counts": {
                "critical": int(eki_summary.get("critical") or 0),
                "error": int(eki_summary.get("error") or 0),
                "warn": int(eki_summary.get("warn") or 0),
            },
            "evidence_refs": _dedupe_evidence(eki_evidence),
        }
    else:
        audits["eki"] = {
            "status": "not_available",
            "ok": None,
            "counts": {"critical": 0, "error": 0, "warn": 0},
            "evidence_refs": [],
        }

    return audits


def _extract_skill_name(hit: Mapping[str, Any]) -> Optional[str]:
    text = str(hit.get("text") or "")
    m_name = re.search(r"(?mi)^name:\s*([A-Za-z0-9_.-]+)\s*$", text)
    if m_name:
        return m_name.group(1).strip()
    md = hit.get("metadata")
    if isinstance(md, dict):
        path = str(md.get("path") or md.get("source") or "")
        m_path = re.search(r"/skills/([^/]+)/", path.replace("\\", "/"))
        if m_path:
            return m_path.group(1).strip()
    return None


def _extract_rationale(hit: Mapping[str, Any], *, gap_code: str) -> str:
    text = str(hit.get("text") or "")
    m_desc = re.search(r"(?mi)^description:\s*\"?(.+?)\"?$", text)
    if m_desc:
        desc = m_desc.group(1).strip()
        if desc:
            return f"{desc} (relevante para {gap_code})"
    for line in text.splitlines():
        line_n = line.strip()
        if line_n and not line_n.startswith("---") and not line_n.lower().startswith("name:"):
            return f"{line_n[:140]} (relevante para {gap_code})"
    return f"Sugerencia basada en recuperación LEANN para {gap_code}."


def _derive_leann_suggestions(
    root: Path,
    *,
    gap_rows: Sequence[Mapping[str, Any]],
    query_leann_fn: Callable[..., List[Dict[str, Any]]] = query_leann,
) -> Dict[str, Any]:
    index_base = root / LEANN_INDEX_REL
    meta = Path(f"{index_base}.meta.json")
    passages = Path(f"{index_base}.passages.jsonl")
    base_evidence = [
        _evidence("leann_index_meta", _to_rel(root, meta)),
        _evidence("leann_index_passages", _to_rel(root, passages)),
    ]
    if not meta.exists() or not passages.exists():
        return {"status": "capability_missing", "suggestions": [], "evidence_refs": _dedupe_evidence(base_evidence)}

    scored: Dict[str, Dict[str, Any]] = {}
    for row in list(gap_rows)[:3]:
        gap_code = str(row.get("gap_code") or "").strip()
        if not gap_code:
            continue
        query = f"{gap_code} mitigation skill for ajax kernel"
        try:
            hits = query_leann_fn(str(index_base), query, top_k=5, fallback_grep=True)
        except Exception:
            return {
                "status": "capability_missing",
                "suggestions": [],
                "evidence_refs": _dedupe_evidence(base_evidence),
            }
        for idx, hit in enumerate(hits or []):
            if not isinstance(hit, dict):
                continue
            name = _extract_skill_name(hit)
            if not name:
                continue
            score = max(1, 5 - idx)
            curr = scored.get(name)
            rationale = _extract_rationale(hit, gap_code=gap_code)
            hit_ref = _evidence("leann_hit", f"{gap_code}:{hit.get('id', idx)}")
            if curr is None:
                scored[name] = {
                    "name": name,
                    "score": score,
                    "rationale": rationale,
                    "evidence_refs": base_evidence + [hit_ref],
                }
            else:
                curr["score"] = int(curr.get("score") or 0) + score
                refs = list(curr.get("evidence_refs") or []) + [hit_ref]
                curr["evidence_refs"] = _dedupe_evidence(refs)

    ranked = sorted(scored.values(), key=lambda item: (-int(item.get("score") or 0), str(item.get("name") or "")))
    suggestions: List[Dict[str, Any]] = []
    for row in ranked[:3]:
        suggestions.append(
            {
                "name": str(row.get("name") or ""),
                "rationale": str(row.get("rationale") or ""),
                "evidence_refs": _dedupe_evidence(row.get("evidence_refs") or [])[:4],
            }
        )
    return {
        "status": "ok",
        "suggestions": suggestions,
        "evidence_refs": _dedupe_evidence(base_evidence),
    }


def _render_brief_md(payload: Mapping[str, Any]) -> str:
    block_a = payload.get("block_a") if isinstance(payload.get("block_a"), dict) else {}
    block_b = payload.get("block_b") if isinstance(payload.get("block_b"), dict) else {}
    block_c = payload.get("block_c") if isinstance(payload.get("block_c"), dict) else {}
    lines: List[str] = []
    lines.append("# Daily Brief")
    lines.append("")
    lines.append(f"- created_utc: {payload.get('created_utc')}")
    lines.append(f"- status: {payload.get('status')}")
    lines.append("- mode: read-only")
    lines.append("")
    lines.append("## A) Estado sistema")
    lines.append(f"- providers: {json.dumps(block_a.get('providers_counts') or {}, ensure_ascii=False)}")
    lines.append(f"- gates: {json.dumps(block_a.get('gates_counts') or {}, ensure_ascii=False)}")
    lines.append(f"- eki: {json.dumps(block_a.get('eki_counts') or {}, ensure_ascii=False)}")
    lines.append("")
    lines.append("## B) Top 3 gaps")
    for row in block_b.get("top_gap_codes") or []:
        lines.append(f"- {row.get('gap_code')} (count={row.get('count')})")
    if not (block_b.get("top_gap_codes") or []):
        lines.append("- none")
    lines.append("")
    lines.append("## C) Top 3 suggestions (LEANN)")
    lines.append(f"- status: {block_c.get('status')}")
    for row in block_c.get("top_skills") or []:
        lines.append(f"- {row.get('name')}: {row.get('rationale')}")
    if not (block_c.get("top_skills") or []):
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def generate_daily_brief(
    root_dir: Path,
    *,
    now_ts: Optional[float] = None,
    lookback_hours: int = DEFAULT_GAP_LOOKBACK_HOURS,
    run_named_audit_fn: Callable[..., Dict[str, Any]] = run_named_audit,
    list_registered_audits_fn: Callable[[], Mapping[str, Any]] = list_registered_audits,
    run_soak_check_fn: Callable[..., Dict[str, Any]] = run_soak_check,
    query_leann_fn: Callable[..., List[Dict[str, Any]]] = query_leann,
) -> Dict[str, Any]:
    root = Path(root_dir)
    now = float(now_ts if now_ts is not None else time.time())
    stamp = _stamp(now)

    concierge_dir = root / "artifacts" / "concierge"
    receipt_dir = root / "artifacts" / "receipts"
    concierge_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    brief_json_path = concierge_dir / f"daily_brief_{stamp}.json"
    brief_md_path = concierge_dir / f"daily_brief_{stamp}.md"
    brief_receipt_path = receipt_dir / f"daily_brief_{stamp}.json"

    audits = _run_system_audits(
        root,
        now_ts=now,
        run_named_audit_fn=run_named_audit_fn,
        list_registered_audits_fn=list_registered_audits_fn,
        run_soak_check_fn=run_soak_check_fn,
    )
    gap_rows = _collect_top_gap_codes_from_receipts(root, now_ts=now, lookback_hours=lookback_hours)
    leann_block = _derive_leann_suggestions(root, gap_rows=gap_rows, query_leann_fn=query_leann_fn)

    status = "ok" if str(leann_block.get("status") or "") == "ok" else "capability_missing"
    providers_counts = audits.get("providers", {}).get("counts") if isinstance(audits.get("providers"), dict) else {}
    gates_counts = audits.get("gates", {}).get("counts") if isinstance(audits.get("gates"), dict) else {}
    eki_counts = audits.get("eki", {}).get("counts") if isinstance(audits.get("eki"), dict) else {}

    block_a_evidence = _dedupe_evidence(
        (audits.get("providers", {}).get("evidence_refs") or [])
        + (audits.get("gates", {}).get("evidence_refs") or [])
        + (audits.get("eki", {}).get("evidence_refs") or [])
    )
    block_b_evidence = _dedupe_evidence(
        [ref for row in gap_rows for ref in (row.get("evidence_refs") or [])]
    )
    block_c_evidence = _dedupe_evidence(
        (leann_block.get("evidence_refs") or [])
        + [ref for row in (leann_block.get("suggestions") or []) for ref in (row.get("evidence_refs") or [])]
    )

    payload: Dict[str, Any] = {
        "schema": "ajax.concierge.daily_brief.v1",
        "created_utc": _utc_now(now),
        "status": status,
        "read_only": True,
        "trigger": "HUMAN_DETECTED",
        "block_a": {
            "title": "Estado sistema",
            "providers_counts": providers_counts,
            "gates_counts": gates_counts,
            "eki_counts": eki_counts,
            "evidence_refs": block_a_evidence,
        },
        "block_b": {
            "title": "Top 3 gaps",
            "window_hours": max(24, int(lookback_hours)),
            "top_gap_codes": gap_rows,
            "evidence_refs": block_b_evidence,
        },
        "block_c": {
            "title": "Top 3 suggestions",
            "status": leann_block.get("status"),
            "top_skills": leann_block.get("suggestions") or [],
            "evidence_refs": block_c_evidence,
        },
        "evidence_refs": _dedupe_evidence(block_a_evidence + block_b_evidence + block_c_evidence),
        "artifacts": {
            "brief_md_path": _to_rel(root, brief_md_path),
            "brief_json_path": _to_rel(root, brief_json_path),
            "receipt_path": _to_rel(root, brief_receipt_path),
        },
    }

    if status != "ok":
        payload["hypothesis"] = "LEANN no disponible para recomendaciones; bloque C degradado a vacío."
        payload["verification_commands"] = [
            "python bin/ajaxctl doctor leann",
            "leann build antigravity_skills_safe --docs .leann/sources/skills/antigravity --include-hidden --file-types .md --force",
        ]

    brief_md_path.write_text(_render_brief_md(payload), encoding="utf-8")
    _write_json(brief_json_path, payload)

    receipt_payload = {
        "schema": "ajax.concierge.daily_brief.receipt.v1",
        "created_utc": _utc_now(now),
        "status": status,
        "trigger": "HUMAN_DETECTED",
        "generated": True,
        "read_only": True,
        "artifacts": payload.get("artifacts"),
        "evidence_refs": payload.get("evidence_refs"),
    }
    _write_json(brief_receipt_path, receipt_payload)

    return {
        "status": status,
        "generated": True,
        "brief_json_path": str(brief_json_path),
        "brief_md_path": str(brief_md_path),
        "receipt_path": str(brief_receipt_path),
        "payload": payload,
    }


def maybe_trigger_human_detected_brief(
    root_dir: Path,
    *,
    now_ts: Optional[float] = None,
    cooldown_s: int = COOLDOWN_SECONDS,
    generator: Callable[..., Dict[str, Any]] = generate_daily_brief,
) -> Dict[str, Any]:
    root = Path(root_dir)
    now = float(now_ts if now_ts is not None else time.time())
    stamp = _stamp(now)

    concierge_dir = root / "artifacts" / "concierge"
    receipt_dir = root / "artifacts" / "receipts"
    state_path = concierge_dir / "daily_brief_state.json"
    receipt_path = receipt_dir / f"daily_brief_{stamp}.json"
    concierge_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)

    state = _safe_json_load(state_path)
    last_emit_ts = None
    try:
        if state.get("last_emit_ts") is not None:
            last_emit_ts = float(state.get("last_emit_ts"))
    except Exception:
        last_emit_ts = None

    if last_emit_ts is not None and now - last_emit_ts < float(max(1, int(cooldown_s))):
        remaining = max(0.0, float(cooldown_s) - (now - last_emit_ts))
        payload = {
            "schema": "ajax.concierge.daily_brief.receipt.v1",
            "created_utc": _utc_now(now),
            "status": "cooldown_skipped",
            "trigger": "HUMAN_DETECTED",
            "generated": False,
            "read_only": True,
            "cooldown_s": int(cooldown_s),
            "cooldown_remaining_s": round(remaining, 3),
            "last_brief_json_path": state.get("last_brief_json_path"),
            "last_brief_md_path": state.get("last_brief_md_path"),
        }
        _write_json(receipt_path, payload)
        return {
            "status": "cooldown_skipped",
            "generated": False,
            "brief_json_path": state.get("last_brief_json_path"),
            "brief_md_path": state.get("last_brief_md_path"),
            "receipt_path": str(receipt_path),
        }

    brief = generator(root, now_ts=now)
    state_out = {
        "schema": "ajax.concierge.daily_brief.state.v1",
        "updated_utc": _utc_now(now),
        "last_emit_ts": now,
        "last_brief_json_path": _to_rel(root, brief.get("brief_json_path")),
        "last_brief_md_path": _to_rel(root, brief.get("brief_md_path")),
        "last_receipt_path": _to_rel(root, brief.get("receipt_path")),
    }
    _write_json(state_path, state_out)
    return brief

