from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

HISTORY_PATH = Path("pre") / "ajax_history_v1.ndjson"
PEPITAS_PATH = Path("Pepitas_Index.json")

HISTORY_MAX_LINES = 4000
HISTORY_MAX_BYTES = 2_000_000
PEPITAS_MAX_ITEMS = 2000

REF_SCORE_MIN = 0.2
MAX_REFS = 8


@dataclass
class SourceScan:
    path: str
    kind: str
    scanned: int
    matched: int
    truncated: bool


@dataclass
class RefCandidate:
    ref_type: str
    ref_id: str
    score: float
    excerpt: str
    text: str
    signals: List[str]


@dataclass
class P0Result:
    context: Dict[str, Any]
    sources: List[SourceScan]
    refs: List[RefCandidate]


def _utc_now(now: Optional[datetime] = None) -> str:
    value = now or datetime.now(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _tokenize(text: str) -> List[str]:
    tokens = re.split(r"[^a-zA-Z0-9_]+", text.lower())
    return [tok for tok in tokens if tok and len(tok) >= 3]


def _score_text(text: str, tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for tok in tokens if tok in text_lower)
    return hits / float(len(tokens))


def _make_excerpt(text: str, limit: int = 140) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def _extract_text(record: Dict[str, Any]) -> str:
    for key in ("text", "summary", "content", "excerpt", "note"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _detect_ref_signals(text: str) -> List[str]:
    lowered = text.lower()
    signals: List[str] = []

    def _add(value: str) -> None:
        if value not in signals:
            signals.append(value)

    if "latency" in lowered and any(
        tag in lowered for tag in ("high", "slow", "timeout", "too high")
    ):
        _add("latency_too_high")
    elif "latency" in lowered:
        _add("latency")

    if "probe" in lowered and any(tag in lowered for tag in ("fail", "error", "timeout")):
        _add("probe_failed")
    elif "probe" in lowered:
        _add("probe_issue")

    if any(tag in lowered for tag in ("timeout", "timed out")):
        _add("timeout")
    if any(tag in lowered for tag in ("fail", "error", "exception")):
        _add("failure")
    if any(tag in lowered for tag in ("success", "ok", "passed")):
        _add("success")

    return signals


def _resolve_source_root(root_dir: Path, env: Optional[Dict[str, str]] = None) -> tuple[Path, str]:
    env_data = env if env is not None else os.environ
    for key in ("LEANN_ROOT", "AJAX_HOME", "MIO_ROOT"):
        raw = env_data.get(key)
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (root_dir / candidate).resolve()
        history = candidate / HISTORY_PATH
        pepitas = candidate / PEPITAS_PATH
        if history.exists() or pepitas.exists():
            return candidate, key
    return root_dir, "root_dir"


def _scan_history(source_root: Path, tokens: List[str]) -> P0Result:
    path = source_root / HISTORY_PATH
    scanned = 0
    matched = 0
    truncated = False
    refs: List[RefCandidate] = []
    sources = [SourceScan(path=str(path), kind="history", scanned=0, matched=0, truncated=False)]

    if not path.exists():
        return P0Result(context={}, sources=sources, refs=[])

    bytes_read = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                scanned += 1
                bytes_read += len(line.encode("utf-8"))
                if scanned > HISTORY_MAX_LINES or bytes_read > HISTORY_MAX_BYTES:
                    truncated = True
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                text = payload.get("text")
                if not isinstance(text, str):
                    continue
                score = _score_text(text, tokens)
                if score < REF_SCORE_MIN:
                    continue
                matched += 1
                ref_id = str(payload.get("id") or f"line_{scanned}")
                signals = _detect_ref_signals(text)
                refs.append(
                    RefCandidate(
                        ref_type="history",
                        ref_id=ref_id,
                        score=score,
                        excerpt=_make_excerpt(text),
                        text=text,
                        signals=signals,
                    )
                )
    except Exception:
        truncated = True

    sources = [
        SourceScan(
            path=str(path), kind="history", scanned=scanned, matched=matched, truncated=truncated
        )
    ]
    return P0Result(context={}, sources=sources, refs=refs)


def _scan_pepitas(source_root: Path, tokens: List[str]) -> P0Result:
    path = source_root / PEPITAS_PATH
    scanned = 0
    matched = 0
    truncated = False
    refs: List[RefCandidate] = []
    sources = [SourceScan(path=str(path), kind="pepitas", scanned=0, matched=0, truncated=False)]

    if not path.exists():
        return P0Result(context={}, sources=sources, refs=[])

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        truncated = True
        sources = [SourceScan(path=str(path), kind="pepitas", scanned=0, matched=0, truncated=True)]
        return P0Result(context={}, sources=sources, refs=[])

    if isinstance(payload, dict):
        for key in ("pepitas", "items", "records", "entries"):
            if isinstance(payload.get(key), list):
                payload = payload.get(key)
                break
    if not isinstance(payload, list):
        sources = [
            SourceScan(path=str(path), kind="pepitas", scanned=0, matched=0, truncated=False)
        ]
        return P0Result(context={}, sources=sources, refs=[])

    for item in payload:
        scanned += 1
        if scanned > PEPITAS_MAX_ITEMS:
            truncated = True
            break
        if not isinstance(item, dict):
            continue
        text = _extract_text(item)
        if not text:
            continue
        score = _score_text(text, tokens)
        if score < REF_SCORE_MIN:
            continue
        matched += 1
        ref_id = str(item.get("id") or item.get("pepita_id") or f"pepita_{scanned}")
        signals = _detect_ref_signals(text)
        refs.append(
            RefCandidate(
                ref_type="pepita",
                ref_id=ref_id,
                score=score,
                excerpt=_make_excerpt(text),
                text=text,
                signals=signals,
            )
        )

    sources = [
        SourceScan(
            path=str(path), kind="pepitas", scanned=scanned, matched=matched, truncated=truncated
        )
    ]
    return P0Result(context={}, sources=sources, refs=refs)


def _select_refs(refs: Iterable[RefCandidate]) -> List[RefCandidate]:
    ranked = sorted(refs, key=lambda ref: (-ref.score, ref.ref_id))
    return ranked[:MAX_REFS]


def _signal_sets(refs: Iterable[RefCandidate]) -> Dict[str, List[str]]:
    signals = set()
    failure_ids = set()
    success_ids = set()
    for ref in refs:
        signals.update(ref.signals)
        if any(
            tag in ref.signals for tag in ("failure", "probe_failed", "latency_too_high", "timeout")
        ):
            failure_ids.add(ref.ref_id)
        if "success" in ref.signals:
            success_ids.add(ref.ref_id)
    return {
        "signals": sorted(signals),
        "failure_ids": sorted(failure_ids),
        "success_ids": sorted(success_ids),
    }


def _detect_gaps(refs: List[RefCandidate], novelty: bool) -> List[Dict[str, str]]:
    gaps: List[Dict[str, str]] = []
    signals = _signal_sets(refs)
    signal_list = signals["signals"]
    failure_ids = signals["failure_ids"]
    success_ids = signals["success_ids"]
    if novelty:
        gaps.append(
            {
                "kind": "unknown",
                "note": "no refs above threshold",
                "signals": [],
                "ref_ids": [],
            }
        )
    if failure_ids:
        gaps.append(
            {
                "kind": "repeat_fail",
                "note": "failure signals present in refs",
                "signals": signal_list,
                "ref_ids": failure_ids,
            }
        )
    if failure_ids and success_ids:
        gaps.append(
            {
                "kind": "contradiction",
                "note": "success and failure signals both present",
                "signals": signal_list,
                "ref_ids": sorted(set(failure_ids) | set(success_ids)),
            }
        )
    return gaps


def build_p0_context(
    root_dir: Path,
    envelope_id: str,
    objective_query: str,
    now: Optional[datetime] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    tokens = _tokenize(objective_query)
    source_root, resolved_from = _resolve_source_root(root_dir, env=env)
    history = _scan_history(source_root, tokens)
    pepitas = _scan_pepitas(source_root, tokens)

    sources_checked = history.sources + pepitas.sources
    refs = _select_refs(history.refs + pepitas.refs)

    novelty = len(refs) == 0
    gaps_detected = _detect_gaps(refs, novelty)

    refs_used = [
        {
            "type": ref.ref_type,
            "id": ref.ref_id,
            "score": round(ref.score, 4),
            "excerpt": ref.excerpt,
        }
        for ref in refs
    ]

    history_scan = (
        sources_checked[0]
        if sources_checked
        else SourceScan(path="", kind="history", scanned=0, matched=0, truncated=False)
    )
    pepitas_scan = (
        sources_checked[1]
        if len(sources_checked) > 1
        else SourceScan(path="", kind="pepitas", scanned=0, matched=0, truncated=False)
    )
    summary = (
        f"P0 scanned {history_scan.scanned} history lines (matched {history_scan.matched}) and "
        f"{pepitas_scan.scanned} pepitas entries (matched {pepitas_scan.matched}); "
        f"found {len(refs_used)} refs above threshold; novelty={str(novelty).lower()}."
    )

    return {
        "schema_version": "0.1",
        "envelope_id": envelope_id,
        "created_utc": _utc_now(now),
        "scoring_method": "token_overlap_v0",
        "score_threshold": REF_SCORE_MIN,
        "objective_query": objective_query,
        "source_resolution": [
            {
                "kind": "history",
                "resolved_from": resolved_from,
                "path": str(source_root / HISTORY_PATH),
            },
            {
                "kind": "pepitas",
                "resolved_from": resolved_from,
                "path": str(source_root / PEPITAS_PATH),
            },
        ],
        "sources_checked": [
            {
                "path": src.path,
                "kind": src.kind,
                "scanned": src.scanned,
                "matched": src.matched,
                "truncated": src.truncated,
            }
            for src in sources_checked
        ],
        "refs_used": refs_used,
        "gaps_detected": gaps_detected,
        "novelty": novelty,
        "summary": summary,
    }
