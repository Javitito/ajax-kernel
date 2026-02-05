from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore


LMSTUDIO_BENCH_SCHEMA = "ajax.lmstudio_bench.v1"
LMSTUDIO_TEST_SCHEMA = "ajax.lmstudio_test.v1"


def _iso_utc(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts or time.time()))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _probe_base_url() -> Optional[str]:
    if requests is None:
        return None
    candidates = []
    base_env = os.getenv("LMSTUDIO_BASE_URL")
    if base_env:
        candidates.append(base_env.strip().rstrip("/"))
    candidates.extend(["http://127.0.0.1:1234", "http://127.0.0.1:1235"])
    for base in candidates:
        try:
            resp = requests.get(f"{base}/v1/models", timeout=2.0)
            if resp.status_code < 400:
                return base
        except Exception:
            continue
    return None


def _list_models(base_url: str) -> List[str]:
    if requests is None:
        return []
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5.0)
        if resp.status_code >= 400:
            return []
        data = resp.json()
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [str(item.get("id")) for item in data["data"] if isinstance(item, dict) and item.get("id")]
        if isinstance(data, dict) and isinstance(data.get("models"), list):
            return [str(item.get("id")) for item in data["models"] if isinstance(item, dict) and item.get("id")]
    except Exception:
        return []
    return []


def _crawl_v1_tests() -> List[Dict[str, Any]]:
    return [
        {
            "id": "A",
            "prompt": 'Return JSON only: {"id":"A","ok":true}',
            "validator": "exact_json",
            "expected": {"id": "A", "ok": True},
        },
        {
            "id": "B",
            "prompt": 'Return JSON only: {"id":"B","options":["one","two","three"]}',
            "validator": "options_max3",
        },
        {
            "id": "C",
            "prompt": 'Return JSON only: {"id":"C","no_se":true,"question":"...?"}',
            "validator": "no_se_question",
        },
        {
            "id": "D",
            "prompt": 'Return JSON only: {"id":"D","summary":"short"}',
            "validator": "summary_len",
            "max_len": 80,
        },
        {"id": "E", "prompt": 'Return JSON: {"id":"E","items":[1,2,3]}', "validator": "exact_id"},
    ]


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _validate_exact_id(doc: Dict[str, Any], expected: str) -> bool:
    return isinstance(doc, dict) and str(doc.get("id") or "") == expected


def _validate_exact_json(doc: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    return isinstance(doc, dict) and doc == expected


def _validate_options_max3(doc: Dict[str, Any], expected_id: str) -> bool:
    if not isinstance(doc, dict) or str(doc.get("id") or "") != expected_id:
        return False
    options = doc.get("options")
    if not isinstance(options, list):
        return False
    return 1 <= len(options) <= 3


def _validate_no_se_question(doc: Dict[str, Any], expected_id: str) -> bool:
    if not isinstance(doc, dict) or str(doc.get("id") or "") != expected_id:
        return False
    if doc.get("no_se") is not True:
        return False
    question = doc.get("question")
    if not isinstance(question, str) or not question.strip():
        return False
    return question.count("?") == 1


def _validate_summary_len(doc: Dict[str, Any], expected_id: str, max_len: int) -> bool:
    if not isinstance(doc, dict) or str(doc.get("id") or "") != expected_id:
        return False
    summary = doc.get("summary")
    if not isinstance(summary, str):
        return False
    return len(summary.strip()) <= max_len


def _validate_doc(test: Dict[str, Any], doc: Dict[str, Any]) -> bool:
    validator = test.get("validator")
    if validator == "exact_json":
        expected = test.get("expected") or {}
        return _validate_exact_json(doc, expected)
    if validator == "options_max3":
        return _validate_options_max3(doc, test.get("id") or "")
    if validator == "no_se_question":
        return _validate_no_se_question(doc, test.get("id") or "")
    if validator == "summary_len":
        return _validate_summary_len(doc, test.get("id") or "", int(test.get("max_len") or 0))
    return _validate_exact_id(doc, test.get("id") or "")


def _classify_error(error: Optional[str], http_status: Optional[int]) -> Optional[str]:
    if error is None and http_status is None:
        return None
    if error:
        lowered = str(error).lower()
        if "timeout" in lowered:
            return "timeout_total"
        if "parse_error" in lowered:
            return "parse_error"
        if "validation_failed" in lowered:
            return "validation_failed"
        if "connection" in lowered or "refused" in lowered:
            return "conn_error"
    if http_status is not None:
        if http_status == 429:
            return "rate_limit"
        if 400 <= http_status <= 499:
            return "http_4xx"
        if 500 <= http_status <= 599:
            return "http_5xx"
    return "unknown"


def _call_chat(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout_s: int,
) -> Tuple[bool, str, Optional[int], Optional[str], int]:
    if requests is None:
        return False, "", None, "requests_unavailable", 0
    payload = {
        "model": model,
        "temperature": 0.0,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    started = time.time()
    try:
        resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
    except requests.Timeout:
        total_ms = int((time.time() - started) * 1000)
        return False, "", None, "timeout_total", total_ms
    except Exception as exc:
        total_ms = int((time.time() - started) * 1000)
        return False, "", None, str(exc)[:200], total_ms
    total_ms = int((time.time() - started) * 1000)
    if resp.status_code >= 400:
        return False, "", resp.status_code, resp.text[:200], total_ms
    try:
        data = resp.json()
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict):
                return True, str(msg.get("content") or ""), resp.status_code, None, total_ms
    except Exception:
        return False, "", resp.status_code, "parse_error", total_ms
    return False, "", resp.status_code, "empty_response", total_ms


def _p95(values: List[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = int(len(ordered) * 0.95) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def _score_model(results: Dict[str, Any], timeout_total_ms: int) -> Tuple[float, float, float, int, float]:
    tests = results.get("tests") or []
    if not isinstance(tests, list) or not tests:
        return 0.0, 0.0, 0.0, 0, 0.0
    success_rates = [t.get("success_rate", 0.0) for t in tests if isinstance(t, dict)]
    parse_rates = [t.get("json_parse_rate", 0.0) for t in tests if isinstance(t, dict)]
    reliability = sum(success_rates) / float(len(success_rates) or 1)
    obedience = sum(parse_rates) / float(len(parse_rates) or 1)
    all_lat = results.get("latencies_ms") or []
    p95_total = _p95([int(v) for v in all_lat if isinstance(v, int)]) if all_lat else 0
    latency_score = max(0.0, 1.0 - min(1.0, float(p95_total) / float(timeout_total_ms)))
    score = (0.70 * reliability) + (0.25 * obedience) + (0.05 * latency_score)
    return reliability, obedience, latency_score, p95_total, score


def run_lmstudio_bench(
    *,
    root_dir: Path,
    suite: str,
    models_filter: str = "all",
    runs: int = 3,
    select_best: bool = False,
) -> Dict[str, Any]:
    if suite != "crawl_v1":
        raise ValueError(f"unsupported_suite:{suite}")
    base_url = _probe_base_url()
    if not base_url:
        raise RuntimeError("lmstudio_unreachable")
    models = _list_models(base_url)
    if not models:
        raise RuntimeError("lmstudio_no_models")
    chosen_models: List[str] = []
    raw_filter = (models_filter or "all").strip()
    if raw_filter == "all":
        chosen_models = models
    elif "," in raw_filter:
        allowed = {m.strip() for m in raw_filter.split(",") if m.strip()}
        chosen_models = [m for m in models if m in allowed]
    else:
        try:
            import re

            pattern = re.compile(raw_filter)
            chosen_models = [m for m in models if pattern.search(m)]
        except Exception:
            chosen_models = [m for m in models if m == raw_filter]
    runs = max(1, int(runs))
    timeout_total_ms = _env_int("AJAX_LMSTUDIO_TOTAL_TIMEOUT_MS", 180000)
    timeout_s = int(timeout_total_ms / 1000)
    tests = _crawl_v1_tests()
    report: Dict[str, Any] = {
        "schema": LMSTUDIO_BENCH_SCHEMA,
        "ts": time.time(),
        "ts_utc": _iso_utc(),
        "suite": suite,
        "runs": runs,
        "base_url": base_url,
        "models": [],
    }
    rankings: List[Dict[str, Any]] = []
    for model_id in chosen_models:
        # warmup
        _call_chat(base_url=base_url, model=model_id, prompt=tests[0]["prompt"], timeout_s=timeout_s)
        results: Dict[str, Any] = {"model": model_id, "tests": [], "latencies_ms": []}
        gate_fail_reason = None
        d_length_ok = True
        for test in tests:
            test_id = test["id"]
            successes = 0
            parse_ok_count = 0
            test_latencies: List[int] = []
            per_run: List[Dict[str, Any]] = []
            for _ in range(runs):
                ok, text, http_status, error, total_ms = _call_chat(
                    base_url=base_url,
                    model=model_id,
                    prompt=test["prompt"],
                    timeout_s=timeout_s,
                )
                test_latencies.append(total_ms)
                results["latencies_ms"].append(total_ms)
                doc = _parse_json(text) if ok else None
                parse_ok = doc is not None
                if parse_ok:
                    parse_ok_count += 1
                valid = bool(doc is not None and _validate_doc(test, doc))
                success = bool(ok and parse_ok and valid)
                if success:
                    successes += 1
                if test_id == "D":
                    if not parse_ok or not valid:
                        d_length_ok = False
                err_source = error
                if ok and not parse_ok:
                    err_source = "parse_error"
                elif ok and parse_ok and not valid:
                    err_source = "validation_failed"
                error_kind = _classify_error(err_source, http_status)
                per_run.append(
                    {
                        "ok": ok,
                        "total_ms": total_ms,
                        "parse_ok": parse_ok,
                        "valid": valid,
                        "error_kind": error_kind,
                        "http_status": http_status,
                    }
                )
            success_rate = successes / float(runs)
            json_parse_rate = parse_ok_count / float(runs)
            p95 = _p95(test_latencies)
            results["tests"].append(
                {
                    "id": test_id,
                    "success_rate": success_rate,
                    "json_parse_rate": json_parse_rate,
                    "latency_p95_ms": p95,
                    "runs": per_run,
                }
            )
        by_id = {t["id"]: t for t in results["tests"]}
        if by_id["A"]["json_parse_rate"] < 1.0:
            gate_fail_reason = "gate_fail_A_parse"
        elif by_id["A"]["success_rate"] < 1.0:
            gate_fail_reason = "gate_fail_A_exact"
        elif by_id["B"]["success_rate"] < 1.0:
            gate_fail_reason = "gate_fail_B_options"
        elif by_id["C"]["success_rate"] < 1.0:
            gate_fail_reason = "gate_fail_C_no_se"
        elif not d_length_ok:
            gate_fail_reason = "gate_fail_D_length"
        eligible = gate_fail_reason is None
        reliability, obedience, latency_score, p95_total, score = _score_model(results, timeout_total_ms)
        results["reliability"] = reliability
        results["obedience"] = obedience
        results["latency_p95_ms"] = p95_total
        results["latency_score"] = latency_score
        results["score"] = score
        results["eligible"] = eligible
        results["gate_fail_reason"] = gate_fail_reason
        report["models"].append(results)
        if eligible:
            rankings.append(
                {
                    "model": model_id,
                    "score": score,
                    "reliability": reliability,
                    "obedience": obedience,
                    "latency_p95_ms": p95_total,
                }
            )
    rankings.sort(key=lambda item: (-item["score"], item["latency_p95_ms"]))
    report["rankings"] = rankings
    out_dir = root_dir / "artifacts" / "benchmarks" / "lmstudio"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = out_dir / f"{ts_label}_bench.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / f"{ts_label}_bench.md"
    md_lines = ["# LM Studio bench", f"- suite: {suite}", f"- runs: {runs}", f"- base_url: {base_url}", ""]
    md_lines.append("## Ranking (eligible only)")
    if rankings:
        for idx, item in enumerate(rankings, start=1):
            md_lines.append(f"{idx}. {item['model']} score={item['score']:.3f} p95_ms={item['latency_p95_ms']}")
    else:
        md_lines.append("No eligible models.")
    md_lines.append("")
    for entry in report["models"]:
        md_lines.append(f"## {entry.get('model')}")
        md_lines.append(
            f"- eligible={entry.get('eligible')} gate_fail_reason={entry.get('gate_fail_reason')} score={entry.get('score'):.3f}"
        )
        md_lines.append(
            f"- reliability={entry.get('reliability'):.3f} obedience={entry.get('obedience'):.3f} latency_p95_ms={entry.get('latency_p95_ms')}"
        )
        for test in entry.get("tests") or []:
            md_lines.append(
                f"  - {test.get('id')}: success_rate={test.get('success_rate')} parse_rate={test.get('json_parse_rate')} latency_p95_ms={test.get('latency_p95_ms')}"
            )
        md_lines.append("")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    latest_path = out_dir / "LMSTUDIO_BENCH_LATEST.json"
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["paths"] = {"json": str(json_path), "md": str(md_path), "latest": str(latest_path)}
    if select_best and rankings:
        best = rankings[0]
        fallback_path = root_dir / "artifacts" / "state" / "fallback_local_model.json"
        fallback_payload = {"model_id": best["model"], "base_url": base_url, "ts_utc": _iso_utc()}
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path.write_text(json.dumps(fallback_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["paths"]["fallback"] = str(fallback_path)
    return report


def run_lmstudio_test(*, root_dir: Path) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests_unavailable")
    fallback_path = root_dir / "artifacts" / "state" / "fallback_local_model.json"
    if not fallback_path.exists():
        raise RuntimeError("fallback_local_model_missing")
    payload = json.loads(fallback_path.read_text(encoding="utf-8"))
    model_id = str(payload.get("model_id") or "").strip()
    base_url = str(payload.get("base_url") or "").strip().rstrip("/")
    if not model_id or not base_url:
        raise RuntimeError("fallback_local_model_invalid")
    timeout_total_ms = _env_int("AJAX_LMSTUDIO_TOTAL_TIMEOUT_MS", 60000)
    timeout_s = int(timeout_total_ms / 1000)
    test = _crawl_v1_tests()[0]
    ok, text, http_status, error, total_ms = _call_chat(
        base_url=base_url, model=model_id, prompt=test["prompt"], timeout_s=timeout_s
    )
    doc = _parse_json(text) if ok else None
    parse_ok = doc is not None
    valid = bool(doc is not None and _validate_doc(test, doc))
    success = bool(ok and parse_ok and valid)
    err_source = error
    if ok and not parse_ok:
        err_source = "parse_error"
    elif ok and parse_ok and not valid:
        err_source = "validation_failed"
    error_kind = None if success else _classify_error(err_source, http_status)
    result = {
        "schema": LMSTUDIO_TEST_SCHEMA,
        "ts": time.time(),
        "ts_utc": _iso_utc(),
        "base_url": base_url,
        "model": model_id,
        "success": success,
        "parse_ok": parse_ok,
        "total_ms": total_ms,
        "error_kind": error_kind,
        "http_status": http_status,
    }
    return result
