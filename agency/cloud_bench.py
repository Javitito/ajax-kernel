from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from agency import provider_scoreboard


CLOUD_CANARY_SCHEMA = "ajax.cloud_canary.v1"
CLOUD_BENCH_SCHEMA = "ajax.cloud_bench.v1"


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


def _extract_codex_jsonl(raw: str) -> Tuple[Optional[str], Optional[str]]:
    last_text: Optional[str] = None
    last_error: Optional[str] = None
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error":
            msg = event.get("message")
            if isinstance(msg, str) and msg.strip():
                last_error = msg.strip()
        if event.get("type") == "item.completed":
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") not in {"agent_message", "assistant_message", "message"}:
                continue
            text = item.get("text") or item.get("content")
            if isinstance(text, str) and text.strip():
                last_text = text.strip()
    return last_text, last_error


def _load_model_providers(root_dir: Path) -> Dict[str, Any]:
    cfg_yaml_path = root_dir / "config" / "model_providers.yaml"
    cfg_json_path = root_dir / "config" / "model_providers.json"
    data: Any = None
    if yaml is not None and cfg_yaml_path.exists():
        try:
            data = yaml.safe_load(cfg_yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            data = None
    if data is None and cfg_json_path.exists():
        try:
            data = json.loads(cfg_json_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    if not isinstance(data, dict):
        return {}
    providers = data.get("providers")
    return providers if isinstance(providers, dict) else {}


def _load_provider_policy(root_dir: Path) -> Dict[str, Any]:
    path = root_dir / "config" / "provider_policy.yaml"
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_local_provider(cfg: Dict[str, Any]) -> bool:
    kind = str(cfg.get("kind") or "").strip().lower()
    if kind == "http_openai":
        base = str(cfg.get("base_url") or "").lower()
        return "localhost" in base or "127.0.0.1" in base
    if kind == "cli":
        cmd = cfg.get("command") or []
        cmd_str = " ".join(str(x) for x in cmd)
        # provider_cli_bridge implies cloud provider
        return "provider_cli_bridge.py" not in cmd_str
    return False


def _select_model(cfg: Dict[str, Any], model_override: Optional[str]) -> Optional[str]:
    if model_override:
        return model_override
    model = cfg.get("default_model")
    if model:
        return str(model)
    models_map = cfg.get("models")
    if isinstance(models_map, dict):
        return str(models_map.get("balanced") or models_map.get("fast") or next(iter(models_map.values()), None))
    return None


def _call_cli(
    cfg: Dict[str, Any],
    *,
    prompt: str,
    model: Optional[str],
    timeout_s: int,
) -> Tuple[bool, str, Optional[str]]:
    kind = str(cfg.get("kind") or "").strip().lower()
    cmd_template = cfg.get("command") or []
    if isinstance(cmd_template, str):
        cmd_template = [cmd_template]
    cmd: List[str] = []
    use_stdin = False
    saw_prompt_placeholder = False
    i = 0
    while i < len(cmd_template):
        token = cmd_template[i]
        next_token = cmd_template[i + 1] if i + 1 < len(cmd_template) else None
        if token in {"--prompt", "-p", "--prompt-file"} and next_token == "{prompt}":
            cmd.append(str(token))
            cmd.append(str(prompt))
            saw_prompt_placeholder = True
            i += 2
            continue
        if token == "{prompt}":
            cmd.append(str(prompt))
            saw_prompt_placeholder = True
            i += 1
            continue
        if token == "{model}":
            cmd.append(str(model or ""))
            i += 1
            continue
        cmd.append(str(token))
        i += 1
    if not cmd:
        return False, "", "cli_command_missing"
    if kind == "codex_cli_jsonl":
        if "--skip-git-repo-check" not in cmd:
            cmd.append("--skip-git-repo-check")
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return False, "", "timeout_total"
        raw = (proc.stdout or "").strip()
        last_text, last_error = _extract_codex_jsonl(raw)
        if proc.returncode != 0:
            err = (last_error or "").strip() or (proc.stderr or "").strip() or raw[:200] or "codex_failed"
            return False, "", err[:200]
        if last_error and not last_text:
            return False, "", last_error[:200]
        return True, (last_text or raw).strip(), None
    if not saw_prompt_placeholder:
        use_stdin = True
    try:
        proc = subprocess.run(
            cmd,
            input=prompt if use_stdin else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "", "timeout_total"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, "", err[:200] if err else "cli_failed"
    return True, (proc.stdout or "").strip(), None


def _call_http_openai(
    cfg: Dict[str, Any],
    *,
    prompt: str,
    model: Optional[str],
    timeout_s: int,
) -> Tuple[bool, str, Optional[int], Optional[str]]:
    if requests is None:
        return False, "", None, "requests_unavailable"
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    api_key_env = cfg.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model or cfg.get("default_model"),
        "temperature": 0,
        "stream": False,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout_s)
    except requests.Timeout:
        return False, "", None, "timeout_total"
    except Exception as exc:
        return False, "", None, str(exc)[:200]
    if resp.status_code >= 400:
        return False, "", resp.status_code, resp.text[:200]
    try:
        data = resp.json()
    except Exception:
        return False, "", resp.status_code, "parse_error"
    text = ""
    try:
        choices = data.get("choices") if isinstance(data, dict) else None
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(msg, dict):
                text = str(msg.get("content") or "")
    except Exception:
        text = ""
    return True, text.strip(), resp.status_code, None


def _classify_error(
    *,
    error: Optional[str],
    http_status: Optional[int],
) -> Optional[str]:
    if error is None and http_status is None:
        return None
    if error:
        lowered = str(error).lower()
        if "policy" in lowered or "blocked" in lowered or "not_allowed" in lowered or "tier_blocked" in lowered:
            return "policy_blocked"
        if "auth_required" in lowered or "auth required" in lowered or "missing_api_key" in lowered:
            return "auth_missing"
        if "auth" in lowered or "unauthorized" in lowered or "forbidden" in lowered:
            return "auth_missing"
        if "terminalquotaerror" in lowered or "quota_exhausted" in lowered or "insufficient_quota" in lowered:
            return "quota_exhausted"
        if "rate" in lowered or "429" in lowered:
            return "rate_limit"
        if "timeout_ttft" in lowered:
            return "timeout_ttft"
        if "timeout_total" in lowered or "timeout" in lowered:
            return "timeout_total"
        if "cli_failed" in lowered or "bridge" in lowered or "provider_cli_bridge" in lowered:
            return "wrapper_error"
        if "validation_failed" in lowered:
            return "validation_failed"
        if "parse_error" in lowered or "json" in lowered:
            return "parse_error"
    if http_status is not None:
        if http_status in {401, 403}:
            return "auth_missing"
        if http_status == 429:
            return "rate_limit"
        if 500 <= http_status <= 599:
            return "http_5xx"
        if http_status >= 400:
            return "http_4xx"
    return "unknown"


def _gate_status(by_id: Dict[str, Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    gate_fail_reason = None
    a_errors = by_id.get("A", {}).get("errors") or []
    if "quota_exhausted" in a_errors:
        gate_fail_reason = "gate_fail_A_quota"
    elif "auth_missing" in a_errors:
        gate_fail_reason = "gate_fail_A_auth"
    elif "policy_blocked" in a_errors:
        gate_fail_reason = "gate_fail_A_policy"
    elif "timeout_total" in a_errors or "timeout_ttft" in a_errors:
        gate_fail_reason = "gate_fail_A_timeout"
    if gate_fail_reason is None and by_id["A"]["json_parse_rate"] < 1.0:
        gate_fail_reason = "gate_fail_A_parse"
    elif gate_fail_reason is None and by_id["A"]["success_rate"] < 1.0:
        gate_fail_reason = "gate_fail_A_exact"
    elif gate_fail_reason is None and by_id["B"]["success_rate"] < 1.0:
        gate_fail_reason = "gate_fail_B_options"
    elif gate_fail_reason is None and by_id["C"]["success_rate"] < 1.0:
        gate_fail_reason = "gate_fail_C_no_se"
    eligible = gate_fail_reason is None
    return eligible, gate_fail_reason


def _load_scoreboard(path: Path) -> Dict[str, Any]:
    return provider_scoreboard.load_scoreboard(path)


def _default_provider_list(
    *,
    root_dir: Path,
    providers_cfg: Dict[str, Any],
    cost_mode: str,
    allow_premium: bool,
) -> List[str]:
    policy_doc = _load_provider_policy(root_dir)
    pref = []
    try:
        from agency import provider_policy  # type: ignore

        rail = os.getenv("AJAX_RAIL") or os.getenv("AJAX_ENV") or os.getenv("AJAX_MODE") or "lab"
        pref = provider_policy.preferred_providers(policy_doc, rail=rail, role="brain")
    except Exception:
        pref = []
    if not pref:
        pref = [name for name, cfg in providers_cfg.items() if isinstance(cfg, dict) and "brain" in (cfg.get("roles") or [])]
    out = []
    for name in pref:
        cfg = providers_cfg.get(name) or {}
        if cfg.get("disabled"):
            continue
        if cost_mode == "save_codex" and name.startswith("codex_") and not allow_premium:
            continue
        if _is_local_provider(cfg):
            continue
        out.append(name)
    return out


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
        {"id": "D", "prompt": 'Return JSON: {"id":"D","summary":"ok"}', "validator": "exact_id"},
        {"id": "E", "prompt": 'Return JSON: {"id":"E","items":[1,2,3]}', "validator": "exact_id"},
    ]


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


def _validate_doc(test: Dict[str, Any], doc: Dict[str, Any]) -> bool:
    validator = test.get("validator")
    if validator == "exact_json":
        expected = test.get("expected") or {}
        return _validate_exact_json(doc, expected)
    if validator == "options_max3":
        return _validate_options_max3(doc, test.get("id") or "")
    if validator == "no_se_question":
        return _validate_no_se_question(doc, test.get("id") or "")
    return _validate_exact_id(doc, test.get("id") or "")


def _parse_json(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def run_cloud_canary(
    *,
    root_dir: Path,
    provider: str,
    model: Optional[str] = None,
    json_output: bool = False,
) -> Dict[str, Any]:
    providers_cfg = _load_model_providers(root_dir)
    cfg = providers_cfg.get(provider)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"provider_not_found:{provider}")
    model_id = _select_model(cfg, model)
    timeout_s = int(_env_int("AJAX_CLOUD_TOTAL_TIMEOUT_MS", 180000) / 1000)
    test = _crawl_v1_tests()[0]
    started = time.time()
    http_status = None
    error = None
    if str(cfg.get("kind") or "").strip().lower() == "http_openai":
        ok, text, http_status, error = _call_http_openai(cfg, prompt=test["prompt"], model=model_id, timeout_s=timeout_s)
    else:
        ok, text, error = _call_cli(cfg, prompt=test["prompt"], model=model_id, timeout_s=timeout_s)
    total_ms = int((time.time() - started) * 1000)
    doc = _parse_json(text) if ok else None
    parse_ok = doc is not None
    valid = bool(doc is not None and _validate_doc(test, doc))
    success = bool(ok and parse_ok and valid)
    err_source = error
    if ok and not parse_ok:
        err_source = "parse_error"
    elif ok and parse_ok and not valid:
        err_source = "validation_failed"
    error_kind = None if success else _classify_error(error=err_source, http_status=http_status)
    payload = {
        "schema": CLOUD_CANARY_SCHEMA,
        "ts": time.time(),
        "ts_utc": _iso_utc(),
        "provider": provider,
        "model": model_id,
        "success": success,
        "parse_ok": parse_ok,
        "ttft_ms": None,
        "total_ms": total_ms,
        "error_kind": error_kind,
        "http_status": http_status,
        "output_excerpt": (text[:200] if isinstance(text, str) else None),
    }
    receipts_dir = root_dir / "artifacts" / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    out_path = receipts_dir / f"cloud_canary_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if json_output:
        payload["receipt_path"] = str(out_path)
    return payload


def run_cloud_bench(
    *,
    root_dir: Path,
    suite: str,
    providers: Optional[List[str]] = None,
    runs: int = 3,
    budget: str = "tiny",
    allow_premium: bool = False,
) -> Dict[str, Any]:
    if suite != "crawl_v1":
        raise ValueError(f"unsupported_suite:{suite}")
    providers_cfg = _load_model_providers(root_dir)
    cost_mode = (os.getenv("AJAX_COST_MODE") or "premium").strip().lower()
    allow_premium = bool(allow_premium)
    if providers is None:
        providers = _default_provider_list(root_dir=root_dir, providers_cfg=providers_cfg, cost_mode=cost_mode, allow_premium=allow_premium)
    runs = max(1, int(runs))
    timeout_total_ms = _env_int("AJAX_CLOUD_TOTAL_TIMEOUT_MS", 180000)
    timeout_s = int(timeout_total_ms / 1000)
    tests = _crawl_v1_tests()
    report: Dict[str, Any] = {
        "schema": CLOUD_BENCH_SCHEMA,
        "ts": time.time(),
        "ts_utc": _iso_utc(),
        "suite": suite,
        "runs": runs,
        "budget": budget,
        "providers": [],
    }
    scoreboard_entries: List[Dict[str, Any]] = []
    for provider in providers:
        cfg = providers_cfg.get(provider)
        if not isinstance(cfg, dict):
            continue
        model_id = _select_model(cfg, None)
        if cost_mode == "save_codex" and provider.startswith("codex_") and not allow_premium:
            continue
        # warmup
        try:
            if str(cfg.get("kind") or "").strip().lower() == "http_openai":
                _call_http_openai(cfg, prompt=tests[0]["prompt"], model=model_id, timeout_s=timeout_s)
            else:
                _call_cli(cfg, prompt=tests[0]["prompt"], model=model_id, timeout_s=timeout_s)
        except Exception:
            pass
        results: Dict[str, Any] = {"provider": provider, "model": model_id, "tests": []}
        total_latencies: List[int] = []
        for test in tests:
            test_id = test["id"]
            successes = 0
            parse_ok_count = 0
            test_errors: List[str] = []
            test_error_raw: List[str] = []
            test_latencies: List[int] = []
            for _ in range(runs):
                started = time.time()
                http_status = None
                error = None
                if str(cfg.get("kind") or "").strip().lower() == "http_openai":
                    ok, text, http_status, error = _call_http_openai(cfg, prompt=test["prompt"], model=model_id, timeout_s=timeout_s)
                else:
                    ok, text, error = _call_cli(cfg, prompt=test["prompt"], model=model_id, timeout_s=timeout_s)
                elapsed_ms = int((time.time() - started) * 1000)
                test_latencies.append(elapsed_ms)
                total_latencies.append(elapsed_ms)
                doc = _parse_json(text) if ok else None
                if doc is not None:
                    parse_ok_count += 1
                valid = bool(doc is not None and _validate_doc(test, doc))
                success = bool(ok and doc is not None and valid)
                if success:
                    successes += 1
                else:
                    err_source = error
                    if ok and doc is None:
                        err_source = "parse_error"
                    elif ok and doc is not None and not valid:
                        err_source = "validation_failed"
                    err_kind = _classify_error(error=err_source, http_status=http_status)
                    if err_kind:
                        test_errors.append(err_kind)
                    if error:
                        test_error_raw.append(str(error)[:200])
            success_rate = successes / float(runs)
            json_parse_rate = parse_ok_count / float(runs)
            p50 = sorted(test_latencies)[len(test_latencies) // 2] if test_latencies else 0
            results["tests"].append(
                {
                    "id": test_id,
                    "success_rate": success_rate,
                    "json_parse_rate": json_parse_rate,
                    "latency_p50_ms": p50,
                    "errors": test_errors,
                    "errors_raw": list(dict.fromkeys(test_error_raw))[:3],
                }
            )
        # aggregate scores
        by_id = {t["id"]: t for t in results["tests"]}
        reliability_gated = sum(by_id[t]["success_rate"] for t in ["A", "B", "C"]) / 3.0
        reliability_all = sum(by_id[t]["success_rate"] for t in ["A", "B", "C", "D", "E"]) / 5.0
        obed = sum(by_id[t]["success_rate"] for t in ["D", "E"]) / 2.0
        latency_p50 = sorted(total_latencies)[len(total_latencies) // 2] if total_latencies else 0
        latency_score = max(0.0, 1.0 - min(1.0, float(latency_p50) / float(timeout_total_ms)))
        score = (0.80 * reliability_all) + (0.15 * obed) + (0.05 * latency_score)
        eligible, gate_fail_reason = _gate_status(by_id)
        gates_passed = eligible
        results["reliability"] = reliability_all
        results["reliability_all"] = reliability_all
        results["reliability_gated"] = reliability_gated
        results["obedience"] = obed
        results["latency_p50_ms"] = latency_p50
        results["latency_score"] = latency_score
        results["score"] = score
        results["gates_passed"] = gates_passed
        results["eligible"] = eligible
        results["gate_fail_reason"] = gate_fail_reason
        report["providers"].append(results)
        scoreboard_entries.append(
            {
                "provider": provider,
                "model": model_id,
                "score": score,
                "reliability": reliability_all,
                "reliability_all": reliability_all,
                "reliability_gated": reliability_gated,
                "obedience": obed,
                "latency_p50_ms": latency_p50,
                "success_rate": reliability_all,
                "success_rate_gated": reliability_gated,
                "json_parse_rate": by_id["A"]["json_parse_rate"],
                "gates_passed": gates_passed,
                "eligible": eligible,
                "gate_fail_reason": gate_fail_reason,
            }
        )
    # artifacts
    out_dir = root_dir / "artifacts" / "benchmarks" / "cloud"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_label = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    json_path = out_dir / f"{ts_label}_bench.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = out_dir / f"{ts_label}_bench.md"
    md_lines = ["# Cloud bench", f"- suite: {suite}", f"- runs: {runs}", f"- budget: {budget}", ""]
    for entry in report["providers"]:
        md_lines.append(f"## {entry.get('provider')}:{entry.get('model')}")
        md_lines.append(
            f"- score: {entry.get('score'):.3f} gates_passed={entry.get('gates_passed')} eligible={entry.get('eligible')} gate_fail_reason={entry.get('gate_fail_reason')}"
        )
        md_lines.append(
            f"- reliability_all: {entry.get('reliability_all'):.3f} reliability_gated: {entry.get('reliability_gated'):.3f} obedience: {entry.get('obedience'):.3f}"
        )
        md_lines.append(f"- latency_p50_ms: {entry.get('latency_p50_ms')}")
        for test in entry.get("tests") or []:
            md_lines.append(
                f"  - {test.get('id')}: success_rate={test.get('success_rate')} parse_rate={test.get('json_parse_rate')} latency_p50_ms={test.get('latency_p50_ms')}"
            )
        md_lines.append("")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    latest_path = out_dir / "CLOUD_BENCH_LATEST.json"
    latest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scoreboard_path = root_dir / "artifacts" / "state" / "provider_scoreboard.json"
    window = _env_int("AJAX_SCOREBOARD_WINDOW", 30)
    provider_scoreboard.update_scoreboard(path=scoreboard_path, entries=scoreboard_entries, window=window)
    report["paths"] = {"json": str(json_path), "md": str(md_path), "latest": str(latest_path), "scoreboard": str(scoreboard_path)}
    return report
