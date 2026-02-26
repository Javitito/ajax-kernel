from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from agency.provider_breathing import _load_provider_configs  # type: ignore
except Exception:  # pragma: no cover
    _load_provider_configs = None  # type: ignore


def _classify_bridge_result(
    *,
    returncode: Optional[int],
    stdout: str,
    stderr: str,
    exc: Optional[Exception],
) -> str:
    if exc is not None:
        text = str(exc).lower()
        if isinstance(exc, subprocess.TimeoutExpired) or "timeout" in text:
            return "TIMEOUT"
        if isinstance(exc, FileNotFoundError) or "not found" in text:
            return "MISSING_BINARY"
        return "RC1"
    if returncode is None:
        return "RC1"
    if returncode != 0:
        if "auth" in stderr.lower() or "api key" in stderr.lower():
            return "AUTH"
        if returncode == 1:
            return "RC1"
        return "RC_NONZERO"
    if not stdout.strip():
        return "OUTPUT_EMPTY"
    if "auth" in stderr.lower() or "api key" in stderr.lower():
        return "AUTH"
    return "OK"


def _build_prompt_cmd(provider: str, cfg: Dict[str, Any]) -> List[str]:
    cmd_template = cfg.get("infer_command") or cfg.get("command") or []
    if isinstance(cmd_template, str):
        cmd_template = [cmd_template]
    if not isinstance(cmd_template, list) or not cmd_template:
        return []
    model = cfg.get("default_model") or cfg.get("model") or ""
    prompt = "ping"
    cmd: List[str] = []
    for token in cmd_template:
        if token == "{model}":
            cmd.append(str(model))
        elif token == "{prompt}":
            cmd.append(prompt)
        else:
            cmd.append(str(token))
    return cmd


def _build_probe_cmd(provider: str, cfg: Dict[str, Any]) -> List[str]:
    cmd_template = cfg.get("probe_command") or []
    if isinstance(cmd_template, str):
        cmd_template = [cmd_template]
    if not isinstance(cmd_template, list) or not cmd_template:
        return []
    model = cfg.get("default_model") or cfg.get("model") or ""
    cmd: List[str] = []
    for token in cmd_template:
        if token == "{model}":
            cmd.append(str(model))
        elif token == "{prompt}":
            cmd.append("ping")
        else:
            cmd.append(str(token))
    return cmd


def _cli_infer_has_prompt_placeholder(cfg: Dict[str, Any]) -> bool:
    cmd_template = cfg.get("infer_command") or cfg.get("command") or []
    if isinstance(cmd_template, str):
        cmd_template = [cmd_template]
    if not isinstance(cmd_template, list):
        return False
    for token in cmd_template:
        if "{prompt}" in str(token):
            return True
    return False


def run_bridge_doctor(
    root_dir: Path,
    *,
    providers: Optional[List[str]] = None,
    timeout_s: float = 6.0,
) -> Dict[str, Any]:
    cfgs: Dict[str, Any] = {}
    if _load_provider_configs is not None:
        try:
            cfgs = _load_provider_configs(Path(root_dir))
        except Exception:
            cfgs = {}
    providers_cfg = cfgs.get("providers") if isinstance(cfgs, dict) else {}
    providers_cfg = providers_cfg if isinstance(providers_cfg, dict) else {}
    selected = []
    for name, cfg in providers_cfg.items():
        if providers and name not in providers:
            continue
        kind = str((cfg or {}).get("kind") or "").strip().lower()
        if kind not in {"cli", "codex_cli_jsonl"}:
            continue
        selected.append((name, cfg))
    results: Dict[str, Any] = {}
    configured_cli_providers = len(selected)
    probed_count = 0
    skipped_unprobed_count = 0
    for name, cfg in selected:
        probe_cmd = _build_probe_cmd(name, cfg if isinstance(cfg, dict) else {})
        command_source = "probe" if probe_cmd else "infer"
        if probe_cmd:
            cmd = probe_cmd
            probe_timeout = min(float(timeout_s), float((cfg or {}).get("probe_timeout_seconds") or 2.0))
            timeout_local = max(0.2, probe_timeout)
        else:
            if not _cli_infer_has_prompt_placeholder(cfg if isinstance(cfg, dict) else {}):
                skipped_unprobed_count += 1
                raw_cmd = (cfg or {}).get("infer_command") if isinstance(cfg, dict) else None
                if not raw_cmd:
                    raw_cmd = (cfg or {}).get("command") if isinstance(cfg, dict) else None
                if isinstance(raw_cmd, list):
                    cmd_text = " ".join(shlex.quote(str(x)) for x in raw_cmd)
                elif isinstance(raw_cmd, str):
                    cmd_text = raw_cmd
                else:
                    cmd_text = ""
                results[name] = {
                    "status": "UNPROBED_COMMAND_TYPE",
                    "detail": "command_has_no_prompt_placeholder",
                    "reason": "incompatible_for_subcall",
                    "cmd_template": cmd_text,
                    "command_source": "unprobeable",
                }
                continue
            cmd = _build_prompt_cmd(name, cfg)
            timeout_local = timeout_s
        if not cmd:
            results[name] = {"status": "MISSING_BINARY", "detail": "command_missing", "command_source": command_source}
            continue
        probed_count += 1
        env = os.environ.copy()
        extra_env = cfg.get("env")
        if isinstance(extra_env, dict):
            for k, v in extra_env.items():
                env[str(k)] = str(v)
        cwd = cfg.get("workdir") if isinstance(cfg, dict) else None
        cwd_path = str(Path(root_dir) / cwd) if isinstance(cwd, str) and cwd.strip() else None
        exc = None
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_local,
                env=env,
                cwd=cwd_path,
                check=False,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            status = _classify_bridge_result(
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                exc=None,
            )
            results[name] = {
                "status": status,
                "returncode": proc.returncode,
                "stdout_tail": stdout[-400:],
                "stderr_tail": stderr[-400:],
                "cmd": " ".join(shlex.quote(str(x)) for x in cmd),
                "command_source": command_source,
            }
        except Exception as err:
            exc = err
            status = _classify_bridge_result(returncode=None, stdout="", stderr="", exc=exc)
            results[name] = {
                "status": status,
                "error": str(exc)[:300],
                "cmd": " ".join(shlex.quote(str(x)) for x in cmd),
                "command_source": command_source,
            }
    coverage_ok = not (configured_cli_providers > 0 and probed_count == 0)
    payload_ok = bool(coverage_ok)
    if any(
        isinstance(entry, dict) and str(entry.get("status") or "").strip().upper() not in {"OK", "UP"}
        for entry in results.values()
    ):
        payload_ok = False
    payload = {
        "schema": "ajax.bridge_doctor.v1",
        "ts": time.time(),
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
        "ok": payload_ok,
        "summary": {
            "configured_cli_providers": configured_cli_providers,
            "probed_count": probed_count,
            "skipped_unprobed_count": skipped_unprobed_count,
            "coverage_ok": coverage_ok,
        },
        "providers": results,
    }
    return payload


def persist_bridge_doctor(root_dir: Path, payload: Dict[str, Any]) -> Path:
    out_dir = Path(root_dir) / "artifacts" / "health"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_label = time.strftime("%Y%m%d-%H%M%S", time.gmtime(time.time()))
    path = out_dir / f"bridge_doctor_{ts_label}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt_dir = Path(root_dir) / "artifacts" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt = receipt_dir / f"bridge_doctor_{ts_label}.json"
    receipt.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
