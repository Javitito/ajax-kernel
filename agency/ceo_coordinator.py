#!/usr/bin/env python3
"""CEO Coordinator – light-weight swarm orchestrator.

This module allows the CEO to delegate work to multiple agents (Qwen, Gemini,
Groq, shell executors, etc.) in parallel using a JSON job description.  It does
not rely on Adam/Ajax; instead it launches the configured commands/tools
directly with tight limits (timeouts, token budgets) and produces a compact
summary artifact.

Usage:
    python -m agency.ceo_coordinator path/to/job.json [--output summary.json]

Job JSON schema (minimal):
{
  "goal": "Recover LEANN search",
  "max_parallel": 2,
  "defaults": {
      "timeout_sec": 180,
      "cwd": ".",
      "capture_output": true
  },
  "nodes": [
     {
        "name": "start_embedding_server",
        "command": "source ~/leann-emb-venv2/bin/activate && nohup python -m ...",
        "deps": [],
        "timeout_sec": 60
     },
     {
        "name": "smoke_search",
        "command": "source ~/leann-emb-venv2/bin/activate && OMP_NUM_THREADS=1 timeout 30s leann search TEST 'ASI' --top-k 3",
        "deps": ["start_embedding_server"],
        "allow_fail": false
     }
  ]
}

Each node may specify either "command" (executed via bash -lc) or "args" (list
to run directly). Optional fields: env (dict), deps (list of node names),
timeout_sec, allow_fail, cwd, capture_output, artifact (path to save stdout),
description.

The coordinator runs nodes whose dependencies completed successfully, up to
max_parallel concurrent jobs.  If a node fails and allow_fail is false, the job
aborts.

Outputs:
  - Stdout prints a concise progress log.
  - A JSON summary (default: artifacts/swarm_runs/<timestamp>.json) containing
    overall status and per-node details (stdout/stderr truncated).

This is intentionally light-weight: it expects the caller to provide prompts or
commands that already enforce JSON-only outputs and token/timeout limits for
Qwen/Gemini/Groq wrappers.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class NodeConfig:
    name: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    deps: List[str] = dataclasses.field(default_factory=list)
    timeout: Optional[float] = None
    allow_fail: bool = False
    cwd: Optional[str] = None
    capture_output: bool = True
    artifact: Optional[str] = None
    description: Optional[str] = None

    def validate(self) -> None:
        if not self.command and not self.args:
            raise ValueError(f"Node '{self.name}' must define 'command' or 'args'.")
        if self.command and self.args:
            raise ValueError(f"Node '{self.name}' cannot define both 'command' and 'args'.")


async def _run_node(node: NodeConfig, job_timeout: Optional[float], max_output: int) -> Dict[str, Any]:
    start = time.time()
    env = os.environ.copy()
    env.update(node.env or {})

    stdout_data = b""
    stderr_data = b""
    status = "success"
    returncode: Optional[int] = None
    timeout_sec = node.timeout or job_timeout

    try:
        if node.args:
            proc = await asyncio.create_subprocess_exec(
                *node.args,
                stdout=PIPE if node.capture_output else None,
                stderr=PIPE if node.capture_output else None,
                cwd=node.cwd,
                env=env,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                node.command or "",
                stdout=PIPE if node.capture_output else None,
                stderr=PIPE if node.capture_output else None,
                cwd=node.cwd,
                env=env,
                executable="/bin/bash",
            )

        try:
            stdout_data, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            try:
                stdout_data, stderr_data = await proc.communicate()
            except Exception:  # pragma: no cover - best effort
                pass
            status = "timeout"
        returncode = proc.returncode
        if returncode not in (0, None):
            status = "failed" if status == "success" else status
    except Exception as exc:  # pragma: no cover - unexpected failure
        status = "error"
        stderr_data = (stderr_data or b"") + f"\nException: {exc}".encode()

    duration = time.time() - start

    def _clip(data: Optional[bytes]) -> str:
        if not data:
            return ""
        text = data.decode("utf-8", errors="replace")
        if max_output and len(text) > max_output:
            return text[:max_output] + f"\n... [truncated {len(text) - max_output} chars]"
        return text

    stdout_text = _clip(stdout_data)
    stderr_text = _clip(stderr_data)

    if node.artifact and stdout_text:
        Path(node.artifact).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        Path(node.artifact).write_text(stdout_text, encoding="utf-8")

    return {
        "name": node.name,
        "status": status,
        "returncode": returncode,
        "duration_sec": duration,
        "stdout": stdout_text if node.capture_output else "",
        "stderr": stderr_text if node.capture_output else "",
    }


def _load_job(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_nodes(job_data: Dict[str, Any]) -> List[NodeConfig]:
    defaults = job_data.get("defaults", {})
    timeout_default = defaults.get("timeout_sec")
    cwd_default = defaults.get("cwd")
    capture_default = defaults.get("capture_output", True)

    nodes: List[NodeConfig] = []
    for raw in job_data.get("nodes", []):
        node = NodeConfig(
            name=raw["name"],
            command=raw.get("command"),
            args=raw.get("args"),
            env=raw.get("env", {}),
            deps=raw.get("deps", []),
            timeout=raw.get("timeout_sec", timeout_default),
            allow_fail=raw.get("allow_fail", False),
            cwd=raw.get("cwd", cwd_default),
            capture_output=raw.get("capture_output", capture_default),
            artifact=raw.get("artifact"),
            description=raw.get("description"),
        )
        node.validate()
        nodes.append(node)

    # Validate dependencies
    names = {node.name for node in nodes}
    for node in nodes:
        missing = [dep for dep in node.deps if dep not in names]
        if missing:
            raise ValueError(f"Node '{node.name}' has undefined deps: {missing}")

    return nodes


async def _run_job(job_path: Path, job_data: Dict[str, Any], nodes: List[NodeConfig]) -> Dict[str, Any]:
    goal = job_data.get("goal", "")
    max_parallel = max(1, int(job_data.get("max_parallel", 1)))
    job_timeout = job_data.get("job_timeout_sec")
    max_output = int(job_data.get("max_output_chars", 8000))

    pending = set(node.name for node in nodes)
    scheduled: Dict[str, asyncio.Task] = {}
    results: Dict[str, Dict[str, Any]] = {}
    nodes_by_name = {node.name: node for node in nodes}

    start_ts = time.time()
    print(f"[CEO] Goal: {goal or job_path.name}")

    while len(results) < len(nodes):
        # Schedule ready nodes
        ready = [
            node for node in nodes
            if node.name in pending
            and node.name not in scheduled
            and all(dep in results and results[dep]["status"] == "success" for dep in node.deps)
        ]

        for node in ready:
            if len(scheduled) >= max_parallel:
                break
            print(f"[CEO] ⏳ Starting node '{node.name}'" + (f" – {node.description}" if node.description else ""))
            task = asyncio.create_task(_run_node(node, job_timeout, max_output))
            scheduled[node.name] = task
            pending.remove(node.name)

        if not scheduled:
            # Deadlock (likely due to dependency failure)
            missing = [name for name in pending if any(dep not in results for dep in nodes_by_name[name].deps)]
            raise RuntimeError(f"No runnable nodes (check dependencies and allow_fail flags). Pending: {missing}")

        done, _ = await asyncio.wait(scheduled.values(), return_when=asyncio.FIRST_COMPLETED)
        for name, task in list(scheduled.items()):
            if task in done:
                res = await task
                results[name] = res
                scheduled.pop(name)
                status = res["status"]
                print(f"[CEO] ✅ Node '{name}' completed with status={status} rc={res['returncode']}")
                if status not in {"success"} and not nodes_by_name[name].allow_fail:
                    print(f"[CEO] ❌ Node '{name}' failed and allow_fail is False → aborting job")
                    for other in scheduled.values():
                        other.cancel()
                    await asyncio.gather(*scheduled.values(), return_exceptions=True)
                    return {
                        "goal": goal,
                        "status": "failed",
                        "started_at": start_ts,
                        "finished_at": time.time(),
                        "nodes": list(results.values()),
                    }

    total = time.time() - start_ts
    status = "success" if all(res["status"] == "success" for res in results.values()) else "partial"
    return {
        "goal": goal,
        "status": status,
        "started_at": start_ts,
        "finished_at": time.time(),
        "duration_sec": total,
        "nodes": [results[name] for name in (node.name for node in nodes)],
    }


def _default_summary_path(job_path: Path) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path("artifacts/swarm_runs") / f"{job_path.stem}_{ts}.json"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CEO Swarm Coordinator")
    parser.add_argument("job", type=Path, help="Path to job JSON")
    parser.add_argument("--output", type=Path, help="Optional output summary JSON")
    args = parser.parse_args(argv)

    job_path = args.job.expanduser().resolve()
    if not job_path.exists():
        print(f"Job file not found: {job_path}", file=sys.stderr)
        return 2

    try:
        job_data = _load_job(job_path)
        nodes = _parse_nodes(job_data)
    except Exception as exc:
        print(f"Invalid job definition: {exc}", file=sys.stderr)
        return 2

    output_path = (args.output or _default_summary_path(job_path)).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        summary = asyncio.run(_run_job(job_path, job_data, nodes))
    except Exception as exc:  # pragma: no cover - coordinator failure
        print(f"[CEO] Fatal error: {exc}", file=sys.stderr)
        summary = {
            "goal": job_data.get("goal", job_path.name),
            "status": "error",
            "error": str(exc),
            "nodes": [],
        }

    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[CEO] Summary written to {output_path}")
    return 0 if summary.get("status") == "success" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

