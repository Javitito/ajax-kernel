"""Agent registry discovery and handshake utilities.

The registry probes each configured CLI agent with three light-weight checks:
- ``--version`` for identity and model metadata.
- ``--capabilities`` for the advertised toolset / policies.
- ``--bench`` for a synthetic micro-benchmark (latency, quality, cost estimates).

Every probe expects JSON on stdout. Non-JSON responses are captured verbatim under
``raw`` so we keep full traceability.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


JsonDict = Dict[str, Any]


@dataclass
class RegistrySpec:
    """Agent declaration used for handshake."""

    name: str
    role: str
    command: List[str]

    @property
    def command_str(self) -> str:
        return " ".join(self.command)


@dataclass
class ProbeResult:
    """Detailed outcome of a single probe (version/capabilities/bench)."""

    ok: bool
    payload: Optional[JsonDict] = None
    raw: Optional[str] = None
    stderr: Optional[str] = None
    latency_ms: int = 0

    def to_dict(self) -> JsonDict:
        data: JsonDict = {"ok": self.ok, "latency_ms": self.latency_ms}
        if self.payload is not None:
            data["payload"] = self.payload
        if self.raw:
            data["raw"] = self.raw
        if self.stderr:
            data["stderr"] = self.stderr
        return data


@dataclass
class RegistryEntry:
    """Aggregated handshake data for a single CLI."""

    name: str
    role: str
    command: str
    version: ProbeResult
    capabilities: ProbeResult
    bench: ProbeResult
    error: Optional[str] = None

    def to_dict(self) -> JsonDict:
        data: JsonDict = {
            "name": self.name,
            "role": self.role,
            "command": self.command,
            "version": self.version.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "bench": self.bench.to_dict(),
        }
        if self.error:
            data["error"] = self.error
        return data


@dataclass
class RegistrySnapshot:
    generated_at: float
    entries: List[RegistryEntry] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "generated_at": self.generated_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _split_command(command: str) -> List[str]:
    return shlex.split(command.strip()) if command.strip() else []


def _extra_args(flag: str) -> List[str]:
    if flag.startswith("--"):
        return [flag]
    return ["--" + flag]


def probe_command(command: List[str], flag: str, timeout: float = 10.0) -> ProbeResult:
    """Run ``command + flag`` capturing JSON or raw outputs."""

    if not command:
        return ProbeResult(ok=False, raw="<empty command>")

    full_cmd = command + _extra_args(flag)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            full_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return ProbeResult(ok=False, raw=str(exc))
    except subprocess.TimeoutExpired:
        return ProbeResult(ok=False, raw="timeout", latency_ms=int((time.perf_counter() - start) * 1000))

    latency_ms = int((time.perf_counter() - start) * 1000)
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip() or None

    if completed.returncode != 0 and not stdout:
        return ProbeResult(ok=False, raw=f"exit {completed.returncode}", stderr=stderr, latency_ms=latency_ms)

    payload: Optional[JsonDict] = None
    raw: Optional[str] = stdout or None
    try:
        if stdout:
            payload = json.loads(stdout)
            raw = None
    except json.JSONDecodeError:
        pass

    return ProbeResult(ok=completed.returncode == 0, payload=payload, raw=raw, stderr=stderr, latency_ms=latency_ms)


def collect_registry(specs: Iterable[RegistrySpec]) -> RegistrySnapshot:
    entries: List[RegistryEntry] = []
    for spec in specs:
        version = probe_command(spec.command, "--version")
        capabilities = probe_command(spec.command, "--capabilities")
        bench = probe_command(spec.command, "--bench", timeout=15.0)

        error = None
        if not version.ok and not capabilities.ok:
            error = "version/capabilities probes failed"

        entries.append(
            RegistryEntry(
                name=spec.name,
                role=spec.role,
                command=spec.command_str,
                version=version,
                capabilities=capabilities,
                bench=bench,
                error=error,
            )
        )

    return RegistrySnapshot(generated_at=time.time(), entries=entries)


def write_registry(snapshot: RegistrySnapshot, path: Path) -> None:
    path.write_text(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def discover_specs_from_env() -> List[RegistrySpec]:
    """Build specs using the same environment variables as the broker."""

    specs: List[RegistrySpec] = []
    planner_cmd = os.getenv("AGENCY_PLANNER_CMD")
    executor_cmd = os.getenv("AGENCY_EXECUTOR_CMD")
    verifier_cmd = os.getenv("AGENCY_VERIFIER_CMD")
    council_cmds = os.getenv("AGENCY_COUNCIL_CMDS")

    if planner_cmd:
        specs.append(RegistrySpec(name="planner", role="planner", command=_split_command(planner_cmd)))
    if executor_cmd:
        specs.append(RegistrySpec(name="executor", role="executor", command=_split_command(executor_cmd)))
    if verifier_cmd:
        specs.append(RegistrySpec(name="verifier", role="verifier", command=_split_command(verifier_cmd)))

    if council_cmds:
        for idx, chunk in enumerate(council_cmds.split(";")):
            chunk = chunk.strip()
            if not chunk:
                continue
            specs.append(RegistrySpec(name=f"council_{idx+1}", role="council", command=_split_command(chunk)))

    return specs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Discover CLI agents and record their capabilities")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("registry.json"),
        help="Destination file for the registry snapshot",
    )
    args = parser.parse_args()

    specs = discover_specs_from_env()
    if not specs:
        raise SystemExit("No agents configured via environment variables")

    snapshot = collect_registry(specs)
    write_registry(snapshot, args.output)


if __name__ == "__main__":
    main()
