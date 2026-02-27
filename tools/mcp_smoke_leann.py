from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "agency" / "mcp" / "leann_server.py"


def main() -> int:
    request = {
        "jsonrpc": "2.0",
        "id": "smoke-1",
        "method": "leann.search",
        "params": {"query": "providers audit", "k": 3, "filters": {}},
    }
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        cwd=str(ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    try:
        ready_line = proc.stdout.readline().strip()
        proc.stdin.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
        proc.stdin.flush()
        response_line = proc.stdout.readline().strip()
    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    response = json.loads(response_line or "{}")
    ok = (
        isinstance(response, dict)
        and response.get("jsonrpc") == "2.0"
        and response.get("id") == "smoke-1"
        and isinstance(response.get("result"), dict)
        and "ok" in response.get("result", {})
    )
    print("request:", json.dumps(request, ensure_ascii=False))
    print("ready:", ready_line)
    print("response:", json.dumps(response, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

