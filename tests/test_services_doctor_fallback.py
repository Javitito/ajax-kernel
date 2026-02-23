from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agency import ops_ports_sessions


class _UnauthorizedHealthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            payload = json.dumps({"ok": False, "error": "unauthorized"}).encode("utf-8")
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:  # noqa: A003
        return


def test_probe_local_health_treats_401_as_reachable() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _UnauthorizedHealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        assert ops_ports_sessions._probe_local_health(port, host="127.0.0.1") is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_run_services_doctor_fallback_local_probe(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path

    class _Proc:
        returncode = 1
        stdout = "simulated_failure"
        stderr = "simulated_failure"

    monkeypatch.setattr(ops_ports_sessions, "_run", lambda *args, **kwargs: _Proc())
    monkeypatch.setattr(
        ops_ports_sessions,
        "_probe_local_port_listening",
        lambda port, host="127.0.0.1", timeout_s=0.4: int(port) == 5012,
    )
    monkeypatch.setattr(
        ops_ports_sessions,
        "_probe_local_health",
        lambda port, host="127.0.0.1", timeout_s=1.5: int(port) == 5012,
    )

    out = ops_ports_sessions.run_services_doctor(root, out_dir=root / "artifacts" / "doctor" / "t")

    assert out["health"][5012] is True
    assert out["health"][5010] is False
    assert isinstance(out["ports"].get(5012), dict)
    assert out["ports"][5012].get("LocalProbe") is True
    assert out["ok"] is False
