from __future__ import annotations

import json
import socket
import time
import urllib.request
from pathlib import Path

from agency.lab_dummy_driver import ensure_dummy_driver, is_dummy_driver_simulated, stop_dummy_driver


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url: str, timeout: float = 2.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_dummy_driver_serves_health_and_version(tmp_path: Path) -> None:
    port = _free_port()
    try:
        out = ensure_dummy_driver(tmp_path, host="127.0.0.1", port=port)
        assert out.get("ok") is True

        health = None
        version = None
        deadline = time.time() + 4.0
        while time.time() < deadline:
            try:
                health = _http_json(f"http://127.0.0.1:{port}/health")
                version = _http_json(f"http://127.0.0.1:{port}/version")
                break
            except Exception:
                time.sleep(0.1)
        assert isinstance(health, dict)
        assert health.get("ok") is True
        assert health.get("simulated") is True
        assert isinstance(version, dict)
        assert version.get("ok") is True
        assert version.get("simulated") is True
        assert is_dummy_driver_simulated(tmp_path, port=port, host="127.0.0.1") is True
    finally:
        stop_dummy_driver(tmp_path, port=port)
