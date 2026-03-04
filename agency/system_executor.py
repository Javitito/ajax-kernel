from __future__ import annotations

import subprocess
from typing import Any


class SystemExecutor:
    """
    Thin wrapper around subprocess primitives used by AjaxCore.
    Keeps execution calls centralized for testability and DSE wiring.
    """

    def run(self, *args: Any, **kwargs: Any):
        return subprocess.run(*args, **kwargs)

    def popen(self, *args: Any, **kwargs: Any):
        return subprocess.Popen(*args, **kwargs)

    def check_output(self, *args: Any, **kwargs: Any):
        return subprocess.check_output(*args, **kwargs)
