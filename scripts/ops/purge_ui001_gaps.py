#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agency.ui001_purge import purge_ui001_gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Move UI-001 gap artifacts to attic with index/report.")
    parser.add_argument("--root", default=str(ROOT), help="AJAX_HOME root path.")
    parser.add_argument("--timestamp", default=None, help="Optional UTC compact timestamp override.")
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Disable SHA256 computation for moved files.",
    )
    args = parser.parse_args(argv)

    summary = purge_ui001_gaps(
        Path(args.root),
        timestamp=args.timestamp,
        with_hash=not bool(args.no_hash),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
