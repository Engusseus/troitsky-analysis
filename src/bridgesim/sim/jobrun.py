"""Run a bridgesim command as a background job that outlives the MCP server.

``python -m bridgesim.sim.jobrun OUT_DIR -- CMD...`` runs CMD with its stdout in
``OUT_DIR/result.json`` and its stderr (progress, errors) in ``OUT_DIR/progress.log``, and
writes ``OUT_DIR/status.json`` (exit code, times) when CMD ends, whatever happens. Any MCP
server, a new one included, reads the job's state from these files alone.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    if "--" not in argv:
        print("usage: python -m bridgesim.sim.jobrun OUT_DIR -- CMD...", file=sys.stderr)
        return 2
    i = argv.index("--")
    out = Path(argv[0])
    cmd = argv[i + 1:]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    code = 3
    try:
        with (out / "result.json").open("w") as so, (out / "progress.log").open("w") as se:
            code = subprocess.call(cmd, stdout=so, stderr=se)
    finally:
        (out / "status.json").write_text(json.dumps(
            {"exit_code": code, "started": t0, "finished": time.time()}))
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
