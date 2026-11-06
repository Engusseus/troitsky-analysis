"""Run a bridgesim command as a background job that outlives the MCP server.

``python -m bridgesim.sim.jobrun OUT_DIR -- CMD...`` runs CMD with its stdout in
``OUT_DIR/result.json`` and its stderr (progress, errors) in ``OUT_DIR/progress.log``, and
writes ``OUT_DIR/status.json`` (exit code, times) when CMD ends, whatever happens. Any MCP
server, a new one included, reads the job's state from these files alone.

Jobs share the machine: at most ``slots`` run at once (half the CPU cores, at least one;
``BRIDGESIM_JOB_SLOTS`` overrides it), each with its share of the cores as numba threads.
A job that finds every slot taken waits, marked by ``OUT_DIR/queued``, until one frees:
two crushes started together run side by side at full speed instead of fighting over
every core.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def slots() -> int:
    env = os.environ.get("BRIDGESIM_JOB_SLOTS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, (os.cpu_count() or 1) // 2)


def _slot_dir() -> Path:
    d = Path(os.environ.get("BRIDGESIM_JOB_SLOT_DIR")
             or Path(tempfile.gettempdir()) / "bridgesim-job-slots")
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextlib.contextmanager
def _slot(out: Path):
    """Hold one of the machine's job slots (an exclusive lock on a slot file) while the
    job runs; wait, marked queued, while all are taken."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows: no queue, run at once
        yield
        return
    n = slots()
    marker = out / "queued"
    while True:
        for k in range(n):
            fh = (_slot_dir() / f"slot_{k}.lock").open("w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue
            marker.unlink(missing_ok=True)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        if not marker.exists():
            marker.write_text(json.dumps({"since": time.time(), "slots": n}))
        time.sleep(2.0)


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
        with _slot(out):
            env = dict(os.environ)
            env.setdefault("NUMBA_NUM_THREADS",
                           str(max(1, (os.cpu_count() or 1) // slots())))
            t0 = time.time()
            with (out / "result.json").open("w") as so, \
                    (out / "progress.log").open("w") as se:
                code = subprocess.call(cmd, stdout=so, stderr=se, env=env)
    finally:
        (out / "status.json").write_text(json.dumps(
            {"exit_code": code, "started": t0, "finished": time.time()}))
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
