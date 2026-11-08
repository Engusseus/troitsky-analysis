"""Check that a folder of evidence runs was made on one code: every ``crush.json`` under
it (single crushes and each montecarlo run's) with its ``code_sha``, F_u, governing part
and trustworthiness, and whether its ``code_sha`` is the code checked out now.

    uv run python validation/check_runs.py DIR [--sha SHA] [--json]

Exit code 1 when a run was made on other code (the realism review of round 25: two of
round 25's light-truss runs were made before the code was frozen, while the docs said
they were on it). Quote evidence only from a folder this passes on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def runs(root: Path) -> list[dict]:
    """One record per crush.json under ``root``."""
    out = []
    for f in sorted(root.rglob("crush.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out.append({"path": str(f.relative_to(root)), "error": "unreadable"})
            continue
        if d.get("kind") != "crush":
            continue
        q = d.get("quality") or {}
        out.append({"path": str(f.parent.relative_to(root)),
                    "code_sha": d.get("code_sha", ""),
                    "Fu_kgf": d.get("Fu_kgf"),
                    "governing": d.get("governing", ""),
                    "trustworthy": q.get("trustworthy"),
                    "fidelity": d.get("fidelity"), "seed": d.get("seed")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--sha", help="the code_sha the runs should carry (default: the code "
                                  "checked out now)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.sha:
        sha = a.sha
    else:
        from bridgesim.sim.study import _code_sha

        sha = _code_sha()
    rs = runs(Path(a.dir))
    bad = [r for r in rs if r.get("code_sha") != sha]
    if a.json:
        print(json.dumps({"code_sha": sha, "runs": rs, "other_code": bad}, indent=2))
    else:
        for r in rs:
            mark = "  " if r.get("code_sha") == sha else "!!"
            print(f"{mark} {r['path']}: {r.get('code_sha', '?')} F_u {r.get('Fu_kgf')} "
                  f"({r.get('fidelity')}, seed {r.get('seed')}, trustworthy "
                  f"{r.get('trustworthy')}) {r.get('governing', '')[:70]}")
        print(f"{len(rs)} runs, {len(bad)} on other code than {sha}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
