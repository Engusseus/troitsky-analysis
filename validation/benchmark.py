"""The fixed benchmark every release reports, so its drift is seen (the realism review of
round 26: the model's answers moved from release to release by about as much as its own
stated model error, and the model form was not settled).

    uv run python validation/benchmark.py --out DIR [--cad examples/cad/Bridge_v10.SLDPRT]
        [--only NAME]... [--threads N]
    uv run python validation/benchmark.py --history

Each case is crushed once (seed, fidelity fixed, stopped 2 mm after the failure call);
the table compares F_u with the releases before (``benchmark_history.json``) and names
what governed. Bridge_v10 needs its part file (not in the repository); without it those
cases are skipped. The policy since round 27: a law changes only with data (a coupon, a
node test, or a published test it brings closer without moving the others away), and a
release that moves a case more than 10 % says why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: (name, builder, fidelity, seed): builder returns (cad path, overrides path or None)
CASES = [
    ("bridge_v10_s0_standard", "v10", "standard", 0),
    ("bridge_v10_s1_standard", "v10", "standard", 1),
    ("truss6_fine", "truss6", "fine", 0),
    ("truss10_standard", "truss10", "standard", 0),
    ("published_A_fine", "A", "fine", 0),
    ("published_B_fine", "B", "fine", 0),
    ("published_C_fine", "C", "fine", 0),
    ("published_D_fine", "D", "fine", 0),
]


def truss10(t: float = 10.0, plies: int = 5, span: float = 1000.0, height: float = 140.0,
            panels: int = 4):
    """``layered_truss`` with 10 mm (5-ply) members: at the light-bridge threshold, not
    flagged as light (round 26's check of small glued joints on plates)."""
    from bridgesim.cad.synthetic import _strip, box, upright

    out = []
    w = 52.0
    x0, x1 = -span / 2 - 40, span / 2 + 40
    y0 = 60.0
    yt = y0 + height
    xs = np.linspace(-span / 2 + 100, span / 2 - 100, panels + 1)
    out.append(box("Support L (block)", (x0, 0, -w - 3 * t - 10), (x0 + 60, y0, w + 3 * t + 10)))
    out.append(box("Support R (block)", (x1 - 60, 0, -w - 3 * t - 10), (x1, y0, w + 3 * t + 10)))
    lay = f"{plies}-ply UD"
    for side, s in (("Near", 1.0), ("Far", -1.0)):
        zv = sorted((w * s, (w + t) * s))
        zo = sorted(((w + t) * s, (w + 2 * t) * s))
        zd = sorted(((w + 2 * t) * s, (w + 3 * t) * s))
        out.append(box(f"Bottom Chord ({lay}) - {side}", (x0, y0, zo[0]), (x1, y0 + 20, zo[1])))
        out.append(box(f"Top Chord ({lay}) - {side}", (xs[0] - 25, yt - 20, zo[0]),
                       (xs[-1] + 25, yt, zo[1])))
        for k, xv in enumerate(xs):
            out.append(box(f"Vertical {k} ({lay}) - {side}", (xv - 5, y0, zv[0]),
                           (xv + 5, yt, zv[1])))
        ends = [((x0 + 30, y0 + 10), (xs[0], yt - 10)), ((x1 - 30, y0 + 10), (xs[-1], yt - 10))]
        mid = len(xs) // 2
        inner = [((xs[k], yt - 10), (xs[k + 1], y0 + 10)) for k in range(mid)] + \
            [((xs[k + 1], yt - 10), (xs[k], y0 + 10)) for k in range(mid, len(xs) - 1)]
        for k, (p, q) in enumerate(ends + inner):
            p, q = np.asarray(p, float), np.asarray(q, float)
            dx = np.array([np.sign(q[0] - p[0]) * 10.0, 0.0])
            out.append(upright(f"Diagonal {k} ({lay}) - {side}", _strip(p + dx, q - dx, 10.0),
                               zd[0], zd[1]))
    xf = 0.5 * (xs[:-1] + xs[1:])
    for k, xm in enumerate(xf):
        out.append(box(f"Floor Beam {k} ({lay})", (xm - 5, y0 + 20, -w - 2 * t),
                       (xm + 5, y0 + 20 + t, w + 2 * t)))
    mid = len(xf) // 2
    out.append(box("Deck (2-ply 0-90)", (xf[mid - 1] - 10, y0 + 20 + t, -w),
                   (xf[mid] + 10, y0 + 24 + t, w)))
    mid = len(xs) // 2
    for k in (0, mid, len(xs) - 1):
        out.append(box(f"Top Strut {k} ({lay})", (xs[k] - 5, yt, -w - 2 * t),
                       (xs[k] + 5, yt + t, w + 2 * t)))
    return out


def _inputs(kind: str, out: Path, cad: Path | None):
    from bridgesim.cad.synthetic import layered_truss, write_stl

    if kind == "v10":
        if cad is None or not cad.exists():
            return None
        return cad, HERE / "bridge_v10.overrides.yaml"
    if kind in ("truss6", "truss10"):
        stl = out / f"{kind}.stl"
        write_stl(layered_truss() if kind == "truss6" else truss10(), stl, names=True)
        return stl, None
    import published_tests as pt
    import yaml

    bodies, ov, _ref = pt.CASES[kind]()
    stl, ovf = out / f"{kind}.stl", out / f"{kind}.overrides.yaml"
    pt.write_stl(bodies, stl, names=True)
    ovf.write_text(yaml.safe_dump(ov, sort_keys=False), encoding="utf-8")
    return stl, ovf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out")
    ap.add_argument("--cad", default=str(HERE.parent / "examples/cad/Bridge_v10.SLDPRT"))
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--history", action="store_true", help="print the history and exit")
    a = ap.parse_args()
    hist = json.loads((HERE / "benchmark_history.json").read_text(encoding="utf-8"))
    if a.history:
        print(json.dumps(hist, indent=1))
        return 0
    if not a.out:
        ap.error("--out is required")
    if a.threads:
        import numba

        numba.set_num_threads(max(1, min(a.threads, numba.config.NUMBA_NUM_THREADS)))
    from bridgesim.sim import study

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, kind, fid, seed in CASES:
        if a.only and not any(o in name for o in a.only):
            continue
        inp = _inputs(kind, out, Path(a.cad))
        if inp is None:
            print(f"{name}: skipped (no part file)", flush=True)
            continue
        cad, ovf = inp
        rep, _ = study.crush(cad, overrides=str(ovf) if ovf else None, seed=seed,
                             fidelity=fid, out_dir=str(out / name), stop_at_failure=True,
                             replay=False)
        last = [v for v in hist.get(name, {}).values() if v is not None]
        prev = last[-1] if last else None
        rows[name] = {"Fu_kgf": rep.Fu_kgf, "governing": rep.governing,
                      "trustworthy": rep.quality.trustworthy,
                      "previous_kgf": prev,
                      "drift": round(rep.Fu_kgf / prev - 1, 3) if prev else None}
        print(json.dumps({name: rows[name]}), flush=True)
    (out / "benchmark.json").write_text(json.dumps(
        {"code_sha": study._code_sha(), "cases": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
