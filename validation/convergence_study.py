"""Mesh convergence of the crush simulation at a fixed jack speed.

    uv run python validation/convergence_study.py BRIDGE.SLDPRT [--out results.json]
        [--save runs/] [--scales 1.6,1,0.75] [--seeds 0,1,2]

Crushes the bridge with each element size (x1.6 = quick, x1 = standard, x0.75 = fine
by default) for each seed, all at 100 mm/s, two runs at a time, and prints F_u, the
deflection at F_u and the first failure of each, plus the mean and spread per mesh.
``--seeds -1`` is the perfect bridge (no imperfections, no scatter): with it, a change
of F_u between meshes is discretisation error alone, not a different random bridge.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

NAMES = {1.6: "quick", 1.0: "standard", 0.75: "fine"}


def _name(scale: float) -> str:
    return NAMES.get(scale, f"x{scale:g}")


def _one(args):
    cad, scale, seed, save = args
    import numba

    numba.set_num_threads(2)
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions

    t0 = time.time()
    out = Path(save) / f"{_name(scale)}_seed{seed}" if save else None
    full = seed in (0, -1) and bool(save)  # replay (and the collapse) for one seed
    rep, _ = study.crush(cad, seed=None if seed < 0 else seed, fidelity="standard",
                         mesh_scale=scale, replay=full, out_dir=out,
                         options=CrushOptions(speed_mm_s=100.0,
                                              dt_target=3e-6 * min(1.0, scale / 1.0) ** 1.1,
                                              after_failure_mm=15.0 if full else 2.0))
    first = next((f for f in rep.failures if f.kind != "joint_cracking"), None)
    return {"mesh": _name(scale), "element_scale": scale, "seed": seed,
            "Fu_kgf": rep.Fu_kgf, "deflection_at_Fu_mm": rep.deflection_at_Fu_mm,
            "first_failure": f"{first.kind}: {first.what}" if first else rep.failure_mode,
            "trustworthy": rep.quality.trustworthy,
            "added_mass": rep.quality.added_mass_fraction, "wall_s": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cad")
    ap.add_argument("--out", default="convergence.json")
    ap.add_argument("--scales", default="1.6,1,0.75", help="element sizes (x standard)")
    ap.add_argument("--seeds", default="0,1,2", help="-1: the perfect bridge")
    ap.add_argument("--save", default="", help="folder for each run's crush.json, curves, "
                    "events (and the replay of seed 0 or -1)")
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = [(a.cad, sc, sd, a.save) for sc in scales for sd in seeds]
    res = []
    with cf.ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(_one, jobs):
            res.append(r)
            print(json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("\n| mesh | element size | F_u by seed (kgf) | mean | spread |")
    print("|---|---|---|---|---|")
    for sc in scales:
        F = [r["Fu_kgf"] for r in res if r["element_scale"] == sc]
        print(f"| {_name(sc)} | x{sc:g} | {', '.join(f'{f:.0f}' for f in F)} | "
              f"{statistics.mean(F):.0f} | {max(F) - min(F):.0f} |")


if __name__ == "__main__":
    main()
