"""Mesh convergence of the crush simulation at a fixed jack speed, over several seeds.

    uv run python validation/convergence_study.py BRIDGE.SLDPRT [--out results.json]
        [--save runs/]

Runs the crushing test with the quick, standard and fine meshes (element size x1.6, x1,
x0.75) all at 100 mm/s, for seeds 0, 1 and 2, two runs at a time, and prints F_u, the
deflection at F_u and the first major failure of each, plus the mean and spread per mesh.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

LEVELS = {"quick": 1.6, "standard": 1.0, "fine": 0.75}


def _one(args):
    cad, level, seed, save = args
    import numba

    numba.set_num_threads(2)
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions

    t0 = time.time()
    out = Path(save) / f"{level}_seed{seed}" if save else None
    rep, _ = study.crush(cad, seed=seed, fidelity=level, replay=seed == 0 and bool(save),
                         out_dir=out,
                         options=CrushOptions(speed_mm_s=100.0,
                                              dt_target=2.2e-6 if level == "fine" else 3e-6))
    first = next((f for f in rep.failures if f.kind != "joint_cracking"), None)
    return {"mesh": level, "element_scale": LEVELS[level], "seed": seed,
            "Fu_kgf": rep.Fu_kgf, "deflection_at_Fu_mm": rep.deflection_at_Fu_mm,
            "first_failure": f"{first.kind}: {first.what}" if first else rep.failure_mode,
            "trustworthy": rep.quality.trustworthy,
            "added_mass": rep.quality.added_mass_fraction, "wall_s": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cad")
    ap.add_argument("--out", default="convergence.json")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--save", default="", help="folder for each run's crush.json, curves, "
                    "events (and the replay of seed 0)")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    jobs = [(a.cad, lv, sd, a.save) for lv in LEVELS for sd in seeds]
    res = []
    with cf.ProcessPoolExecutor(2, mp_context=mp.get_context("spawn")) as ex:
        for r in ex.map(_one, jobs):
            res.append(r)
            print(json.dumps(r), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("\n| mesh | element size | F_u by seed (kgf) | mean | spread |")
    print("|---|---|---|---|---|")
    for lv, sc in LEVELS.items():
        F = [r["Fu_kgf"] for r in res if r["mesh"] == lv]
        print(f"| {lv} | x{sc} | {', '.join(f'{f:.0f}' for f in F)} | "
              f"{statistics.mean(F):.0f} | {max(F) - min(F):.0f} |")


if __name__ == "__main__":
    main()
