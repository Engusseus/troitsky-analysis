"""Published crush tests of craft-stick trusses, rebuilt as bridgesim models.

There is no physical test of a team's bridge yet. These three bridges were built and
loaded to failure by others and published with their geometry, stick, load and failure:

A  Lamberti, Perius, Carvalho & Silva (2025), "Ponte trelicada de palitos de picole",
   JESTA 3 e86766, doi:10.5902/2965694X86766. Warren, 6 panels of 100 mm, 3-ply pine
   members (120 x 8 x 2 mm sticks, cyanoacrylate), dead weights hung from the centre
   bottom node in 20 kgf steps: held 140, failed at 160 kgf, in a top-chord panel and a
   compression diagonal next to the centre, after a lateral drift from 100 kgf.
B  Silva, Dantas, Morais, Sousa & Timoteo Junior (2026), "Estimativa da carga de ruptura
   em pontes com palitos de picole", Rev. Ensino Eng. 45:389, doi:10.37702/
   REE2236-0158.v45p389-402.2026. Warren, 4 panels of 260 mm, 230 mm deep, pine sticks
   1.94 x 8.58 mm, epoxy-putty nodes, load hung from the centre bottom nodes: 710 N, a
   5-ply compression diagonal buckling out of plane where its outer plies were butted.
C  Moraes, Freitas, Moraes, Freire Jr, Silva & Goncalves Filho (2019), "Procedimento de
   analise experimental e numerico de uma ponte executada com palitos de picole",
   Atena Editora, doi:10.22533/at.ed.30619250631. Howe, 6 panels of 80 mm, 80 mm deep,
   3-ply members (120 x 7.8 x 2 mm sticks, wood glue), a rebar across the top chords at
   mid-span pressed by a test machine: 0.49 kN peak at 3.8 mm (a measured curve), broken
   at the bottom chord and the foot of a vertical.

Geometry from the papers' drawings and photos (tags in the comments: S stated, D from a
drawing, P from a photo, A assumed), as extracted for this repository (round 21). The
loads were hung from a node or pressed on a bar, not on a 200 x 90 mm plate: a rigid
hanger (an I-shaped block) rests on the loaded node and stands clear above the bridge,
and the plate pushes on its top. The ends rest on support blocks (bearing, no glue).
Material: the papers' stick sizes and the pine values they quote (E, f_c); everything
else the shipped placeholders (craft-stick birch, PVA), so these are blind predictions.

    uv run python validation/published_tests.py --case A --out DIR [--fidelity standard]
        [--seeds 0,1,2] [--write-only]

writes DIR/<case>.stl and DIR/<case>.overrides.yaml and crushes it (study.crush), then
prints F_u against the test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bridgesim.cad import CadBody
from bridgesim.cad.mesh import tessellate
from bridgesim.cad.parasolid import Body, Face
from bridgesim.cad.synthetic import box, upright, write_stl


def hanger(name: str, outline_zy, x0: float, x1: float) -> CadBody:
    """An extrusion along x of the polygon ``outline_zy`` ((z, y) points) from x0 to x1:
    the load hanger's I-shaped profile, which spans across the two trusses."""
    up = upright(name, [(-z, y) for z, y in outline_zy], x0, x1)
    R = np.array([[0.0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # (x, y, z) -> (z, y, -x)
    faces = [Face(f.kind, [lp @ R.T for lp in f.loops], normal=R @ f.normal,
                  origin=R @ f.origin) for f in up.faces]
    body = Body(index=0, name=name, faces=faces)
    return CadBody(name, tessellate(body), faces, 0, named_in_cad=True)


def _i_profile(foot: float, stem: float, cap: float, y0: float, y1: float, y2: float,
               y3: float):
    """(z, y) outline of an I: a foot +-foot wide from y0 to y1, a stem +-stem to y2, a cap
    +-cap to y3."""
    return [(-foot, y0), (foot, y0), (foot, y1), (stem, y1), (stem, y2), (cap, y2),
            (cap, y3), (-cap, y3), (-cap, y2), (-stem, y2), (-stem, y1), (-foot, y1)]


COMMON_OVERRIDES = {
    # the supports bear only; the hanger is pinned to the node it loads (the wire or bar
    # passes through it), so it cannot topple; the press's 4.2 kg plate is not there
    "glue": {"Support* | *": 0},
    "params": {"crusher.plate_mass_kg": 0.05},
    "bodies": {"Support*": {"kind": "block"}, "Load hanger*": {"kind": "block"}},
    "acknowledge": ["rigid_blocks", "rigid_under_plate:*", "material:density",
                    "material:measure", "end_grain_joints", "weak_joints", "rule:*",
                    "plate_fit", "plate_edges"],
}


def _merge(*docs: dict) -> dict:
    out: dict = {}
    for d in docs:
        for k, v in d.items():
            if isinstance(v, dict):
                out.setdefault(k, {}).update(v)
            elif isinstance(v, list):
                out[k] = list(dict.fromkeys(out.get(k, []) + v))
            else:
                out[k] = v
    return out


def lamberti_2025():
    """A: Warren, 6 x 100 mm bottom panels, 100 mm between chord centre lines (S, D), 80
    mm wide (S), every member 3 sticks face to face, 8 mm deep in the truss plane, 6 mm
    across it (S, P); diagonals on the chords' outer faces, mitred where they meet (P,
    A); cross sticks at the top nodes and at four bottom nodes (S: top and bottom; their
    places A)."""
    bodies = []
    for side, s in (("N", 1.0), ("F", -1.0)):
        def zr(a, b, s=s):
            return (min(s * a, s * b), max(s * a, s * b))
        z0, z1 = zr(28, 34)
        bodies.append(box(f"BotChord {side} (3-ply UD)", (0, 0, z0), (600, 8, z1)))
        bodies.append(box(f"TopChord {side} (3-ply UD)", (45.5, 100, z0), (554.5, 108, z1)))
        d0, d1 = zr(34, 40)
        rise = [(0, 0), (0, 12.9), (47.5, 108), (50, 108), (50, 95.1), (2.5, 0)]
        fall = [(50, 108), (52.5, 108), (100, 12.9), (100, 0), (97.5, 0), (50, 95.1)]
        for k in range(6):
            dx = 100.0 * k
            bodies.append(upright(f"Diag {side} {2 * k + 1} (3-ply UD)",
                                  [(x + dx, y) for x, y in rise], d0, d1))
            bodies.append(upright(f"Diag {side} {2 * k + 2} (3-ply UD)",
                                  [(x + dx, y) for x, y in fall], d0, d1))
    for xn in (50, 150, 250, 350, 450, 550):
        bodies.append(box(f"TopX {xn} (1-ply UD)", (xn - 4, 108, -40), (xn + 4, 110, 40)))
    for xn in (0, 200, 400, 600):
        x0, x1 = (0, 8) if xn == 0 else (592, 600) if xn == 600 else (xn - 4, xn + 4)
        bodies.append(box(f"BotX {xn} (1-ply UD)", (x0, 8, -34), (x1, 10, 34)))
    bodies += [box("Support L", (-20, -10, -60), (50, 0, 60)),
               box("Support R", (550, -10, -60), (620, 0, 60))]
    # the wire through the centre bottom node (x 300): the hanger's foot glued 40 mm along
    # the two bottom chords (so the plate on its cap cannot tip it: dead weights pull
    # straight down), its stem between the trusses, its cap 20 mm above the cross sticks
    bodies.append(hanger("Load hanger", _i_profile(33, 10, 45, 8, 14, 130, 140), 280, 320))
    ov = _merge(COMMON_OVERRIDES, {"params": {
        "material.E_MPa": 8846.0, "material.f_c_MPa": 31.5,  # pine, IPT values (S)
        "material.density_kg_m3": 500.0}})
    ref = {"case": "A", "source": "Lamberti et al. 2025, JESTA 3 e86766",
           "Fu_kgf": 160.0, "Fu_range_kgf": [140.0, 160.0],
           "failure": "top chord panel next to the centre and a compression diagonal, "
                      "after a lateral drift from 100 kgf",
           "parts": ["TopChord", "Diag"], "mass_g": 180.0,
           "load": "dead weights hung from the centre bottom node, 20 kgf steps"}
    return bodies, ov, ref


def silva_2026(width: float = 150.0):
    """B: Warren, 4 x 260 mm bottom panels, 230 mm between chord centre lines (S), sticks
    1.94 x 8.58 mm (S): top chord 8 plies, bottom chord and diagonals 5 (S), 8.58 mm deep
    in the truss plane; epoxy-putty lumps at the nodes (P), rigid, glued to the wood;
    the truss spacing not stated (``width``, A: 150 mm); cross sticks at the top nodes
    (A)."""
    bodies = []
    t = 1.94
    for side, zc in (("N", width / 2), ("F", -width / 2)):
        bodies.append(box(f"BotChord {side} (5-ply UD)", (0, 0, zc - 2.5 * t),
                          (1040, 8.58, zc + 2.5 * t)))
        bodies.append(box(f"TopChord {side} (8-ply UD)", (108, 230.0, zc - 4 * t),
                          (932, 238.58, zc + 4 * t)))
        for xn in (260, 520, 780):
            bodies.append(box(f"Putty {side} B{xn}", (xn - 22, 8.58, zc - 10),
                              (xn + 22, 30.6, zc + 10)))
        bodies.append(box(f"Putty {side} B0", (-25, 8.58, zc - 10), (30, 40, zc + 10)))
        bodies.append(box(f"Putty {side} B1040", (1010, 8.58, zc - 10),
                          (1065, 40, zc + 10)))
        for xn in (130, 390, 650, 910):
            bodies.append(box(f"Putty {side} T{xn}", (xn - 22, 208, zc - 13),
                              (xn + 22, 230.0, zc + 13)))
        diags = [[(15.3, 40), (110.2, 208), (120.1, 208), (25.2, 40)],
                 [(149.8, 208), (250.1, 30.6), (240.2, 30.6), (139.9, 208)],
                 [(269.9, 30.6), (370.2, 208), (380.1, 208), (279.8, 30.6)],
                 [(409.8, 208), (510.1, 30.6), (500.2, 30.6), (399.9, 208)],
                 [(529.9, 30.6), (630.2, 208), (640.1, 208), (539.8, 30.6)],
                 [(669.8, 208), (770.1, 30.6), (760.2, 30.6), (659.9, 208)],
                 [(789.9, 30.6), (890.2, 208), (900.1, 208), (799.8, 30.6)],
                 [(919.8, 208), (929.6, 208), (1024.7, 40), (1014.9, 40)]]
        for k, poly in enumerate(diags, 1):
            bodies.append(upright(f"Diag {side} D{k} (5-ply UD)", poly, zc - 2.5 * t,
                                  zc + 2.5 * t))
    for xn in (130, 390, 650, 910):
        bodies.append(box(f"TopX {xn} (1-ply UD)", (xn - 4.3, 238.58, -width / 2 - 7.76),
                          (xn + 4.3, 240.52, width / 2 + 7.76)))
    bodies += [box("Support L", (-45, -10, -width), (30, 0, width)),
               box("Support R", (1010, -10, -width), (1085, 0, width))]
    # the metal bar through the centre bottom nodes (x 520): on the putty lumps there
    w2 = width / 2 + 10
    bodies.append(hanger("Load hanger", _i_profile(w2, 10, 45, 30.6, 36.6, 260, 270),
                         500, 540))
    ov = _merge(COMMON_OVERRIDES, {
        "bodies": {"Putty*": {"kind": "block"}},
        # the putty wraps round each diagonal's end (P): the end face glued to the lump
        # stands for that, x5 (its four sides, ~4 x 9 x 10 mm against the 85 mm^2 end)
        "glue": {"Diag* | Putty*": 5.0},
        "params": {"material.E_MPa": 13000.0, "material.f_c_MPa": 31.5,
                   "material.density_kg_m3": 500.0}})
    ref = {"case": "B", "source": "Silva et al. 2026, Rev. Ensino Eng. 45:389",
           "Fu_kgf": 710.0 / 9.80665, "failure": "compression diagonal D3 buckling out "
           "of plane where its outer plies were butted at mid-length",
           "parts": ["Diag"], "load": "bar through the centre bottom nodes",
           "width_mm": width}
    return bodies, ov, ref


def moraes_2019(width: float = 85.0):
    """C: Howe, 6 x 80 mm panels, 80 mm between chord centre lines, verticals at every
    station, 3-ply members 7.8 x 6 mm (S), 7.8 mm deep in the truss plane (A), web
    members cut to fit between the chords, glued on their end grain (A); cross members at
    every station top and bottom (D); 85 mm wide (P)."""
    bodies = []
    zi, zo = width / 2 - 6.0, width / 2
    for side, s in (("N", 1.0), ("F", -1.0)):
        z0, z1 = (zi, zo) if s > 0 else (-zo, -zi)
        bodies.append(box(f"BotChord {side} (3-ply UD)", (-3.9, 0, z0), (483.9, 7.8, z1)))
        bodies.append(box(f"TopChord {side} (3-ply UD)", (-3.9, 80, z0), (483.9, 87.8, z1)))
        for xi in range(0, 481, 80):
            bodies.append(box(f"Vertical {side} {xi} (3-ply UD)", (xi - 3.9, 7.8, z0),
                              (xi + 3.9, 80, z1)))
        up_ = [(3.9, 7.8), (3.9, 13.3), (70.6, 80), (76.1, 80), (76.1, 74.5), (9.4, 7.8)]
        dn = [(243.9, 80), (249.4, 80), (316.1, 13.3), (316.1, 7.8), (310.6, 7.8),
              (243.9, 74.5)]
        for k, dx in enumerate((0, 80, 160)):
            bodies.append(upright(f"Diag {side} L{k} (3-ply UD)",
                                  [(x + dx, y) for x, y in up_], z0, z1))
            bodies.append(upright(f"Diag {side} R{k} (3-ply UD)",
                                  [(x + dx, y) for x, y in dn], z0, z1))
    for xi in range(0, 481, 80):
        bodies.append(box(f"TopX {xi} (3-ply UD)", (xi - 3.9, 80, -zi), (xi + 3.9, 87.8, zi)))
        bodies.append(box(f"BotX {xi} (3-ply UD)", (xi - 3.9, 0, -zi), (xi + 3.9, 7.8, zi)))
    bodies += [box("Support L", (-15, -10, -60), (30, 0, 60)),
               box("Support R", (450, -10, -60), (495, 0, 60))]
    # the rebar across the two top chords at node 11 (x 240), pressed by the platen: a
    # bar 10 mm deep on the top chords, the plate on it
    bodies.append(box("Load hanger", (235, 87.8, -zo - 5), (245, 97.8, zo + 5)))
    ov = _merge(COMMON_OVERRIDES, {"glue": {"Load hanger | *": 0}, "params": {
        "material.E_MPa": 7350.0, "material.density_kg_m3": 500.0}})
    ref = {"case": "C", "source": "Moraes et al. 2019, Atena Editora ch. 31",
           "Fu_kgf": 490.0 / 9.80665, "failure": "the bottom chord and the foot of a "
           "vertical", "parts": ["BotChord", "Vertical"],
           "stiffness_kgf_per_mm": 9.6, "deflection_at_Fu_mm": 3.8,
           "load": "test machine platen on a rebar across the top chords at mid-span"}
    return bodies, ov, ref


CASES = {"A": lamberti_2025, "B": silva_2026, "C": moraes_2019}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fidelity", default="standard")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--write-only", action="store_true")
    a = ap.parse_args()
    import yaml

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bodies, ov, ref = CASES[a.case]()
    stl, ovf = out / f"{a.case}.stl", out / f"{a.case}.overrides.yaml"
    write_stl(bodies, stl, names=True)
    ovf.write_text(yaml.safe_dump(ov, sort_keys=False), encoding="utf-8")
    (out / f"{a.case}.reference.json").write_text(json.dumps(ref, indent=2))
    if a.write_only:
        print(stl, ovf)
        return
    from bridgesim.sim import study

    for sd in (int(s) for s in a.seeds.split(",")):
        rep, _ = study.crush(stl, overrides=str(ovf), seed=sd, fidelity=a.fidelity,
                             out_dir=str(out / f"{a.case}_{a.fidelity}_seed{sd}"),
                             stop_at_failure=True, replay=False)
        print(json.dumps({"case": a.case, "seed": sd, "fidelity": a.fidelity,
                          "Fu_kgf": rep.Fu_kgf, "test_Fu_kgf": round(ref["Fu_kgf"], 1),
                          "ratio": round(rep.Fu_kgf / ref["Fu_kgf"], 3),
                          "governing": rep.governing, "competing": rep.competing[:3],
                          "deflection_at_Fu_mm": rep.deflection_at_Fu_mm,
                          "mass_kg": rep.mass_kg,
                          "trustworthy": rep.quality.trustworthy,
                          "reasons": rep.quality.reasons}), flush=True)


if __name__ == "__main__":
    main()
