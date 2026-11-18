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
   mid-span pressed by a test machine: 0.49 kN peak at 3.8 mm on the measured curve (the
   text also gives 0.421 kN for the rupture), broken at the bottom chord and the foot of
   a vertical.

Two semi-blind cases, their geometry set from the builder's photos and figures before any
run (no failure place published):
D  Boon (2025), "Basic Warren Truss Popsicle Bridge (Short)", garrettsbridges.com: ten
   jig-built bridges of 44 uncut sticks, Elmer's Glue-All, loaded on top: 37.7-53.6 kg.
E  Boon (2024), "Short Howe Truss Popsicle Bridge", garrettsbridges.com: Howe, 13.5 x 4 x
   3.5 in, node positions from the author's truss diagrams, members from his photo;
   nine bridges in three sets of three, Elmer's Glue-All, a Pitsco bridge tester. By
   stick count the photo, which this rebuilds, is Set 2's reinforced version (57 sticks):
   61.2-83.9 kg peak (the reference); Set 1 (49 sticks, 41.3-54.4 kg) and Set 3 (Set 1
   sanded) are stored beside it (see boon_howe_2024).
Considered and not rebuilt (round 29): the six 50 cm deck models of ENS Paris-Saclay's
"Move Your Bridge" (2019) and violet's 25 cm one: their reports give no dimensions of
these models (height, panels, layers, width, supports, load block), only small photos.

Geometry from the papers' drawings and photos (tags in the comments: S stated, D from a
drawing, P from a photo, A assumed), as extracted for this repository (round 21). The
loads were hung from a node or pressed on a bar, not on a 200 x 90 mm plate: a rigid
hanger (an I-shaped block) rests on the loaded node (dry on A's chords, glued to B's
putty lumps) and stands clear above the bridge, and the plate pushes on its top; C's
rebar is a 10 mm bar resting dry on the top chords. The ends rest on support blocks
(bearing, no glue).
Material: the papers' stick sizes and the pine values they quote (E, f_c), A's glue
(cyanoacrylate) from its maker's data; everything else the shipped placeholders
(craft-stick birch, PVA). Each test is loaded at its own pace (crusher.test_rate_mm_per_min,
the glue's rate law). A and B are predictions made without their results; C is a
calibration case: round 21 chose its end-grain glue law after seeing C (validation
README 3.14), so its agreement does not validate the model.

    uv run python validation/published_tests.py --case A --out DIR [--fidelity standard]
        [--seeds 0,1,2] [--threads 2] [--write-only]
    bridgesim montecarlo DIR/A.stl -o DIR/A.overrides.yaml -n 10 --fidelity fine \
        --sample-placeholders --out DIR/A_mc10      (each case)
    uv run python validation/published_tests.py --summarize --out DIR

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
from bridgesim.cad.synthetic import box, prism, upright, write_stl


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
    # the supports bear only; the hanger is glued to the node it loads (the bar passes
    # through B's putty lumps) unless the case says it bears dry (A, C); the press's
    # 4.2 kg plate is not there. The supports are blocks (since round 28 cells of wood: a
    # support pier is wood); the hanger stands for a steel bar, a wire's yoke or the
    # machine's own loading block: one rigid body, as every block was until round 27
    "glue": {"Support* | *": 0},
    "params": {"crusher.plate_mass_kg": 0.05},
    "bodies": {"Support*": {"kind": "block"}, "Load hanger*": {"kind": "rigid"}},
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
    # the wire through the centre bottom node (x 300): the hanger's foot bears on the two
    # bottom chords there over 40 mm, dry (until round 21 glued, which stiffened the
    # chords over the laps that fail; an 8 mm foot glued rocked over as an inverted
    # pendulum at ~24 kgf in round 22, which dead weights hanging below never do), its
    # stem between the trusses, its cap 20 mm above the cross sticks. The paper (p. 7,
    # fig. 4) passes a 6 mm^2 copper wire through the central node of both trusses, where
    # the diagonals' feet fill the V 4.9 mm above the chords: the load entered the
    # diagonals there, not only the chords, which this foot forces through the 36 mm^2
    # laps of the diagonals' feet (the published-test check of round 25). A seat in the
    # diagonals' crotch was tried in round 25: wedged in the V it spread the diagonals'
    # tips and slipped out of its dry contacts at ~24 kgf, and a wire through a drilled
    # node, a pin through every part there, needs bodies that overlap: A keeps this foot,
    # its load path a known mismatch that reads low
    bodies.append(hanger("Load hanger", _i_profile(33, 10, 45, 8, 14, 130, 140), 280, 320))
    ov = _merge(COMMON_OVERRIDES, {
        "glue": {"Load hanger | *": 0},
        "params": {
        "material.E_MPa": 8846.0, "material.f_c_MPa": 31.5,  # pine, IPT values (S)
        "material.density_kg_m3": 500.0,
        # cyanoacrylate (S: "Almasuper Ultrarrapido", held with clothes pegs): lap shear
        # on pine 10-12 MPa by ISO 4587 with the wood failing (Ceys Superunick data sheet
        # 2020), less for laps glued by hand (A: 8); a glassy adhesive, not PVA: no rate
        # law (A); dead weights in 20 kgf steps (S), so the pace does not matter then
        "material.glue.tau_g_MPa": 8.0, "glue.rate_exponent": 0.0,
        # dead weights (S): a crack lets the load drive on (the press stiffness's low
        # end, round 25)
        "crusher.stiffness_N_per_mm": 2000.0}})
    ref = {"case": "A", "source": "Lamberti et al. 2025, JESTA 3 e86766",
           "Fu_kgf": 160.0, "Fu_range_kgf": [140.0, 160.0],
           "failure": "top chord panel next to the centre and a compression diagonal, "
                      "after a lateral drift from 100 kgf",
           "parts": ["TopChord", "Diag"], "place": ["TopChord"], "mass_g": 180.0,
           "load": "dead weights hung from the centre bottom node, 20 kgf steps",
           "glue": "cyanoacrylate"}
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
        # (epoxy putty, not wood: one rigid body each, as until round 27)
        "bodies": {"Putty*": {"kind": "rigid"}},
        # the putty wraps round each diagonal's end (P): the end face glued to the lump
        # stands for that, x5 (its four sides, ~4 x 9 x 10 mm against the 85 mm^2 end)
        "glue": {"Diag* | Putty*": 5.0},
        # every joint here is epoxy putty on wood, not white glue: the bond keeps the 2.0
        # MPa the stand-in above was set with (the white-glue placeholder until round 28;
        # with round 29's 3.0 the x5 would stand for 15 MPa, and B read 161.6 kgf, 2.23
        # of its test, on seed 0 at fine)
        "params": {"material.E_MPa": 13000.0, "material.f_c_MPa": 31.5,
                   "material.glue.tau_g_MPa": 2.0,
                   "material.density_kg_m3": 500.0,
                   # a load hung from the nodes: the softest loading there is (a crack
                   # lets the load drive on), the press stiffness's low end (round 25)
                   "crusher.stiffness_N_per_mm": 2000.0}})
    ref = {"case": "B", "source": "Silva et al. 2026, Rev. Ensino Eng. 45:389",
           "Fu_kgf": 710.0 / 9.80665, "failure": "compression diagonal D3 buckling out "
           "of plane where its outer plies were butted at mid-length",
           "parts": ["Diag"], "place": ["member", "Diag"],
           "load": "bar through the centre bottom nodes",
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
        "material.E_MPa": 7350.0, "material.density_kg_m3": 500.0,
        # a 30 t universal press (S), its rate not stated: at a usual 1-2 mm/min the
        # record's 3.8 mm to the peak took 2-4 min; the model's ~1 mm at 0.5 mm/min ~2
        "crusher.test_rate_mm_per_min": 0.5,
        # a universal testing machine's frame: stiff (round 25)
        "crusher.stiffness_N_per_mm": 20000.0}})
    ref = {"case": "C", "source": "Moraes et al. 2019, Atena Editora ch. 31",
           "Fu_kgf": 490.0 / 9.80665, "Fu_range_kgf": [421.0 / 9.80665, 490.0 / 9.80665],
           "calibration": "round 21 chose the end-grain glue law after seeing this test",
           "failure": "the bottom chord and the foot of a "
           "vertical", "parts": ["BotChord", "Vertical"],
           # the foot of a vertical on the bottom chord: both in the label (a diagonal's
           # joint on the chord is not it, the adversarial review of round 23)
           "place": ["BotChord", "Vertical"],
           "stiffness_kgf_per_mm": 9.6, "deflection_at_Fu_mm": 3.8,
           "load": "test machine platen on a rebar across the top chords at mid-span"}
    return bodies, ov, ref


def boon_2025():
    """D: Boon, "Basic Warren Truss Popsicle Bridge (Short)", garrettsbridges.com, last
    modified 13 Feb 2025 (https://garrettsbridges.com/photos/popsicle-bridges/
    basic-warren-popsicle-bridge/): 10 bridges of 44 uncut craft sticks and Elmer's Glue-
    All, built in jigs, 13.5 in long, 12 in span, 4 in high, 3.5 in wide (S), loaded on top
    by the author's machine without the card deck (S): 40.9-53.6 kg, mean 45.5, CoV 0.10
    (S); where they broke is not given. A semi-blind case (round 27): the geometry was set
    from the page's three photos before any run. Each side a Warren of single sticks, three
    bottom panels of one stick's length, apexes over their middles (P); the bottom chord
    in two layers, three sticks full length and two inside them from the first to the last
    apex, so each layer's butt joints are spliced by the other (P: web2); the top chord two
    sticks between the end apexes (P), doubled (A: a single stick would buckle out of
    plane at about half the chord force at the test's load; 40 sticks placed against the
    44 stated); diagonals on the chords' outer face (P), mitred where two meet at a node
    (A, as A's); cross sticks under the four bottom nodes and on top at the end apexes and
    either side of the middle one (P), two lateral braces on top in the end panels (P);
    the end portals' braces left out (A). Sticks 114.3 x 9.5 x 2.0 mm (S: standard 4.5 in;
    width and thickness A). The load: a 76 x 102 mm block on the middle top cross sticks
    (A: the author loads 'the top of the bridge', his top-loading page stacks weights on
    the middle of the top); the ends rest on blocks under the end cross sticks, 12 in
    between their inner edges (S, A)."""
    s, w, t = 114.3, 9.5, 2.0
    H, W = 101.6, 88.9
    from shapely.geometry import Polygon
    from shapely.geometry import box as sbox

    def strip(p, q, x0, x1):
        """A stick of length s centred between nodes p and q, clipped to x0..x1 and to
        the truss's height."""
        p, q = np.asarray(p, float), np.asarray(q, float)
        c, d = 0.5 * (p + q), (q - p) / np.linalg.norm(q - p)
        a, b = c - d * s / 2, c + d * s / 2
        n = np.array([-d[1], d[0]]) * w / 2
        poly = Polygon([a - n, b - n, b + n, a + n]).intersection(sbox(x0, 0.0, x1, H))
        return [tuple(v) for v in np.asarray(poly.exterior.coords)[:-1]]

    bodies = []
    zd, zm, zi = (W / 2 - t, W / 2), (W / 2 - 2 * t, W / 2 - t), (W / 2 - 3 * t, W / 2 - 2 * t)
    bot = [(k * s, w / 2) for k in range(4)]
    top = [((k + 0.5) * s, H - w / 2) for k in range(3)]
    nodes = [bot[0], top[0], bot[1], top[1], bot[2], top[2], bot[3]]
    for side, sg in (("N", 1.0), ("F", -1.0)):
        def zr(z, sg=sg):
            return (min(sg * z[0], sg * z[1]), max(sg * z[0], sg * z[1]))
        z0, z1 = zr(zm)
        for k in range(3):
            bodies.append(box(f"BotChord {side} {k + 1} (1-ply UD)", (k * s, 0, z0),
                              ((k + 1) * s, w, z1)))
        for k in range(2):
            bodies.append(box(f"TopChord {side} {k + 1} (1-ply UD)", ((k + 0.5) * s, H - w, z0),
                              ((k + 1.5) * s, H, z1)))
        z0, z1 = zr(zi)
        for k in range(2):
            bodies.append(box(f"BotSplice {side} {k + 1} (1-ply UD)", ((k + 0.5) * s, 0, z0),
                              ((k + 1.5) * s, w, z1)))
            bodies.append(box(f"TopChord {side} {k + 1} inner (1-ply UD)",
                              ((k + 0.5) * s, H - w, z0), ((k + 1.5) * s, H, z1)))
        z0, z1 = zr(zd)
        for k in range(6):
            p, q = nodes[k], nodes[k + 1]
            bodies.append(upright(f"Diag {side} {k + 1} (1-ply UD)",
                                  strip(p, q, p[0], q[0]), z0, z1))
    for k, xc in enumerate((w / 2, s, 2 * s, 3 * s - w / 2)):
        bodies.append(box(f"BotX {k + 1} (1-ply UD)", (xc - w / 2, -t, -s / 2),
                          (xc + w / 2, 0, s / 2)))
    for k, xc in enumerate((s / 2, 1.5 * s - 27, 1.5 * s + 27, 2.5 * s)):
        bodies.append(box(f"TopX {k + 1} (1-ply UD)", (xc - w / 2, H, -s / 2),
                          (xc + w / 2, H + t, s / 2)))
    za = W / 2 - 2.0
    for k, (xa, xb) in enumerate(((s / 2 + w / 2 + 4, 1.5 * s - 27 - w / 2 - 4),
                                  (2.5 * s - w / 2 - 4, 1.5 * s + 27 + w / 2 + 4))):
        L = np.hypot(xb - xa, 2 * za)
        d = np.array([xb - xa, -2 * za]) / L
        n = np.array([-d[1], d[0]]) * w / 2
        a, b = np.array([xa, za]), np.array([xb, -za])
        bodies.append(prism(f"TopBrace {k + 1} (1-ply UD)",
                            [tuple(a - n), tuple(b - n), tuple(b + n), tuple(a + n)],
                            H, H + t))
    bodies += [box("Support L", (-25, -t - 10, -60), (19.05, -t, 60)),
               box("Support R", (3 * s - 19.05, -t - 10, -60), (3 * s + 25, -t, 60))]
    bodies.append(box("Load hanger", (1.5 * s - 38.1, H + t, -50.8),
                      (1.5 * s + 38.1, H + t + 20, 50.8)))
    ov = _merge(COMMON_OVERRIDES, {"glue": {"Load hanger | *": 0}, "params": {
        # 44 sticks, 54.56 g with the glue (S): ~1.24 g a stick of 2172 mm^3
        "material.density_kg_m3": 570.0}})
    held = [40.9, 53.6, 46.4, 50.5, 37.7, 47.7, 42.7, 46.4, 44.1, 44.5]
    ref = {"case": "D", "source": "Boon 2025, garrettsbridges.com, Basic Warren (Short)",
           "Fu_kgf": float(np.mean(held)), "Fu_range_kgf": [float(np.mean(held))] * 2,
           "replicates_kgf": held,
           "replicate_cov": round(float(np.std(held, ddof=1) / np.mean(held)), 3),
           "failure": "not given", "parts": [], "place": None, "mass_g": 54.56,
           "load": "the author's machine on the top of the bridge (block assumed)",
           "glue": "Elmer's Glue-All (PVAc)", "semi_blind": True}
    return bodies, ov, ref


def _round_stick_xy(x0: float, x1: float, rising: bool, length: float, width: float,
                    y_lo: float, y_hi: float, bottom: float | None = None):
    """(x, y) outline of a whole craft stick (semicircular ends of radius width / 2)
    standing in the truss plane, its horizontal extent exactly x0..x1, rising to the right
    if ``rising``: a web member laid whole between its neighbours, across the chords. Its
    angle follows from (length - width) cos(angle) + width = x1 - x0. Centred on
    y_lo..y_hi, or its lowest point at ``bottom``; clipped to y_lo..y_hi. The ends' arcs
    have a corner at every 22.5 degrees, so no facet next to a neighbour's face lies
    parallel to it."""
    from shapely.geometry import Polygon
    from shapely.geometry import box as sbox

    r, a_len = width / 2, length - width
    c = (x1 - x0 - width) / a_len
    sn = float(np.sqrt(1.0 - c * c))
    d = np.array([c, sn if rising else -sn])
    yc = 0.5 * (y_lo + y_hi) if bottom is None else bottom + 0.5 * (a_len * sn + width)
    m = np.array([0.5 * (x0 + x1), yc])
    phi = float(np.degrees(np.arctan2(d[1], d[0])))
    pts = []
    for end, lo in ((m + d * a_len / 2, phi - 90), (m - d * a_len / 2, phi + 90)):
        angs = [lo] + [22.5 * k for k in range(-16, 17) if lo < 22.5 * k < lo + 180]
        for g in angs + [lo + 180]:
            pts.append(end + r * np.array([np.cos(np.radians(g)), np.sin(np.radians(g))]))
    poly = Polygon(pts).intersection(sbox(x0 - 1.0, y_lo, x1 + 1.0, y_hi))
    return [tuple(v) for v in np.asarray(poly.exterior.coords)[:-1]]


def _stick_plan(p, q, length: float, width: float):
    """(x, z) outline of a stick of ``length`` x ``width`` lying flat, centred between
    the plan points p and q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    c, d = 0.5 * (p + q), (q - p) / np.linalg.norm(q - p)
    a, b = c - d * length / 2, c + d * length / 2
    n = np.array([-d[1], d[0]]) * width / 2
    return [tuple(a - n), tuple(b - n), tuple(b + n), tuple(a + n)]


def _brace_plan(x0: float, x1: float, z0: float, z1: float, length: float, width: float):
    """(x, z) outline of a whole flat stick laid from the line z = z0 to the line z = z1
    (centred between them, its ends running past them) that fills x0..x1 exactly, its z0
    end at x0: a lateral brace fitted between the sticks beside it. The stick's x extent
    for a run D between the lines, (D length + |z1 - z0| width) / hypot(D, z1 - z0),
    grows with D, so D is found by bisection."""
    dz = abs(z1 - z0)
    lo, hi = 0.0, x1 - x0
    for _ in range(60):
        D = 0.5 * (lo + hi)
        if (D * length + dz * width) / np.hypot(D, dz) < x1 - x0:
            lo = D
        else:
            hi = D
    xa = 0.5 * (x0 + x1 - D)
    return _stick_plan((xa, z0), (xa + D, z1), length, width)


def boon_howe_2024():
    """E: Boon, "Short Howe Truss Popsicle Bridge", garrettsbridges.com, last modified 1
    Apr 2024 (https://garrettsbridges.com/photos/popsicle-bridges/short-howe-truss-
    popsicle-bridge/), the same builder, jigs and glue as D. Stated (S): 13.5 in long, 4 in
    high, 3.5 in wide; in 2024 nine bridges in three sets of three, Elmer's Glue-All, 48 h
    of drying, the author's assembly jigs, tested on "the Pitsco Bridge tester machine";
    first failure / peak in lb: Set 1 (the base design of the store's blueprints, 49
    unmodified sticks) 120/120, 88/98, 91/91; Set 2 (57 sticks: "additional lateral
    bracing and more sticks to reinforce the top chord") 135/135, 88/135, 132/185; Set 3
    (Set 1 with the glued faces sanded flat) 158/158, 118/146, 142/142. A first failure is
    "the first joint popped loose" (S: no other kind in any set); Set 2's were "a tension
    member joint, mostly along the top chord near the bridge ends" (S); Set 1's and Set
    3's places, and every peak's, are not given. A semi-blind case (round 29): geometry set
    from the page's figures before any run, no crush made while setting it.

    The truss (D: the page's three 'Average Peak Load' figures, the author's 2-D truss
    diagrams of this bridge, drawn on a 1 in grid whose node 0 to node 5 spans the stated
    13.5 in and whose height is the stated 4 in): six bottom nodes at 0, 2.5, 5.25, 8.25,
    11, 13.5 in, four top nodes over the inner four, inclined end posts 0-6 and 5-9,
    verticals 1-6, 2-7, 3-8, 4-9 and the diagonals 1-7, 4-8 (Howe) and 2-8 in the middle
    panel; supports at nodes 0 and 5, the load at the top nodes 7 and 8. The chords lie
    inside the stated 4 in, the stated 3.5 in over the web members' outer faces (as D; the
    end posts' second sticks stand outside it, A). How each member is made (P: the page's
    studio photo, Howe-Truss-Popsicle-Stick-Bridge.jpg): verticals and diagonals single
    sticks, end posts two sticks face to face (a glue seam along each), the bottom chord
    and the top chord two layers (two top edges seen; the bottom chord's outer layer butted
    between nodes 1 and 2 and between 3 and 4, i.e. three full sticks), web members on the
    chords' outer faces, the verticals running past the bottom chord (their lower ends
    below it; A: flush with the top chord's top, the stick's extra 12.7 mm below), the
    diagonals and end posts whole round-ended sticks lying across the chords' depth, a
    diagonal's ends against the verticals beside it (P), so each is laid here whole between
    its neighbours, 0.25 mm clear of them (A), its angle set by the gap (the end panels'
    diagonals then span the chords' full depth, their tips trimmed by ~1 mm; the middle one
    stops 2 mm short of each chord's edge), the end posts from the bridge's end (their feet
    on its underside) to the verticals 1 and 4. Two sticks cannot share node 6 in one
    layer: the verticals 1 and 4 stand beside the end posts' tops, their outer faces on the
    nodes 6 and 9 (4.8 mm inside the diagram's node; P: V1 stands inside the end post's top
    in the photo; A: where exactly). Where a layer's sticks lie when the photo does not
    show it (A): the bottom chord's inner layer two sticks from 0.5 to 2.5 stick lengths
    (as D's, the same builder's), each top-chord layer two uncut sticks centred on mid-span
    (the page: 'unmodified' sticks; two of them span the 8.5 in between nodes 6 and 9 only
    with their joint within 6 mm of mid-span; the photo seems to show one layer's end near
    node 8 and a chord end beyond the end post's top, which two uncut sticks a layer cannot
    do: how the end posts' tops lap on the top chord is not resolved by the photo, here
    over 6 mm of the chord's ends). The middle diagonal runs from node 7 to node 3 on the
    near side of the photo (N, P) and from node 2 to node 8 on the far side (F, A: both
    sides built alike on the blueprint and turned to face each other, which puts it the
    other way, as the X in the 2011 test photo's middle panel shows). Floor: cross sticks
    lying on the bottom chords at the two ends and beside the verticals 2 and 3 on their
    end sides (P; 3 and 4 mm clear, A), under the optional card deck, which was not on the
    test bridges (the same builder's D page: "did not have the optional cardstock roadbed";
    here A). Top (P): a cross stick on the chords at each top node and a Warren of four
    flat sticks between them (far node 6 - near node 7 - far mid-span - near node 8 - far
    node 9), all on one level, each whole stick fitted beside the next (A: 1.5 mm apart).
    The end portals' braces (two crossed sticks in each, P) are left out (A, as D's).
    Sticks 4.5 x 3/8 x 1/16 in (S: the builder's store page for this bridge's plans,
    gb_howe.html: "Standard Popsicle Sticks (4.5 in long, 3/8 in wide, 1/16 in thick)"; D
    assumed 2.0 mm).

    52 sticks placed (per side: bottom chord 3 + 2, top chord 2 + 2, end posts 2 x 2,
    verticals 4, diagonals 3; floor 4; top 8); with the 4 portal braces the photo shows
    56, against 49 stated for Set 1 and 57 for Set 2: by count the photographed bridge is
    the reinforced Set 2 (one stick unseen), and Set 1 lacks 8 of its sticks, 'more sticks
    ... in the top chord' and 'additional lateral bracing' which the page does not show.
    The reference below is therefore Set 2 (decided on the stick count alone, before any
    run); ``set1_*`` and ``set3_*`` hold the other sets', whose bridge this is not.

    Loading (A): the photos of a test on the page are video stills of the 2011 first
    version (50 sticks, Weldbond, a barbell plate on the top chords, the ends on wooden
    planks), not the 2024 tester. The author's diagrams load the top nodes 7 and 8, and
    the same builder's D page: "my testing machine which applies a load to the top of the
    bridge". So a rigid 4 x 3 in block (D's size, its 4 in along the span to reach both
    nodes) lies dry on the top sticks between nodes 7 and 8. Pitsco's Structures Testing
    Instrument manual (not this page) loads a bridge through a roadbed block inside it, on
    its floor, ~2 in shorter than the roadbed, pulled down by a rod: not modelled. The
    ends rest on blocks under the bottom chords, 12 in between their inner edges (A: D's
    stated span on the same builder's machine; Pitsco's manual: the ends overlap the
    supports 1/2-1 in), 70 mm tall so that the verticals' lower ends hang clear (the
    tester has no floor under the span). Glue: Elmer's Glue-All (S), the shipped PVA
    placeholder values; Set 3's sanding is not modelled."""
    from shapely.geometry import Polygon
    from shapely.geometry import box as sbox

    s, w, t = 114.3, 9.525, 1.5875
    H, W = 101.6, 88.9
    L = 13.5 * 25.4
    xt = [2.5 * 25.4, 5.25 * 25.4, 8.25 * 25.4, 11.0 * 25.4]  # nodes 6-9 (and 1-4 below)
    # the verticals: on the nodes 7 and 8; at the nodes 6 and 9 beside the end posts' tops
    xv = [xt[0] + w / 2, xt[1], xt[2], xt[3] - w / 2]
    vl = [x - w / 2 for x in xv]  # the verticals' faces
    vr = [x + w / 2 for x in xv]
    c = 0.25  # a web member's rounded ends this close to the verticals beside it
    bodies = []
    for side, sg in (("N", 1.0), ("F", -1.0)):
        def zr(a, b, sg=sg):
            return (min(sg * a, sg * b), max(sg * a, sg * b))
        zw = zr(W / 2 - t, W / 2)            # web members, on the chords' outer face
        zo = zr(W / 2, W / 2 + t)            # the end posts' second stick
        zc = zr(W / 2 - 2 * t, W / 2 - t)    # chords, outer layer
        zi = zr(W / 2 - 3 * t, W / 2 - 2 * t)  # chords, inner layer
        for k in range(3):
            bodies.append(box(f"BotChord {side} {k + 1} (1-ply UD)", (k * s, 0, zc[0]),
                              ((k + 1) * s, w, zc[1])))
        x0 = 0.5 * L - s  # two uncut sticks centred on mid-span
        for k in range(2):
            bodies.append(box(f"BotSplice {side} {k + 1} (1-ply UD)",
                              ((k + 0.5) * s, 0, zi[0]), ((k + 1.5) * s, w, zi[1])))
            bodies.append(box(f"TopChord {side} {k + 1} (1-ply UD)",
                              (x0 + k * s, H - w, zc[0]), (x0 + (k + 1) * s, H, zc[1])))
            bodies.append(box(f"TopChord {side} {k + 1} inner (1-ply UD)",
                              (x0 + k * s, H - w, zi[0]), (x0 + (k + 1) * s, H, zi[1])))
        # end posts: from the bridge's end, their feet on its underside, to the verticals 1
        # and 4; each two sticks of one outline face to face
        ends = (("L", 0.0, vl[0] - c, True), ("R", vr[3] + c, L, False))
        for name, xa, xe, up in ends:
            outline = _round_stick_xy(xa, xe, up, s, w, 0.0, H, bottom=0.0)
            bodies.append(upright(f"EndPost {side} {name} (1-ply UD)", outline, *zw))
            bodies.append(upright(f"EndPost {side} {name} outer (1-ply UD)", outline, *zo))
        for k, x in enumerate(xv):
            bodies.append(box(f"Vertical {side} {k + 1} (1-ply UD)", (x - w / 2, H - s, zw[0]),
                              (x + w / 2, H, zw[1])))
        # diagonals, each whole between two verticals: 1-7 rising, 7-3 falling (N) or 2-8
        # rising (F), 8-4 falling
        for k, up in enumerate((True, side == "F", False)):
            bodies.append(upright(f"Diag {side} {k + 1} (1-ply UD)",
                                  _round_stick_xy(vr[k] + c, vl[k + 1] - c, up, s, w, 0.0, H),
                                  *zw))
    # floor: cross sticks lying on the bottom chords, at the ends 3 mm clear of the end
    # posts' feet, and beside the verticals 2 and 3 on their end sides, 4 mm clear
    foot = Polygon(_round_stick_xy(0.0, vl[0] - c, True, s, w, 0.0, H, bottom=0.0))
    xf = foot.intersection(sbox(-1.0, 0.0, L, w + t)).bounds[2] + 3.0
    for k, xa in enumerate((xf, vl[1] - 4.0 - w, vr[2] + 4.0, L - xf - w)):
        bodies.append(box(f"BotX {k + 1} (1-ply UD)", (xa, w, -s / 2), (xa + w, w + t, s / 2)))
    # top: a cross stick at each top node (on the verticals' tops), and a Warren of whole
    # flat sticks between them (far F at -z, near N at +z), each 1.5 mm clear of the next
    for k, x in enumerate(xv):
        bodies.append(box(f"TopX {k + 1} (1-ply UD)", (x - w / 2, H, -s / 2),
                          (x + w / 2, H + t, s / 2)))
    za, g, xm = W / 2 - 2 * t, 1.5, 0.5 * L
    for k, (xa, xe, z0, z1) in enumerate(((vr[0] + g, vl[1] - g, -za, za),
                                          (vr[1] + g, xm - g / 2, za, -za),
                                          (xm + g / 2, vl[2] - g, -za, za),
                                          (vr[2] + g, vl[3] - g, za, -za))):
        bodies.append(prism(f"TopBrace {k + 1} (1-ply UD)",
                            _brace_plan(xa, xe, z0, z1, s, w), H, H + t))
    bodies += [box("Support L", (-25, -70, -60), (L / 2 - 6 * 25.4, 0, 60)),
               box("Support R", (L / 2 + 6 * 25.4, -70, -60), (L + 25, 0, 60))]
    bodies.append(box("Load hanger", (xm - 50.8, H + t, -38.1),
                      (xm + 50.8, H + t + 20, 38.1)))
    ov = _merge(COMMON_OVERRIDES, {"glue": {"Load hanger | *": 0}, "params": {
        # no mass stated for this bridge: D's 54.56 g for 44 of the same builder's sticks
        # with the glue (S there), 1.24 g a stick of 1728 mm^3 (4.5 x 3/8 x 1/16 in)
        "material.density_kg_m3": 718.0}})

    def kgf(lb):
        return [round(v * 0.45359237, 2) for v in lb]

    peak, first = kgf([135, 135, 185]), kgf([135, 88, 132])
    mean = float(np.mean(peak))
    ref = {"case": "E", "source": "Boon 2024, garrettsbridges.com, Short Howe Truss "
           "Popsicle Bridge, Set 2",
           "Fu_kgf": mean, "Fu_range_kgf": [mean, mean], "replicates_kgf": peak,
           "replicate_cov": round(float(np.std(peak, ddof=1) / mean), 3),
           "first_failure_kgf": first,
           "failure": "the first failure a joint popping loose (every set); Set 2's a "
                      "tension member joint, mostly along the top chord near the bridge "
                      "ends; where the peak's failure was, not given",
           "parts": [], "place": None, "mass_g": None,
           "load": "the Pitsco bridge tester; a block on the top nodes 7 and 8 assumed "
                   "(the author's truss diagrams load those nodes)",
           "glue": "Elmer's Glue-All (PVAc)", "semi_blind": True,
           "sticks_stated": 57, "sticks_placed": 52, "sticks_in_photo": 56,
           "version_note": "by stick count the photographed bridge, which this rebuilds, is "
                           "Set 2's reinforced version (57 stated, 56 seen, the 4 portal "
                           "braces left out); Set 1 (49) differs in its top chord and "
                           "lateral bracing, not shown",
           "set1_replicates_kgf": kgf([120, 98, 91]),
           "set1_first_failure_kgf": kgf([120, 88, 91]),
           "set1_note": "the base design, 49 sticks: not this geometry",
           "set3_replicates_kgf": kgf([158, 146, 142]),
           "set3_first_failure_kgf": kgf([158, 118, 142]),
           "set3_note": "Set 1's design with the glued faces sanded flat"}
    return bodies, ov, ref


CASES = {"A": lamberti_2025, "B": silva_2026, "C": moraes_2019, "D": boon_2025,
         "E": boon_howe_2024}


def summarize(out: Path, case: str, mc: str = "mc10") -> dict:
    """A case's montecarlo (``<case>_<mc>/seed_*/crush.json``, placeholders drawn) against
    its test: the runs' F_u over the test's (median and range), where the test falls
    among the runs (the share of runs below it), and the share of runs failing at the
    test's place (``place``: words every governing label must hold; 'member' for a
    member's own failure). A distribution, not seed 0 (the realism review of round 22:
    one seed misrepresented trusses whose seeds spread 2x). The runs are those montecarlo
    keeps (since round 25): a run crushed again at half the jack speed (``seed_N_slow``)
    in place of its first, untrustworthy runs left out while at least half remain (until
    round 24 the table counted them all while the text said they were left out: the
    adversarial review of round 24); with the montecarlo's own report, its recommended
    prediction and total range against the test."""
    ref = CASES[case]()[2]
    base = out / f"{case}_{mc}"
    found: dict[str, tuple] = {}
    for f in sorted(base.glob("seed_*/crush.json")):
        name = f.parent.name
        seed = name.removesuffix("_slow")
        if name.endswith("_slow") or seed not in found:
            d = json.loads(f.read_text())
            found[seed] = (d["Fu_kgf"], d.get("governing", ""), d["quality"]["trustworthy"])
    every = list(found.values())
    if not every:
        return {"case": case, "runs": 0}
    ok = [r for r in every if r[2]]
    runs = ok if len(ok) >= max(3, (len(every) + 1) // 2) else every
    F = np.array([r[0] for r in runs])
    lo, hi = ref.get("Fu_range_kgf", [ref["Fu_kgf"]] * 2)
    place = ref.get("place") or []

    def at_place(g: str) -> bool:
        kind, _, what = g.partition(": ")
        return all((w in kind) if w == "member" else (w in what) for w in place)

    out = {"case": case, "runs": len(runs), "test_Fu_kgf": [round(lo, 1), round(hi, 1)],
           "Fu_kgf": [round(float(x), 1) for x in sorted(F)],
           "ratio_median": round(float(np.median(F)) / hi, 2),
           "ratio_range": [round(float(F.min()) / hi, 2), round(float(F.max()) / lo, 2)],
           "share_below_test": round(float(np.mean(F < lo)), 2),
           "share_above_test": round(float(np.mean(F > hi)), 2),
           "share_at_test_place": (round(float(np.mean([at_place(r[1]) for r in runs])), 2)
                                   if place else None),
           "untrustworthy": int(sum(not r[2] for r in every)), **_mc_summary(base, lo, hi)}
    reps = ref.get("replicates_kgf")
    if reps:
        # a test built several times (D): its scatter build to build against the model's
        # over seeds alone (<case>_<mc>_seeds: the imperfections and joint-to-joint glue,
        # the placeholders fixed), and the share of the runs inside the builds' range
        out["test_cov"] = ref.get("replicate_cov")
        out["share_in_test_range"] = round(float(np.mean(
            (F >= min(reps)) & (F <= max(reps)))), 2)
        seeds = sorted(base.parent.glob(f"{base.name}_seeds/seed_*/crush.json"))
        Fs = []
        for f in seeds:
            d = json.loads(f.read_text())
            if d["quality"]["trustworthy"]:
                Fs.append(d["Fu_kgf"])
        if len(Fs) >= 3:
            out["seeds_only_kgf"] = [round(float(x), 1) for x in sorted(Fs)]
            out["seeds_only_cov"] = round(float(np.std(Fs, ddof=1) / np.mean(Fs)), 3)
            out["seeds_only_median_over_test"] = round(float(np.median(Fs)) / hi, 2)
    return out


def _mc_summary(base: Path, lo: float, hi: float) -> dict:
    """The montecarlo report's recommended prediction over the test and whether the test
    falls inside its total range (the realism review of round 24: A's did not)."""
    f = base.parent / f"{base.name}.json"
    try:
        d = json.loads(f.read_text())
    except (OSError, ValueError):
        return {}
    out = {"recommended_over_test": round(d["recommended_prediction_kgf"] / hi, 2)}
    if d.get("total_p05_kgf") is not None:
        out["total_range_kgf"] = [d["total_p05_kgf"], d["total_p95_kgf"]]
        out["test_in_total_range"] = bool(d["total_p05_kgf"] <= hi
                                          and lo <= d["total_p95_kgf"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", choices=sorted(CASES))
    ap.add_argument("--summarize", action="store_true",
                    help="summarize each case's montecarlo in --out (<case>_mc10)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fidelity", default="standard")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--write-only", action="store_true")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads for each crush (default: all cores)")
    a = ap.parse_args()
    if a.summarize:
        for c in sorted(CASES):
            print(json.dumps(summarize(Path(a.out), c)))
        return
    if a.case is None:
        ap.error("--case is required unless --summarize")
    if a.threads:
        import numba

        numba.set_num_threads(max(1, min(a.threads, numba.config.NUMBA_NUM_THREADS)))
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
