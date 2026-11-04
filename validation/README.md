# Validation

bridgesim's predictions are only useful if the method is checked. Two kinds of evidence:

## 1. Textbook checks (automated, run in CI)

| Check | Reference | Test |
|---|---|---|
| Simply supported beam, mid-span point load: $\delta = PL^3/(48EI)$ within 0.5 % | Hibbeler, *Mechanics of Materials*, App. C | `tests/test_beam_validation.py` |
| Pin-jointed planar Warren truss: chord forces by the method of sections, $F = M/h$, within 1 % | Hibbeler, *Structural Analysis*, Ch. 3 | `tests/test_truss_validation.py` |
| Euler buckling of one stick: $P_{cr} = \pi^2 EI/L^2 \approx 49.75$ N (5.07 kgf) | Timoshenko & Gere | `tests/test_buckling.py` |
| Global buckling eigen-solver: pinned column $\pi^2EI/L^2$, flagpole $\pi^2EI/(2L)^2$, within 0.5 % | Euler; McGuire, Gallagher & Ziemian, Ch. 9 | `tests/test_stability.py` |
| Rectangle section properties and torsion constant | Roark, Table 10.7 | `tests/test_sections.py` |
| Pynite local-axis orientation (b ≠ d cantilevers) | Pynite 3.2.0 source | `tests/test_orientation.py` |
| Plate load distribution sums to $P$ and is symmetric | statics | `tests/test_loads.py` |

`textbook/` is for worked examples (hand calculations, spreadsheets) that back these tests.

## 2. Physical tests (your team, planned support in v0.2)

`physical/` holds CSV templates. Fill them with your own tests, then update
`materials/*.yaml` (and mark values `source: measured`). Suggested tests:

* **Stick bending (E and f_b)**: three-point bending of single sticks,
  $E = F L^3 / (48\,\delta\,I)$, $f_b = M_{max}/S = (F_{max} L/4)/(b t^2/6)$.
* **Glued lap joint (τ_g)**: two sticks overlapped by a known length, pulled to failure:
  $\tau_g = F_{max} / (\text{overlap} \times \text{width})$.
* **Short column (f_c) and buckling**: laminated stacks of increasing length.
* **Prototype crush**: a sub-assembly or a scaled bridge, compared with the prediction.

## 3. Crush simulation (`bridgesim crush`): verification, convergence, validation

The crush simulation is checked in three layers, as in ASME V&V 10: are the equations
solved right (code verification), is the mesh and the time compression fine enough for
this bridge (solution verification), and are they the right equations for popsicle sticks
and white glue (validation, which needs physical tests).

### 3.1 Code verification (automated, `tests/test_sim.py`)

| Check | Reference | Result | Test |
|---|---|---|---|
| Cantilever, small tip load | $PL^3/(3EI)$ | 0.04 % (test: 1 %) | `test_cantilever_small_deflection` |
| Cantilever, large deflection (elastica) | Bisshopp & Drucker (1945) | tip deflection within 1 %, shortening within 3 % | `test_cantilever_elastica` |
| Euler buckling of a stick (bow L/1000) | $\pi^2EI/L^2$ | between -10 % and +2 % | `test_euler_buckling_of_a_stick` |
| Deep beam, shear deformation (depth/span 1/4, E/G 16) | Timoshenko: $PL^3/48EI + PL/4\kappa GA$ | < 1 % on 4 and 16 elements; strength unchanged | `test_deep_beam_includes_shear_deformation` |
| Stick, 3-point bending stiffness | $PL^3/(48EI)$ | 0.03 % (test: 3 %) | `test_three_point_bending` |
| Stick bending strength | the material's $f_b$ | within 10 % | `test_stick_bending_strength_matches_f_b` |
| Single-lap glued joint | $\tau_g$ of the material file | within 5 % | `test_lap_shear_reproduces_tau_g` |
| DKT plate patch test, layered laminate stiffness | constant curvature; CLT A, B, D | exact | `test_dkt_patch_test`, `test_classical_lamination_theory` |
| Shell element zero-energy modes | 6 rigid-body modes | exactly 6 | `test_shell_element_has_only_rigid_body_zero_modes` |
| Deep laminated beam (span/depth 2.5) strong in bending | splits in shear at $\tfrac43 f_v A$ | within 0.5 % on 6 and 12 elements | `test_deep_laminated_beam_fails_in_shear` |
| Cracking plate strip, 5 and 2.5 mm elements | strength and work of fracture $G_f W t$ | strength within 4 %, work within 10 % | `test_shell_fracture_energy_is_mesh_independent` |
| Energy conservation, free vibration of a bridge (beams, shells, glue) | kinetic + strain = const | spread < 0.5 % | `test_energy_is_conserved_without_damping` |
| Glue-line bearing by grain direction | Hankinson | end grain $f_c$, side grain $f_{c,90}$, a 0/90 plate edge their mean | `test_glue_bearing_follows_the_grain` |
| Compiled kernels = NumPy reference | - | relative 1e-9 (forces), 1e-8 (shell damage) | `test_compiled_kernels_match_numpy`, `test_compiled_shell_damage_matches_numpy` |
| End to end: CAD beam on two piers under the plate | beam theory (with shear) and $f_b$ | stiffness within 5 %, strength between $f_c$ and $f_b$ | `test_simply_supported_beam_matches_beam_theory` |
| Calibration loop | coupons reproduce the fitted data | < 1 % | `test_calibration_closes_the_loop` |

### 3.2 Solution verification (per bridge)

* **Time compression**: `crush --rate-check` runs the test again at half the jack speed;
  `quality.rate_check` reports the change of F_u (Bridge_v6, standard fidelity: see
  below). More than 5 % makes the result untrustworthy.
* **Mesh**: `validation/convergence_study.py BRIDGE.SLDPRT` crushes the bridge with the
  quick, standard and fine meshes (element size x1.6, x1, x0.75), all at 100 mm/s, for
  seeds 0, 1 and 2, and prints F_u and the first failure of each. Results for Bridge_v6
  are in 3.4.

### 3.3 Validation plan (needs your tests)

Validation goes up the usual hierarchy; each level uses the data and the score bridgesim
already computes, so it can be repeated after every change of material or method.

| Level | Physical test | Template | Command | Accept when |
|---|---|---|---|---|
| Coupon | 3-point bending of single sticks (5+), glued single-lap joints (5+); stick tension, short-stack compression, a block pressed flat, end-to-end joints and lap load-slip records (3+ each) | `stick_bending_template.csv`, `glue_joint_template.csv`, `stick_tension_template.csv`, `stick_compression_template.csv`, `crush_across_template.csv`, `end_grain_template.csv`, `lap_slip_template.csv` | `bridgesim calibrate` then `bridgesim lab --data` | lab reality-gap score >= 90 on the calibration set, >= 75 on specimens not used to calibrate |
| Joint | a node (two members and their gusset) pulled or crushed to failure | `crush_test_template.csv` | crush the joint's CAD, `bridgesim compare` | failure load within 15 %, same failure mode |
| Sub-assembly or prototype | a truss panel or a scaled bridge crushed to failure | `crush_test_template.csv` | `bridgesim compare` | F_u within 15 %, stiffness within 25 %, same part fails first |
| Competition bridge | the §12.5 test itself (peak load and the part that broke) | `crush_test_template.csv` | `bridgesim compare` | F_u inside the `montecarlo` range |

Make the bridge-level comparisons **blind**: save the `crush.json` (and the
`montecarlo` range) before the physical test, then compare. A model tuned after seeing
the test is calibrated, not validated.

**Status**: no physical data has been run through this plan yet. Until then the crush
simulation is verified but not validated, and its absolute F_u carries the uncertainty
of the placeholder material values (`bridgesim sensitivity` shows how much).

### 3.4 Bridge_v6: mesh convergence and time compression

`validation/convergence_study.py examples/cad/Bridge_v6.SLDPRT` (a real team design, 100
bodies, 7.15 kg; placeholder materials), every mesh at a jack speed of 100 mm/s.

**The perfect bridge** (`--seeds -1`: no imperfections, no scatter, so a change between
meshes is discretisation alone):

| element size | x2 | x1 (standard) | x0.75 (fine) | x0.5 |
|---|---|---|---|---|
| F_u (kgf) | 2717 | 2686 | 2657 | 2359 |
| deflection at F_u (mm) | 12.2 | 11.8 | 10.7 | 4.0 |
| first part to fail (load, kgf) | deck stringer shear (2297) | hub tie strap joint (2068) | hub tie strap crack (1760) | hub tie strap crack (1339) |

From x2 to x0.75 F_u converges (-1.1 % per step), and the load-deflection curves agree
up to 4 mm (2340-2410 kgf there). At x0.5 the bridge collapses early: the first crack,
in the hub tie strap at the end of its glued lap to the hub spreader, starts at a lower
load the finer the mesh (2068, 1760, 1339 kgf). That is the known limit of a **local**
damage model at a stress concentration: the crack band makes the *energy* of a crack
independent of the mesh, not *where and when it starts* at a sharp stress raiser,
whose resolved peak grows as the elements shrink. The same holds in LS-DYNA's MAT_054
or Ansys' progressive damage: the element size is tied to the material's scale and not
refined below it. bridgesim's finest fidelity is therefore x0.75, where every plate
element is still larger than the plate is thick; x0.5 puts plate elements below their
own thickness (the 6.2 mm tie strap gets 4.7 mm elements), outside the model's range.

A nonlocal (process-zone) crack onset was tried: on a plate pulled through a third of
its end (`lab.shell_strip_tension(h, L=40, W=30, angles=(90,), weak=1, grip=1/3)`), the
local peak load falls 48.1, 36.3, 28.8 N with 5, 2.5, 1.25 mm elements, the process-zone
version stays at 59.8, 59.2, 58.4 N. But holding the across-grain and shear modes at
their strengths until the zone cracks doubled the strength of a 45 degree strip (where
those modes interact), so it was not adopted.

A membrane with drilling rotations (Allman's triangle) was also implemented and tested.
A strip bent in its plane gives 0.64, 0.87 and 0.96 of the beam deflection with 1, 2 and
4 elements across (the constant-strain triangle: 0.23, 0.54, 0.82), and the work of
fracture of a cracking strip stayed mesh-independent. But on Bridge_v6 it made the fine
mesh weaker than the standard one: the perfect bridge gave 2731 and 2519 kgf (-7.8 %),
2707 and 2572 kgf with a stiffer drilling penalty (-5.0 %), against 2684 and 2673 kgf
(-0.4 %) with constant-strain membranes (all at 100 mm/s). The better membrane resolves
the stress concentration where the hub tie strap's glued lap ends, and a local crack
onset there then depends on the mesh, the limit described above. It was not adopted:
it needs a mesh-objective crack onset first. Until then, plates made of narrow arms are
too stiff in their plane, and inspect asks about them (`narrow_plates`).

**Three seeds** (imperfections and scatter; each seed is the same imperfect bridge at
every mesh, its random values being keyed on the CAD, not the mesh):

| mesh (element size) | F_u by seed 0 / 1 / 2 (kgf) | mean | spread |
|---|---|---|---|
| quick (x1.6) | 2234 / 2389 / 1972 | 2198 | 417 |
| standard (x1.0) | 2163 / 2288 / 1883 | 2111 | 405 |
| fine (x0.75) | 2157 / 2310 / 1898 | 2122 | 412 |

| mesh | seed | F_u (kgf) | deflection at F_u (mm) | first part to fail |
|---|---|---|---|---|
| quick | 0 | 2234 | 3.39 | joint failure: End Post - L Near \| T1 Cover - L Near |
| quick | 1 | 2389 | 4.26 | member shear: Deck Stringer - R Far |
| quick | 2 | 1972 | 3.02 | joint failure: End Post - L Near \| T1 Cover - L Near |
| standard | 0 | 2163 | 3.36 | joint failure: Hub Tie Strap In - Far \| Hub Spreader - L |
| standard | 1 | 2288 | 3.50 | plate crack: Hub Tie Strap In - Far |
| standard | 2 | 1883 | 3.02 | joint failure: End Post - L Near \| T1 Cover - L Near |
| fine | 0 | 2157 | 3.45 | joint failure: Hub Tie Strap In - Far \| Hub Spreader - L |
| fine | 1 | 2310 | 3.69 | plate crack: Hub Tie Strap In - Near |
| fine | 2 | 1898 | 3.09 | member shear: Deck Stringer - L Near |

* **Per seed**, the standard mesh is within 1 % of the fine one (-0.3 %, +1.0 %, +0.8 %);
  quick is 3.4-3.9 % high, and all three meshes rank the seeds in the same order. (An
  earlier version keyed each joint's random glue strength on its mesh points, so a seed
  drew a different bridge at each fidelity and seemed to move 14 % between meshes.)
* The spread between seeds (~400 kgf, ~20 %) is real: three failure sites compete within
  ~15 % (the deck stringers under the plate, the end-post joint to its locating plate, the
  hub tie strap), and the imperfections decide which goes first. Give F_u as the
  `montecarlo` range and judge design changes with `ab` over several seeds.
* The imperfect bridge fails at ~2110 kgf, the perfect one at ~2690: imperfections (bow,
  out-of-level supports, plate placement) and glue/strength scatter cost ~20 %.
* Every run passed the quasi-static check (kinetic / external work < 0.1) with < 1 %
  added mass.
* **Time compression** (`crush --fidelity standard --seed 0 --rate-check`): F_u 2163 kgf
  at 100 mm/s, 2129 kgf at 50 mm/s (-1.6 %); kinetic / external work 0.002.
* **Design loop**: the `strengthen` what-if of a quick crush of Bridge_v6 (seed 0: the
  End Post joints to their locating plates x1.5 glue and the Top Strut Inner x1.25 section,
  on every copy), tested with `ab -n 3 --fidelity quick`: F_u +13.5 % (90 % interval
  +3.7 % to +23.4 %, stronger with 3 of 3 seeds); the deck stringers under the plate then
  govern.

### 3.5 The linear screen against the crush

`bridgesim screen` solves the same model linearly (seconds) and says where damage starts.
On Bridge_v6 (perfect bridge, standard mesh, 6 s) it puts the bridge's stiffness under the
plate at 678 kgf/mm (the crush's load against jack travel, ~450 kgf/mm, includes the
press's own compliance, ~2000 kgf/mm, and the 0.3 mm seating). Its most loaded parts are the
four end-grain joints of the deck stringers to the cross-beam under the plate (their
limit at 150 kgf), where the crush's glue damage starts first (50-190 kgf), and then the
joints of the hub tie straps to the hub spreader (667 kgf), the first joints the crush
separates (~790 kgf). It does not give F_u: those first joints let go long before the
peak (~2690 kgf), as the load finds other paths, which only the crush follows
(`test_linear_screen_matches_beam_theory` checks its stiffness and fibre limits against
beam theory within 3 %).
