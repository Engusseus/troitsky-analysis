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
| F_u (kgf) | 2710 | 2684 | 2673 | 2429 |
| deflection at F_u (mm) | 12.6 | 7.6 | 10.2 | 4.8 |
| first part to let go (load, kgf) | deck stringer shear (2269) | hub tie strap joint (792) | hub tie strap joint (878) | hub tie strap joint (892) |
| what governs the collapse | deck stringer shear | cross-beam to deck plate joint | deck stringer to deck rib joint | deck stringer shear |

From x2 to x0.75 F_u converges (-1.0 %, then -0.4 %), and the first joint to let go (the
hub tie strap's glued lap to the hub spreader, at ~30 % of F_u) is the same from x1 down.
What governs the collapse changes with the mesh because several parts under the plate
fail within a few % of each other (the deck stringers, their end joints to the
cross-beam, the deck plate's joint to it); F_u does not. At x0.5 F_u drops 9 %: plate
elements are then smaller than their plates are thick (the 6.2 mm tie strap gets 4.7 mm
elements), and crack onset at a stress concentration becomes local. That is the known
limit of a **local** damage model: the crack band makes the *energy* of a crack
independent of the mesh, not *where and when it starts* at a sharp stress raiser,
whose resolved peak grows as the elements shrink. The same holds in LS-DYNA's MAT_054
or Ansys' progressive damage: the element size is tied to the material's scale and not
refined below it. bridgesim's finest fidelity is therefore x0.75, where every plate
element is still larger than the plate is thick.

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
| quick (x1.6) | 2218 / 2389 / 1984 | 2197 | 405 |
| standard (x1.0) | 2165 / 2306 / 1891 | 2121 | 414 |
| fine (x0.75) | 2147 / 2299 / 1917 | 2121 | 383 |

| mesh | seed | F_u (kgf) | deflection at F_u (mm) | what governs the collapse |
|---|---|---|---|---|
| quick | 0 | 2218 | 3.90 | member shear: Deck Stringer - R Near |
| quick | 1 | 2389 | 4.61 | member shear: Deck Stringer - R Far |
| quick | 2 | 1984 | 3.61 | joint failure: End Post - L Near \| T1 Cover - L Near |
| standard | 0 | 2165 | 3.86 | plate crack: Hub Tie Strap In - Far |
| standard | 1 | 2306 | 4.08 | plate crack: Hub Tie Strap In - Far |
| standard | 2 | 1891 | 3.48 | plate crack: Hub Tie Strap In - Far |
| fine | 0 | 2147 | 4.06 | joint failure: cross-beam \| Deck Plate - R |
| fine | 1 | 2299 | 4.33 | joint failure: cross-beam \| Deck Stringer - R Near |
| fine | 2 | 1917 | 3.68 | joint failure: cross-beam \| Deck Plate - R |

(The cross-beam is the body named "Body 21531", the 221 x 88 x 63 mm piece under the
plate, now a deep member: as a rigid block, quick seed 0 gave 2268 kgf, as a member 2237
kgf, -1.4 %.)

* **Per seed**, the standard mesh is within 1.3 % of the fine one (+0.8 %, +0.3 %,
  -1.3 %); quick is 2.4-4.9 % high, and all three meshes rank the seeds in the same
  order.
* The spread between seeds (~400 kgf, ~20 %) is real: several failure sites compete
  within ~15 % (the deck stringers under the plate and their joints to the cross-beam,
  the end-post joint to its locating plate, the hub tie strap), and the imperfections
  decide which goes first. Give F_u as the `montecarlo` range and judge design changes
  with `ab` over several seeds.
* The imperfect bridge fails at ~2120 kgf, the perfect one at ~2680: imperfections (bow,
  out-of-level supports, plate placement) and glue/strength scatter cost ~20 %.
* Every run passed the quasi-static check (kinetic / external work < 0.1) with < 1 %
  added mass.
* **Time compression** (`crush --fidelity standard --seed 0 --rate-check`): F_u 2165 kgf
  at 100 mm/s, 2137 kgf at 50 mm/s (-1.3 %); kinetic / external work 0.002.
* **Design loop**: the `strengthen` what-if of a quick crush of Bridge_v6 (seed 0: the
  deck stringers x1.25 section, their end joints to the cross-beam and the End Post joints
  to their locating plates x1.5 glue, on every copy), tested with `ab -n 3 --fidelity
  quick`: F_u +7.0 %, stronger with 3 of 3 seeds, but the 90 % interval (-7.4 % to
  +21.5 %) includes 0: not proven with 3 seeds. Seed by seed it is +16.8 % where the End
  Post joint governed (seed 2) and only +3.3 % and +1.0 % where the deck stringers under
  the plate did: there the stringer, its end joint and the cross-beam fail within a few %
  of each other, so strengthening two of them moves the failure to the third.

### 3.5 The linear screen against the crush

`bridgesim screen` solves the same model linearly (seconds) and says where damage starts.
On Bridge_v6 (perfect bridge, standard mesh, 6 s) it puts the bridge's stiffness under the
plate at 678 kgf/mm (the crush's load against jack travel, ~450 kgf/mm, includes the
press's own compliance, ~2000 kgf/mm, and the 0.3 mm seating). Its most loaded parts are the
four end-grain joints of the deck stringers to the cross-beam under the plate (their
limit at 150 kgf), where the crush's glue damage starts at 102-132 kgf (only a hub cover
lip joint, which the screen ranks lower, starts earlier, at 34 kgf), and then the
joints of the hub tie straps to the hub spreader (667 kgf), the first joints the crush
separates (792 kgf). It does not give F_u: those first joints let go long before the
peak (2684 kgf), as the load finds other paths, which only the crush follows
(`test_linear_screen_matches_beam_theory` checks its stiffness and fibre limits against
beam theory within 3 %).
