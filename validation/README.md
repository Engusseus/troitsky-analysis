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
  quick, standard and fine meshes (element size x1.6, x0.75, x0.5 of the original 30 mm
  members; `--scales` for others),
  all at 100 mm/s, for seeds 0, 1 and 2, and prints F_u and the first failure of each;
  `--amplitude 0.1` scales the imperfections down for a nearly perfect bridge. Results
  for Bridge_v6 are in 3.4.

### 3.3 Validation plan (needs your tests)

Validation goes up the usual hierarchy; each level uses the data and the score bridgesim
already computes, so it can be repeated after every change of material or method.

| Level | Physical test | Template | Command | Accept when |
|---|---|---|---|---|
| Coupon | 3-point bending of single sticks (5+), glued single-lap joints (5+); stick tension, short-stack compression, a block pressed flat, end-to-end joints, lap load-slip records and short-beam shear of a laminate (3+ each) | `stick_bending_template.csv`, `glue_joint_template.csv`, `stick_tension_template.csv`, `stick_compression_template.csv`, `crush_across_template.csv`, `end_grain_template.csv`, `lap_slip_template.csv`, `short_beam_shear_template.csv` | `bridgesim calibrate` then `bridgesim lab --data` | lab reality-gap score >= 90 on the calibration set, >= 75 on specimens not used to calibrate |
| Joint | a node (two members and their gusset) pulled or crushed to failure | `crush_test_template.csv` | crush the joint's CAD, `bridgesim compare` | failure load within 15 %, same failure mode |
| Sub-assembly or prototype | a truss panel or a scaled bridge crushed to failure | `crush_test_template.csv` | `bridgesim compare` | F_u within 15 %, stiffness within 25 %, same part fails first |
| Competition bridge | the §12.5 test itself (peak load and the part that broke) | `crush_test_template.csv` | `bridgesim compare` | F_u inside the `montecarlo` range |

Make the bridge-level comparisons **blind**: save the `crush.json` (and the
`montecarlo` range) before the physical test, then compare. A model tuned after seeing
the test is calibrated, not validated.

**Status**: no physical data has been run through this plan yet. Until then the crush
simulation is verified but not validated, and its absolute F_u carries the uncertainty
of the placeholder material values (`bridgesim sensitivity` shows how much).

**Plausibility, not validation.** Published Troitsky results put Bridge_v6's prediction
in the range real bridges reach, nothing more: the 2025 winner (BCIT) was a 7.2 kg
bridge that carried 1587 kg ([BCIT news](https://educationnewscanada.com/article/organization/24081/1131794/BCIT-Engineering-students-win-big-at-the-Troitsky-Bridge-Building-Competition.htm));
the record is 4889 kg (University of Toronto, 2026), after 3700 kg (McMaster, 2018)
([U of T news](https://news.engineering.utoronto.ca/u-of-t-team-sets-new-record-at-premier-student-bridge-building-competition/)).
Bridge_v6, 7.15 kg, is predicted at 1.9-2.2 t over seeds (standard fidelity, placeholder
materials, the craft-stick E): the same order as the 7.2 kg winner and below the record. These are other
designs, built with measured sticks and glue, so this checks only that the predicted
loads are not out of scale.

### 3.4 Bridge_v6: mesh convergence and time compression

`validation/convergence_study.py examples/cad/Bridge_v6.SLDPRT` (a real team design, 100
bodies, 7.15 kg; placeholder materials), every mesh at a jack speed of 100 mm/s. Element
sizes are given as a multiple of the original standard (30 mm members): since round 13,
on the evidence below, the standard fidelity is x0.75 and the fine one x0.5 (x1 was the
standard mesh until then, x0.75 the fine one). Round 13:
a frame's plies keep the sheet's axis, torsion loads each family of planes by its own
shear, wood bends over before it crushes, correlated scatter. The one `important`
question, the cross-beam under the plate ("Body 21531"), the designer answered: a
laminated block of sticks, 31-ply UD, which is the model these runs used (identical,
array for array, with and without the answer).

**A nearly perfect bridge** (`--seeds 0 --amplitude 0.1`: every imperfection and scatter
of seed 0 at a tenth). A perfect bridge is symmetric, near and far, left and right:
which of its identical sides gives way first is a bifurcation that round-off decides,
so a different mesh also breaks the symmetry at a different moment. A tenth of the
imperfections lets the seed pick the side at every mesh, the bridge still nearly
perfect:

| element size | x2 | x1 | x0.75 (standard) | x0.5 (fine) |
|---|---|---|---|---|
| F_u (kgf) | 2063 | 2362 | 2270 | 2403 |
| deflection at F_u (mm) | 5.5 | 5.7 | 5.7 | 6.2 |
| first joint to separate (kgf) | 956 | 725 | 776 | 724 |
| what governs the collapse | inner hub tie strap splits | Hub Core splits (far) | outer hub tie strap breaks | Hub Core splits (near) |

The perfect bridge itself (`--seeds -1`) gives 2351 and 2257 kgf at x1 and x0.75,
within 0.6 % of these. (Round 12's perfect bridge, 2079 / 2512 / 2272 kgf at x2 / x1 /
x0.75, swung by +21 % and -10 %.) From x1 down, F_u stays within +-3 % of the mean of
the three (2345 kgf), x0.75 lowest and x0.5 highest: the nearly perfect bridge's hub
parts give way within a few % of each other, and the mesh tips which. The load curves
of x1, x0.75 and x0.5 lie on top of each other up to 90 % of the peak (1951 kgf at 5.5
mm of jack travel on all three); they part only in which hub part gives way first and
when. The hub is where it fails at every mesh but x2.

**x0.5 and local crack onset.** A plate left as a shell cracks where its strain first
reaches the strength in one element: the crack band makes the *energy* of a crack
independent of the mesh, not *when it starts* at a sharp stress raiser, whose resolved
peak grows as elements shrink (the known limit of a local damage model, as in LS-DYNA's
MAT_054 or Ansys' progressive damage). Until round 12 this showed at x0.5 (round 9: F_u
-9 %), and x0.75 was the finest fidelity. Since the narrow plates are frames (round 11)
the stress raisers that mattered are no longer resolved by shells, and x0.5 behaves
like x0.75 (above and below). (A first x0.5 run gave 2088
kgf, -8 %: frames then merged every node within 0.3 x the arm width, so at x0.5 the hub
plates' frames lost most of their elements. Fixed in round 13; the quick, standard and
fine models are unchanged by the fix, `test_a_fine_frame_keeps_its_elements`.)

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
onset there then depends on the mesh, the limit described above. It was not adopted.
Since round 11 the plates made of narrow arms are frames instead (above), which removes
the in-plane locking where it mattered without resolving those stress concentrations.

**Three seeds** (imperfections and scatter; each seed is the same imperfect bridge at
every mesh, its random values being keyed on the CAD, not the mesh):

| mesh (element size) | F_u by seed 0 / 1 / 2 (kgf) | mean | spread |
|---|---|---|---|
| quick (x1.6) | 2038 / 1973 / 1679 | 1897 | 359 |
| x1 (the standard mesh until round 13) | 2279 / 2311 / 1851 | 2147 | 460 |
| standard (x0.75) | 2197 / 2131 / 1823 | 2050 | 375 |
| fine (x0.5) | 2189 / 2079 / - | | |

| mesh | seed | F_u (kgf) | deflection at F_u (mm) | what governs the collapse | `competing` |
|---|---|---|---|---|---|
| quick | 0 | 2038 | 6.07 | joint failure: End Post - L Near \| T1 Cover - L Near | 8 |
| quick | 1 | 1973 | 4.65 | member crushing: Hub Cheek - Far | 8 |
| quick | 2 | 1679 | 4.36 | joint failure: End Post - L Near \| T1 Cover - L Near | 4 |
| x1 | 0 | 2279 | 5.67 | joint failure: End Post - L Near \| T1 Cover - L Near | 6 |
| x1 | 1 | 2311 | 5.55 | member fracture: Hub Core - Far | 8 |
| x1 | 2 | 1851 | 4.49 | joint failure: End Post - L Near \| T1 Cover - L Near | 3 |
| standard (x0.75) | 0 | 2197 | 5.54 | member shear: Hub Core - Near | 7 |
| standard (x0.75) | 1 | 2131 | 5.26 | member shear: Hub Core - Far | 2 |
| standard (x0.75) | 2 | 1823 | 4.49 | joint failure: End Post - L Near \| T1 Cover - L Near | 1 |
| fine (x0.5) | 0 | 2189 | 5.42 | member shear: Top Rail Plate - R Far | 0 |
| fine (x0.5) | 1 | 2079 | 4.96 | member shear: Hub Core - Far | 0 |
(`competing`: the other parts that broke within 10 % of F_u.)

* **Per seed**, x0.75 is within 2.5 % of x0.5 (+0.4 %, +2.5 %), and x1 is 4-11 % higher
  (+4.1 %, +11.2 %): F_u falls toward a limit as the mesh is refined, by 82 then 8 kgf
  (seed 0) and 180 then 52 kgf (seed 1). Hence x0.75 is the standard fidelity since round
  13 and x0.5 the fine one. On the nearly perfect bridge the three lie within +-3 % with
  no trend (above): there several hub parts give way within a few % of each other, and
  the mesh tips which. The quick mesh is 7-8 % lower than x0.75 on every seed (it is for
  ranking designs on the same mesh). All meshes rank the seeds alike.
* **Where it fails**: seed 2 at the End Post joint on the quick, x1 and x0.75 meshes;
  seed 1 in the Hub Core at x1, x0.75 and x0.5; seed 0 at the End Post joint (quick,
  x1), the Hub Core (x0.75) or a top rail plate (x0.5), three sites within a few % of
  each other. `diagnose` (below) measures which of them the load depends on; design
  decisions go through it and `ab`, not one run's label.
* **The strict reading** (`Fu_first_joint_failure_kgf`, the first joint to separate):
  615 / 718 / 466 kgf at x0.75, 579 / 663 at x0.5, 572 / 582 / 428 at x1, 826 / 887 /
  779 at quick (the inner hub tie straps' laps to the hub spreader, which the coarse
  quick frames load less): read it at standard or fine, as an order of magnitude.
* The spread between seeds (~375 kgf, ~18 % at x0.75) comes from the imperfections
  (bows, the lean, out-of-level supports, plate placement) and the scatter, with a
  bridge-wide glue factor (15 %) on top of the joint-to-joint one: repeat builds of one
  design differ by about that much. Give F_u as the `montecarlo` range and judge design
  changes with `ab` over several seeds.
* The imperfect bridge fails at ~2050 kgf (x0.75, mean of 3 seeds), the nearly perfect
  one at ~2270: imperfections and scatter cost ~10 %.
* Every run passed the quasi-static check (kinetic / external work < 0.1) with < 0.3 %
  added mass.
* **Time compression** (`crush --fidelity standard --seed 0 --rate-check`, x0.75): F_u
  2197 kgf at 100 mm/s, 2182 kgf at 50 mm/s (-0.7 %); extrapolated to a jack at rest
  2168 kgf. (On the x1 mesh: 2279 and 2243 kgf, -1.6 %.)
* **The load curve**: along the jack travel, no step: the largest change within 0.01 mm
  of travel before the failure call is within the bridge's stiffness x 0.01 mm + 1 %.
  It bends over before the peak: the secant stiffness from 50 to 90 % of the peak is
  14 % below that from 10 to 50 % (quick, seed 0: 376 and 439 kgf/mm).
* **Stick modulus**: with the material file's 10 GPa instead of the 7.1 GPa crush
  placeholder (the published craft-stick value), standard seed 0 gives F_u 2202 kgf
  (+0.2 %) and a deflection at F_u of 4.17 mm instead of 5.54: the strength is set by
  the joints and the splits, the deflection by E.
* **What limits the bridge** (`diagnose -n 2 --fidelity quick --candidates 3`, seeds 1
  and 2): strengthening the End Post joints to their T1 covers (glue x1.5) raises F_u by
  5.1 % (0.0 % and +10.2 %); the joints of the top rail plates to the T-nodes and the
  T1 gussets, by -1.4 % and -1.9 % (the failure moves to the hub).
* **Design loop**: the `strengthen` what-if of a quick crush of Bridge_v6 (seed 0: the
  inner hub tie straps, hub cheeks and hub covers x1.25 in section, on every copy),
  tested with `ab -n 3 --fidelity quick`: F_u +7.0 % (+12.5 %, +0.1 %, +8.2 %), 90 %
  interval -3.6 % to +17.5 %: not proven with 3 seeds; the failure moved to the T-nodes
  on 2 of 3 seeds (`failure_moved`).

### 3.5 The linear screen against the crush

`bridgesim screen` solves the same model linearly (seconds) and says where damage starts.
On Bridge_v6 (perfect bridge, standard mesh, 5 s) it puts the bridge's stiffness under the
plate at 497 kgf/mm (the crush's load against jack travel, 375 kgf/mm, includes the
press's own compliance and the 0.3 mm seating; against the deck's deflection the crush
gives 471 kgf/mm). Its first limit is the Euler buckling of an arm of the thin +-45 End
Diaphragms (35 kgf: a brace that carries little and that the crush follows through
without a failure), then the four end-grain joints of the deck stringers to the
cross-beam under the plate (87-94 kgf), where the crush's glue damage starts (98 kgf),
the top rail plates' buckling (146-148 kgf), and later the joints of the inner hub tie
straps to the hub spreader (597-634 kgf), where the crush's glue starts to fail (628
kgf) and the first joint separates (724 kgf). It does not give F_u: those first joints
let go long before the peak, as the load finds other paths, which only the crush follows
(`test_linear_screen_matches_beam_theory` checks its stiffness and fibre limits against
beam theory within 3 %).
