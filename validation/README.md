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
bodies, 7.15 kg; placeholder materials), every mesh at a jack speed of 100 mm/s.

**The perfect bridge** (`--seeds -1`: no imperfections, no scatter, so a change between
meshes is discretisation alone):

| element size | x2 | x1 (standard) | x0.75 (fine) |
|---|---|---|---|
| F_u (kgf) | 2079 | 2512 | 2272 |
| deflection at F_u (mm) | 6.3 | 9.7 | 5.7 |
| what governs the collapse | hub cheek crushing | outer hub tie strap breaks (far side) | Hub Core arm breaks (near side) |

Round 12 (the stick modulus 7.1 GPa, interlaminar strength by ply angle, the Hub Core a
frame, ply splits judged over a section depth). The perfect bridge is symmetric, near
and far, left and right: which of its identical sides gives way first is a bifurcation
that round-off decides, and here the standard mesh lets the far hub go, the fine mesh
the near one, 10 % lower. (Up to the hub the two sequences match: the first joint to
let go, the hub tie strap's lap to the hub spreader, at 706 and 770 kgf; the deck
stringers' joints to the cross-beam at 1083 and 1163 kgf.) Round 11 found 2300 / 2411 /
2419 kgf, round 10 2703 / 2640 / 2674; with which side goes first left to chance, the
perfect bridge does not measure the mesh error. The seeds below do: their imperfections
pick the side, the same at every mesh. The bridge is judged failed when the load falls
30 % below its peak (§12.5's "moment of failure").

Below x0.75 the plates left as shells get elements smaller than they are thick, and
crack onset at a stress concentration becomes local. That is the known limit of a
**local** damage model: the crack band makes the *energy* of a crack independent of the
mesh, not *where and when it starts* at a sharp stress raiser, whose resolved peak grows
as the elements shrink. The same holds in LS-DYNA's MAT_054 or Ansys' progressive damage:
the element size is tied to the material's scale and not refined below it. bridgesim's
finest fidelity is therefore x0.75, where every plate element is still larger than the
plate is thick (round 9: F_u -9 % at x0.5). For the members, ply splits are judged on the
shear averaged over a section depth along the member (ASSUMPTIONS-SIM.md section 4), so
they do not start earlier on a finer mesh.

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
| quick (x1.6) | 2031 / 1903 / 1739 | 1891 | 292 |
| standard (x1.0) | 2218 / 2162 / 1860 | 2080 | 358 |
| fine (x0.75) | 2199 / 2095 / 1813 | 2036 | 386 |

| mesh | seed | F_u (kgf) | deflection at F_u (mm) | what governs the collapse | `competing` |
|---|---|---|---|---|---|
| quick | 0 | 2031 | 6.07 | member shear: Hub Tie Strap In - Far | 8 |
| quick | 1 | 1903 | 4.52 | joint failure: cross-beam \| Deck Stringer - L Far | 8 |
| quick | 2 | 1739 | 4.63 | joint failure: End Post - L Near \| T1 Cover - L Near | 8 |
| standard | 0 | 2218 | 5.49 | member shear: Top Rail Plate - R Near | 3 |
| standard | 1 | 2162 | 5.11 | member shear: cross-beam | 1 |
| standard | 2 | 1860 | 4.61 | joint failure: End Post - L Near \| T1 Cover - L Near | 2 |
| fine | 0 | 2199 | 5.60 | joint failure: End Post - L Near \| T1 Cover - L Near | 8 |
| fine | 1 | 2095 | 5.02 | member shear: Hub Core - Far | 3 |
| fine | 2 | 1813 | 4.54 | joint failure: Hub Cheek - Far \| Deck Rib Hub - L | 6 |

(The cross-beam is the body named "Body 21531", the 221 x 88 x 63 mm piece under the
plate, a deep member. `competing`: the other parts that broke within 10 % of F_u.)

* **Per seed**, the standard mesh is within 3.2 % of the fine one (+0.8 %, +3.2 %,
  +2.6 %), the means within 2.2 % (2080 and 2036 kgf); the quick mesh is 4-9 % lower
  (it is for ranking designs on the same mesh). All three rank the seeds alike. (Before
  the splits were judged over a section depth, standard was up to 8.8 % off the fine
  mesh, seed 1.)
* **The strict reading** (`Fu_first_joint_failure_kgf`, the first joint to separate):
  546 / 565 / 405 kgf at standard, 597 / 730 / 444 at fine, 849 / 921 / 794 at quick:
  read it at standard or fine, and as an order of magnitude.
* **Where it fails**: several parts break within a few % of F_u (`competing`), and
  which one starts the collapse changes with the mesh: on seed 2 standard and quick name
  the End Post joint to its T1 Cover, which fine names on seed 0. F_u does not depend
  on which goes first (per seed within 3.2 %); design decisions go through `diagnose`
  and `ab`, not one run's label.
* The spread between seeds (~360 kgf, ~17 %) comes from the imperfections (bows, the
  lean, out-of-level supports, plate placement) and the scatter, now with a bridge-wide
  glue factor (15 %) on top of the joint-to-joint one: repeat builds of one design differ
  by about that much. Give F_u as the `montecarlo` range and judge design changes with
  `ab` over several seeds.
* The imperfect bridge fails at ~2080 kgf, the perfect one at ~2270-2510: imperfections
  and scatter cost ~10-15 %.
* Every run passed the quasi-static check (kinetic / external work < 0.1) with < 1 %
  added mass.
* **Time compression** (`crush --fidelity standard --seed 0 --rate-check`): F_u 2218 kgf
  at 100 mm/s, 2217 kgf at 50 mm/s (-0.06 %); extrapolated to a jack at rest 2215 kgf.
* **The load curve**: along the jack travel, no step: the largest change within 0.01 mm
  of travel before the failure call is within the bridge's stiffness x 0.01 mm + 1 %.
* **Stick modulus**: with the material file's 10 GPa instead of the 7.1 GPa crush
  placeholder (the published craft-stick value), standard seed 0 gives F_u 2243 kgf
  (+1.1 %) and a deflection at F_u of 4.17 mm instead of 5.49: the strength is set by the
  joints and the splits, the deflection by E.
* **What limits the bridge** (`diagnose -n 2 --fidelity quick --candidates 3`, seeds 1
  and 2): strengthening the hub covers (section x1.25) raises F_u by 5.1 % (+3.4 % and
  +6.7 %, both seeds), the hub cheeks by 4.9 % (+11.5 %, -1.7 %), the top rail plates by
  2.4 %.
* **Design loop**: the `strengthen` what-if of a quick crush of Bridge_v6 (seed 0: the
  hub covers x1.25 in section, the inner hub tie straps' laps to the hub spreader x1.5
  glue, on every copy), tested with `ab -n 3 --fidelity quick`: F_u +4.0 % (+9.6 %,
  -0.5 %, +2.8 %), 90 % interval -4.7 % to +12.6 %: not proven with 3 seeds; the failure
  moved on 2 of 3 seeds (`failure_moved`).

### 3.5 The linear screen against the crush

`bridgesim screen` solves the same model linearly (seconds) and says where damage starts.
On Bridge_v6 (perfect bridge, standard mesh, 5 s) it puts the bridge's stiffness under the
plate at 501 kgf/mm (the crush's load against jack travel, 373 kgf/mm, includes the
press's own compliance and the 0.3 mm seating; against the deck's deflection the crush
gives 464 kgf/mm). Its first limit is the Euler buckling of an arm of the thin +-45 End
Diaphragms (34 kgf: a brace that carries little and that the crush follows through
without a failure), then the four end-grain joints of the deck stringers to the
cross-beam under the plate (91-98 kgf), where the crush's glue damage starts (103-108
kgf), and later the joints of the hub tie straps to the hub spreader (568 kgf), the
first joints the crush separates (706 kgf). It does not give F_u: those first joints let
go long before the peak, as the load finds other paths, which only the crush follows
(`test_linear_screen_matches_beam_theory` checks its stiffness and fibre limits against
beam theory within 3 %).
