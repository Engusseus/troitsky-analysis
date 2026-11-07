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

### 3.4 Bridge_v6: mesh convergence, uncertainty and time compression

`validation/convergence_study.py examples/cad/Bridge_v6.SLDPRT` (a real team design, 100
bodies, 7.15 kg; placeholder materials), every mesh at a jack speed of 100 mm/s. Element
sizes are a multiple of the original 30 mm members: the standard fidelity is x0.75, the
fine one x0.5, quick x1.6. Round 14: the glue lines creep in physical test time, a
poorly glued joint is softer as well as weaker, the supports, the plate and the glue
lines bear by the grain of the ply (and of the frame arm) at each point, an element bent
past 1 rad is reported as erosion, and the load-drop criterion waits until the bridge
has bedded in on its supports (it judged one seed failed at 17.8 kgf while it rocked onto
the corner that stood off the platform). The cross-beam under the plate ("Body 21531") is
what the designer answered: a laminated block of sticks, 31-ply UD.

(The Bridge_v6 numbers below are from the round-14 code. Round 15 changed what touches
them only slightly and has not been re-run on Bridge_v6, whose file was not at hand: its
period under the plate, ~8 ms, is below the 10 ms reference, so its jack speeds are
unchanged; a load drop now has to last that period before it is the failure; a joint
carrying under 2 % of F_u no longer governs; the split check weights its averaging window
by overlap; and the placeholder priors are per value, wider for the glue (§3.6).)

**Five seeds** (imperfections and scatter; each seed is the same imperfect bridge at
every mesh, its random values being keyed on the CAD, not the mesh):

| mesh (element size) | F_u by seed 0 / 1 / 2 / 3 / 4 (kgf) | mean |
|---|---|---|
| quick (x1.6) | 2006 / 1986 / 1671 / 2161 / 2041 | 1973 |
| standard (x0.75) | 2251 / 2120 / 1887 / 2318 / 2061 | 2127 |
| fine (x0.5) | 2282 / 2097 / 2042 / 2403 / 2289 | 2223 |

| seed | standard: what governs (`competing`) | fine: what governs (`competing`) | fine vs standard |
|---|---|---|---|
| 0 | joint: cross-beam \| Deck Stringer (6) | the same joint (5) | +1.4 % |
| 1 | Hub Core splits (2) | Hub Core splits (0) | -1.1 % |
| 2 | joint: cross-beam \| Deck Stringer (3) | joint: End Post \| Heel Cheek (3) | +8.2 % |
| 3 | Hub Cheek splits (7) | Hub Core splits (1) | +3.7 % |
| 4 | Hub Core splits (6) | Heel Cover cracks (8) | +11.1 % |

* **Mesh**: the fine mesh is +4.6 % above standard on average (-1.1 % to +11.1 %, seed
  by seed), the same part governs on 2 of 5 seeds, and on the others another of the
  competing sites (`competing`, 2-8 parts within 10 % of F_u) goes first. Read F_u at
  standard with a discretisation uncertainty of about 5 %, on the low (safe) side. Round
  13, before the round-14 physics, x0.75 was within 2.5 % of x0.5 on two seeds. The
  quick mesh is 1-11 % below standard (mean 7 %): for ranking designs on one mesh. The
  load curves of the meshes coincide up to ~90 % of the peak.
* **The strict reading** (`Fu_first_joint_failure_kgf`, the first joint to separate):
  622 / 735 / 458 / 734 / 671 kgf at standard, 584 / 672 / 436 / 698 / 631 at fine
  (within 10 % of each other), 824-989 at quick: read it at standard or fine.
* Every run passed the quasi-static check (kinetic / external work < 0.1) with < 0.3 %
  added mass.

**The uncertainty of F_u** (`montecarlo -n 20 --fidelity standard
--sample-placeholders`: seeds 1-20, each also drawing the nine placeholder material values
around their placeholder, lognormal 20 %: E, f_t, f_c, f_v, f_c90, tau_g, the glue's
G_IIc, G_f, the end-grain factor): F_u p05 / p50 / p95 = 1720 / 2095 / 2317 kgf (min
1679, max 2432), -18 % / +11 % around the median; the Load Prediction Form value (1/F
weighted) 2069 kgf; judged at the first separated joint, median 666 kgf. Where the
collapse starts, by part over the 20 runs (`governing_parts`): the Hub Core splitting 7,
the End Post joints to their covers and gussets 5, the top rail plates splitting 2, the
End Pieces splitting 2, four others once. That is the failure site as a distribution:
the hub core and the End Post joints are the bridge's weak spots, each going first on
about a third of the builds. The placeholder values move F_u less than the build
scatter: the range is set by the imperfections and the joint-to-joint glue scatter as
much as by the material values.

**What to measure first** (`sensitivity --fidelity quick --seeds 2`, each value -20 % and
+20 %, F_u averaged over two seeds; elasticity = % change of F_u per % change of the
value): the lap joints' glue strength tau_g 0.19, the glue's shear toughness G_IIc 0.15,
the sticks' shear strength f_v 0.13, the stick modulus 0.11, f_t 0.08; the end-grain
factor, the wood's fracture energy, the glue's creep and shear stiffness, the rolling
shear, f_c and the laminate bond line within the noise of two seeds (|elasticity| <=
0.03, ~2 % of F_u). Lap joints (strength and load-slip) and stick shear are the tests that
would narrow the prediction most.

**A nearly perfect bridge** (`--seeds 0 --amplitude 0.1`: every imperfection and scatter
of seed 0 at a tenth): F_u 2353 / 2247 / 2402 kgf at x1 / x0.75 / x0.5, the Hub Core
splitting first at all three meshes (far, far, near side), the first joint separating at
725 / 778 / 724 kgf: standard is 4.5 % below x1 and 6.5 % below x0.5. (A perfect bridge
is symmetric: which of its identical sides gives way first is a bifurcation that
round-off decides; a tenth of the imperfections lets the seed decide.)

**x0.5 and local crack onset.** A plate left as a shell cracks where its strain first
reaches the strength in one element: the crack band makes the *energy* of a crack
independent of the mesh, not *when it starts* at a sharp stress raiser, whose resolved
peak grows as elements shrink (the known limit of a local damage model, as in LS-DYNA's
MAT_054 or Ansys' progressive damage). Until round 12 this showed at x0.5 (round 9: F_u
-9 %), and x0.75 was the finest fidelity. Since the narrow plates are frames (round 11)
the stress raisers that mattered are no longer resolved by shells, and x0.5 is not
weaker than x0.75 (above: +4.6 % on average); on one seed of five a shell (a heel cover)
cracks first at x0.5, at a load above standard's F_u. (A first x0.5 run gave 2088
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

**Time compression and the rest** (standard, seed 0):

* **Rate check** (`crush --fidelity standard --seed 0 --rate-check`): F_u 2251 kgf at 100
  mm/s, 2226 kgf at 50 mm/s (-1.1 %); extrapolated to a jack at rest 2200 kgf. The glue
  creep runs in physical test time (the jack's travel at the real 10 mm/min), so it is
  the same at both speeds.
* **The load curve**: along the jack travel, no step: the largest change within 0.01 mm
  of travel before the failure call is within the bridge's stiffness x 0.01 mm + 1 %.
  It bends over before the peak: the secant stiffness from 50 to 90 % of the peak is
  10 % below that from 10 to 50 % (quick, seed 0: 389 and 433 kgf/mm).
* **Stick modulus**: with the material file's 10 GPa instead of the 7.1 GPa crush
  placeholder, standard seed 0 gives F_u 2302 kgf (+2.2 %) and a deflection at F_u of 4.52
  mm instead of 5.81: the strength is set by the joints and the splits, the deflection by E.
* **What limits the bridge** (`diagnose -n 2 --fidelity quick --candidates 3`, seeds 1
  and 2): strengthening the inner hub tie straps (section x1.25) raises F_u by 2.8 %
  (+1.3 % and +4.4 %), their laps to the hub spreader (glue x1.5) by 2.8 %, the hub
  cheeks' joints to the deck rib by -0.4 %: no single part carries the bridge; the hub
  and the End Post joints share it (above).
* **Design loop**: the `strengthen` what-if of a quick crush of Bridge_v6 (seed 0: the
  hub cheeks and covers x1.25 in section, the hub cheeks' joints to the deck rib x1.5
  glue, on every copy), tested with `ab -n 3 --fidelity quick`: F_u +3.3 % (+5.3 %,
  +1.1 %, +3.6 %), 90 % interval -0.3 % to +6.9 %: stronger on all three seeds but not
  proven; the failure moved on all three (`failure_moved`).

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
beam theory within 3 %). Since round 15 the screen takes faces that touch but are not
glued (glue 0: a cut end left dry) as bearing in compression only, as the crush does;
before, it listed them as glue limits at ~1 kgf, and a team's design report had to
footnote it (`test_screen_takes_unglued_faces_as_bearing_only`). And the plate rests on
the deck points at or below its underside: it took the highest points under its
footprint, and on a truss with a strut over the deck balanced on that strut's line.

### 3.6 Soft and light bridges: the jack speed (round 15)

Bridge_v6 is stiff and heavy: its lowest mode under the 4.2 kg plate rings in ~8 ms and
it fails after ~6 mm of jack travel, so a jack at 100 mm/s moves 0.8 mm per period and
the crush is quasi-static (rate check -1.1 %). A light truss or a soft beam rings ten to
twenty times slower, and the same jack outran it. Three synthetic bridges
(`bridgesim.cad.synthetic`), crushed with the round-15 jack:

* **A light Pratt truss** (`layered_truss()`: two trusses of 3-ply members, chords,
  verticals and diagonals in three glued layers, 1 m span, 104 mm apart so the plate fits
  between them; period under the plate 74 ms). Quick fidelity, seed 0, `--rate-check`,
  as each round-15 change came in:

  | jack | F_u at speed / half speed (kgf) | change | extrapolated to rest |
  |---|---|---|---|
  | slowed in proportion to the period, at most 4x (50 / 25 mm/s) | 12.98 / 12.17 | +6.6 % | 11.36 |
  | + the plate damped relative to the jack, not to the platform | 12.45 / 11.90 | +4.5 % | 11.36 |
  | + started over two periods, sped up after a hold over one | 12.50 / 11.88 | +5.3 % | 11.25 |
  | + at most 16x, speeding up with the travel (27 / 13.5 mm/s) | 11.93 / 11.53 | +3.5 % | 11.13 |

  The jack-at-rest estimate holds at 11.1-11.4 kgf throughout while the run at speed
  comes down to it; the last run passes the 5 % rate criterion (56 s on 2 cores). It
  fails where such a truss does: the end-grain glue of the top struts on the verticals
  lets go at 3.9 mm of travel, and the far truss, no longer braced, rolls over.
  **The final truss** (floor beams resting on the bottom chords mid-panel; until then
  they hung from the verticals by end-grain glue, and the floor barely held the 4.2 kg
  plate: one run collapsed under it at 0.1 kgf of jack load), with glued joints that
  resist turning (§3.7), rings in 45 ms under the plate. Seed 0, with `--rate-check`:

  | fidelity | jack | F_u at speed / half speed | change | governs |
  |---|---|---|---|---|
  | quick | 44 / 22 mm/s | 14.0 / 13.6 kgf | +3.3 % | the deck's glued edges to the middle verticals |
  | standard | 24 / 12 mm/s | 34.3 / 34.5 kgf | -0.5 % | a floor beam's joint to a chord |
  | fine | 12 mm/s | 39.8 kgf | - | the deck's edges; a floor beam splitting competes |

  The jack is quasi-static at every fidelity (the rate checks), but **the mesh is not
  converged for a bridge this small**: standard is 14 % below fine, and quick, whose
  48 mm elements are coarser than the deck is wide, finds a different and much weaker
  failure (a deck of four elements across at quick does not change it: 12.6 kgf). The
  element sizes were set on Bridge_v6, whose members are 20-30 mm deep and plates
  hundreds of mm across; this truss's members are 10 mm wide and its glued joints 40-60
  mm^2. For such a bridge read F_u at fine, and rank designs at standard, not quick.
  (Before the truss was made to fit the plate it stood 86 mm apart: the plate started
  inside its verticals and threw them off; such a design now gets an important
  `plate_fit` question. Its diagonals had been cut 12 mm short at both ends and held by
  ~2 mm of glue: F_u 5.4 kgf, the inspect `weak_joints` question naming them.)
* **Two stringers over a metre with a deck** (`simple_beam_bridge()`, period 165 ms).
  At a fixed 150 mm/s it "collapsed" at 5.7 kgf with kinetic / external work 0.12 (not
  quasi-static); the jack slowed for its period carries 8.8 kgf at 30 mm of travel and
  has not failed. Started at full speed, the plate's kick set it ringing by ~4 mm: its
  load against deflection started at 5.5 kgf (14 kgf at 600 mm/s) instead of zero; now
  at 0.1-0.3 kgf, the slope the same at 37.5 and 150 mm/s within 1.5 %
  (`test_a_soft_bridge_is_loaded_without_a_kick`).
* **A laminated beam on two piers** (period 79 ms; `test_simply_supported_beam_matches_
  beam_theory`): it bends to the 50 mm limit at 142 kgf without breaking, and is
  quasi-static at 100 mm/s (kinetic / work 0.0006). Slowed 4x for its period alone,
  its 60 mm of travel took 2.4 s of simulated time, more than the 400 000-step budget:
  the run stopped at 30 mm and reported no failure (fixed: the budget follows the
  slow-down, and a run that reaches it says so). With the jack speeding up past 6 mm x
  79 / 10 of travel, the test runs in 77 s instead of 110 s.

### 3.7 Bridge_v10: a team's Rev N, and joints that turn (round 15)

`examples/cad/Bridge_v10.SLDPRT` (a team's Revision N, kept out of the repository like
Bridge_v6): a Pratt-like truss of 13-ply members, 1050 mm between pier centres, 535 mm
high, 7.90 kg; the floor beam under the jack (a block) bears through tenons on sills in
node blocks at the centre node M. The overrides follow the team's design report
(section 10.1): the cut ends of the bottom chord and of the deck plies at the field
joints touch but are not glued (glue 0), and the heel seat with its chord and the floor
beam with its tenons are one piece (glue x10). The reader found 69 of the 70 bodies the
file names (one half of a portal X brace could not be put together; inspect now says
so, `cad_unread`). Its lowest mode under the plate rings in 6.7 ms, so its jack speed
and ramps are those of a stiff bridge.

* **Screen** (seconds): 674 kgf/mm under the plate; the first glue limits at 340 kgf
  (the deck's bottom plies to the inner centre plates) and 449 kgf (the diagonal feet in
  node M), which the team's report also lists (496 kgf). Until round 15 the dry cut
  ends came first, at ~1 kgf, and the report had to footnote it.
* **Crush, standard**, seeds 0 and 1:

  | model | seed 0 | seed 1 |
  |---|---|---|
  | previous release (round 14) | 1360 kgf | 1333 kgf |
  | round 15, each small glued patch one point (a hinge) | 1397 | 1332 |
  | round 15, two points per side at the cell midpoints | 1534 | 1876 |
  | round 15, three points per side at the cell midpoints | 1784 | 1899 |
  | **round 15, two points per side, exact second moment** | **1769** | **1895** |
  | round 15, three points per side, exact second moment | 1778 | - |

  The rest of round 15 (jack, holds, drop rule, plate damping, split window) moves this
  stiff bridge by less than 3 % (rows 1 and 2). What moves it is how a glued patch
  smaller than an element resists turning: the diagonal feet glued into the stub at
  node M (475 mm^2 each, end grain) were hinges, and the node's glue lines peeled
  apart around them. With the patch's own turning stiffness the answer converges (two
  and three points per side within 0.5 %). Seed 0 then fails at node M of the near
  truss (the diagonal feet into the stub), seed 1 by the deck cracking under the plate
  (all three layers of the middle panel) with node M holding: two failure modes within
  7 %. The deck deflection at F_u is 2.7 and 3.3 mm.
* **The spread** (`montecarlo -n 8 --fidelity standard --sample-placeholders`: seeds
  1-8, each also drawing the nine placeholder material values with their own priors):
  F_u 932 to 2297 kgf, median 1641; with the mesh and model-form error, 865 to 2507 kgf
  (p05 / p95); the Load Prediction Form value 1503 kgf (1/F weighted), 1075 kgf judged
  at the first separated joint. Where it starts (`governing_parts`): the diagonal feet in
  node M on 4 of 8 runs, the deck under the plate on 2, the end post's joint to the
  gusset at node A on 2. The range is set mostly by the glue values not yet measured
  (lap strength +-40 %, toughness +-50 %): lap-joint tests would narrow it most.
* **Against the team's own runs.** The design report gives 1632 and 1548 kgf (bridgesim
  v0.2 on a scripted 70-body model of the same design). On this SLDPRT with the report's
  settings the previous release gives 1360 and 1333 kgf: the models differ (the missing
  brace half, and settings the report does not list), so the report's numbers are not
  reproduced here. Both agree on where it fails: at node M of one truss.

### 3.8 Round 16: the load reading, plates on plates, gussets that are plates

Three things a review of round 15 measured on Bridge_v10, now changed:

* **The reported load cut at breaks.** The raw jack force of standard seed 0 (round 15
  model) rose smoothly to 1879 kgf, held there while node M let go, then fell; the
  reported F_u was 1769 kgf (-6 %) and the governing joint's break read 1393 kgf, as if
  after the peak, because the 0.9 mm smoothing window reached across the drop. It no
  longer does: each stretch between breaks is smoothed on its own. The strict reading
  is the highest load up to the first separation (it was the load at that moment, on the
  falling branch: 1075 kgf for the Load Prediction Form against 1503).
* **Plates glued face to face meshed node on node.** Bridge_v10's three-layer deck under
  the plate (a tongue and groove into the middle panel) hung its glue and its layers on
  nodes off-centre: the stiffness under the plate was 360 / 623 / 737 / 816 kgf/mm at
  quick / x1 / standard / fine, now 838 / 819 / 836 / 833.
* **The centre gussets are plates.** plate_M (470 x 140 mm with 60 mm tails) was a
  single beam (a "slender" frame by its median width, set by the tails); a plate counts
  as slender now only if 90 % of its arms are at most three elements wide.

Bridge_v10 with the round-16 model (the same file and settings as §3.7):

| | quick seed 0 | standard seed 0 | fine seed 0 | standard seed 1 |
|---|---|---|---|---|
| F_u (reported / raw peak) | 2210 / 2213 kgf | 2154 / 2154 kgf | 2127 kgf | 2532 kgf |
| secant stiffness 10-50 % | 798 kgf/mm | 793 kgf/mm | - | - |
| deflection at F_u | 3.0 mm | 2.9 mm | 2.9 mm | 3.7 mm |
| governs | End Post R Far to its gusset at node A | the same joint | the same joint | End Post L Near to its gusset at node A |
| strict reading | 2210 | 2154 | 2127 | 2532 |

The mesh is converged on seed 0: standard is 1.3 % above fine and quick 3.9 %, all
three failing at the same joint at 4.1-4.2 mm of jack travel. In round 15 quick was 4.6x
softer, had 2.9x the deflection and failed elsewhere, and on seed 1 fine was 3.4 %
below standard (round-15 model): standard is not on the safe side of fine, on this
bridge it is 1-3 % above it. The failure
moved from node M (round 15: the diagonal feet) to node A, where the end post's glue to
the gussets lets go: with the gussets as plates, node M carries its load through them,
as a review found by setting them to shells by hand (2065 kgf, node A, with the old
reading). F_u is higher than round 15's 1769 / 1895 kgf by the reading (+6 %), the
gussets and the deck; the seeds differ by 17 %. With the round's last changes (end
grain by the angle of the cut, copies, draws) standard gives 2156 and 2529 kgf on seeds
0 and 1, at the same joints.

The spread with the round-16 model and its last changes (end grain by the angle of the
cut, the wood's strengths drawn with its modulus, G with E; `montecarlo -n 8 --fidelity
standard --sample-placeholders`): F_u 1323 to 2674 kgf, median 1730; with the mesh and
the model-form error (now 25 %), 1043 to 3359 kgf (p05 / p95); the Load Prediction Form
value 1699 kgf, and the strict reading the same (the first joint now goes at the peak in
every run: in round 15 the strict prediction was 1075 kgf, read on the falling branch).
Where it starts, by part and its copies (`governing_parts`, copies now grouped by the
words naming their side): the End Post's joints to the node A gussets on 4 of 8 runs,
the diagonal feet at node M on 2, a gusset or centre-plate crack on 2 (one of them
flagged: the crack starts in elements larger than the crack band). The range is still
set mostly by the unmeasured glue: lap-joint tests would narrow it most.

