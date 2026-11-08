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


### 3.9 Round 17: fit-up, bearing, butt joints by statics, literature values

Four reviews of round 16 (realism, adversarial, ambitious, literature) led to these
changes; each has a test in `tests/test_sim.py`:

* **Fit-up gaps.** Each end-grain face (a member's end bearing or glued on another part)
  stands off its seat by a seeded gap, 0 at the edge that touches up to 0-0.3 mm at the
  other (`scatter.bearing_gap_mm`), dry where it is wider than PVA bridges (0.15 mm,
  `glue.gap_fill_mm`). With the round-16 materials it lowered Bridge_v10 quick seed 0
  from 2210 to 1971 kgf (-11 %); with the round-17 model, over seeds 1-3 (`ab -n 3
  --fidelity quick`, gaps against none), F_u changes by +0.1 % (90 % interval -1.5 % to
  +1.7 %) but the failure moved on all three seeds: the gaps decide where Bridge_v10
  gives way more than how much it carries. (The realism review's probe, every knee face
  dry with a uniform 0.3 mm, gave -22 %: a cut end out of square touches at one edge and
  bears from the start, with ~15 % of its face dry on average.)
* **Bearing crushing is reported** (`bearing_crushing`, a face crushed 0.1 mm deep on
  average) and the screen gives each face's peak bearing utilisation: on Bridge_v10 the
  End Post's faces on the knees are the first to reach the wood's crushing strength
  (~1900 kgf if the bridge stayed linear), then the floor beams on the pier legs.
* **Butt joints by statics.** A brick pattern repeats every stick, so every other ply is
  cut at the same section: a 13-ply member keeps 6/13 (46 %) of its tension strength,
  not the 12/13 the round-16 formula gave (the adversarial review's finding: on one of
  round 16's Monte Carlo draws -39 %). The weakest path through the plies is found
  exactly (`butt_path_capacity`), and the glue between plies shears at the laminate's
  interlaminar strength, as where members split. Stick plates longer than a stick get
  the same reduction of their fibre tension strength (a deck about half).
* **Literature values** (the research review): the across-grain values from 2 mm birch
  veneer data (f_t90 1.5, f_c90 6, f_v 8, rolling shear 3 MPa), the glue's shear
  toughness 1.2 N/mm (pressed PVAc 2.1, wood in mode II ~1), peel 0.66 x shear capped
  at the wood's tension across the face, the end-grain factor at most 1, wider priors
  (the stick modulus 35 %: clamped tests on craft sticks 5-7 GPa, veneer 14.6; the lap
  joints' glue 50 %). f_t and f_c are unchanged: the veneer's failure strains at the
  craft sticks' measured E give 40 and 24 MPa.
* **Elements past the crack band** have their strength lowered so they dissipate G_f
  (they kept it and dissipated up to 3.5x G_f; with f_v 8 MPa the shear crack band is
  ~14 mm, smaller than most standard elements).
* **The load reading**: a local linear fit (the reading at a break was pinned to the
  raw, ringing sample) and, from each break through its hold, the load at the break
  plus the raw force's change since, over the period before (each event in a hold reads
  its own moment). The rate check extrapolates along the square of the speed.
* **Montecarlo**: untrustworthy runs left out of the range and the prediction, the
  recommended prediction over the runs times the model-form error with a bootstrap
  interval, a rigid block under the plate named in the note; the kinetic-energy limit
  of a quasi-static run is 5 % (Abaqus/Explicit's guideline).
* Verification against closed-form fracture mechanics: simulated laps of 60 and 110 mm
  carry the share of tau x area generalised Volkersen theory gives, within 12 %
  (0.68 / 0.70 against 0.64 / 0.77 at 110 mm for G_IIc 0.6 / 1.2 N/mm).

What each change did to Bridge_v10 (quick seed 0, the round-17 model with one change
backed out at a time; one seed, so a change smaller than the ~20 % between seeds is
only indicative):

| round-17 model | F_u (kgf) | governs |
|---|---|---|
| all changes | 2383 | a crack in the centre plate plate_M (in elements past the crack band: flagged) |
| without fit-up gaps | 2290 (-3.9 %) | the floor beam's joint to a pier leg |
| G_IIc 0.6 (round 16) | 2185 (-8.3 %) | End Post to its gusset at node A |
| f_c90 5 (round 16) | 2335 (-2.0 %) | End Post to its gusset at node A |
| peel 0.5 x shear, f_t90 1 (round 16) | 2254 (-5.4 %) | the floor beam's joint to a pier leg |
| f_v 6, rolling shear 2 (round 16) | 2383 (0 %) | the same plate crack |
| no butt joints | 2538 (+6.5 %) | End Post to its gusset at node A |

The round-16 model gave 2210 kgf here: the literature values of the glue (toughness,
peel) raise F_u by about as much as butt joints by statics lower it. At standard seed 0
the round-17 model gives 2457 kgf (round 16: 2156), trustworthy, governed by node A
again (End Post R Far to its inner gusset), with the plate_M cracks and the diagonal
feet at node M competing within 10 %.

### 3.10 Round 18: the reading, the attribution, the stick modulus

Four reviews of round 17 (realism 5/10). What they found, measured on the round-17 model,
and what changed:

* **The mesh at one jack speed.** Bridge_v10 seed 0 at 100 mm/s: quick 2337 kgf (a floor
  beam's joint to a pier leg governs, the plate_M cracks compete), standard 2457 kgf (the
  End Post's gusset at node A): quick is 4.9 % low and fails elsewhere. The round-16
  agreement of the three meshes (2210 / 2154 / 2127 kgf) was at 200 / 100 / 50 mm/s, the
  jack speed changing with the mesh. AGENTS.md said they agreed within 1 % at one speed;
  that came from round 16's model and is withdrawn.
* **The trust flag on plate cracks** looked at every mode: with f_v 8 MPa nearly every
  plate element was past the shear crack band, so every plate crack was flagged, also
  plate_M's, which is not a shear crack (halving G_v or lowering f_v left F_u exactly as
  it was). The §3.9 table's "flagged" for the round-17 model, and 5 of the 6 quick runs
  of the fit-up A/B, were flagged for that reason. Each crack now records its mode, and
  only a crack in elements past the band of that mode is flagged.
* **The reading** could stand above the force the jack applied (a Monte Carlo run 1553
  against a raw maximum of 1495 kgf; a light truss +6.6 %) and, in a hold, half a period
  late, which made a joint on the other truss "govern" standard seed 0 after the centre
  plate and node M's diagonal feet had started the fall. The reading is now kept within
  the raw force's envelope, and the governing break is found on the raw force.
* **tau_peak_over_avg** was not recomputed when round 17 changed the glue's toughness
  and peel: every joint was ~5 % stronger than tau_g (`bridgesim lab`: 2.10 MPa against
  2.0). It is 1.10 now, and its test uses the simulation's own glue values.
* **The stick modulus** (the literature review): the 7.1 GPa the crush used came from a
  stick weighted at a table edge, and the 5.0 GPa cited beside it from sticks soaked in
  water; dry craft sticks measured by vibration are 11.6-14.8 GPa. The crush uses 11 GPa,
  and the Monte Carlo draws E around it (it drew around the material file's 10 GPa while
  every crush used 7.1). With it, the fracture energies (fibre tension 3 -> 8, across the
  grain 0.3 -> 0.45, in-plane shear 1.0 -> 1.3 N/mm), the bond line between plies (6 -> 4
  MPa: craft-stick laminates glued with Elmer's split at 3.7 MPa), a strength taper for
  glue in a fit-up gap (dry only from ~1 mm, not from 0.15) and the model-form error (25
  -> 30 %: the model moved 14-30 % per release).
* **Smaller**: the screen's supports slide where the thrust is more than friction holds
  (a light truss read as an arch, 4 % stiffer); quick is flagged on a light bridge
  (members under 15 mm deep: the light truss read 49 % low); `ab`'s verdict says when its
  runs are untrustworthy; the Monte Carlo draws the values that set where a bridge fails
  (f_t90, the bond line, G_v, peel, the fit-up gap); the screen gives the bridge's tap-test
  frequencies (Bridge_v10: 47.8, 54.9, 76.5 Hz ...; a beam on knife edges within 5 % of
  Rayleigh's); Bridge_v10's butt joints follow the team's report (designed for 75 % of the
  tension strength: 40 mm between the joints of neighbouring plies, 73 % by statics).

The round-17 Monte Carlo, stopped after 3 of 8 runs when its model was superseded: 2737,
1553, 1452 kgf.

Bridge_v10 with the round-18 model (the team's butt-joint stagger, 40 mm; the same file
and other settings as §3.7):

* **Standard seed 0**: F_u 2483 kgf (round 17: 2457), trustworthy, at node A: the End
  Post R Far's glue to its outer gusset, the inner one competing; the End Post's face on
  the knee starts crushing just before (2473 kgf). Deflection at F_u 2.4 mm, the screen's
  stiffness 1134 kgf/mm (the sticks at 11 GPa), its lowest tap-test mode 47.8 Hz.
* **The mesh at one jack speed** (seed 0, 100 mm/s, `convergence_study.py --overrides
  --speed 100`): quick 2506, standard 2483, fine 2479 kgf, all three at the same joint
  with the same competitor, the deflection at F_u 2.4-2.5 mm: standard is 0.2 % above fine
  and quick 1.1 %. (With round 17's model quick had been 4.9 % low and failed elsewhere.)
* **The rate check** (standard, seed 0): 2445 kgf at 50 mm/s against 2483 at 100 (-1.6 %),
  the same joint; extrapolated to a jack at rest along the square of the speed, 2432 kgf.
  Read F_u at standard with ~2 % for the jack's speed and ~1 % for the mesh.
* **Montecarlo** (`-n 8 --fidelity standard --sample-placeholders`, now 16 values drawn):
  F_u 1226 to 3299 kgf, median 1954, every run trustworthy; with the mesh and the model-form
  error (30 %) 939 to 3997 kgf (p05 / p95). The Load Prediction Form value is 1545 kgf
  (with the model-form error; 1871 over the runs alone), and other runs could move it
  between 1294 and 1908 kgf (bootstrap): eight runs do not pin it. Where it starts: the
  diagonal feet at node M (Stub) on 2 runs, node A's gussets on 2, the knee diagonals'
  glue to the centre plates on 2, a node block splitting on 1, a centre plate cracking on
  1. Six failure sites within one design: which goes first depends on the glue and the
  build, which only tests settle.
* **The light truss** (`layered_truss()`, members 6 mm deep, seed 0) with the round-18
  model: quick 17.9 kgf (a deck joint; flagged: the mesh too coarse for it), standard 34.5
  (a floor beam splitting in shear), fine 44.6 kgf (the other floor beam splitting): fine
  is 29 % above standard (round 15: 14.0 / 34.3 / 39.8). Not converged at standard: since
  round 18 a crush whose elements are more than 3x its members' median depth is flagged
  (quick and standard here, not fine; Bridge_v10's members are 20 mm deep, so not there).

### 3.11 Round 19: one-piece parts, the size of a lap, gaps in compression, stick sheets

Four reviews of round 18 (realism 5/10: validation -2.5, placeholders -0.75, compact
parts -0.75, solution verification -0.5, an idealised curve -0.5) and what changed:

* **One piece drawn as two bodies** (`glue: {"A | B": solid}`): the face between a
  floor beam and its tenon, or a heel seat and its chord, breaks at the wood's own
  strength (shear f_v / 1.5 averaged, as the code check of a notched end; tension across
  it) instead of 10x glue, which made it unbreakable. On Bridge_v10 the tenon roots carry
  about a quarter of the jack load each in the model (the floor beam also bears on the
  parts it touches) and reach their limit at ~6900 kgf in the linear screen: not
  governing, but no longer impossible.
* **Larger laps are weaker** (a Weibull size effect, m = 6, drawn 4-10 by montecarlo):
  the 1320 mm^2 gusset laps at node A hold 0.73 of tau_g; the 200 mm^2 coupon tau_g is
  measured on still reproduces it.
* **Butt joints in compression bear only if their gap closes** before the member
  crushes (~0.1 mm) or glue fills it (0.15 mm): seeded per member from 0 to 0.3 mm, so
  about half the laminated members keep their butt-joint factor in compression too (73 %
  for Bridge_v10's 40 mm stagger); `butt_gap_mm: 0` for ends sanded tight.
* **Stick sheets with unglued edges** (0/90 plates, the default; asked as `plate_edges`)
  carry in-plane shear like cross-laminated timber with unglued narrow faces: x0.68 of
  the shear stiffness, half the shear strength.
* **Bearing is capped** at the wood's crushing strength along the grain (a face pressed
  on end grain plateaus; densified wood across the grain stops there): it hardened
  without limit, up to ~10 f_c, and propped up the Monte Carlo's top runs.
* **The reading's envelope stops at a break** (a bounce after a joint let go read a
  light truss 4 % high); **values typed under params stay placeholders** (still drawn by
  montecarlo) unless listed under `measured:` (calibrate writes that); the crush's 10.5
  GPa replaces the material file's E only while it holds its shipped value; **the
  stick modulus** is 10.5 GPa (a static value, 5 % below the low end of dry sticks by
  vibration); **montecarlo also draws** the glue lines' stiffness and the size-effect
  modulus; a part drawn a little off its neighbour is tied by breakable dabs of glue and
  asked about as important (it was an unbreakable tie); a joint alone between two bodies
  keeps its random draw when an edit moves it; the coarse-mesh flag looks at the members
  that fail; calibrate fits tau_g through the measured glue law and takes a toughness only
  from a record with a softening branch.

Evidence on round 19's model (Bridge_v10, the team report's overrides with the tenons and
heel seats `solid`, joints 40 mm apart): at 100 mm/s, quick / standard / fine read
1944 / 1850 / 1880 kgf on seed 0 and 2009 / 1839 / 1892 on seed 1; standard and fine
failed at the same joint (a diagonal's foot on the stub at node M, with its two gusset
laps competing), fine +1.6 % and +2.9 % above standard; quick 5-9 % high, failing at a
competing node-M lap on seed 0. The size effect had moved the failure from node A (the
End Post on its knee, round 18) to node M. An 8-run Monte Carlo drew the glue's strength
below its placeholder in every run (seeds 1-8, a 1-in-256 chance): its placeholder draws
are now a Latin hypercube over the runs (each value once in each of n equal-probability
strata of its prior). Its rerun and the rate check were stopped for round 20, whose
literature changes move every number again.

### 3.12 Round 20: the literature review's corrections

The literature review of round 19 checked round 19's numbers against primary sources
and found four of them wrong in the direction of a weaker bridge, and the glue law's
shape missing:

* **Butt joints bear in compression up to 0.5 mm.** Berard et al. (2011, J Wood Sci
  57:107), the source round 19 cited for "1 mm gaps never closed", found 0.25 and 0.5 mm
  gaps in 2 mm veneer laminates closing early after first yield; tight butt joints keep
  0.94-0.96 of the compression strength (Dansoh et al. 2003). Members keep 0.95 up to 0.5
  mm, their tension factor from 1 mm, linear between (`wood.butt_*`): hand-cut gaps
  (0-0.3 mm) now nearly always bear; rounded stick ends (~2 mm) do not.
* **Butt joints in tension** carry 0.8 of the statics (`wood.butt_tension_efficiency`):
  in stick-thin plies the glue cracks from the joints' tips and the fracture runs into
  the plies (Jung & Murphy 1983; Dansoh et al. 2003: 0.53 against 0.67 of the section).
* **The lap size effect is milder and bounded**: m 10 (6-20) instead of 6, carried no
  further than 3000 mm^2 (Flaig & Blass 2013: none on pressed crossings of 1600-29,000
  mm^2); Debertolis et al. 2025's "1-3 MPa" were design inputs, not measurements
  (coupons cut from their gussets held 5.1 MPa, CoV 0.27). **Plates glued face to face**
  (Bridge_v10's deck plies, 71,370 mm^2) are laminations, at the interlaminar strength (4
  MPa) and a pressed line's stiffness, not hand-made laps at 0.75 MPa.
* **2-ply stick sheets** whose edges are not glued also fail by torsion of the glued
  crossings (Boggian et al. 2019): 0.83 x the crossing's strength, 3.3 MPa against half of
  f_v, 4.
* **The glue law bends before its peak** (trilinear: straight to 0.65 of the strength,
  peak at 1.5 x the straight line's slip, the same G_c): a pressed PVAc line bends from
  60-70 % of its peak and peaks at 0.05-0.07 mm (Serrano 2000). The lap coupon now needs
  a peak/average ratio of 1.075 (1.10 bilinear).
* **Stiffness** in `compare` is the 10-40 % secant (EN 12512, as Debertolis et al. report
  glued trusses), on the straight part of the glue law.
* **Plausibility.** Troitsky results with a mass: BCIT 2025, 7.2 kg, 1587 kgf
  (efficiency ~220; second place 1111 kgf); McMaster 2025, ~6.35 kg, > 272 kgf; the
  2026 record, U of T, 4889 kgf (mass not published, predicted 4000 by the team). Glued
  timber trusses tested to failure were "mostly linear elastic until failure, which
  occurred in a brittle manner", their secant at the peak 0.84-0.92 of the 10-40 %
  secant (Debertolis et al. 2025). Nothing published gives a hand-made PVA lap on craft
  sticks a strength (tau_g stays a 2.0 MPa placeholder, drawn 50 %), nor a popsicle-stick
  truss a load-deflection record.

Evidence on round 20's model (Bridge_v10 with the same overrides; runs in the
developer's `r20_runs/bridge_v10/`): at 100 mm/s, quick / standard / fine read
2162 / 2061 / 2093 kgf on seed 0 and 2232 / 2000 / 2069 on seed 1, standard and fine at
the same joint (a knee-diagonal's foot on the stub at node M, the two gusset laps on
that diagonal competing), fine +1.5 % and +3.4 % above standard, quick 5-12 % high;
standard +11 % and +9 % above round 19 (the milder size effect, laminations and butt
joints bearing); standard seed 0 at 50 mm/s 2035 kgf (-1.3 %, the same joint; at rest,
extrapolated, ~2026). The curve bends little before its peak: its secant at the peak is
0.98-1.00 of the 10-40 % secant against the jack's travel and 0.94-0.98 against the
deflection, stiffer than glued timber trusses tested to failure (0.84-0.92, Debertolis et
al. 2025), though 70-100 glue lines pass their own peak locally (the ends of laps) before
F_u, from ~4 % of it (the ambitious and adversarial reviews): their share of the
bridge's compliance is small. At node M F_u is a redistribution: the diagonal's
end-grain foot on the stub reaches its glue limit in the linear screen at ~450 kgf and
sheds its load into the diagonal's two gusset laps, which go at ~2050 kgf. The Monte
Carlo (8 standard runs, the placeholders a Latin hypercube): 1023-2866 kgf, median 1851,
recommended prediction 1386 kgf [1123-1710], total range 816-3321 (node M's foot
governing 4 of 8, its gusset lap 1, a plate_M crack 1, a diagonal 1, node A 1).

### 3.13 Round 21: the glue's time to failure, hand-made laminations, moisture

From the literature review of round 20:

* **The glue's strength follows its time to failure**, tau(t) = tau(t_ref) (t /
  t_ref)^-0.1 (+26 % per decade faster; PVAc +20 % at 16x the rate, Wernersson 1994, Lund
  TVSM-1006; -40 % at a tenth of the rate, Serrano 2000), on the crush's physical test time (the jack's
  travel at `crusher.test_rate_mm_per_min`, 10 mm/min until someone times the crusher),
  against the 60 s of the coupons tau_g describes; a face through one piece of wood
  follows the wood's 0.02. The crush reports `test_time_at_Fu_s`; the glue and
  short-beam templates take `time_to_failure_s` and `calibrate` brings each test to 60
  s; montecarlo draws the exponent (0.065-0.22) and the crusher's pace.
* **Laminations made by hand**: `glue.interlaminar_MPa` 3.5 (4 was Karplus's pressed
  beams, an upper bound; hand clamps cost 5-10 % against the optimum, Rabiej & Behm 1992,
  uneven fit more).
* **The glue law's shape**: peak at 2.0x the straight line's slip (Serrano's lines peak at
  ~2.5x their own initial slope, Wernersson's with a negligible elastic part); drawn over
  0.45-0.75 (onset) and 1.1-3.8 (peak); the lap coupon's peak/average ratio is 1.065.
* **Moisture**: `wood.moisture_excess_pct` (0) for members laminated days before the
  test (a PVA line brings ~+8 points of moisture to a 2 mm ply): E and G -2 %, f_c -5.7 %,
  f_c90 -5 %, f_v -2.4 % per point (USDA Wood Handbook eq. 5-3).
* **Hobby stick bridges** (Boon 2024, garrettsbridges.com: 9 Howe trusses, Elmer's,
  48 h cure): the first failure was always a joint; 4 of 9 survived it (first failure at
  0.65-0.90 of the peak); sanding the glued faces flat raised the mean peak 44 %; the
  peak's CoV 6-19 % per group (log-sd 0.14 pooled), inside the model's ~20 % seed spread.
  A Warren stick bridge failed at 710 N against 1780 N predicted on its gross section:
  its compression diagonal buckled at stick joints on its outer faces (Silva et al. 2026,
  Rev. Ensino Eng. 45:389), which the model's butt joints do not do (they scale strength,
  not the bending stiffness at the joint).

From the adversarial review of round 20:

* **The governing joint among those that break together** is the one that could carry
  the most: on Bridge_v10 the diagonal's end-grain foot on the stub (475 mm^2, under 600
  N of glue) and its two gusset laps let go in the same instant, and the foot was named;
  strengthening it moved F_u +1 %, the laps +8.7 % (quick seed 0). The strengthen what-if
  covers every joint of the governing part that broke with it.
* **The prediction's errors are median-centred** (as the placeholder draws): mean-centred,
  a model-form sigma of 0.3 had put the recommended prediction 4.5 % lower for a
  convention. **The mesh's bias** is taken out of it (fine reads +2.3 % above standard
  over 10 seed pairs, quick 1-12 % above standard: `uncertainty.mesh_bias_*`), leaving the
  5 % scatter; the meshes are not monotone (no Richardson estimate), and the docs no longer
  quote round 18's agreement.
* **montecarlo draws the joint-to-joint glue scatter** (`scatter.glue_cov` 0.30, drawn
  0.17-0.53: workmanship, not the weakest-link scatter of the size effect; 0.12 instead
  of 0.30 raised quick F_u 6-16 %), and `sensitivity` tests it and the glue lines'
  stiffness by default.
* **A joint on a laminate's ply face** is capped at what the laminate passes to its other
  plies (its interlaminar strength x n / (n - 1)): the shells do not delaminate by
  themselves, and a run had drawn a gusset lap at 5.1 MPa on a gusset of 2.5 MPa rolling
  shear. **A lamination** covers at least half the smaller plate: Bridge_v10's deck
  tongues (3240 mm^2, slid in on build day) are laps (4943 N), not laminations (12,960 N).
* **calibrate-bridge** lists the fitted values as measured and sets the model-form error
  to 15 % (it left every value a placeholder at 30 %, so a fit narrowed nothing).
* **compare** matches the part that broke by its body name (a `part` column: a name, a
  glob or `A | B`) or by words without a node's letter or 'foot', 'centre', 'node'
  ('diagonal foot at the centre node, far side, right' scored 0 against the right joint);
  a mismatch no longer advises lowering the glue.
* Mirrored plates list their plies from the same side (facing away from the CAD origin,
  not as round-off picks between their two equal faces: ±0.4 % on Bridge_v10); a plate's
  butt-joint factor takes the weaker direction after its efficiency; the confirm step
  quotes the current mesh numbers.

From the realism review of round 20 (5/10: validation -2.5, placeholders -0.75, compact
parts -0.5, an idealised curve -0.5, the governing label -0.25, light bridges -0.25,
stale docs -0.25):

* **Members' elements at most 2x their depth** at standard (in proportion at the other
  fidelities; members 20 mm deep and more unchanged). The light truss (6 mm members, 12 mm
  elements) read 46.2 kgf at standard against 35.8 at fine (8 mm), failing elsewhere: not
  converged at standard, so such members are flagged below 1.5 depths per element (fine
  passes); read a light bridge at fine.
* **Butt joints cost stiffness**: shear lag round each joint, ~0.88 of E for 13 plies.
* **A block can be given its grain** (`grain: x | y | z`, a direction, or one per layer):
  Bridge_v10's heel seats bear on end grain in 6 of their 13 layers, as the team's report
  lays them (its overrides in `r21_runs`).
* **An end-grain glue line pulled apart** holds 0.1 x the wood's tension strength along
  the grain (~4 MPa; USDA: up to 25 % with special techniques), capped by the other side
  pulled across its face (f_t90 on a chord's side): it held 0.7 MPa (the side-grain peel x
  the end-grain factor), which the published test C below showed far too weak (a
  vertical's end-grain foot carried 2.6 MPa before it broke). `calibrate` takes the factor
  from sticks glued end to end and pulled.

Evidence on round 21's model (Bridge_v10, the team report's overrides plus its heel
seats' layers, `r21_runs/bridge_v10/`): at 100 mm/s quick / standard / fine read 2300 /
2276 / 2271 kgf on seed 0 and 2281 / 2165 / 2235 on seed 1 (fine -0.2 % and +3.2 %
against standard, all trustworthy), failing at node M's gusset (plate_M cracking) or the
diagonal's laps on it, where the adversarial review found the load depends; the real test
would reach the peak ~24 s in (the glue ~9 % stronger than in 60 s coupons). Standard
seed 0: 2061 kgf in round 20, +10 %.

### 3.14 Validation against published tests (rounds 21-24)

No team has crushed a bridge for this repository yet, but three craft-stick trusses were
built, loaded to failure and published with their geometry, stick, glue, load and
failure. `validation/published_tests.py` rebuilds them as bridgesim models from the
papers' drawings and photos (each value tagged stated, drawing, photo or assumed in its
docstrings) and loads them as they were loaded: a rigid hanger on the loaded node (dry on
A's bottom chords, glued to B's putty lumps), C's rebar resting dry on the top chords, the
ends on support blocks (bearing only). Material: the papers' stick sizes and the pine
values they quote, A's superglue from its maker's data (8 MPa, no PVA rate law),
everything else the shipped placeholders (white glue at 2.0 MPa, birch). Each test runs
at its own pace (`crusher.test_rate_mm_per_min`: C's press ~2 min to the peak; A's dead
weights and B's unknown pace leave the default, A's glue having no rate law).

**None of the three is a blind validation.** A's rebuild changed after round 21's
comparison (its hanger, its glue); B's putty stand-in (the end face glued x5) came with
its first comparison, and its governing joint is that one; C chose round 21's end-grain
glue law, and round 22's size effect across the grain gives the 2.6 MPa C's foot held: C
is a calibration case. They are listed for what they show about the model's form.

**As distributions** (round 24, `r24_runs/published`, one frozen code: `montecarlo -n 10
--fidelity fine --sample-placeholders` per case, the members 6-9 mm deep, summarised by
`published_tests.py --summarize`; the realism review of round 22: one seed misrepresented
trusses whose seeds alone spread 2x):

| Test | Glue | Test F_u (kgf) | 10 runs (kgf) | Median / test | Test among the runs | Runs failing where the test did |
| --- | --- | --- | --- | --- | --- | --- |
| A, Lamberti et al. 2025: Warren, 6 x 100 mm, 3-ply, dead weights | superglue | 140-160 | 23-61 | 0.26 | above all 10 | 0 of 10 (test: top chord and a diagonal after a lateral drift; model: the diagonals' laps at the centre) |
| B, Silva et al. 2026: Warren, 4 x 260 mm, 5-8 ply, putty nodes | wood glue, epoxy putty | 72.4 | 40-215 | 1.18 | 4 runs below, 6 above | 0 of 10 (test: a diagonal buckling at its butted plies; model: putty joints) |
| C, Moraes et al. 2019: Howe, 6 x 80 mm, 3-ply, test machine (calibration) | wood glue | 42.9 (text) - 50 (curve) | 12-78 | 0.40 | 8 runs below, 2 above | 4 of 10 at a vertical's foot on the bottom chord (test: the bottom chord and a vertical's foot) |

Round 24's runs are all trustworthy for A (round 23's ten were flagged at 5.0-5.6 % added
mass: a scaling round that would pass 5 % is no longer taken); 3 of B's and 1 of C's are
flagged not quasi-static (kinetic energy 5-8 % of the work as a weak joint snaps) and
left out of their ranges where enough remain. Round 23 (`r23_runs/published`, the joint
faces' constant stiffness) read 0.24 / 1.24 / 0.40, C at a vertical's foot in 3 of 10
(round 23 counted any bottom-chord joint, 7 of 10: the adversarial review of round 23).

Seed 0 alone (round 22, the placeholders as shipped, fine): A 33.5, B 85.6, C 28.2 kgf
(0.21-0.24, 1.18, 0.56-0.66).

What they say:

* **B's and C's loads fall inside the model's range, A's far above it**; the ranges
  are wide (a factor of 3-6 over 10 runs, the placeholders drawn), and the medians read
  0.26, 1.18 and 0.40 of the tests.
* **Two of three failure mechanisms are wrong.** A's test failed in its members after a
  lateral drift, the model at its joints; B's in a diagonal buckling at its butted outer
  plies, the model at a putty joint. Neither can happen in the model as built: A's drift
  needs the dead-weight hanger to swing, which a rigid hanger pushed by the plate does
  not; B's buckling needs the diagonal's outer plies to drop out of its bending stiffness
  at their butt joints near mid-length, which the model's butt joints (a factor over the
  whole member) cannot do: Silva et al.'s own back-analysis with 3 of the 5 plies over
  the length reproduces the test within 3.5-15 % (the literature review of round 22).
  C fails at a vertical's foot on the bottom chord, where its test did, in 4 of 10 runs.
* **A is far below its test**. Its laps are 36 mm^2 of superglue on members crossing at
  ~62 degrees; to carry the test's load they would hold ~12 MPa, above what the maker
  gives for pine with the wood failing (10-12 MPa, parallel laps): either the rebuild's
  laps are smaller than the real ones (the node geometry is read from photos, confidence
  low; superglue wicks along the edges) or the test's joints held more than any
  glue-on-stick lap the literature reports.
* **Stiffness** (C's record, 9.6 kgf/mm between 10 and 40 % of the peak, crosshead
  travel): the model reads 34.8 kgf/mm by jack travel (round 22, seed 0), 3.6 x stiffer,
  and peaks at 0.88 mm of travel against the record's 3.8. The rebuild's flat bar leaves
  out the rebar's indentation into the 6 mm chords across the grain (round bar, f_c90 6
  MPa: 0.18 mm between 10 and 40 % of the peak, 1.3 mm at it), which takes the model's
  secant to ~26 kgf/mm. The record itself is weak evidence of the truss's stiffness: the
  crosshead of a 30 t press over 0.05-0.2 kN (0.02-0.07 % of its range), stiffer at its
  peak (13.2 kgf/mm) than over 10-40 %, the mark of seating; a rigid-joint model of
  full-scale glued trusses read 0.82-0.96 of their stiffness measured on the bridge
  (Debertolis et al. 2025), and glued joints' slip is negligible in a truss's deflection
  (Hansen 1965). The glue lines would have to be ~10x softer to match it, which would
  lower Bridge_v10's F_u by ~17 % (the realism review of round 22): it is not done.
* **The scatter**: over the three seed-0 results the log of fine / test has a mean of
  -0.65 and a standard deviation of 0.87; over the two white-glue tests 0.42-0.52.
  Small trusses (members 6-9 mm deep, joints of 36-90 mm^2), partly read from photos,
  outside Bridge_v10's range (13-ply members, gusset laps of ~2100 mm^2): they set no
  bias to apply to it. `uncertainty.model_cov`, 0.40 since round 22, is a judgement
  informed by them, not a measurement.

Round 21's table (the same rebuilds, with A's hanger glued 40 mm along the chords over
the laps that fail, A at white glue's 2.0 MPa and every test at 10 mm/min) read 0.18,
1.18 and 0.66 at fine on seed 0; its C "improvement from 0.55" compared round 20 at
standard with round 21 at fine (the realism and adversarial reviews of round 21).

### 3.15 Round 22: crossed laps, small joints, the reviews of round 21

From the four reviews of round 21 (realism 5/10: validation -2.5, placeholders -0.75,
an idealised response -0.5, compact parts and laminates -0.5, light bridges -0.25,
validation-claim slips -0.5) and its literature review:

* **Laps whose grains cross hold less** than the coupon's parallel sticks: Hankinson's
  form with 0.38 at 90 degrees, exponent 2.35 (Padilha et al. 2023, pine blocks bonded at
  0-90 degrees: 10.5 / 8.4 / 6.4 / 5.1 / 3.9 MPa), averaged over each side's plies: a
  member on a 0/90 gusset at 45 degrees 0.62, a stringer along a 0/90 deck 0.69.
  Laminations keep their interlaminar strength. This is the change that moves Bridge_v10
  most (below).
* **Small joints pull on little wood**: the tension across the grain that caps peel and
  end-grain joints follows a weakest-link size effect, f_t90 x (4800 mm^3 / V)^(1/8), V
  the joint's area x 2 mm, 1 to 2.5 (birch veneer m 5-11, Kruger et al. 2025; Barrett
  1974): 1.64 x for a 47 mm^2 foot.
* **The glue's scatter no longer lifts the wood**: a joint drawn strong has a stronger
  glue line, not stronger wood behind it (its caps shrink with a draw below 1 only); a
  user's glue_factor (a what-if's bigger area) still scales both.
* **A partial lamination goes over from a lap gradually** (30-70 % of the smaller plate
  covered), not at a 2.5x step at half.
* **A member splits one glue line at a time**: into halves (bending stiffness x 1/4), and
  further only while the shear, which falls with the stiffness, still reaches the
  strength; it went to its separate plies at once (1/169 for 13 plies).
* **The governing label**: when a joint lets go first, the strongest of the joints
  letting go with it governs; a plate crack or member break with no glue capacity no
  longer takes over from joints that broke before it (the adversarial review's proven
  bug: Bridge_v10's node M was labelled `plate_M` cracking 0.1 ms after its laps let go).
* **Light bridges**: the crush reports `quality.min_fidelity` (the coarsest mesh fine
  enough for the members that fail) and its `diagnose` and `predict` steps use it;
  montecarlo says NOT RELIABLE when fewer than half its runs are trustworthy. The
  reading's window is a quarter of the travel to the raw peak on a short rise (never
  under a period of the ringing): 0.9 mm had read 3-7 kgf on test C before the jack
  touched it. The fundamental period takes the deck's sag under the plate, not the
  plate's, when the plate pivots on a line contact (C: 5677 ms, now ~1.4 ms).
* **montecarlo draws** the end-grain tension factor (50 %), the size effect's modulus
  (30 %), the crossed-lap ratio (25 %) and the wood's moisture (additively: max(0, 2 z)
  points, half the runs dry).
* **ab warns** when bodies of B have no namesake in A: each body's and joint's scatter
  is seeded by its name, so a renamed body draws new luck.
* **Docs**: n = 0.1 is +26 % per decade, not +16 %; a lap failing in 10 s reads ~20 %
  above one of 60 s; Bridge_v10 peaks ~17 s into the test (23 s in round 21), not 13; the rate law adds ~5 %
  to Bridge_v10's F_u by convention while tau_g is a placeholder.

Not done, and why: butt joints at their place in a member (the literature review's
calculation gives 0.64-0.84 of the buckling load from the lost stiffness alone, while
test B's diagonal failed at ~0.2 of it, which needs a crookedness of L/50-L/90 that
nobody has measured: modelling the joints would not reach B's failure, and B's model
fails first at its putty joint anyway); deformable blocks; delamination inside a shell.

Evidence on round 22's model (Bridge_v10, the same overrides as round 21, `r22_runs/
bridge_v10/`), 100 mm/s, every run trustworthy and failing at node M (the diagonal's
end-grain foot on the stub lets go, then within 0.06 mm of travel its two laps on the
0/90 `plate_M` gusset, which govern): quick / standard / fine read 1693 / 1651 / 1651
kgf on seed 0 and 1803 / 1638 on seed 1. Standard seed 0 read 2276 kgf in round 21:
-27 %, nearly all from the crossed laps (the diagonal crosses the gusset's plies at
~45 degrees: 0.62 of its laps' strength). The real test would reach the peak ~17 s in.
`montecarlo -n 8 --fidelity standard --sample-placeholders` (`mc8.json`): the runs
spread 807-2757 kgf (median 1697), every run trustworthy, failing at node M in 7 of 8
(the diagonal's gusset laps 3, its foot 2, `plate_M` cracking 2) and at node A once;
recommended prediction 1103 kgf (interval 882-1463), total range with the mesh and the
model-form error (now 40 %) 607-3633 kgf. Round 21's three finished runs read 1150-2288.

### 3.16 Round 23: what a crossed lap loses, each glue line's own clock

From the four reviews of round 22 (realism 5/10: validation -2.5, placeholders -0.75, a
response too stiff and linear -0.75, idealisation gaps -0.5, small bridges -0.25,
validation reporting -0.25) and its literature review:

* **A crossed lap loses what its wood loses, as much as the wood fails.** Round 22 took
  Padilha et al.'s pine blocks (0.38 at 90 degrees, a strong adhesive failing in the
  wood) on every ply pair of both sides; the literature review found that the extreme
  case (birch veneer keeps 0.62-0.72 of its shear strength at 45-90 degrees, Wang et al.
  2024; where the glue line itself gives way the angle matters little), and the
  adversarial review that it alone set Bridge_v10's answer (-27 %). Now 1 - w (1 - f),
  f the face ply's strength at its angle (birch veneer), w the share of the lap failing
  in the wood (0.5, drawn 0-1, so "no effect" is sampled), on the two plies the glue lies
  on (a 0/90 plate's face ply; `90-0` names a plate with a 90 ply on the face its normal
  points away from); peel no longer weakened by it. Rolling shear 4 MPa (birch veneer
  4.6, silver birch 3.5), up from 3.0.
* **Each glue line keeps its own clock**: its strength follows its time under load from
  when it first took a tenth of its strength (the realism review), not the test's
  elapsed time (a line picking up load when its neighbour breaks was as old as the test).
* **A lamination is a plate on a plate, by geometry**: a strap narrow enough to be a
  frame of members was always a lap, 4.8x weaker than one 10 mm wider (the adversarial
  review); now one stick wide is a lap like tau_g's coupon, three sticks wide a
  lamination, ramping between; a lamination holds no more than its two facing plies
  take at the angle they cross at.
* **inspect's joint capacities are what the crush holds**: the sum over each joint's
  glue points (end grain, crossing, fit-up gaps, scatter), not area x the face's average
  (the diagonal's end-grain foot read 872 N where its points hold 520); `weak_joints`
  uses them.
* **sensitivity ranks 18 values by default**, adding the six that moved F_u most in
  round 22 and were never ranked (the wood-failure share, the rate exponent, the
  crusher's pace, the two size effects, the end-grain tension factor).
* **crush.json records `code_sha`**, the code's own fingerprint: every run below has
  6a81ef9f7a7532fc; the committed code's is ad8581714e3823b0, which differs from it by a
  line wrap and a comment in `study.py` only (`r23_runs/code_after_runs.diff`: undo it
  and `_code_sha()` reads 6a81ef9f7a7532fc again). The full suite ran on the runs' code
  (587 passed, 8 skipped, in two invocations: `test_sim.py`, then the rest), and the
  crush-report tests again after the line wrap.
* **Docs**: the rate law adds ~11 % on Bridge_v10 (round 22), not ~5 %; design
  comparisons inherit an uncertain law at the part they change (glue x1.5 at node M:
  +32 % with round 22's crossed-lap law, +8 % without it), so `ab` there is re-run with
  that law at its other end; `uncertainty.model_cov` is labelled a judgement; the
  stiffness evidence (a rigid-joint model of full-scale glued trusses read 0.82-0.96 of
  their LVDT stiffness, Debertolis et al. 2025; test C's crosshead record is dominated
  by seating).

**A deformable floor beam** (the realism review: the rigid block under the plate):
Bridge_v10 quick seed 0 with Floor Beam M as a 15-ply UD plate spanning between the
trusses read 1017 kgf against 2079 rigid, cracking inside the beam from 8.3 kN, where the
team's hand checks give it 42 kN in bending and 25.6 kN in shear at its notched ends:
thin-plate elements for a 30 mm thick part of 162 mm span, glued at its edges to rigid
tenons, concentrate the stress where they join. The rigid block joins its tenons through
'solid' faces (the wood's shear across the tenon's root, f_v / 1.5: ~3500 kgf, the
team's governing 1.02x check at f_v 6 MPa), so it is kept, and the beam's bending is
left to the hand check.

Evidence on round 23's model (Bridge_v10, the same overrides as rounds 21-22,
`r23_runs/bridge_v10/`), 100 mm/s, every run trustworthy and failing at node M's gusset
laps (the diagonal's foot lets go, then its laps on the 0/90 `plate_M`): quick /
standard / fine read 2079 / 2021 / 2014 kgf on seed 0 (fine -0.4 %) and 2142 / 1944
(quick, standard) on seed 1. Standard seed 0 read 1651 kgf in round 22 and 2276 in
round 21: the crossed laps now hold 0.83 of a parallel lap at the diagonal's angle
(w 0.5) against 0.62-0.64 in round 22. `montecarlo -n 8 --sample-placeholders` (`mc8.json`):
the runs spread 1061-2796 kgf, every run trustworthy, at node M in 7 of 8 (its laps 5,
the foot 1, `plate_M` cracking 1) and at node A once; recommended prediction 1387 kgf
(interval 1188-1669), total range with the mesh and model-form error 795-3865 kgf.

The published tests as distributions (`r23_runs/published`, 10 runs each at fine,
placeholders drawn): §3.14. B's and C's loads fall inside the runs (B 4 runs below, 6
above; C 8 below, 2 above), A's above all of them; C fails where its test did in 7 of
10 runs, A and B never.

### 3.17 Round 24: how stiff a face is, a floor beam that bends

From the four reviews of round 23 (realism 5/10: validation -2.5, placeholders -0.75,
idealisation -0.75, too stiff and linear -0.5, small bridges -0.25, reporting slips
-0.25):

* **How stiff a face is.** Every glue face had one normal stiffness (150 N/mm^3), a
  side-grain face pressed across the grain as stiff as end grain (the realism review:
  E_T over the depth under a 780 mm^2 knee face is ~17-50 N/mm^3). The glue line and its
  surfaces are now in series with the wood on either side, its modulus at its grain's
  angle to the face over the depth the stress reaches (its extent, or half the face's
  width): ~16 N/mm^3 for a member's lap on a 0/90 gusset, ~28 for a knee face, ~110-140
  end grain to end grain. Bridge_v10 quick seed 0: the secant at the peak is now 0.85
  of the 10-40 % secant by deflection (0.89 by jack travel) against 0.94 (0.97) in round
  23, inside glued trusses' 0.84-0.92 (Debertolis et al. 2025); its stiffness by
  deflection 687 kgf/mm against 925, its screen stiffness ~750 against ~1000 kgf/mm, its
  first mode 39 Hz against 47; F_u 1715 against 2079 kgf (rigid floor beam), the faces
  pressed across the grain at node M passing more of their load through the glue in
  shear (the diagonal's foot on the stub now goes first). `wood.E_T_MPa`, which sets it,
  is drawn by montecarlo (35 %).
* **A bar can be told which way it spans** (`axis: x | y | z | [x, y, z]`): the floor
  beam under the jack, wider (200 mm) than it is long between its trusses (162 mm), was
  laid along its outline's length by `kind: bar`, the tool's own advice (the adversarial
  review: -20 % at the wrong joint). Bridge_v10's reference now models it as the team's
  report builds it, a bar along z laminated in vertical layers (one ply through its
  depth: no glue line in its bending shear plane), its tenons rigid and one piece with
  it: quick seed 0 reads 1830 kgf against 1715 with the rigid block (+7 %, the same
  joint), and the `rigid_under_plate` question no longer stands. As a thin-plate shell it
  had read 1017 (round 23).
* **Where the bridge sits in the CAD changes no joint**: a 0/90 plate's plies are listed
  from the face toward the bridge's middle (was: away from the CAD origin; moving the
  bridge 300 mm changed 37 joints by up to 16 %), and a joint sharing its two bodies with
  others draws its scatter by its rank along the bridge (was: its centre rounded to 10 mm,
  halves to even; a 5 mm shift or reading STL instead of SLDPRT redrew 15 joints).
* **Light trusses get trustworthy runs**: a mass-scaling round that would pass the 5 %
  limit is not taken, the time step comes down instead (test A's ten runs were all
  flagged at 5.0-5.6 %; now none is).
* **Each glue line's clock** starts at a tenth of its mixed-mode onset (it started at
  ~2.6 % of a shear line's strength, the peel limit setting the threshold).
* **Asked and noted**: `crusher:pace` asks the team to time the crusher (the glue's
  strength follows the pace: 3.8x slower read -10 % on Bridge_v10); montecarlo's note names
  the params values it redraws though the user set them (list them under `measured:`);
  C's failure place is a vertical's foot on the bottom chord (both in the label), not
  any bottom-chord joint.
* **Docs**: the mesh studies of rounds 22-24 run each fidelity at its own speed (quick
  200, fine 50 mm/s), not one; standard and fine fail at node M's laps, one truss's or its
  mirror's; rolling shear is 4 MPa wherever quoted; `model_cov`'s basis is the published
  tests' distributions; the stale knee-crushing and floor-beam claims are corrected; one
  body of Bridge_v10 (`Portal X R Half b`) is unread in every run (`cad_unread`).

Not done, and why: continuum elements for compact parts (the floor beam is a bar along
its span instead; node blocks, tenons and heel seats stay rigid), butt joints at their
places (round 25, with the literature review of round 23's compression rule), the
end-grain tension law (kept: its one test, C, is a calibration case).

Evidence on round 24's model (Bridge_v10, the reference overrides of rounds 21-23 with
the floor beam as a bar along z, `r24_runs/bridge_v10/`, every crush.json with `code_sha`
350d3ffeeaf6d557, the committed code's), each fidelity at its own speed, every run
trustworthy and failing at node M (the diagonal's foot on the stub, or its laps on the
0/90 `plate_M`): quick / standard / fine read 1830 / 1725 / 1701 kgf on seed 0 (fine -1.4
%) and 1927 / 1796 (quick, standard) on seed 1; the secant at the peak is 0.83-0.89 of
the 10-40 % secant by deflection (0.89-0.92 by jack travel), the deflection at F_u 3.0-3.4
mm. Standard seed 0 read 2021 kgf in round 23 (rigid floor beam, constant face
stiffness). `montecarlo -n 8 --sample-placeholders` (`mc8.json`): the runs spread
1079-2541 kgf, every run trustworthy and at node M (the diagonal's foot 2, its laps 3,
`plate_M` cracking 3); recommended prediction 1263 kgf (interval 1123-1477), total range
750-3444 kgf. The full suite ran on this code (592 passed, 8 skipped, in two
invocations).

The published tests as distributions (`r24_runs/published`, 10 runs each at fine,
placeholders drawn): §3.14. A 23-61 kgf (median 0.26 of its test, every run now
trustworthy), B 40-215 (1.18; 3 runs not quasi-static), C 12-78 (0.40; at a vertical's
foot on the bottom chord, where its test failed, in 4 of 10).
