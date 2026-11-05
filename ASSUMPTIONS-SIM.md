# Crush simulation: method and assumptions

`bridgesim crush` simulates the Troitsky crushing test (rulebook §12.5) on a bridge read
from SolidWorks. This page states what it models, how, and what it does not. Present the
relevant parts in your Design Validation (§10). Units: mm, N, MPa, s.

## 1. Geometry: from SolidWorks to solids

* `.SLDPRT` files are read directly: the Parasolid model inside the file is decoded
  (faces, loops, edges, vertices, planes, cylinders, circles) and every solid body is
  rebuilt exactly, with its SolidWorks body name. Each body is checked to be a closed,
  outward-oriented surface with positive volume; a face that cannot be decoded is closed
  over (only seen on 3 mm floss holes) and reported.
* STL (all bodies in one file) and Parasolid `.x_b` are also read. STL bodies have no
  names, so their layups become questions.
* Assemblies (`.SLDASM`) are not read: save the assembly as a multi-body part.

## 2. Idealisation: what each body becomes

| Body shape (from its faces) | Model | Why |
|---|---|---|
| extrusion at least 2x longer than its section, section no flatter than 1:4 (a deep cross-beam 221 x 88 x 63 mm is one) | **bar**: chain of corotational fibre beams on its centre line, ~25-30 mm elements (>= 4), real section and length, Timoshenko (shear-flexible) | members carry axial force + bending; buckling needs >= 4 elements |
| slender but tapered extrusion (a wedge, a lip or filler cut along a diagonal: 40-75 % of its bounding box) | **tapered bar**: as above, each element with the width and centre offset of the profile at its position (same volume as the solid) | still a member; a rigid block would make the load path too stiff |
| thin extrusion (width >= 3 x thickness) | **shell**: CST membrane + DKT bending triangles on the mid-surface, meshed from the real outline (holes > element size kept), integrated ply by ply (layered) | gussets, webs, decks, diaphragms |
| plate made of narrow arms (an L-shaped leg, an X, a curved rib, a strut-like plate): arms narrower than 1.6 standard shell elements, or 3 when they are slender (centre lines >= 4x their width), arms covering >= 80 % of the plate and carrying what is glued to it (a part glued off the arms, a wing of a hub plate, gets an arm of its own out to it, if that runs inside the plate) | **frame**: fibre beams along the centre lines of its arms (the medial axis of its outline, `sim/skeleton.py`), each with the arm's local width and the plate's thickness, plies stacked through the thickness with their grain along the arm (or the span), butt joints per arm as for bars, arms sharing nodes where they meet, glue on rigid arms to the nearest node | a strip of constant-strain triangles one or two elements wide is 2-4x too stiff in in-plane bending, and a 20 mm thick plate is no thin plate; a beam is exact there (`test_frame_plate_bends_like_a_beam`: beam theory within 3 %, the same strip as shells 70 % too stiff). The `frame_plates` question lists them; a plate whose glued parts cannot be reached that way stays a shell and the `narrow_plates` question asks about it |
| compact piece (less than 2x longer than its section, not thin), a body named a block, short toothpick | **rigid body** with its real mass and inertia | much stiffer than the members; a question lists those in the load path |
| floss (by name) | **cable**: tension only, snaps at its breaking load | §8.4.5 |

Laminates come from the body name (`10-ply UD`, `2-ply 0-90`, `4-ply +-45`, `block`,
`round toothpick`, `waxed floss`) or the overrides file; an unknown one is assumed UD
along the long axis and reported as a question. Stiffness from classical lamination
theory with orthotropic plies (E_L, E_T, G_LT, nu_LT), including the membrane-bending
coupling of unsymmetric layups (a 2-ply 0/90 plate bends when stretched). UD bars and
plates take their grain along the long axis (or the global X with "along the span"; for
non-rectangular UD plates, along the nearest long edge). The order of the plies through
the thickness is not in the CAD: it is taken from the name, listed from the face the
plate's normal points away from.

**Shear deformation of members.** Wood is soft in shear (E/G ~ 16), so a deep laminated
member (a 47.5 mm chord between nodes 300 mm apart) deflects ~25 % more in bending than
Euler-Bernoulli theory says. Each bar element is a Timoshenko beam (shear area 5/6 A,
G = G_LT of the layup): a constant shear strain takes its share of the end rotations, the
exact two-node split phi / (2 (1 + phi)) with phi = 12 EI / (k G A L^2) while elastic,
corrected each step so the shear force k G A gamma equals the bending shear (M_i + M_j) / L
once the fibres crack or crush. Tested: a 50 mm deep beam over 200 mm matches
Timoshenko theory within 1 % on 4 or 16 elements, and its bending strength is the same as
without shear deformation (`test_deep_beam_includes_shear_deformation`).

**Laminated members longer than one stick.** A ply longer than a stick (115 mm) has butt
joints, and its load goes round each joint through the glue lines to the neighbouring
plies. With the joints of neighbouring plies a distance s apart, the weakest tension path
cuts k plies at their joints and shears the glue between them:
P_k = (n - k) P_ply + (k - 1) tau_g w s; the member's tension strength is the smallest
P_k (k = 1 is the net section without one ply). By default the plies are assumed laid in
a brick pattern (s = half a stick), which gives (n - 1)/n of the solid strength; closer
joints let the glue govern (10 plies with joints 11.5 mm apart: 25 %). Each such member
gets a question (`butt_joints:`), answered with `butt_stagger_mm` in the overrides (0 for
plies without joints). Shear across the thickness through which the plies are stacked
runs along the glue lines, so there the shear strength is min(f_v, tau_g); shear in the
other direction crosses the plies and is resisted by the wood (f_v). Butt joints are
assumed tight in compression (they bear).

## 3. Joints

* Every pair of coplanar touching faces of two bodies is a **glued patch** of the exact
  overlap shape. The glue is a grid of cohesive springs (one per node of the finer mesh,
  each with its share of the area) attached to both bodies through rigid arms at their
  real positions: joint eccentricity, peel from bending and the joint's rotational
  stiffness all come out of the geometry.
* Glue law: penalty stiffness K_n, K_s (glue line + adjacent wood), quadratic initiation
  (sigma/sigma_max)^2 + (tau/tau_max)^2 = 1, linear softening with mixed-mode fracture
  energy (Benzeggagh-Kenane, G_Ic, G_IIc). tau_max is tau_g of the material file times
  the peak/average ratio of the simulated 20 mm lap test (`bridgesim lab`), so a joint
  reproduces the measured tau_g (checked in the tests). Peel strength = 0.5 tau_max.
* A broken glue line keeps **bearing in compression** and **slides with friction**
  (mu = 0.4): pieces still lean on each other after the glue lets go.
* The wood at a glue line **crushes** when the joint is pressed together, at the
  strength of each side for the direction of the joint (Hankinson's formula between f_c
  along the grain and f_c90 across it, averaged over the plies; the weaker side governs),
  and keeps the permanent set: a member's end pressed into the face of a plate crushes
  the plate across its grain (f_c90), two members end to end bear at f_c. Crushed wood
  **compacts**: the crushing stress rises with the crushed share of the depth that can
  crush (the weaker part's extent along the joint normal, at most
  `wood.crush_zone_depth_factor` = 3 times its thickness), and once crushed by
  `wood.densification_strain` (0.45) of it the wood is densified and about as stiff as
  wood across the grain: a block cannot push through a plate
  (`test_glue_line_crushing_densifies`). A crushed pair keeps bearing; the crushed part's
  own section is not reduced.
* **End grain** holds glue poorly: the glue strength of a joint is multiplied by
  1 - e (1 - `glue.end_grain_factor`), e being the share of a side's plies meeting the
  joint end-on (0.5 placeholder: a member's end glued flat gets half, a 0/90 plate's edge
  three quarters). Joints that are mostly end grain are listed in a question.
* **Shear lag and the mesh.** The glue points of a joint hang on the nearest nodes of
  each part (rigid arms), so how much shear lag a joint shows depends on how many nodes
  that is. The lap coupon with its glue hung on 1, 2, 3, 4, 5 and 9 nodes per side needs a
  peak/average ratio of 1.01, 1.11, 1.20, 1.15, 1.16 and 1.15 to give tau_g; the model
  uses 1 + 0.155 (1 - exp(-2.2 (n - 1))) with n the larger node count of the two sides
  (within 4 % of each), so a joint's capacity (tau_g times its area) does not jump
  when the mesh changes.
* A joint is reported **cracking** when 30 % of its glued area has let go and
  **separated** (`joint_failure`) at 90 % (area-weighted over its glue points); the
  first separation sets `Fu_first_joint_failure_kgf`.
* Bodies that touch nothing are tied to the nearest part (and reported); floss ends
  attach to the part they touch.

## 4. Materials (placeholders until measured)

Wood along the grain: E, f_t, f_c, f_v from `materials/popsicle_birch.yaml`. For scale,
the USDA *Wood Handbook* (FPL-GTR-190, Table 5-3a, clear wood at 12 % moisture) gives
paper birch E = 11.0 GPa, modulus of rupture 85 MPa, compression parallel to grain 39 MPa,
shear 8.3 MPa (yellow birch: 13.9 GPa, 114, 56, 13 MPa). Craft sticks are thin, cut across
growth rings and have defects, so the placeholders (10 GPa; f_b 50, f_c 30, f_v 6 MPa)
are deliberately lower. The one published measurement on craft sticks we found gives
E = 7.1 +- 0.2 GPa (cantilever tests, *Emergent Scientist* 1, 3 (2017), "the popsicle
stick cobra"), 29 % below the material file's 10 GPa, which the linear analysis keeps.
The crush uses 7.1 GPa (and G = E/16) while the material's E is a placeholder
(`wood.E_placeholder_MPa`; a measured E from `calibrate` or `material.E_MPa` in the
overrides replaces it): stiffness, buckling and deflections scale with it, so measure
your own sticks. f_t (40 MPa) is below f_b on purpose: a stick
in tension fails at its weakest cross-section (slope of grain, defects), which bending
loads only at one face, so graded timber has f_t,0 = 0.6 f_m (EN 338), while clear wood
has the opposite. `bridgesim sensitivity` shows how much each
placeholder moves F_u for your design. Your own stick tests replace them
(`bridgesim calibrate`). In the fibre
sections: linear elastic; tension softens to zero after f_t with fracture energy G_f
(3 N/mm, element-length regularised); compression yields at f_c and softens to 70 %
(crushing). A single stick so modelled breaks in 3-point bending at the material's
f_b (within 4 %; the test allows 10 %): compression yielding on top and tension softening below give
the bending strength, as in wood. A member **splits in shear** when 1.5 V/A + T/W_t
reaches the strength of the glue planes between its plies, or f_v across them. The glue
planes of a laminate (pressed while they cured, long and continuous) are not a hand-made
lap joint between parts (tau_g, an average over the lap): they split at the lower of the
bond line's strength (`glue.interlaminar_MPa`, 6 MPa: PVA bond lines in pressed wood
laminates fail in the wood next to them) and what the weakest ply takes there, the
interlaminar shear running along its grain (f_v) and across it (rolling shear,
`wood.f_rolling_MPa`, 2 MPa) in proportion to its angle: f_v for a UD stack, the
rolling shear for a ply across the member (0-90), in between for +-45. A short-beam
shear test of a laminate measures it (`short_beam_shear_template.csv`, `calibrate`).
(Until round 12 it was 1.155 tau_g, 2.3 MPa, for every laminate: most frame arms and
many members split long before the peak.) The shear is compared averaged, by length,
over the elements of the member within half a section depth of each (`solver.
split_members`): a split needs the shear over a member's depth, not on the one element
next to a glued joint, which resolves a sharper peak the finer the mesh
(`test_members_split_on_their_shear_over_a_section_depth`). It then splits at once, as a
laminated beam does in a short-span shear test. When a member breaks, the stress wave can
split its neighbours in the same instant; the crush then names the break, not the splits,
as the cause. A split member is
not deleted: it delaminates (its n plies then bend each about their own axis, 1/n^2 of
the bending stiffness through the plies, half across the width for a split) and keeps
its axial strength, so a split member still props what it carried. Glue and floss: see
`materials/simulation.yaml`, every value with its source and how to measure it.

**Plates (layered shells).** At each of the 3 in-plane points of a triangle and at 3
points through each ply group (Simpson's rule, exact for the elastic bending
stiffness), the strain is turned into the grain axes and each failure mode has its own
damage, in the spirit of LS-DYNA's MAT_054 or Abaqus' Hashin damage but with an
independent maximum-strain criterion per mode: fibre tension softens to zero with
fracture energy G_f (3 N/mm); fibre compression crushes to a plateau at 70 % of f_c;
across-grain tension (sticks parting edge to edge, f_t90 = 1 MPa) softens with G_f90
(0.3 N/mm); across-grain compression yields at f_c90; in-plane shear softens with G_v
(1 N/mm). Cracks close: a broken ply still carries compression. The softening strain of
each element is set from its size (crack band, Bažant & Oh 1983), so the energy a crack
dissipates does not depend on the mesh: a plate strip pulled apart dissipates
G_f x width x thickness within 2 % with 5, 2.5 and 1.25 mm elements
(`test_shell_fracture_energy_is_mesh_independent`). Elements larger than the crack band
of a mode allows (snap-back) keep their strength and fail as brittle as possible; they are
counted in the warnings. An element is removed only when every ply is torn wide open
(twice the strain at which it carries nothing) or when it is distorted beyond 0.5 rad.

**Scatter** (seeded, `--seed`; `--seed -1` for a perfect bridge): per body, E (CoV 12 %)
and strength (CoV 18 %); per joint, glue strength (CoV 30 %); per bar, an initial bow of
L/400 on average (0.3-1.7 of that) in a random direction; per bridge, the plate placed up
to 3 mm off-centre, the support corners up to 0.5 mm out of level, and the whole bridge
leaning out of its plane by up to 0.005 rad either way (`scatter.lean_rad`, the initial
inclination of EN 1995-1-1 5.4.4: height x angle at every node and glue point alike, so
the parts stay together and unstressed). The arms of a frame are bowed like members (the
arm's length / 400 on average, across it, keyed on the arm). The glue strength has a
bridge-wide factor too (`scatter.glue_bridge_cov`, 15 %: the glue batch, the curing, the
builders' hands), on top of the joint-to-joint scatter, so repeat builds differ by more
than independent joints would make them. Each value is drawn
from a stream named after its body or joint (a joint by its two bodies and where they
overlap in the CAD), so a seed is the same imperfect bridge at every fidelity and after
an edit elsewhere in the design. Monte Carlo over seeds gives the spread of F_u.

## 5. The test

* The bridge stands on the crusher's flat steel base platform: unilateral contact with
  Coulomb friction (mu = 0.35), no anchorage (§8.3). Every vertex of every body, and a
  grid on downward faces at table level, can touch it.
* Gravity acts on everything. The bridge first settles under its own weight; a bridge
  that sags more than 20 mm fails by *instability* ("failure under self-weight") and
  gets F_u = 0.
* The 200 x 90 mm loading plate (§12.5; two 15 mm steel plates, 4.2 kg, Appendix B) is a
  rigid body resting on the deck at mid-span and the deck centreline. The jack pushes on
  its hemispherical seat, so the plate tilts freely; plate-bridge contact is unilateral
  with friction (mu = 0.3) on any part of the bridge that reaches the plate.
* **Seating and an out-of-level bridge.** A bearing face beds in on the steel: its contact
  stiffness grows from 0 to full over the first `contact.seating_mm` (0.3 mm: roughness,
  glue beads, fibres crushing at the first touch), which gives the soft start ("toe") of
  a real load-deflection curve. And a built bridge never stands on its four support
  corners at once: with a seed, each corner is lifted by up to
  `scatter.support_level_mm` (0.5 mm, one corner not at all), so the bridge first rocks
  on three corners until the load beds it down. Both are assumed values: read them off
  the start of a real crush curve and off the bridge on a flat table.
* **Bearing**: each contact point with the platform or the plate carries at most the
  crushing strength of the wood under it times the area it stands for (f_c90 for wood
  pressed across the grain, f_c for a post standing on its end grain): a plate edge
  digging into the deck, or a support corner, crushes locally instead of carrying an
  unlimited load.
* **Contact between parts that were not touching at the start** (a broken chord falling
  on a cross-beam, a deck sagging onto a strut): penalty contact of every node and beam
  mid-point against beam capsules and shell slabs of the other bodies
  (`bridgesim/sim/contact.py`), frictionless. Parts touching at the start interact
  through their glue springs instead, which keep bearing and sliding with friction after
  the glue breaks; a broken glue pair whose two points have moved apart (3 mm), turned
  (74 degrees) or passed through each other stops acting.
* The jack moves down at constant speed (displacement control); its force is the
  load-cell reading. The crusher + load cell stiffness is 20 kN/mm, its sideways
  stiffness at the plate 2 kN/mm (`crusher.stiffness_N_per_mm`,
  `crusher.lateral_N_per_mm`: measure them on your press).
* **The load reading.** The jack force rings with stress waves that a real test at
  mm/min never shows (time is compressed). Two filters of it: *online*, a one-pole
  low-pass over 0.3 mm of jack travel, which decides when to hold the jack and when the
  bridge is judged failed, as a load cell would in real time; and *reported*, the raw
  force smoothed by a centred Hann window three times as wide (0.9 mm of travel at full
  speed), along the jack travel: the load against travel, as a test record plots it. It
  rejects the ringing as well but has no lag, and a hold (the jack nearly still while
  something breaks) is one point of the curve, weighted by the little travel it covers,
  so the load can neither jump nor keep rising at constant travel there (round 11
  smoothed the moving and held stretches apart, which left steps of 4-6 % at their
  boundaries). The curve, F_u, the deflection at F_u and the load of every event are the
  reported one. (Until round 10
  the online reading was reported: the same F_u within 0.2 % on Bridge_v6, but its lag
  put the rising branch 0.3 mm late, the secant stiffness 7 % low.)
* **Failures run their course at constant load**, as in the real test where the jack
  moves mm per minute: whenever something breaks (a joint lets go, a member or plate
  cracks, floss snaps) the simulated jack slows smoothly to 2 % of its speed and waits
  until nothing more has broken for 4 ms and the bridge has settled (its kinetic energy
  back within twice that of steady loading, its load steady within 1 % over the last
  2 ms), at most 30 ms, then speeds up again (`quality.holds`). The curve therefore does
  not show the bridge ringing after a break as a load rising at constant travel. Without this, a failure that takes ~30 ms to develop (a heel joint
  letting go and the supports sliding out) would be outrun by the jack, and F_u would
  depend on the jack speed.
* The **deflection** reported is that of the deck points the plate rests on when the
  jack starts, followed as material points (not the jack stroke, which also includes the
  20 kN/mm machine compliance).
* **F_u** = peak jack force up to the moment the bridge is judged failed: the load falling
  30 % below its peak (collapse: a fracture the judges see and the load cell shows, even
  if the bridge then picks up some load again; §12.5 takes the load "at the moment of
  failure"), mid-span deflection > 50 mm, or contact of a
  part with the platform other than the original supports (§12.5) or with the press
  uprights (debris flying off a broken part does not count). A glued joint letting
  go inside the bridge is also a §12.5 failure ("shear, including joint failure"), but
  the judges only see it when the bridge visibly gives way, which shows as the load
  drop; the load at the first complete joint separation is reported separately
  (`Fu_first_joint_failure_kgf`). Every joint failure and member split is logged with
  the load at which it happened.
* **What governed** (`governing`): the break that started the fall of the load to the
  failure call, i.e. the first break from the moment the reported load last stood within
  5 % of F_u (a quarter of the online filter's time constant earlier, since the centred
  smoothing leads a sudden drop). A part that broke before that was survived: the load
  came back, as it did on Bridge_v6 seed 1 after a deck-stringer joint let go at the peak
  (5 % drop), before the End Post joints went at 96 % of the peak and the load fell 40 %.
  A member split in the same 2 ms as a member fracture is put down to the fracture; a
  clean break comes before a joint that only cracked or a member crushing. Only what
  happened up to the failure call counts: what breaks after it is the collapse. When
  nothing broke, the joint whose glue let go during the fall (at least 20 % of its area
  between the load leaving its peak and the call) governs, although it has not reached
  the 30 % of `joint_cracking` yet; else "buckling or sway". A member **crushing** (a
  fibre folding in compression past f_c, beyond its onset strain) is an event too: a
  laminated beam that fails in bending by crushing on its compression face is named so,
  and its what-if is a bigger section, not more glue. `competing` lists the other parts
  that broke within 10 % of F_u: the failure can start at one of them on another seed or
  mesh; `diagnose` measures which one F_u depends on.
* **Rate check** (`--rate-check`): the crush again at half the jack speed. F_u is
  roughly linear in the speed over this range, so `Fu_extrapolated_kgf` = 2 F(v/2) -
  F(v) estimates it for a jack at rest (the real test's mm/min).

## 6. Numerics

* Shell drilling rotations (about the plate's normal) are tied to the element's in-plane
  rigid rotation (Hughes & Brezzi 1989) with a small penalty (5 % of E t x area), so
  every element has exactly its 6 rigid-body zero-energy modes (tested); the penalty is
  released in an element that has started to crack. An earlier formulation left the
  nodes of a glued plate free to spin about the normal, and that drift made joints fail
  earlier the slower the test ran.
* Explicit central-difference time integration (like LS-DYNA): each step computes the
  internal forces of the deformed structure, so fracture, contact and collapse need no
  convergence. Large rotations are exact (corotational elements, rotation matrices
  updated by the exponential map).
* Stable time step from the true highest frequency of the model (Lanczos on the
  linearised system); light nodes that limit it get extra mass (selective mass scaling,
  reported, limited to 5 %; rotational inertia up to 50x). A run that ever diverges is
  restarted with half the step.
* Damping: mass-proportional (alpha = 25 1/s; ~5 % of critical for the bridge's first
  modes), material damping beta = 0.1 dt (only for the highest, numerical frequencies)
  and contact damping. The kinetic energy of the bridge (falling debris excluded) is
  reported against the work done by the jack while the load rises to 95 % of its peak
  (`kinetic_to_work_max`): below 0.1 the loading is quasi-static. The collapse after the
  peak is dynamic, as in the real test: once the bridge is judged failed (F_u is known)
  the mass damping drops to 2 1/s, so the wreckage falls at its real speed in the replay
  instead of the ~0.4 m/s that the loading-phase damping (g / alpha) would allow. `--rate-check` crushes the bridge again at half
  the jack speed and reports how much F_u changes (`quality.rate_check`).
* **Bifurcations.** When two failure modes compete (say a joint at one heel and a
  gusset at mid-span reach their strength at similar loads), a small difference - the
  jack speed, the random seed, a 1 % change of a strength - can decide which goes first,
  and F_u jumps between the two outcomes. Real bridges do the same: two copies of one
  design fail at different loads. So give F_u as the range from `montecarlo`, compare
  designs with `ab` (paired seeds), and read `failure_sequence` to see which modes
  compete.
* **Mesh convergence** (`validation/convergence_study.py`; Bridge_v6 in
  `validation/README.md` §3.4): seed by seed the standard mesh is within 3.2 % of the
  fine mesh, the quick mesh 4-9 % lower, the seeds ranked alike. The perfect bridge is
  symmetric: which side gives way first is a bifurcation round-off decides, so its F_u
  moves by up to 10 % between meshes and does not measure the mesh error. Which part
  goes first (`governing`) changes with the mesh: several compete within a few %
  (`competing`).
* **Crack onset is local.** A plate cracks where its strain first reaches the strength
  in one element; the crack band makes the energy of the crack independent of the mesh,
  not its onset at a sharp stress raiser (the end of a glued lap, a re-entrant corner),
  which comes at a lower load the finer the mesh: at element size x0.5 the Bridge_v6 tie
  strap starts cracking at 1339 kgf instead of 1760 at x0.75. Do not refine beyond
  `fine` (x0.75, elements still larger than the plate thickness). A nonlocal
  process-zone onset was tried and not adopted (it doubled the strength of off-axis
  plies; `validation/README.md` §3.4).
* Time is compressed: the simulated jack moves at ~100 mm/s instead of the real
  mm/min. The material laws have no rate effect, so this changes nothing as long as the
  run stays quasi-static; the replay converts back to real test time.
* **Verified** (the equations are solved right) in `tests/test_sim.py`: cantilever
  (0.04 % vs PL^3/3EI), large-deflection elastica (Bisshopp-Drucker, < 1 %), Euler
  buckling of a stick, a deep beam with shear deformation (Timoshenko, < 1 % on 4 or 16
  elements), 3-point bending stiffness (0.03 %; the lab coupon starts slowly so
  its stiffness is quasi-static), a stick's bending strength = f_b, lap-joint strength,
  energy conservation (kinetic + strain energy constant to 0.5 % in free vibration of a
  bridge with beams, shells and glue), DKT patch test, layered shell = exact laminate
  stiffness, exactly 6 rigid-body zero modes per shell element, mesh-independent fracture
  energy of a plate strip, a member dropped on another stopped by contact, compiled
  kernels identical to the NumPy reference (damage histories included).
* **End to end** (CAD file -> crushing test -> report): a laminated beam on two narrow
  piers under the rigid plate has the stiffness of beam theory within 5 % (measured:
  1.8 %) and breaks at the section's bending strength
  (`test_simply_supported_beam_matches_beam_theory`); `calibrate` followed by `lab` on
  the same coupon data gives < 1 % error (`test_calibration_closes_the_loop`).
* **Not yet validated** against physical tests (whether the equations are the right ones
  for popsicle sticks): no load-deflection record of a real Troitsky crush test was
  available. `bridgesim lab --data` and `bridgesim compare` measure that gap as soon as
  you have coupon or prototype data, and `bridgesim calibrate` closes it.

## 7. Not modelled (yet)

* Glue lines cover the whole touching area. Starved or gappy joints come in through
  `glue_factor` in the overrides, or through the glue strength itself: tau_g is the
  average over the nominal overlap of a hand-made lap, so once measured on laps made
  like the bridge's, incomplete coverage is in it (a separate coverage factor would
  count it twice).
* Splitting of a member along the grain other than by shear, knots and stick-to-stick
  variation inside one body; butt joints in plates (in bars and frame arms only). A member that splits
  in shear switches to its split section at once (no fracture energy for the split
  itself); its axial strength is kept.
* A crushed part keeps its full section in the rest of the model: crushing is in the
  glue line between two parts (permanent set, compaction), not a loss of the part's
  own area.
* Thick plates that are not made of arms (a wide gusset 20 mm thick) use thin-plate
  (Kirchhoff) elements with constant-strain membranes, at least 1.5 x their thickness in
  size (a membrane with drilling rotations was tried and not adopted:
  `validation/README.md` §3.4); plates of narrow arms are frames (section 2). Compact
  blocks (a node block, the block under the plate; slender or tapered pieces are bars)
  are rigid: they crush only where they bear on the plate or the platform, and their
  glue lines crush (by grain direction, as above); a question lists the ones in the load
  path, answered with `kind: bar` or `kind: shell` when one is really a member.
* Friction in the contact between parts that were not touching at the start.
* Debris (pieces with nothing left holding them) falls under gravity and lands on the
  platform, but does not hit the bridge or the plate.
* Creep, humidity, and the time the bridge waits loaded. White (PVA) glue creeps and the
  real test takes minutes, so a real curve bends over more before the peak and the
  deflection at the peak is larger than simulated; slip in imperfect joints (gaps, glue
  starved at the edges) adds to it. The simulated curve is that of perfect joints:
  the toe from seating, then near-linear to ~90 % of the peak.
