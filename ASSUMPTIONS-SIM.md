# Crush simulation: method and assumptions

`bridgesim crush` simulates the Troitsky crushing test (rulebook §12.5) on a bridge read
from SolidWorks. This page states what it models, how, and what it does not. Present the
relevant parts in your Design Validation (§10). Units: mm, N, MPa, s.

## 1. Geometry: from SolidWorks to solids

* `.SLDPRT` files are read directly: the Parasolid model inside the file is decoded
  (faces, loops, edges, vertices, planes, cylinders, circles) and every solid body is
  rebuilt exactly, with its SolidWorks body name. Each body is checked to be a closed,
  outward-oriented surface with positive volume; a face that cannot be decoded is closed
  over and reported (3 mm floss holes; on Bridge_v10 18.5 % of a portal brace's
  surface, `cad_approximated`). A body the file names whose faces cannot be put
  together is missing from the model and asked about (`cad_unread`: one of
  Bridge_v10's 70, half a portal brace).
* STL (all bodies in one file) and Parasolid `.x_b` are also read. STL bodies have no
  names, so their layups become questions.
* Assemblies (`.SLDASM`) are not read: save the assembly as a multi-body part.

## 2. Idealisation: what each body becomes

| Body shape (from its faces) | Model | Why |
|---|---|---|
| extrusion at least 2x longer than its section, section no flatter than 1:4 (a deep cross-beam 221 x 88 x 63 mm is one) | **bar**: chain of corotational fibre beams on its centre line, ~25-30 mm elements (>= 4; no longer than 2x the member's depth at standard since round 21, for light members), real section and length, Timoshenko (shear-flexible) | members carry axial force + bending; buckling needs >= 4 elements |
| slender but tapered extrusion (a wedge, a lip or filler cut along a diagonal: 40-75 % of its bounding box) | **tapered bar**: as above, each element with the width and centre offset of the profile at its position (same volume as the solid) | still a member; a rigid block would make the load path too stiff |
| thin extrusion (width >= 3 x thickness) | **shell**: CST membrane + DKT bending triangles on the mid-surface, meshed from the real outline (holes > element size kept), integrated ply by ply (layered) | gussets, webs, decks, diaphragms |
| plate made of narrow arms (an L-shaped leg, an X, a curved rib, a strut-like plate): arms narrower than 1.6 standard shell elements, or 3 when they are slender (centre lines >= 4x their width, and 90 % of their length no wider than 3 elements: a 140 mm gusset with 60 mm tails is a plate, not a strut; it was a single beam until round 16), arms covering >= 80 % of the plate and carrying what is glued to it (a part glued off the arms, a wing of a hub plate, gets an arm of its own out to it, if that runs inside the plate) | **frame**: fibre beams along the centre lines of its arms (the medial axis of its outline, `sim/skeleton.py`), each with the arm's local width and the plate's thickness, plies stacked through the thickness: a UD frame's sticks along each arm (it is built arm by arm), a cross-ply or angle-ply frame a sheet cut to shape whose plies keep the plate's long axis in every arm (an arm at 45 degrees to a +-45 plate is 0/90; asked as `frame_plies:`, `grain: arm` if they were laid along each arm; `test_frame_plies_follow_the_sheet_or_the_arm`), or the span; butt joints per arm as for bars, arms sharing nodes where they meet, glue on rigid arms to the nearest node | a strip of constant-strain triangles one or two elements wide is 2-4x too stiff in in-plane bending, and a 20 mm thick plate is no thin plate; a beam is exact there (`test_frame_plate_bends_like_a_beam`, a strip 40 mm deep on knife edges, ~2.5 elements across: beam theory within 3 %, the same strip as shells 20 % too stiff; the 70 % reported until round 15 was mostly the strip's ends glued to 10 mm piers, which fix them partly, for frames as for shells since round 15). The `frame_plates` question lists them; a plate whose glued parts cannot be reached that way stays a shell and the `narrow_plates` question asks about it |
| compact piece (less than 2x longer than its section, not thin), a body named a block, short toothpick | **rigid body** (its faces bear at f_c90 unless its fibres are given along an axis, `grain: x | y | z`: a heel seat whose layers run along its post bears on end grain, Hankinson between) with its real mass and the mean of its principal moments of inertia (one value for all axes: the loading is quasi-static, kinetic energy below 5 % of the jack's work (on Bridge_v10 under 1 %), so a block's rotational inertia only enters the time step) | much stiffer than the members; a question lists those in the load path |
| floss (by name) | **cable**: tension only, snaps at its breaking load | §8.4.5 |

Laminates come from the body name (`10-ply UD`, `2-ply 0-90`, `4-ply +-45`, `block`,
`round toothpick`, `waxed floss`) or the overrides file; an unknown one is assumed UD
along the long axis in plies of a stick's thickness (2 mm) glued together, since a part
of a stick bridge is built of sticks (its glue lines can split), and reported as an
important question; a layup named without a ply count (`UD`, `0-90`) gets its plies
from the thickness the same way. Stiffness from classical lamination
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
plies. The member's tension strength is that of its weakest path through the plies
(`butt_path_capacity`): a path crosses each ply either at one of its joints (free) or
through its fibres (P_ply), and between neighbouring plies shears their glue line over
the distance between the two crossings, at the laminate's interlaminar strength (the
pressed bond line or the plies next to it, where a member splits: below). Ply i's joints
lie at i s, repeating every stick, and the weakest path is found over those positions
(dynamic programming). By default the plies are laid in a brick pattern (s = half a
stick), so every other ply is cut at the same section: a 13-ply member keeps 6 plies
there, 46 % of its solid tension strength, a 4-ply one 50 % (until round 16 the formula
counted one cut ply per section, (n - 1)/n, 92 % for 13 plies: twice what statics
allows, found by the adversarial review). Joints a little apart from ply to ply (a
staircase) give the zigzag path through k joints, (n - k) P_ply + (k - 1) tau w s
(`test_butt_joints_cut_the_plies_whose_joints_line_up`). Each such member gets a question
(`butt_joints:`), answered with `butt_stagger_mm` in the overrides (0 for plies without
joints). **Plates laid from sticks** have them too: in each direction in which a plate is
longer than one stick, its sticks of that direction (rows side by side, plies through the
thickness) form strands with joints staggered row to row and ply to ply by
`butt_stagger_mm` (brick by default), the path between them shearing at the laminate's
interlaminar strength (rolling shear through a cross ply); the lower factor of the two
directions scales the fibre tension strength of every ply of the plate (a shell keeps
one fibre strength per element): about half for a deck in a brick pattern. The statics
is then taken at `wood.butt_tension_efficiency` (0.8, since round 20): in plies as thin
as a stick the glue line cracks from a joint's tips at 24-73 % of the ultimate load and
the fracture runs into the plies, so net-section estimates that hold for 1/4 in veneers
fail for 1/10 in ones (Jung & Murphy 1983, Wood Fiber Sci. 15(2):116); one of three
laminations cut kept 0.53 of the tension strength against 0.67 of its section (Dansoh,
Koizumi & Hirai 2003, J Wood Sci 49:405): a 13-ply member in a brick pattern keeps
0.8 x 6/13 = 37 % of its solid tension strength. Butt joints cost **stiffness** too
(since round 21): near each joint the cut ply hands its force to its neighbours over a
shear-lag length 1 / lambda, lambda^2 = 2 K / (E t) on the pressed glue lines either side
(`glue.lamination_K_s_N_per_mm3`), so a member longer than a stick has E x 1 / (1 + 2 n
/ ((n - 1) lambda L_stick)): ~0.88 for 13 plies, 0.84 for 3 (`butt_stiffness_factor`;
gluing the butts raised such laminates' stiffness 13-39 %, Dansoh et al. 2003; closing
their gaps 9 %, Berard et al. 2011). Shear across
the thickness through which the plies are stacked runs along the glue lines, so there the
shear strength is min(f_v, the bond line); shear in the other direction crosses the plies
and is resisted by the wood (f_v). A butt joint bears in compression when its gap
closes before the member is damaged: in 10-ply laminates of 2 mm veneer, gaps of 0.25
and 0.5 mm closed early after first yield and 1 mm gaps never did (Berard et al. 2011,
J Wood Sci 57:107), and tight butt joints kept 0.94-0.96 of a jointless laminate's
compression strength (Dansoh et al. 2003). So up to `wood.butt_bear_gap_mm` (0.5) the
member keeps `wood.butt_bearing_efficiency` (0.95) of its crushing strength; from
`wood.butt_open_gap_mm` (1.0) the ply's force goes round the joint through the glue
lines as in tension and the member keeps its butt-joint tension factor in compression
too; linear between (`butt_gap_mm` the designer's gap, 0 when the ends are sanded
tight, ~2 for rounded stick ends; else seeded per member from 0 to
`scatter.bearing_gap_mm`, 0.3 mm, so a hand-cut member nearly always bears;
`test_butt_joints_with_a_gap_are_out_in_compression_too`). Until round 18 butt joints
were assumed tight in compression; round 19 let them bear only up to 0.15 mm (a misread
of Berard et al., which the round-19 literature review corrected).

## 3. Joints

* Every pair of coplanar touching faces of two bodies is a **glued patch** of the exact
  overlap shape. The glue is a grid of cohesive springs (one per node of the finer mesh,
  or between two members a grid at their element size with at least two points across
  each side of the patch, `glue.points_per_side`, spaced so that their equal shares of
  the area have the patch's own second moment: two at the Gauss points) attached to
  both bodies through rigid arms at their real positions: joint eccentricity, peel from
  bending and the joint's rotational stiffness all come out of the geometry. Until
  round 15 a patch smaller than an element was one spring, a hinge. On Bridge_v10 (a
  team's Rev N) seed 0 then gave F_u 1397 kgf; with the glued diagonal feet at its
  centre node resisting turning, 1769 kgf with two points per side and 1778 with three:
  converged (two points at the cell midpoints, 75 % of the turning stiffness, gave
  1534; validation README §3.7).
* **Plates glued face to face are meshed node on node** over their overlap (a deck
  laminated from separate ply bodies, tongue and groove included; a gusset on a plate):
  finest first, each takes the nodes of those meshed before it, then each its partners'
  final ones, its own outline kept. The glue then joins nodes straight across. Hung on
  nodes off-centre by rigid arms, a load through the stack bent each thin layer between
  them, a compliance that grew with the element size: Bridge_v10's stiffness under the
  plate was 360 / 623 / 737 / 816 kgf/mm at x1.6 / x1 / x0.75 / x0.5, now 838 / 819 /
  836 / 833 (`test_stacked_plates_are_meshed_node_on_node`: a three-ply deck with a
  tongue and groove on a block, 9x too soft at quick before).
* Glue law: penalty stiffness K_n, K_s (glue line + adjacent wood), quadratic initiation
  (sigma/sigma_max)^2 + (tau/tau_max)^2 = 1, linear softening with mixed-mode fracture
  energy (Benzeggagh-Kenane, G_Ic, G_IIc). Since round 20 the law bends before its peak
  (trilinear): straight to `glue.prepeak_onset` (0.65) of the strength, then bending over
  to the strength at `glue.peak_slip_ratio` (2.0 since round 21; 1.5 in round 20) x the
  slip a straight line would reach it at (~0.075 mm for a hand-made lap), then softening
  linearly; the work done to separation is still G_c. A PVA bond line is a thermoplastic
  that yields before it parts: a pressed PVAc line's load-slip curve bends from 60-75 % of
  its peak and peaks at 0.05-0.07 mm, ~2.5x the slip of its own initial slope (Serrano
  2000, Lund TVSM-1012, Fig. 4.10), or from 0.4-0.75 of its peak at 0.04-0.135 mm with a
  negligible elastic part (Wernersson 1994, Lund TVSM-1006); montecarlo draws both
  (0.45-0.75, 1.1-3.8). A joint near its strength slips more
  and passes load to its neighbours, and a bridge that fails at its joints bends over
  before its peak (`test_the_glue_law_bends_before_its_peak`). A glue line counts as
  failing (`glue_damage`, the joint's share let go) only past its peak. tau_max is tau_g of the material file times
  the peak/average ratio of the simulated 20 mm lap test (`bridgesim lab`), so a joint
  reproduces the measured tau_g (checked in the tests). Peel strength = the lower of
  0.66 tau_max (a pressed PVAc bond line: 6.29 / 9.46 MPa, Serrano 2000) and the wood's
  tension across the face on its weaker side (f_t90 on side grain, Hankinson towards f_t
  on end grain): on 2 mm sticks the wood next to the glue parts first, so a stronger
  glue does not raise a lap's peel strength past its sticks'.
* **Workmanship**: each joint's glue strength scatters (`scatter.glue_cov`, 30 %, and a
  bridge-wide factor), and so does its shear stiffness, in proportion: a joint glued
  over less of its face (starved edges, gaps) is softer as well as weaker. Bearing in
  compression keeps the whole contact.
* **Creep**: PVA is a thermoplastic and creeps. Each glue line's shear slip has a Kelvin
  element in series with its spring (`glue.creep_factor` phi = 1: under a steady load
  the slip grows toward (1 + phi) x the elastic slip, retardation `glue.creep_time_s` =
  600 s), run in physical test time: the jack's travel at the real test rate
  (`crusher.test_rate_mm_per_min`), so it does not depend on the simulated jack speed.
  The elastic part of the slip carries the stress and drives the damage, so creep makes
  a joint softer, not weaker (`test_glue_creeps_in_test_time`: the relaxation of a held
  glue line matches the Kelvin solution within 0.1 %). Over the 10-40 s a test takes to
  its peak at 10 mm/min it adds a few % to the joints' slip; measure it with a lap
  joint held at a constant load.
* **Rate of loading** (since round 21): a PVA line's strength depends on its time to
  failure, tau(t) = tau(t_ref) (t / t_ref)^-n, `glue.rate_exponent` n = 0.1 (+16 % per
  decade faster, Wernersson 1994, Table 4.1; -40 % at a tenth of the rate, Serrano 2000:
  0.065-0.22, drawn by montecarlo), t_ref the time to failure of the lap tests tau_g
  describes (`glue.coupon_time_s`, 60 s). The crush runs on physical test time (the
  jack's travel at `crusher.test_rate_mm_per_min`, 10 mm/min, a placeholder nobody has
  timed, drawn 50 %): at each moment the glue holds what a lap failing then would show,
  so a bridge that peaks 13 s into the test (Bridge_v10) is ~16 % stronger at its glue
  than 60 s coupons, one that takes 3 min ~10 % weaker. A face through one piece of wood
  follows the wood's own exponent (`wood.rate_exponent`, 0.02: ~4 % per decade, USDA Wood
  Handbook ch. 5). The coupons and the screen take tau_g (no physical time); the crush
  reports when the real test would reach F_u (`test_time_at_Fu_s`). Give each glue test
  its `time_to_failure_s` (glue_joint and short_beam_shear templates): `calibrate` brings
  every one to t_ref (`test_the_glue_is_as_strong_as_a_test_failing_when_the_crush_does`).
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
  own section is not reduced. A face whose wood has crushed 0.1 mm deep on average is
  reported (`bearing_crushing`, with the load then): the first non-linearity of a node
  that bears, often before any glue lets go (on Bridge_v10 standard seed 0 the End
  Post's face on the knee at 99.6 % of F_u, just before its gussets' glue lets go). A strengthen step then does not offer more glue on that face (it
  changes nothing the grain takes): a bigger bearing face or the grain along the force.
  `screen` lists each face's `bearing` utilisation (glued or dry). A compact block bears
  across its grain (f_c90) unless a layup override gives it a grain (`layup: "13-ply
  UD"`, kind block).
* **End grain** holds glue poorly: the glue strength of a joint is multiplied by
  1 - e (1 - `glue.end_grain_factor`), e being the share of a side's plies meeting the
  joint end-on (0.5 placeholder: a member's end glued flat gets half, a 0/90 plate's edge
  three quarters). Joints that are mostly end grain are listed in a question.
* **End grain pulled apart** (since round 21): a glue line on end grain loaded normal to
  the end holds `glue.end_grain_tension_factor` (0.1) x the wood's tension strength along
  the grain (~4 MPa for craft sticks; end-grain butt joints reach ~25 % of it with special
  techniques, USDA Wood Handbook ch. 10), capped by the other side pulled across its face
  (f_t90 where a vertical's end sits on a chord's side): the fibres run along the pull,
  so the side-grain peel rule (x the end-grain factor: 0.7 MPa until round 21) was a tenth
  of it, and a published Howe truss's end-grain vertical foot held 2.6 MPa
  (`validation/README.md` §3.14). Shear on end grain keeps the end-grain factor.
* **Fit-up of bearing faces.** A member's end cut to bear or be glued on another part is
  not square: each end-grain face touches at one edge and stands off its seat by a seeded
  gap (0 to `scatter.bearing_gap_mm`, 0.3 mm: a 1 degree error over a 20 mm deep end is
  0.35 mm) at the other edge of its outline, in a random direction across it. The glue
  that fills the gap is weaker the wider it is (`gap_strength`): full strength up to
  `glue.gap_fill_mm` (0.15 mm, the strongest bond lines are 0.08-0.15 mm), 0.9 at 0.25 mm,
  0.6 at 0.5 mm and nothing from 1 mm (shimmed block-shear tests, Kurt 2006: -10 % at
  0.25, -39 % at 0.48 mm; a 0.4 mm PVAc line carried 1.38 MPa for 70 days, USDA FPL
  RP-400). Where it is too wide the face is dry: its glue points start broken and their
  two sides are held apart by the gap, so it bears only once the gap has closed. Dry glue
  points count neither as glued area nor as glue that failed. The perfect bridge
  (`--seed -1`) has none (`test_cut_ends_stand_off_their_seats_by_a_seeded_gap`). (Round
  17 took a face as dry wherever the gap passed 0.15 mm, which the literature review found
  too severe for PVAc; with the round-16 materials that had lowered Bridge_v10 quick seed
  0 by 11 %.)
* **Size effect.** A hand-made lap larger than the coupon tau_g is measured on (200 mm^2,
  the 20 x 10 mm lap) holds less on average: (A / 200)^(-1/m), a Weibull size effect of
  modulus `glue.weibull_m` (10 since round 20; range 6-20, drawn by montecarlo), 0.83
  for Bridge_v10's 1320 mm^2 gusset laps (more starved spots and uneven clamping over a
  larger area: Tannert, Vallee & Hehl 2012), carried no further than
  `glue.size_cap_mm2` (3000 mm^2: pressed glued crossings of CLT showed no size effect
  from 1600 to 29,000 mm^2, Flaig & Blass 2013). The fracture part of the size effect,
  shear lag along a long lap, is in the glue law itself (generalised Volkersen, §3.9 of
  the validation README), so m is the statistical part only. (Round 19 used m 6 and
  cited Debertolis et al. 2025 for gussets holding "1-3 MPa": those were their design
  inputs, not measurements; coupons cut from the gussets held 5.1 MPa, CoV 0.27.)
* **Laminations.** Two plates glued face to face (a deck's plies drawn as separate
  bodies, a gusset flat on a plate) are glued as the plies inside one laminated body:
  at `glue.interlaminar_MPa` (3.5 MPa) with no size effect and a pressed line's shear
  stiffness (`glue.lamination_K_s_N_per_mm3`, 200 N/mm^3: a pressed PVAc line ~150
  secant, 300-400 initially, Serrano 2000, in series with two 2 mm sticks in shear), as
  strong as the glue lines inside one laminated member (where those split); the layered
  shells themselves do not delaminate, so a bond line drawn inside one plate never fails
  (below). A lamination covers at least half of the smaller plate (`LAMINATION_COVER`):
  a short overlap of two plates (a deck's tongue slid into its groove on build day) is a
  hand-made lap (the adversarial review of round 20). Round 19 gave
  Bridge_v10's 71,370 mm^2 deck plies 0.375 x tau_g = 0.75 MPa as hand-made laps, a
  fifth of the same line inside one body (the literature review).
  `test_plates_glued_face_to_face_are_laminations`; inspect marks them `lamination`, and
  each joint's `shear_capacity_N` is its area times the strength its face is given.
* **Through the laminate** (since round 21): a joint glued on a ply face of a laminated
  member or plate puts its load into that ply, which hands (n - 1) / n of it to the
  others across its first glue line: the joint is no stronger than the laminate's
  interlaminar strength x n / (n - 1) (2 x the rolling shear of a 2-ply 0/90 gusset,
  3 MPa: 6 MPa; 13/12 x 3.5 MPa for a 13-ply member). The layered shells do not
  delaminate by themselves (no interlaminar failure in a shell element), so without this
  a gusset lap drawn at 5.1 MPa held on a 0/90 gusset of 2.5 MPa rolling shear (the
  adversarial review of round 20) (`test_laps_are_capped_by_what_the_laminate_passes_on`).
* **One piece drawn as two bodies** (a tenon and its beam, a seat laminated with its
  chord): the glue override `solid` gives the face between them the wood's own strength
  instead of a glue line's (round 18 used 10x glue, which made a tenon's root
  unbreakable): shear f_v / 1.5 averaged over the section, as the code check of a notched
  end (EN 1995-1-1 6.1.7, 6.5.2), tension across the face by the grain on either side (a
  block with no grain known has its fibres across it), the wood's fracture energies, no
  fit-up gap and no glue scatter (`test_a_solid_joint_breaks_at_the_wood_strength`).
* **Shear lag and the mesh.** The glue points of a joint hang on the nearest nodes of
  each part (rigid arms), so how much shear lag a joint shows depends on how many nodes
  that is. The lap coupon with its glue hung on 1, 2, 3, 5 and 9 nodes per side needs a
  peak/average ratio of 1.00, 1.045, 1.10, 1.08 and 1.075 to give tau_g with the round-20
  glue law (`glue.tau_peak_over_avg` 1.075; 1.10 with the bilinear law, whose sharper
  peak concentrates the shear at a lap's ends); the model uses 1 + 0.075 (1 - exp(-2.2
  (n - 1))) with n the larger node count of the two sides (within 3 % of each), so a joint's
  capacity (tau_g times its area) does not jump when the mesh changes. The ratio depends
  on the glue's toughness and peel strength: round 17 changed them and kept 1.155, which
  left every joint ~5 % stronger than tau_g (the adversarial review); the test now uses
  the simulation's own glue values.
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
growth rings and have defects: one craft stick broke at 0.52 of a solid birch strip of
its size in 3-point bending (Karplus 1996, ASEE, "Craft Stick Beams"), so the strengths
are about half of clear birch: f_b 50, f_t 40, f_c 30 MPa. The across-grain values
follow 2 mm birch veneer data (Krüger et al. 2025, doi:10.5281/zenodo.17201843) and the
Wood Handbook: f_v 8 MPa (veneer 9.2), f_t90 1.5 MPa (veneer 1.1-2.0), f_c90 6 MPa
(veneer 6.9, clear birch 4.1-7.4), rolling shear 3 MPa (silver birch ~3.5, Collins & Fink
2025). The stick modulus is about as high as clear birch's: dry craft sticks are 11-15 GPa
(the wave speed of 2.2 mm birch sticks measured by vibration, 4600 m/s, Sautel et al.,
Am. J. Phys. 85, 783 (2017), Table I: E = rho c0^2 = 11.6-14.8 GPa at 550-700 kg/m^3;
2 mm veneer 14.6 GPa). The crush uses 10.5 GPa (the low end, less 5 % for a modulus from vibration against a static one) and G = E/16, while the
material's E is a placeholder (`wood.E_placeholder_MPa`; the material file's 10 GPa is the
linear analysis's; a measured E from `calibrate` or `material.E_MPa` in the overrides
replaces it). Until round 17 the crush used 7.1 GPa, from a stick weighted at a table edge
(*Emergent Scientist* 1, 3 (2017)), which reads low because the edge lets the stick turn,
and cited Sautel's 5.0 GPa, which was for sticks soaked in water for three weeks (the
literature review of round 17). Stiffness, buckling and deflections scale with E, so
measure your own sticks: a free tap test (simulation.yaml) or 3-point bending. f_t (40
MPa) is below f_b on purpose: a stick in tension fails at its weakest cross-section (slope
of grain, defects), which bending loads only at one face, so graded timber has f_t,0 = 0.6
f_m (EN 338), while clear wood has the opposite. `bridgesim sensitivity` shows how much each
placeholder moves F_u for your design. Your own stick tests replace them
(`bridgesim calibrate`). In the fibre
sections: linear elastic; tension softens to zero after f_t with fracture energy G_f
(8 N/mm, element-length regularised); compression is straight to the proportional limit
(`wood.fc_proportional`, 0.65 f_c: clear wood 0.6-0.75, *Wood Handbook*), bends over
smoothly to f_c (reached at 1.35 f_c / E, unloading at E), then crushes, softening to
70 % (`test_wood_softens_before_it_crushes`), so a member near its crushing strength
loses stiffness before it crushes (tangent-modulus buckling). A single stick so modelled breaks in 3-point bending at the material's
f_b (within 4 %; the test allows 10 %): compression yielding on top and tension softening below give
the bending strength, as in wood. A member **splits in shear** when the shear on a
family of planes along it reaches their strength: on the glue planes between its plies,
1.5 V/A plus the torsional shear acting on those planes; across its width (a crack
through every ply), 1.5 V/A plus the torsional shear on those. Saint-Venant torsion of a
rectangle (Roark; `materials.rect_torsion`) puts the peak shear on the planes normal to
the short side and 0.74-1 of it on the others, so a flat laminate twisted shears its
glue lines at about 3/4 of the peak (`test_torsion_shears_each_plane_by_saint_venant`).
Across the width every ply has to fail: f_v for plies along or across the member, the
fibres' strength for plies crossing the plane at 45 degrees (a +-45 pair carries
in-plane shear as fibre tension and compression, netting analysis min(f_t, f_c) / 2),
a quadratic interaction between, averaged over the plies
(`test_plies_resist_a_split_across_the_width_with_their_fibres`). The glue
planes of a laminate (pressed while they cured, long and continuous) are not a hand-made
lap joint between parts (tau_g, an average over the lap): they split at the lower of the
bond line's strength (`glue.interlaminar_MPa`, 3.5 MPa since round 21: craft-stick laminates glued with Elmer's and pressed over their whole length split at 3.7 MPa, an upper bound, and hand clamps or weights press less evenly; PVA bond lines in pressed wood
laminates fail in the wood next to them) and what the weakest ply takes there, the
interlaminar shear running along its grain (f_v) and across it (rolling shear,
`wood.f_rolling_MPa`, 3 MPa) in proportion to its angle: f_v for a UD stack, the
rolling shear for a ply across the member (0-90), in between for +-45. A short-beam
shear test of a laminate measures it (`short_beam_shear_template.csv`, `calibrate`).
(Until round 12 it was the hand-made lap joint's average strength, 1.155 tau_g = 2.3
MPa, which confuses an average over a lap with stress peaks at its ends and partial
coverage with a pressed, continuous bond line. A PVAc bond line pressed while it cures
is stronger than birch in shear: EN 204 asks >= 10 MPa of a D1 PVAc adhesive in the EN
205 lap-shear test on beech, so the line fails in the wood next to it, at about f_v.) The shear is compared averaged, by length,
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
fracture energy G_f (8 N/mm); fibre compression crushes to a plateau at 70 % of f_c;
across-grain tension (sticks parting edge to edge, f_t90 = 1.5 MPa) softens with G_f90
(0.45 N/mm); across-grain compression yields at f_c90; in-plane shear softens with G_v
(1.3 N/mm). Cracks close: a broken ply still carries compression. A 0/90 sheet of sticks with unglued edges (`glue.plate_edges_glued` 0, asked as `plate_edges`) carries in-plane shear through the glued crossings of its plies, like cross-laminated timber with unglued narrow faces: ~0.68 of the shear stiffness and half the shear strength (Bogensperger, Moosbrugger & Silly 2010), or less where the glued crossings fail first in torsion: tau_T = 3 tau t* / a with t* = n t / (n - 1) per glued interface (Boggian, Andreolli & Tomasi 2019), which caps a 2-ply sheet of 2 x 10 mm sticks at 0.83 x the crossing's torsional strength, min(the bond line 4 MPa, 2.4 x rolling shear: Flaig & Blass 2013 measured torsion 3.5-4.7 MPa against rolling shear 1.4-1.5 on CLT crossings), 3.3 MPa against half of f_v, 4 (round 20). The softening strain of
each element is set from its size (crack band, Bažant & Oh 1983), so the energy a crack
dissipates does not depend on the mesh: a plate strip pulled apart dissipates
G_f x width x thickness within 2 % with 5, 2.5 and 1.25 mm elements
(`test_shell_fracture_energy_is_mesh_independent`). An element larger than the crack
band a mode allows (h_max = 2 E G_f / f^2: fibre tension ~105 mm, in-plane shear ~27 mm
with the round-18 values) cannot soften steeply enough; its strength in that mode is
lowered until its steepest softening dissipates exactly G_f, f' = f sqrt(rf / 1.05)
(Bažant & Oh 1983; Turon et al. 2007 do the same for cohesive elements on coarse meshes;
`test_elements_past_the_crack_band_dissipate_the_fracture_energy`). That is also about
what an energy criterion gives for a crack the size of the element, so a coarse mesh
cracks at a lower stress where a fine one resolves the stress peak. Until round 16 such
elements kept their strength and dissipated up to 3.5x G_f (on Bridge_v10 at standard,
5515 elements in shear), a coarse plate tougher than a fine one. They are counted in the
warnings, and a crush whose failure starts as a crack in such a plate is flagged
untrustworthy when the cracked elements are past the crack band of the mode they cracked
in (each crack records it: fibre tension, across the grain or shear): then its onset
depends on the mesh. With f_v 8 MPa nearly every standard element is past the shear
crack band, so until round 17 any plate crack was flagged, also one in fibre tension
that the mesh resolves (`test_a_crack_is_flagged_only_past_the_crack_band_of_its_own_
mode`). An element is removed only when every ply is torn wide open
(twice the strain at which it carries nothing) or when it is distorted beyond 0.5 rad.

**Scatter** (seeded, `--seed`; `--seed -1` for a perfect bridge): per body, E (CoV 12 %)
and strength (CoV 18 %), correlated (`scatter.E_strength_corr`, 0.7: stiffer wood is
stronger, MOE-MOR r = 0.6-0.8, which machine stress grading relies on; paired bending
tests measure it, `calibrate`), both for a single stick: a member of n plies averages its
sticks, which are partly alike (`scatter.ply_corr`, 0.3: one batch, often one tree), so
its CoV is that x sqrt((1 + (n - 1) 0.3) / n), 61 % of it for 10 plies (the laminating
effect on the scatter; on the mean strength it is not modelled), `test_stick_scatter_is_
correlated_and_averaged_by_lamination`; per joint, glue strength (CoV 30 %); per bar, an initial bow of
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
`scatter.amplitude` scales every seeded imperfection and scatter at once (0.1: a nearly
perfect bridge whose asymmetry is still the seed's, for a mesh or imperfection-
sensitivity study: `validation/convergence_study.py --amplitude`).

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
  crushing strength of the wood under it times the area it stands for, for the angle of
  each ply's grain to the vertical (Hankinson, averaged over the plies, as at the glue
  lines: f_c90 for wood pressed across the grain, f_c for a post standing on its end
  grain, a standing +-45 plate in between; `test_platform_bearing_follows_the_grain`);
  for a frame whose sticks follow its arms, the grain of the arm at that point (a UD
  L-shaped leg stands on its sticks' end grain: `test_a_frame_leg_bears_on_its_sticks_
  end_grain`), there and at its glue lines: a plate edge
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
  boundaries). **The window never reaches across a break**: each stretch between the
  holds (each starting when something breaks) is smoothed on its own, so a load that
  rises to a break and falls there reads its peak where it was. Until round 16 the
  window averaged across the drop: Bridge_v10's raw jack force peaked at 1879 kgf, the
  reported F_u was 1769 (-6 %, up to -9 % for a sharp peak), and its governing joint,
  which broke at the peak, read 1393 kgf, as if after it
  (`test_the_peak_before_a_break_is_read_where_it_was`). The smoothing is a
  Hann-weighted **local linear fit** whose window keeps its width near the end of a
  stretch, so the load at a break is the trend's end, not the raw sample there (round 16
  reflected the stretch oddly at its ends, which pinned the reading at each break to the
  raw, ringing force); only the record's start, at rest, is kept as it is. **Within a
  hold** the travel stands still while the load falls in time as parts let go, so there
  it is smoothed along time, over two periods of the bridge's lowest mode: each event
  in a hold reads the load at its moment (along the travel, every event of a hold read
  the hold's start; `test_the_load_at_a_break_and_in_a_hold_is_smoothed_not_raw`).
  Last, **the reading is kept within the raw force's envelope** over a centred period of
  the ringing: it never stands above the highest force the jack applied around that
  moment, nor below the lowest. A force that rounds over just before a break read 4-7 %
  high from the straight line fitted to the end of its stretch (the round-17 realism
  review: a Monte Carlo run of Bridge_v10 at 1553 kgf against a raw maximum of 1495, a
  light truss +6.6 %), and a fit after a survived break read the collapse that came
  later back at the break (`test_the_reading_stays_within_the_force_the_jack_applied`).
  The curve, F_u, the
  deflection at F_u and the load of every event (the last reading at or before it) are
  the reported one; `curve.csv` and the report's curve also give the raw force
  (`load_raw_kgf`). (Until round 10
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
  failure; the fall has to last one period of the bridge's lowest mode, so the dip of
  a bridge ringing after a break is not taken for its failure, and the failure is dated
  from the start of the fall; if the jack's travel ends while the load is still down,
  that was the failure), mid-span deflection > 50 mm, or contact of a
  part with the platform other than the original supports (§12.5) or with the press
  uprights (debris flying off a broken part does not count). A glued joint letting
  go inside the bridge is also a §12.5 failure ("shear, including joint failure"), but
  the judges only see it when the bridge visibly gives way, which shows as the load
  drop; the load at the first complete joint separation is reported separately
  (`Fu_first_joint_failure_kgf`). Every joint failure and member split is logged with
  the load at which it happened.
* **What governed** (`governing`): the break that started the fall of the load to the
  failure call, i.e. the first break from the moment the raw jack force, averaged over a
  centred period of the bridge's ringing, last stood within 5 % of F_u (a quarter of the
  online filter's time constant earlier). In round 17 that moment came from the reported
  load, which within a hold reads half a period late: on Bridge_v10 standard seed 0 the
  crack of the centre plate and the diagonal feet at node M, which started the fall, were
  taken as survived and a joint on the other truss was named (the realism review). A part that broke before that was survived: the load
  came back, as it did on Bridge_v6 seed 1 after a deck-stringer joint let go at the peak
  (5 % drop), before the End Post joints went at 96 % of the peak and the load fell 40 %.
  A member split in the same 2 ms as a member fracture is put down to the fracture; a
  clean break comes before a joint that only cracked or a member crushing, and of the
  breaks in the same instant the one that could carry the most (a diagonal's end-grain
  foot and its two gusset laps letting go together: the laps; strengthening the foot alone
  moved Bridge_v10's F_u 1 %, the laps 9 %, the adversarial review of round 20; the
  strengthen what-if covers every joint of the governing part that broke with it). Only what
  happened up to the failure call counts: what breaks after it is the collapse. When
  nothing broke, the joint whose glue let go during the fall (at least 20 % of its area
  between the load leaving its peak and the call) governs, although it has not reached
  the 30 % of `joint_cracking` yet; else "buckling or sway". A joint whose glue can carry
  under 2 % of F_u (a small end-grain joint, a toothpick's end) neither governs nor
  competes: its break is a symptom of the bridge deforming, not the cause of the fall
  (on a light Pratt truss the review crushed, such a joint was named governing). A member **crushing** (a
  fibre folding in compression past f_c, beyond its onset strain) is an event too: a
  laminated beam that fails in bending by crushing on its compression face is named so,
  and its what-if is a bigger section, not more glue. `competing` lists the other parts
  that broke within 10 % of F_u: the failure can start at one of them on another seed or
  mesh; `diagnose` measures which one F_u depends on.
* **Rate check** (`--rate-check`): the crush again at half the jack speed. What the
  speed adds to F_u is inertia, which grows with the kinetic energy, the square of the
  speed (Bridge_v10 quick seed 0, adversarial review of round 16: -2.4 % from 200 to 100
  mm/s, -0.5 % from 100 to 50), so `Fu_extrapolated_kgf` = (4 F(v/2) - F(v)) / 3
  estimates it for a jack at rest (the real test's mm/min). Until round 16 it was
  extrapolated linearly (2 F(v/2) - F(v)), which overshot.

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
  and contact damping. The loading plate is damped relative to the jack's motion, not
  to the platform: damping its steady descent would add alpha x 4.2 kg x the jack speed
  to the load (0.5 kgf at 50 mm/s, nothing for Bridge_v6, a tenth of a light truss's
  strength). The kinetic energy of the bridge (falling debris excluded) is
  reported against the work done by the jack while the load rises to 95 % of its peak
  (`kinetic_to_work_max`): below 0.05 the loading is quasi-static (a few % at most, the
  Abaqus/Explicit guideline for quasi-static analyses). The collapse after the
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
  `validation/README.md` §3.4, five seeds): the fine mesh (x0.5) is +4.6 % above the
  standard one (x0.75) on average, -1.1 % to +11 % seed by seed, and the same part
  governs on 2 of 5 seeds; on the others another of the sites competing within a few %
  (`competing`: the hub core, the End Post joints, the cross-beam's joints to the deck
  stringers) goes first. On Bridge_v10 at one jack speed (100 mm/s) quick / standard /
  fine gave 2506 / 2483 / 2479 kgf in round 18, and 2162 / 2061 / 2093 (seed 0) and
  2232 / 2000 / 2069 (seed 1) in round 20 (`validation/README.md` §3.10, §3.12), standard
  and fine at the same joint. The meshes are not monotone (standard the lowest of the
  three in rounds 19-20), so there is no Richardson estimate: over 10 seed pairs fine
  reads +2.3 % above standard on average (-1.1 to +11 %), which `montecarlo` takes out
  of its prediction (`uncertainty.mesh_bias_standard`, 0.02; quick, 1-12 % above
  standard, `mesh_bias_quick` -0.04), leaving ~5 % scatter (`uncertainty.mesh_cov`). Fine
  is not shown converged either (the adversarial review of round 20). Until round 16 the
  quick mesh was 4.6x too soft (plates on plates, below), and the round-16 spread was
  mostly the jack speed, which each fidelity changes with the mesh. A seed whose failure starts as a plate crack converges less
  well (the round-18 adversarial review: quick +6.6 % over standard, at the other end of
  the bridge). On Bridge_v6 the load curves coincided up to ~90 % of the peak, and the
  first joint to separate agreed within 10 %. **A small, light bridge converges less
  well**: the element sizes were set on Bridge_v6 (members 20-30 mm deep); on a light
  truss of 6 mm members (round 18) quick read 17.9, standard 34.5, fine 44.6 kgf and
  x0.35 44.8 (fine converged). Since round 21 a member's beam elements are at most 2x its
  depth at standard (12 mm on the light truss, in proportion at the other fidelities;
  members 20 mm deep and more unchanged), and the light truss read 46.2 kgf at standard
  (a deck-to-vertical joint) against 35.8 at fine (a floor beam splitting in shear):
  still not converged at standard, so read such a bridge's F_u at fine. A crush whose
  elements are more than 3x the depth of the members that fail, or 1.5x for members
  shallower than 10 mm, is flagged untrustworthy. A perfect bridge is symmetric, so which side gives way first is
  a bifurcation round-off decides; `--amplitude 0.1` keeps the seed's asymmetry at a
  tenth of its imperfections for a mesh study.
* **Uncertainty** (`montecarlo -n 20 --sample-placeholders`, standard): with the build
  scatter and the nine placeholder material values drawn per run, Bridge_v6's F_u was
  1720 / 2095 / 2317 kgf at p05 / p50 / p95 (-18 % / +11 %; round 14, every value drawn
  at 20 %); `sensitivity` ranks the lap joints' glue strength and toughness and the
  sticks' shear strength first. Each placeholder is drawn with its own prior, about the
  half-width of the literature or craft-stick range at 90 %: the stick modulus 30 % (dry
  craft sticks 11-15 GPa by vibration; until round 18 the draws were centred on the
  material file's 10 GPa while each crush used 7.1, now on the crush's own 11), f_t 30 %,
  f_c, f_v, f_c90, f_t90, rolling shear and peel 25 %, the pressed bond line 30 %, the
  glue's shear toughness, the plies' shear fracture energy and the end-grain factor
  45 %, the across-grain fracture energy 35 %, the wood's fracture energy 40 %, the
  fit-up gap 50 %, the lap joints' glue strength 50 % (hand-made PVA laps on craft
  sticks: nothing published, bounded at 2-6 MPa by the bond line's, the stick surface's
  and the peel strengths). Since round 18 the draws include the values that set where a
  bridge fails (f_t90, the bond line, G_v, peel, the fit-up gap: the adversarial review
  found f_v draws inert in members, whose glue lines split first). Since round 19 the
  draws are a Latin hypercube over the runs: each value falls once in each of n
  equal-probability strata of its prior, so a few runs still cover its range (with plain
  draws seeds 1-8 all drew the glue's strength below its placeholder, a 1-in-256 chance
  that pulled the median down). The values F_u is most sensitive to are the least known. `montecarlo` also gives a **total range**
  (`total_p05_kgf`, `total_p95_kgf`): the runs' F_u, corrected for the mesh's bias
  (`uncertainty.mesh_bias_*`: +2 % at standard, -4 % at quick, toward fine), each times a
  lognormal mesh error (`uncertainty.mesh_cov`, 5 %, the scatter of fine against
  standard) and a model-form error, both median-centred as the placeholders are (until
  round 21 mean-centred, which put the prediction 4.5 % lower for a convention: the
  adversarial review of round 20). The model-form error
  (`uncertainty.model_cov`, 30 % since round 18: Bridge_v10's F_u moved 14-30 % per
  release over rounds 15-17; what the equations leave out for a real bridge, until a
  physical test measures it; JCSS Probabilistic Model Code part 3.9 gives 10-15 % for
  mature, tested connection models, so not less before tests; `calibrate-bridge` sets it
  to 15 % once a test of the bridge has been fitted, and lists the fitted values as
  measured, so montecarlo stops redrawing them): the range to give a team
  before any test, the runs' range being what the model knows it does not know. The
  **recommended prediction** (the 1/F-weighted median, which minimises the expected
  §12.7 error) is taken over the runs times that same error, which pulls it below the
  runs' own (`recommended_prediction_runs_kgf`), and comes with a bootstrap interval
  over the runs (`recommended_interval_kgf`: wide with 8 runs in two clusters, as on
  Bridge_v10). Runs flagged untrustworthy are listed apart and left out while at least
  half the runs are trustworthy, and a rigid block under the loading plate is named in
  the note (it cannot fail in any run).
* **Crack onset is local.** A plate left as a shell cracks where its strain first
  reaches the strength in one element; the crack band makes the energy of the crack
  independent of the mesh, not its onset at a sharp stress raiser (the end of a glued
  lap, a re-entrant corner), which comes at a lower load the finer the mesh. The narrow
  plates where that mattered on Bridge_v6 are frames since round 11, and its x0.5 mesh
  converges with x0.75 (above); on a design whose shells have sharp stress raisers in
  the load path, check with `--fidelity fine` and `convergence_study.py --scales
  1,0.75,0.5`. A nonlocal process-zone onset
  was tried and not adopted (it doubled the strength of off-axis plies;
  `validation/README.md` §3.4).
* Time is compressed: the simulated jack moves at ~100 mm/s instead of the real
  mm/min. The material laws have no rate effect, so this changes nothing as long as the
  run stays quasi-static; the replay converts back to real test time. **The jack speed
  follows the bridge**: the speed of each fidelity (200 / 100 / 50 mm/s) is for a bridge
  whose lowest mode under the plate (the plate's mass plus half the bridge's on the
  bridge's linear stiffness under the plate, as `screen` computes it) has a period of
  at most 10 ms (Bridge_v6: ~8 ms) that fails after a few mm of jack travel (Bridge_v6:
  ~6). How far a crush is from quasi-static is set by how far the jack moves in one
  period of the bridge's ringing against the travel reached (the lag of the bridge
  behind the jack, relative): a softer or lighter bridge is crushed slower in
  proportion to its period, and its jack speeds up again in proportion to the travel
  once that passes 6 mm (a soft bridge that fails at large deflection needs no slow
  jack there), at most 16x slower (`quality.speed_mm_s` at the start,
  `quality.fundamental_period_ms`). A bridge still too fast is caught by the
  quasi-static check (`kinetic_to_work_max`) and `--rate-check` measures what is left.
  Such a bridge is also started over two of its periods and sped up again after a hold
  over one (a stiff one over 20 ms and 2 ms): a change of speed made faster than the
  period kicks the plate and sets the bridge ringing by about the speed x period / 2 pi
  (two stringers over a metre, period 0.17 s: 4 mm at 150 mm/s, and the load line
  started at 5.5 kgf instead of zero, a bridge that carries ~9), and a linear ramp over
  whole periods leaves no ringing.
  `test_a_soft_bridge_is_loaded_without_a_kick`.
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
  variation inside one body (a plate's butt joints are one factor on its fibre tension
  strength, not resolved joint by joint). A member that splits
  in shear switches to its split section at once (no fracture energy for the split
  itself, as a pressed glue line fails brittly in a short-beam shear test; the energy it
  releases goes into the dynamics and the damping); its axial strength is kept.
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
* Creep of the wood and the time the bridge waits loaded (the glue's creep and its
  rate-dependent strength are modelled, section 3, in physical test time). Moisture is
  one input, `wood.moisture_excess_pct` (0): how many points wetter the bridge's wood is
  than its coupons' (members laminated days before the test: a PVA line brings ~+8
  points to a 2 mm ply, evened out only after a week), lowering E and G 2 %, f_c 5.7 %,
  f_c90 5 % and f_v 2.4 % per point (USDA Wood Handbook eq. 5-3, paper birch); make the
  coupons alongside the members, or weigh a member daily after gluing until its mass is
  steady. The simulated curve shows the toe from seating, then is near-linear: on
  Bridge_v10 (round 20) its secant at the peak is 0.98-1.00 of the 10-40 % secant
  against the jack's travel and 0.94-0.98 against the deflection, though 70-100 glue lines
  pass their own peak locally (the ends of laps) before F_u: their share of the bridge's
  compliance is small. Glued timber trusses tested to failure were "mostly linear elastic
  until failure, which occurred in a brittle manner", their secant at the peak 0.84-0.92
  of the 10-40 % secant (Debertolis et al. 2025, 7 m span): the model is stiffer to its
  peak than that, and a real stick bridge (creep under a pumped jack, joints that seat,
  members bowing) may bend more before its peak. Stick coupons loaded at the test's own pace (failure in
  about the same minutes as the crush test; EN 408 asks 300 +- 120 s) carry the wood's
  creep of that duration in the E that `calibrate` fits, so a calibrated model includes
  it to first order; what is left out is how creep grows with the stress near the peak.
