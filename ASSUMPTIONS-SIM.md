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
| slender extrusion, section no flatter than 1:4 | **bar**: chain of corotational fibre beams on its centre line, ~25-30 mm elements (>= 4), real section and length | members carry axial force + bending; buckling needs >= 4 elements |
| thin extrusion (width >= 3 x thickness) | **shell**: CST membrane + DKT bending triangles on the mid-surface, meshed from the real outline (holes > element size kept) | gussets, webs, decks, diaphragms |
| compact piece, block, short toothpick | **rigid body** with its real mass and inertia | much stiffer than the members |
| floss (by name) | **cable**: tension only, snaps at its breaking load | §8.4.5 |

Laminates come from the body name (`10-ply UD`, `2-ply 0-90`, `4-ply +-45`, `block`,
`round toothpick`, `waxed floss`) or the overrides file; an unknown one is assumed UD
along the long axis and reported as a question. Stiffness from classical lamination
theory with orthotropic plies (E_L, E_T, G_LT, nu_LT); strength by first-ply max strain
along the fibres and in shear (across-the-grain cracking only counts when no ply runs
along the load). UD bars and plates take their grain along the long axis (or the global
X with "along the span"; for non-rectangular UD plates, along the nearest long edge).

## 3. Joints

* Every pair of coplanar touching faces of two bodies is a **glued patch** of the exact
  overlap shape. The glue is a grid of cohesive springs (one per node of the finer mesh,
  each with its share of the area) attached to both bodies through rigid arms at their
  real positions: joint eccentricity, peel from bending and the joint's rotational
  stiffness all come out of the geometry.
* Glue law: penalty stiffness K_n, K_s (glue line + adjacent wood), quadratic initiation
  (sigma/sigma_max)^2 + (tau/tau_max)^2 = 1, linear softening with mixed-mode fracture
  energy (Benzeggagh-Kenane, G_Ic, G_IIc). tau_max is tau_g of the material file times
  1.17, the ratio found by the simulated 20 mm lap test (`bridgesim lab`), so the
  simulated lap test reproduces the measured tau_g (checked in the tests). Peel strength = 0.5 tau_max.
* A broken glue line keeps **bearing in compression** and **slides with friction**
  (mu = 0.4): pieces still lean on each other after the glue lets go.
* Bodies that touch nothing are tied to the nearest part (and reported); floss ends
  attach to the part they touch.

## 4. Materials (placeholders until measured)

Wood along the grain: E, f_t, f_c, f_v from `materials/popsicle_birch.yaml`. For scale,
the USDA *Wood Handbook* (FPL-GTR-190, Table 5-3a, clear wood at 12 % moisture) gives
paper birch E = 11.0 GPa, modulus of rupture 85 MPa, compression parallel to grain 39 MPa,
shear 8.3 MPa (yellow birch: 13.9 GPa, 114, 56, 13 MPa). Craft sticks are thin, cut across
growth rings and have defects, so the placeholders (10 GPa; f_b 50, f_c 30, f_v 6 MPa)
are deliberately lower. Your own stick tests replace them (`bridgesim calibrate`). In the fibre
sections: linear elastic; tension softens to zero after f_t with fracture energy G_f
(3 N/mm, element-length regularised); compression yields at f_c and softens to 70 %
(crushing). Members split when 1.5 V/A + T/W_t > f_v. Plates crack when a ply strain
limit is passed (its stiffness falls to zero over a small strain range). Glue and floss: see
`materials/simulation.yaml`, every value with its source and how to measure it.

**Scatter** (seeded, `--seed`; `--seed -1` for a perfect bridge): per body, E (CoV 12 %)
and strength (CoV 18 %); per joint, glue strength (CoV 30 %); per bar, an initial bow of
L/400 on average (0.3-1.7 of that) in a random direction. Monte Carlo over seeds gives
the spread of F_u.

## 5. The test

* The bridge stands on the crusher's flat steel base platform: unilateral contact with
  Coulomb friction (mu = 0.35), no anchorage (§8.3). Every vertex of every body, and a
  grid on downward faces at table level, can touch it.
* Gravity acts on everything. The bridge first settles under its own weight (a bridge
  that keeps sagging fails by *instability*).
* The 200 x 90 mm loading plate (§12.5; two 15 mm steel plates, 3 kg, Appendix B) is a
  rigid body resting on the deck at mid-span and the deck centreline. The jack pushes on
  its hemispherical seat, so the plate tilts freely; plate-bridge contact is unilateral
  with friction (mu = 0.3) on any part of the bridge that reaches the plate.
* The jack moves down at constant speed (displacement control); its force is the
  load-cell reading. The crusher + load cell stiffness is 20 kN/mm.
* **F_u** = peak jack force up to the moment the bridge is judged failed: a load drop to
  half the peak after the peak (collapse), mid-span deflection > 50 mm, or contact of a
  part with the platform other than the original supports (§12.5). Joint failures and
  member splits are logged with the load at which they happened.

## 6. Numerics

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
  and contact damping. The kinetic energy is reported against the work done by the jack
  (`kinetic_to_work_max`): below 0.1 the test is quasi-static.
* Time is compressed: the simulated jack moves at ~100 mm/s instead of the real
  mm/min. The material laws have no rate effect, so this changes nothing as long as the
  run stays quasi-static; the replay converts back to real test time.
* Validated in `tests/test_sim.py`: cantilever (0.04 % vs PL^3/3EI), large-deflection
  elastica (Bisshopp-Drucker, < 1 %), Euler buckling of a stick, 3-point bending
  stiffness, lap-joint strength, energy conservation, DKT patch test, compiled kernels
  identical to the NumPy reference.

## 7. Not modelled (yet)

* Glue lines are perfect over the whole touching area: starved or gappy joints only
  through `glue_factor` in the overrides.
* Splitting of a member along the grain other than by shear, local crushing under a
  concentrated load at a joint, knots and stick-to-stick variation inside one body.
* Thick plates (gussets 20 mm thick) use thin-plate (Kirchhoff) elements, sized at
  >= 2.5 x their thickness.
* Contact between two bridge parts that were not touching at the start (only parts that
  were touching keep bearing on each other after their glue fails).
* Creep, humidity, and the time the bridge waits loaded.
