# Modelling assumptions (bridgesim v0.1)

Every number bridgesim predicts rests on the assumptions below. They also appear in the
app (*Assumptions & method* tab) and in the exported Design Validation report. If you
use the tool for your Technical Presentation (rulebook §10), present these assumptions
and say which ones you replaced with your own measurements.

Units: mm, N, MPa (= N/mm²); mass in kg. 1 kgf = 9.80665 N.
Axes: X along the bridge, Y up (table at Y = 0), Z across.

## 1. Material (placeholders)

All values in `materials/popsicle_birch.yaml` are **assumed placeholders**, not data:

| Property | Value | Notes |
|---|---|---|
| E | 10,000 MPa | Measure: three-point bending, $E = F L^3 / (48\,\delta\, I)$ |
| G | E/16 = 625 MPa | Typical wood ratio |
| ν | 0.3 | Only used by the solver's material definition |
| f_t (tension) | 40 MPa | Parallel to grain |
| f_c (compression) | 30 MPa | Crushing, parallel to grain |
| f_b (bending) | 50 MPa | Modulus of rupture |
| f_v (shear) | 6 MPa | Parallel to grain, $\tau = 1.5V/A$ |
| density | 650 kg/m³ | |
| τ_g (glue shear) | 2.0 MPa | Lap-joint average shear strength |
| glue mass fraction | 0.10 | |
| glued overlap / faces | 20 mm / 2 | Per member end |
| stick | 115 × 10 × 2 mm | |

The app shows a banner and an *assumed* badge on every value until you mark it *measured*.

## 2. Structural model

1. **Linear elastic, small displacements.** One linear static solve (Pynite) at the
   reference load $P_{ref}$ = 1000 N. No P-Δ, no initial imperfections, no material
   non-linearity, no joint slip, no creep.
2. **Frame elements on centre lines.** Each member is a prismatic 3D Euler-Bernoulli frame
   element joining node centres. Joint size (eccentricity, gussets) is ignored.
3. **Laminated sections act as one solid rectangle.** n sticks glued face to face are a
   single b × d rectangle with perfect glue lines (full composite action). This
   over-estimates stiffness and strength of poorly glued laminates.
4. **No splice weakening.** Members longer than one stick (115 mm) must be spliced; v0.1
   does not reduce their strength or stiffness at splices.
5. **Section orientation.** The depth d lies in the member's primary bending plane
   (Pynite local y): vertical for horizontal members, in the truss plane for truss
   members. `flat` stacks grow in depth (b = 10, d = 2n); `on_edge` stacks grow in width
   (b = 2n, d = 10). Verified by `tests/test_orientation.py`.
6. **Joints rigid by default.** Glued joints are closer to rigid than pinned. The
   `pinned` switch releases the in-plane moment (Rz) at both ends of diagonals and
   verticals, and both bending moments of bracing members. Chords, floor beams, struts
   and piers remain continuous, otherwise the end portals become a mechanism.
7. **Deck is non-structural.** It transfers the plate load to the floor beams as simply
   supported strips and adds mass, but has no stiffness in the model.
8. **Self-weight ignored.** A 2 kg bridge weighs about 20 N, well under 1 % of a typical
   failure load.
9. **Torsion ignored** in the member checks (it is still in the stiffness model).

## 3. Load (rulebook §12.5)

10. **Crusher plate**: size taken from the rules file (`crushing`, 200 × 90 mm for 2027),
    uniform load over its length along X, centred at mid-span (midpoint
    between the support groups). Self-aligning hemispherical loading point
    assumed, so no eccentricity.
11. **Load path**: the deck spans as simply supported strips between floor beams
    (lever rule, see `loads.py`). All load goes to the floor-beam **centre** nodes
    (z = 0). Real load is spread over 90 mm, so the floor-beam moment is over-estimated
    by up to about 25 % (conservative).

## 4. Supports (rulebook §8.3: no anchorage)

12. One end's pier bases restrain DX, DY, DZ; the other end's restrain DY, DZ
    (longitudinal roller). Rotations are free.
13. These restraints are **bilateral** in the model, but the real bridge only rests on
    the platform. The analysis therefore **warns** when a support needs a hold-down force
    (uplift) or a horizontal force above an assumed friction coefficient μ = 0.4
    (sliding, e.g. arch thrust). Such results are unconservative; never ignore the warning.

## 5. Capacity checks and failure load

14. Capacities (wood parallel to grain):
    $N_{t,R} = f_t A$; $N_{c,R} = \min(f_c A, P_{cr})$ with
    $P_{cr} = \pi^2 E I_{min}/(KL)^2$; $M_R = f_b S$; $V_R = f_v A / 1.5$.
15. **Buckling** uses the full member length, K = 1 (editable per member) and the weaker
    axis. Conservative for chords braced at every node; unconservative for crooked
    sticks (imperfections arrive in v0.2).
16. **Interaction**: $U = |N|/N_R + |M_y|/M_{R,y} + |M_z|/M_{R,z}$, linear, without moment
    amplification. Member shear (resultant of the two local shears, $1.5V/A \le f_v$)
    and glued-joint shear are separate checks.
17. **Glued joints (placeholder)**: capacity $\tau_g \times$ overlap $\times w \times$ faces,
    checked against the resultant member-end force. For members in a truss plane
    (diagonals, verticals, chords, piers) $w = d$, the in-plane depth: gussets lying in the
    truss plane can only glue to that face (for an on-edge stack it is the outer stick's
    broad face; the other face is made of stick edges). For floor beams and bracing
    $w = \max(b, d)$, the broad face of the laminate. By default the chord groups are
    treated as continuous (no joint check); `glue.exclude_groups` in the material file sets
    which groups, and the report lists them. Your real joint detail decides the true glue
    area: test it. A member with a gusset or a longer lap can state its own glued area per
    end (`glue_area_mm2` in the bridge file); it then replaces overlap × w × faces, and
    that member's joints are checked even if its group is otherwise continuous.
18. **Global (system) buckling**: linear eigenvalue analysis $(K + \lambda K_g)\phi = 0$ of
    the whole frame, with $K_g$ from the member axial forces at $P_{ref}$. It catches modes
    the member check cannot, such as sway of the legs or lateral buckling of an unbraced top
    chord. During buckling the pier bases at **both** ends are assumed held in X by
    friction (no horizontal force is needed at the onset of buckling). Pier bases have no
    rotational restraint (conservative). Each member is split into two elements, which
    puts system modes within about 0.2 % of a fine mesh; buckling of a single member
    between joints is left to the member check. Perfect geometry: this is an upper bound
    on the real instability load. When $F_{u,p}$ exceeds half of $F_{cr}$ the results warn
    that imperfections would be amplified by about $1/(1 - F_{u,p}/F_{cr})$ (P-Δ is not
    modelled in v0.1).
19. **First failure = bridge failure.** $F_{u,p} = \min(P_{ref}/\max U_i,\ P_{ref}\cdot
    50/\delta_{ref},\ F_{cr})$. Redistribution after the first member fails (common for
    bracing) is ignored, and so is "contact with the crusher" failure.

## 6. Mass and measurements

20. Mass = density × (Σ member volumes on centre-line lengths + deck plate of clear width ×
    length × thickness + extra wood) × (1 + glue fraction). Centre-line lengths double count
    the wood inside joints (slightly conservative for §8.8). Extra wood is everything that
    is not a modelled member, such as gusset and splice plates or blocks, listed in the
    bridge file under `extra_wood` (volume and count of each piece). It adds mass but no
    stiffness or strength.
21. Rule measurements treat members as solid prisms. Values are rounded like the judges
    do (nearest mm, nearest 0.01 kg) before checking. Ambiguous rules use the stricter
    reading; see `docs/open-questions.md`. A member ending on the table is taken as cut
    flush there over an end zone of two section sizes (max(b, d)); beyond that its section
    must stay above the table. A member reaching the table at more than about 14° always
    passes; a shallower one may fail the platform check, since it would lose much of its
    section to the cut.

## Deviations from the original v0.1 specification

* Added a **global elastic buckling** analysis (rulebook "instability" failure mode).
  Without it, the default design's 180 mm piers looked fine as K = 1 members but actually
  sway longitudinally at about 234 kgf, below the strength estimate; the default piers were
  stiffened (10 sticks flat) as a result.
* Added a member **shear** check ($f_v$ = 6 MPa, assumed) because the rulebook names
  shear as a failure mode (§12.5).
* Joint check uses the **resultant** end force (axial and shear) rather than axial force
  alone (floor beams load their joints mainly in shear), on the glued face described in
  item 17.
* The glue mass fraction is applied to the **deck** as well as the members.
* The span rule checks both the inner-face clear span and the pier centre-to-centre span
  (stricter reading of §8.2.1.1 vs. Figure 2).
