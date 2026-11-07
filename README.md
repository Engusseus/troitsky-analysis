# bridgesim: free 3D analysis for Troitsky popsicle-stick bridges

**bridgesim** is a free, open-source tool for teams in the
Troitsky Bridge Building Competition (ECA, Concordia
University). Design a popsicle-stick bridge, and bridgesim will:

* run a **3D frame finite-element analysis** of the whole bridge under the Crushing Day
  load (a 200 × 90 mm plate at mid-span, rulebook §12.5);
* show **axial force, shear, bending moment and utilisation** in every member, the
  deformed shape, and which member fails first, and how;
* predict the **ultimate load** $F_{u,p}$ (for the Load Prediction Form, §12.7), the
  **mass** and the **structural efficiency** $\eta_s = F_u/m$ (§12.6);
* check the bridge against the **2027 dimensional rules** (§8), with penalties and bans;
* export **shear and bending-moment diagrams** for the poster (§12.4) and an
  **"Assumptions, Method, Results" report** for the Design Validation (§10).

No expensive commercial software is needed. It runs on any laptop, and can run fully in
the browser.

> **Status: v0.2.** Two tools in one package:
> * **Crush simulation of SolidWorks bridges** (new): drop in your `.SLDPRT` and get a
>   nonlinear, dynamic simulation of the §12.5 crushing test, in which glue joints peel
>   and break, members buckle and crack, floss snaps and the bridge collapses, with an
>   interactive 3D replay. See [Crush simulation of a SolidWorks bridge](#crush-simulation-of-a-solidworks-bridge).
> * **Linear frame analysis** of centre-line models (YAML / parametric Warren truss), with
>   first-failure prediction.
>
> **All material values are placeholders until you test your own sticks and glue.** See
> [Limitations](#v01-limitations).

## Quick start

The quickest route is [uv](https://docs.astral.sh/uv/) (it installs the right Python for
you). No uv? See [Without uv](#without-uv-plain-python-and-pip) below.

```bash
git clone https://github.com/engusseus/troitsky-analysis.git
cd troitsky-analysis
uv sync                                          # install
uv run streamlit run app/streamlit_app.py        # web app at http://localhost:8501
uv run bridgesim run examples/warren_2027.yaml   # command line
```

Command line:

```bash
uv run bridgesim run examples/warren_2027.yaml --out results/   # report, CSVs, PNGs
uv run bridgesim check my_bridge.yaml                           # rules only (exit 1 if any rule fails)
uv run bridgesim generate warren --span-mm 1100 --out my_bridge.yaml
```

### Without uv (plain Python and pip)

With Python 3.11 or newer (for example from python.org) you can install into a virtual
environment inside the repository instead. Nothing is installed system-wide.

Windows (PowerShell or Command Prompt):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[app]"   # drop [app] for the command line only
.venv\Scripts\bridgesim.exe run examples\warren_2027.yaml --out results
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

macOS and Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[app]"           # drop [app] for the command line only
.venv/bin/bridgesim run examples/warren_2027.yaml --out results
.venv/bin/python -m streamlit run app/streamlit_app.py
```

The console script is `.venv\Scripts\bridgesim.exe` on Windows and `.venv/bin/bridgesim`
elsewhere: use it wherever this README says `uv run bridgesim`. pip installs the newest
versions that `pyproject.toml` allows rather than the exact ones in `uv.lock` (Pynite is
pinned in both).

### Default design (placeholder materials, illustrative only)

The bundled `examples/warren_2027.yaml` (span 1150 mm, 8 panels, 260 mm deep, deck 170 mm
wide at 200 mm) passes every computable 2027 rule:

| Result | Value |
|---|---|
| Predicted ultimate load $F_{u,p}$ | **≈ 316 kgf** (3097 N) |
| Governs | glued joints of the end diagonals, followed by floor-beam bending under the plate (86 % utilised) |
| Global (system) buckling load | ≈ 430 kgf (the whole bridge swaying lengthwise on its piers); $F_{u,p}$ is 73 % of it, so the app warns that imperfections would be amplified |
| Mass | 2.11 kg (≈ 1290 sticks) |
| Efficiency $\eta_s$ | ≈ 149 kgf/kg |
| Mid-span deflection at $F_{u,p}$ | 3.9 mm (limit 50 mm) |

These numbers use **assumed** material strengths and will change once you enter measured
values.

## Crush simulation of a SolidWorks bridge

```bash
uv sync --extra sim                               # numba: ~10x faster simulation
uv run bridgesim inspect Bridge_v6.SLDPRT         # what was understood + questions
uv run bridgesim crush Bridge_v6.SLDPRT --out results/   # minutes; results/replay.html
uv run streamlit run app/streamlit_app.py         # page "SolidWorks crush test": drag & drop
```

**What happens to your file.** The SolidWorks part is read directly (the Parasolid model
inside it; no SolidWorks licence needed). Every solid body is rebuilt with its name. Each
body becomes what an engineer would model it as: a **member** (corotational
shear-flexible beam with a fibre section that cracks in tension and crushes in
compression; tapered members follow their profile), a **plate** (shell
elements meshed on its real outline, holes included, that crack ply by ply, each
crack dissipating its fracture energy whatever the element size; where a crack starts
still depends on the mesh), a **rigid block**, or **floss** (a tension-only cable
that snaps). The layup comes from the body name: `10-ply UD`, `2-ply 0-90`, `4-ply +-45`,
`block`, `round toothpick`, `waxed floss`. Laminates use classical lamination theory;
members longer than one stick lose tension strength at their butt joints. Every place
where two bodies touch face to face becomes a **glue joint** of the exact overlap shape.
It peels and slides, fails in mixed mode, and keeps bearing with friction once broken.
Parts that meet during the collapse push on each other (contact), and wood crushes
locally under the plate and at the supports (across or along the grain, as it bears).
The glue lines crush by grain direction too, and end-grain joints are weaker.

**The test, as at the competition.** The bridge stands on the press's flat steel
platform with friction and no anchorage (its corners a little out of level, its
bearing faces seating in), and settles under its own weight. The 200 x 90
mm loading plate is placed on the deck at mid-span; it is a rigid body that tilts on the
jack's hemisphere. The jack then moves down at constant speed, and its force is the
load-cell reading. The run stops after the bridge has failed by a §12.5 criterion: a load
drop (collapse), more than 50 mm of deflection, or contact with the platform. You get
the **predicted ultimate load**, the **failure sequence** (what broke, in order, at what
load), the load–deflection curve, the 2027 rules measured on the real solids, and an
**interactive replay** of the crush, in real test time or faster.

**For AI agents.** Every command has `--json`, `bridgesim schema` prints the JSON
schemas, and `bridgesim mcp` serves the same functions as MCP tools. `inspect` returns
**questions** when the design is ambiguous or suspect. Examples: an unnamed body, parts
that overlap, a joint much weaker than its member, a failing rule, an assumed density
pushing the mass over 6 kg. The answers go in an overrides YAML. Every report ends
with `next_steps`: what to do next, with the exact command, and for the parts that broke
first a what-if (a thicker member, a bigger glued area) that `bridgesim ab` tests
before anyone touches the CAD. `bridgesim compare` scores the simulation against a real
crush test (or places it in the `montecarlo` range), `bridgesim calibrate-bridge` fits
the model to it, and `bridgesim lab` and `bridgesim calibrate` do the same at the coupon
level; `bridgesim sensitivity` says which material value to measure first. See
[AGENTS.md](AGENTS.md).
The method and every assumption are in [ASSUMPTIONS-SIM.md](ASSUMPTIONS-SIM.md). The
simulation is verified against textbook solutions in the tests but **not yet validated
against a real crush test**: with placeholder materials, treat F_u as an estimate and
rely on comparisons between designs.

## The app

* **Design panel**: span, panels, truss height, deck height and width, bracing, joint
  fixity, and sticks per member for every group (`flat` or `on_edge` stacks). Or upload a
  `bridge.yaml`.
* **Material panel**: every value is editable and carries an *assumed* / *measured*
  badge. A banner stays up while anything is assumed. You can also upload a material YAML.
* **3D view**: members coloured by utilisation at the predicted failure load. The
  governing member is shown in red and labelled. Toggle between deformed (scaled),
  first global buckling mode, and undeformed views. Hover a member for its force,
  capacity and governing check. The crusher plate is drawn on the deck.
* **Results**: $F_{u,p}$ in kgf and N, mass, $\eta_s$, governing member and mode, and
  deflection at $F_{u,p}$ against 50 mm.
* **Rule check**: ✓/✗, penalty and rulebook section for each rule, with total penalty and
  bans shown prominently. A failed geometric rule names what causes it (the members in
  the clear-span box or the cart path, the highest member, ...). Ambiguous rules are
  flagged.
* **Tabs**: load–deflection, the 10 most critical members, poster-ready SFD/BMD (whole
  bridge, a chord, or one member), A/B design comparison, exports (bridge.yaml, CSV,
  PNG, HTML/Markdown report), and assumptions & method.

### In the browser (no install)

The app can run entirely in the browser through [stlite](https://github.com/whitphx/stlite)
(Streamlit on Pyodide). See [docs/web-spike.md](docs/web-spike.md). A GitHub Pages
workflow (`.github/workflows/pages.yml`) is ready but **manual**: it only publishes when
a maintainer enables Pages and runs it. The first visit downloads about 60 MB.

## How it works

1. **Model.** Every member is a 3D frame element on its centre line, with the rectangular
   section of its glued stick stack. Joints are rigid by default, with a pinned option for
   comparison. Pier bases rest on the platform: one end is pinned, the other slides
   longitudinally, and nothing is anchored (§8.3). Solver:
   [Pynite](https://github.com/JWock82/Pynite) 3.2.0.
2. **Load path.** The crusher plate's load is spread over 200 mm and passed through the
   (non-structural) deck to the floor beams as simply supported strips:
   $$R = \frac{P}{200}(b-a), \quad F_i \mathrel{+}= R\frac{x_{i+1}-\bar x}{p}, \quad F_{i+1} \mathrel{+}= R\frac{\bar x - x_i}{p}$$
3. **Checks.** Each member gets tension $f_t A$, compression $\min(f_c A,\ \pi^2 E I_{min}/(KL)^2)$,
   bending $f_b S$ and shear capacities, plus a glued-joint check. Utilisation is
   $$U_i = \frac{|N|}{N_R} + \frac{|M_y|}{M_{R,y}} + \frac{|M_z|}{M_{R,z}}$$
4. **Global buckling ("instability", §12.5).** A linear eigenvalue analysis of the whole
   frame finds system modes that single-member checks miss, such as sway of tall legs or
   sideways buckling of an unbraced top chord:
   $$(K + \lambda K_g)\,\phi = 0, \qquad F_{cr} = \lambda_{cr} P_{ref}$$
   Each mode is described by its direction and the nodes that move most (for example
   "sideways (Z) movement, largest at T3n, T3f, T4n, T4f"), with the first three loads.
   `--out` writes the first mode's node movements (`buckling_mode.csv`), the loads and
   shapes of the first three modes (`buckling_loads.csv`) and a plan and elevation of
   the first mode (`buckling_mode.png`).
5. **Load-factor method.** One linear solve at $P_{ref}$ = 1000 N. Because everything
   scales with load,
   $$F_{u,p} = \min\left(\frac{P_{ref}}{\max_i U_i},\ P_{ref}\frac{50\text{ mm}}{\delta_{ref}},\ F_{cr}\right)$$
   The first member, joint, global buckling or the 50 mm deflection limit (§12.5) governs.

Every assumption is listed in [ASSUMPTIONS.md](ASSUMPTIONS.md).

### Gussets, plates and glued areas

Members are modelled on their centre lines, so wood that is not a member (gusset and
splice plates, blocks, dowels) would be missing from the mass. List it in the bridge file;
it adds mass and sticks, but no stiffness:

```yaml
extra_wood:
  - {id: end_gusset, volume_mm3: 14400, count: 8}   # 120 x 60 x 2 mm plate, 8 of them
  - {id: splice_plate, volume_mm3: 4600, count: 24}
```

The placeholder joint check assumes a glued area of overlap × width × faces per member
end. Where a gusset or a long lap gives a member much more, add `glue_area_mm2` to that
member's line: the total glued area at one end, over all its glued faces. For a 10 mm deep
diagonal lapped 60 mm under gussets on both faces, 60 × 10 × 2 = 1200 mm²:

```yaml
- {id: d0un, i: B0n, j: T0n, section: diagonal, group: diagonal, glue_area_mm2: 1200}
```

That area then replaces the default, and the member's joints are checked even if its
group is otherwise treated as continuous. Base the number on the glued area you actually
build, and on a joint test if you can.

### Plates, decks and other materials

Every member uses the bridge material unless its section says otherwise. A part that is
not sticks laid along the member, such as a 0/90 or ±45 cross-laminated stick plate, can
set its own E, G and strengths in MPa. Any value left out comes from the bridge material:

```yaml
sections:
  - {id: gusset_plate, b_mm: 60, d_mm: 4, E_MPa: 5000, G_MPa: 150, f_t_MPa: 20}
```

The deck is non-structural by default. If it is glued to both trusses, it also works as a
horizontal shear panel (a diaphragm) that stops the trusses moving sideways relative to
each other. Leaving it out can make the sideways (system) buckling load much lower than
the real one. To include it, give the deck's in-plane shear modulus and the nodes where
it is glued, one on each side of the deck at every station along the bridge:

```yaml
deck:
  # x_start_mm, x_end_mm, top_elevation_mm, clear_width_mm, ... as before
  diaphragm:
    G_MPa: 150                              # in-plane shear modulus of the deck plate
    connected_to: [B0n, B0f, B1n, B1f, B2n, B2f]   # ... every station along the bridge
```

Each deck panel between two stations then resists racking with the plate's shear
stiffness $G t w / a$ in the global buckling analysis. Here $a$ is the panel length, and
$t$ and $w$ are the plate's thickness and width: the deck's thickness and clear width
unless the diaphragm gives `thickness_mm` or `width_mm` (for a plate narrower than the gap
between the trusses). The deck resists shear only. It takes none of the chords' force and
changes no member force or utilisation; it only raises the buckling load. In-plane shear
stiffness depends a lot on how the plate is laid up and glued, so measure G if you can,
for example with a racking test of a square of your deck plate. The deck's own bending
and its strength are not modelled.

Members are shear-rigid (Euler-Bernoulli) frame elements, because Pynite has no
shear-deformable beam. Do not model a wide plate as one wide member: its shear flexibility
would be missing, and it would come out far too stiff. Use the diaphragm, or sections
with their own E and G.

### Slender bracing: tension-only members

Thin X-bracing straps get squeezed whenever the chords shorten under load. The member
check would then report a strap that only bows elastically by a millimetre or two as the
governing member, and adding bracing would seem to lower the predicted load. Mark such
members as tension-only:

```yaml
- {id: tb0a, i: T0n, j: T1f, section: top_brace, group: top_brace, tension_only: true}
```

A tension-only member that would be in compression is left out, and the bridge is
re-analysed without it, for the static analysis and the global buckling analysis alike.
The results list these *slack* members, and the 3D view draws them dashed. In tension the
member is checked like any other. Use the flag only for members that may bow without the
bridge failing, not for ones the bridge needs to push (piers, posts, struts). When a
brace governs by buckling, the results suggest the flag.

Under the centred crusher load the chords shorten, so often *both* straps of an X go
slack, and the bracing then does nothing in the global buckling analysis. But in a
sideways buckling mode one strap of each X is stretched and works again. So when slack
straps cross, the results give global buckling as a bracket. The lower end leaves them
out, and F_u,p uses it. The upper end has one strap of each X working (both straps at
half stiffness, which is the same thing when the X racks). The real value lies between
the two, which shows whether the X is worth its mass.

## v0.1 limitations

Stated plainly:

* **Placeholder materials.** E = 10 GPa (11 GPa in the crush: dry craft sticks are 11-15 GPa by vibration), f_t = 40, f_c = 30, f_b = 50, f_v = 8 MPa, glue
  τ = 2 MPa: textbook-order guesses, **not data**. Popsicle sticks vary a lot between brands
  and batches.
* **Linear, first-failure.** No P-Δ, no imperfections, no joint slip, no load
  redistribution after a member fails, and no contact-with-crusher failure. Real bridges
  can carry more after a brace buckles (mark slender braces
  [tension-only](#slender-bracing-tension-only-members)), or less if they are crooked or
  badly glued.
* **Plates only as a deck diaphragm.** Plates are not modelled as plates. A deck glued to
  both trusses can be a [shear diaphragm](#plates-decks-and-other-materials); other plates
  are members with their own E and G, or mass only (`extra_wood`). Leaving out a plate
  that stiffens the bridge makes the global (sideways) buckling result conservative.
  Members are shear-rigid, so very deep, short members are too stiff.
* **Idealised joints and laminates.** Glue lines are perfect, splices are not weakened,
  and the joint check is a placeholder (a member under a gusset can state its own glued
  area, see [Gussets, plates and glued areas](#gussets-plates-and-glued-areas)).
* **Geometry input** is the parametric Warren generator or a hand-written / exported
  `bridge.yaml`. Direct import from AutoCAD, Revit or SolidWorks is on the roadmap (DXF
  centrelines first).
* **Not checked:** deck level, warping and smoothness, the moving-load test itself, glue
  and floss usage, pre-fab box size, and assembly rules (listed in the app as "not checked").

## Measuring your own material (do this first)

1. **E (stiffness).** Three-point bending of single sticks: span $L$, load $F$,
   mid-span deflection $\delta$, $I = b t^3/12$:
   $$E = \frac{F L^3}{48\,\delta\,I}$$
2. **f_b (bending strength).** Same test to failure: $f_b = \frac{F_{max} L / 4}{b t^2/6}$.
3. **τ_g (glue).** Lap joints pulled apart: $\tau_g = F_{max}/(\text{overlap} \times \text{width})$.
4. **f_t, f_c.** Tension and short-column tests on laminated stacks.
5. **Density.** Weigh 100 sticks and divide by their volume.

Enter the values in the app (tick *measured*) or edit `materials/popsicle_birch.yaml` and
set `source: measured`. Templates for your test data are in
[`validation/physical/`](validation/physical/).

## Validation status

Automated in CI (`uv run pytest`, ~6 min on 4 cores; `uv run pytest -m "not slow"` skips
the dozen tests that run crush simulations, ~2 min, for quick checks while you work), see
[validation/README.md](validation/README.md):

* simply supported beam deflection $PL^3/48EI$ within 0.5 %;
* global buckling eigen-solver against Euler for a pinned column;
* pin-jointed Warren truss chord forces by the method of sections within 1 %;
* Euler buckling of one stick, $P_{cr} \approx 49.75$ N (5.07 kgf);
* rectangle and torsion section properties, Pynite axis orientation, load distribution;
* rule boundaries (span, deck width, mass table, clear opening), YAML round-trip, the
  default design passing all rules, and an app smoke test.

**Not yet validated against physical crush tests.** Please contribute data!

## For next year's team: updating the rules

All rules live in [`rules/troitsky_2027.yaml`](rules/troitsky_2027.yaml) with their
rulebook sections. Copy it to `troitsky_2028.yaml`, update the numbers, and the app picks
it up. See [CONTRIBUTING.md](CONTRIBUTING.md). Ambiguities we found are in
[docs/open-questions.md](docs/open-questions.md).

## Roadmap

* **v0.2:** P-Δ analysis; initial imperfections (bow L/300–L/500) on compression members;
  rotational springs at joints; Pratt and Howe generators; materials fitted from test CSVs;
  pre-fab segmentation into 500 × 400 × 300 mm pieces (§7.1.2).
* **v0.3:** progressive failure (remove the failed member or joint, re-solve, keep loading
  until a mechanism or 50 mm) for a real load–deflection curve and collapse replay.
* **v0.4:** Monte Carlo over measured material scatter, giving a distribution of $F_u$:
  the median for the Load Prediction Form, the spread for the Design Validation.
* **v1.0:** CAD import (DXF centrelines via ezdxf, from AutoCAD / Revit / SolidWorks
  exports), PDF report, docs site.
* **v2:** in-browser 3D editor (three.js).

## Contributing

Issues and pull requests are welcome, especially physical test data. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Credits and disclaimer

* Built on [Pynite](https://github.com/JWock82/Pynite) by D. Craig Brinck (MIT), plus
  NumPy, SciPy, Pydantic, Plotly, Matplotlib and Streamlit.
* bridgesim is **not affiliated with the Troitsky organizers** or ECA Concordia. The
  **official rulebook is authoritative**. If this tool disagrees with it, the rulebook wins.
  Please open an issue.
* If you use bridgesim in your Design Validation, identify it (§10.2), for example:
  "bridgesim v0.1 (open source, MIT) using the Pynite 3.2.0 frame solver".
* License: [MIT](LICENSE).
