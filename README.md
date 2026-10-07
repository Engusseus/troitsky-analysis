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

> **Status: v0.1.** Linear analysis, first-failure prediction, one parametric generator
> (through Warren truss) plus any bridge described in a YAML file. **All material values
> are placeholders until you test your own sticks and glue.** See
> [Limitations](#v01-limitations).

## Quick start

You need [uv](https://docs.astral.sh/uv/) (it installs the right Python for you).

```bash
git clone https://github.com/engusseus/troitsky-analysis.git
cd troitsky-analysis
uv sync                                          # install
uv run streamlit run app/streamlit_app.py        # web app at http://localhost:8501
uv run bridgesim run examples/warren_2027.yaml   # command line
```

Command line:

```bash
uv run bridgesim run examples/warren_2027.yaml --out results/   # report, CSV, SFD/BMD PNG
uv run bridgesim check my_bridge.yaml                           # rules only (exit 1 if penalised)
uv run bridgesim generate warren --span-mm 1100 --out my_bridge.yaml
```

### Default design (placeholder materials, illustrative only)

The bundled `examples/warren_2027.yaml` (span 1150 mm, 8 panels, 260 mm deep, deck 170 mm
wide at 200 mm) passes every computable 2027 rule:

| Result | Value |
|---|---|
| Predicted ultimate load $F_{u,p}$ | **≈ 368 kgf** (3607 N) |
| Governs | floor-beam bending under the plate, closely followed by diagonal glued joints and top-chord buckling |
| Global (system) buckling load | ≈ 438 kgf (the whole bridge swaying lengthwise on its 180 mm piers) |
| Mass | 2.11 kg (≈ 1290 sticks) |
| Efficiency $\eta_s$ | ≈ 174 kgf/kg |
| Mid-span deflection at $F_{u,p}$ | 4.5 mm (limit 50 mm) |

These numbers use **assumed** material strengths and will change once you enter measured
values.

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
  bans shown prominently. Ambiguous rules are flagged.
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
5. **Load-factor method.** One linear solve at $P_{ref}$ = 1000 N. Because everything
   scales with load,
   $$F_{u,p} = \min\left(\frac{P_{ref}}{\max_i U_i},\ P_{ref}\frac{50\text{ mm}}{\delta_{ref}},\ F_{cr}\right)$$
   The first member, joint, global buckling or the 50 mm deflection limit (§12.5) governs.

Every assumption is listed in [ASSUMPTIONS.md](ASSUMPTIONS.md).

## v0.1 limitations

Stated plainly:

* **Placeholder materials.** E = 10 GPa, f_t = 40, f_c = 30, f_b = 50, f_v = 6 MPa, glue
  τ = 2 MPa: textbook-order guesses, **not data**. Popsicle sticks vary a lot between brands
  and batches.
* **Linear, first-failure.** No P-Δ, no imperfections, no joint slip, no load
  redistribution after a member fails, and no contact-with-crusher failure. Real bridges
  can carry more after a brace buckles, or less if they are crooked or badly glued.
* **Idealised joints and laminates.** Glue lines are perfect, splices are not weakened,
  and the joint check is a placeholder.
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

Automated in CI (`uv run pytest`), see [validation/README.md](validation/README.md):

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
