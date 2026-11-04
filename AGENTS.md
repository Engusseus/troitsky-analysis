# Using bridgesim from an AI agent

This file is for AI agents (Claude Code, Codex, an MCP client, a script) helping a
Troitsky team design and test a popsicle-stick bridge. It says what to call, what comes
back, and how to turn the answers into a loop that gets the simulation closer to reality.

Everything below works on a **SolidWorks part file (`.SLDPRT`)** read directly (no
SolidWorks needed), a Parasolid `.x_b`, or an STL with all bodies in one file (mm).

## The loop

```
1. inspect   -> what bridgesim understood + the questions it could not answer
2. ask       -> put the questions to the designer, write the answers in overrides.yaml
3. crush     -> simulated §12.5 crushing test: ultimate load, failure sequence, replay
4. improve   -> change the CAD (or overrides) and crush again; compare designs
5. calibrate -> after coupon tests (sticks, glue joints): fit the material
6. compare   -> after a real crush test: reality-gap score and what to measure next
```

## Commands (all accept `--json`: one JSON document on stdout, nothing else)

| Command | What it does | Time |
|---|---|---|
| `bridgesim inspect BRIDGE.SLDPRT [-o overrides.yaml] --json` | Bodies and how each is modelled, glued joints, loading position, rule check from the solids, `questions` | seconds |
| `bridgesim crush BRIDGE.SLDPRT [-o overrides.yaml] [--seed N] [--fidelity quick|standard|fine] [--out DIR] --json` | Simulated crushing test; with `--out`: `crush.json`, `curve.csv`, `replay.html`. Use `--fidelity quick` to screen design changes (coarser mesh, faster jack), `standard` for the answer, `fine` to check convergence | minutes |
| `bridgesim compare DIR/crush.json my_test.csv --json` | Reality gap vs a physical crush test | seconds |
| `bridgesim lab [--data validation/physical] --json` | Simulated coupon tests (stick bending, lap joint, stick column) and their gap to the team's coupon data | ~20 s |
| `bridgesim calibrate validation/physical --out my_material.yaml` | Fit E, strengths, glue strength from coupon CSVs | seconds |
| `bridgesim montecarlo BRIDGE.SLDPRT -n 8 --json` | Spread of F_u over imperfections and material scatter; value for the Load Prediction Form | n x crush |
| `bridgesim schema inspect|crush|reality_gap|lab|montecarlo|question` | JSON Schema of each report | instant |
| `bridgesim mcp` | The same as MCP tools (`pip install "bridgesim[mcp]"`) | - |

Exit codes: `0` ok, `2` bad input (message on stderr), `3` simulation failed.
Units are in every field name (`_mm`, `_kgf`, `_N`, `_g`, `_s`, `_MPa`).

MCP tools: `inspect_bridge`, `crush_bridge` (blocking), `start_crush` + `crush_status`
(background, poll until `done`: use these if your tool calls time out), `compare_with_test`,
`lab_tests`, `calibrate_material`, `get_schema`.

Register the MCP server with Claude Code:

```bash
claude mcp add bridgesim -- uv run --directory /path/to/troitsky-analysis bridgesim mcp
```

## Speed and determinism

`inspect` takes seconds. A crush of a 100-body bridge takes roughly 5-20 minutes at
`standard` fidelity on a laptop with numba (`uv sync --extra sim`), a few minutes at
`quick`; a small bridge takes under a minute. Results are deterministic for a given
`--seed` (which sets the imperfections, the material scatter and the plate placement);
`--seed -1` gives the perfect, centred bridge. Compare two designs with the same seed, and
use `montecarlo` for the spread.

## Questions: when the design is not what it should be

`inspect` and `crush` return `questions`. Each has an `id`, a `topic`
(`material`, `connection`, `geometry`, `rules`, `design`), a `severity`
(`important` > `check` > `info`), the `question` in plain words, what was `assumed`
meanwhile, and `answer_with`: an overrides fragment showing how to answer. Typical ones:

* a body whose name does not say its layup (`layup:<body>`);
* a part that touches nothing, or a floss end tied to nothing (`floating:`, `loose:`,
  `floss_end:`);
* two solids that overlap in the CAD (`overlap:`) - usually a modelling error;
* a member whose glued area is much weaker than the member (`weak_joint:`) - a missing
  gusset?
* a failing rule (`rule:<section>`), the mass with an assumed density
  (`material:density`), placeholder material values (`material:measure`).

Ask the `important` ones first. Do not guess an answer the designer can give.

## Overrides YAML

```yaml
bodies:                     # by exact body name or glob pattern
  "Body 21531": {layup: "10-ply UD"}            # what an unnamed body is made of
  "Saddle*": {kind: block}                      # bar | shell | block | cable | ignore
  "Hub Tie Strap*": {glue_factor: 0.7}          # poorly glued part (x glue strength)
  "Top Chord*": {strength_factor: 0.9, grain: span}
glue:
  "Top Chord (10-ply UD 20.9x47.5) - Far | T-Node (10-ply) - R Far": 0.5   # one joint
params:                     # any key of materials/simulation.yaml, or material.<key>
  floss.strands: 6
  material.density_kg_m3: 590
  material.glue.tau_g_MPa: 2.6
```

Body names are exactly as `inspect` lists them (SolidWorks body names; mirrored copies
get "(mirror copy)").

## What the crush report means

* `Fu_kgf`: predicted ultimate load (§12.5): the peak jack force before the bridge is
  judged failed (load drop, 50 mm deflection, or contact with the platform).
* `failure_sequence`: what broke, in order, with the load at that moment. The first
  entries near the peak are the reason the bridge failed: strengthen *that*.
* `criteria`: when each §12.5 criterion (deflection, contact, joint/shear, instability)
  was first met.
* `quality`: trust the numbers only if `added_mass_fraction` < 0.05 and
  `kinetic_to_work_max` < 0.1 (quasi-static). `notes` lists automatic restarts.
* `assumed_parameters`: values still not measured. With placeholders the absolute load
  is uncertain (maybe +-40 %), while design comparisons (A vs B) are much more reliable.
* `rules`: the 2027 dimensional rules measured on the solids (penalties, bans).

## Quantifying how close to reality the model is

1. Coupon level: fill `validation/physical/stick_bending_template.csv` and
   `glue_joint_template.csv`, run `bridgesim lab --data validation/physical --json`:
   each specimen gets simulated-vs-measured errors and a 0-100 reality-gap score.
   `bridgesim calibrate` then fits the material; re-run `lab` to confirm.
2. Bridge level: crush a prototype (or a sub-assembly), log load and mid-span deflection
   in `crush_test_template.csv` (a row with `event = failure` and the broken part in
   `notes`), then `bridgesim compare results/crush.json my_test.csv --json`: errors on
   ultimate load, stiffness, deflection at peak, curve shape and failure location, a
   0-100 score, the §12.7 prediction error, and advice on what to measure next.

## Python API

```python
from bridgesim.sim import study
rep = study.inspect("Bridge_v6.SLDPRT")                     # InspectReport
crush, raw = study.crush("Bridge_v6.SLDPRT", overrides={"params": {"floss.strands": 6}},
                         out_dir="results")                  # CrushReport, CrushResult
gap = study.compare(crush, "my_test.csv")                    # RealityGap
print(crush.model_dump_json(indent=2))
```

## Limits to tell the designer about

Read [ASSUMPTIONS-SIM.md](ASSUMPTIONS-SIM.md). In short: material values are placeholders
until measured; glue lines are perfect over the whole touching area (no starved joints);
members are fibre beams (no local crushing under concentrated loads, no splitting along
the grain except through shear); thick plates use thin-plate elements; time is
compressed (the jack moves ~mm/s instead of mm/min; the materials have no rate effect).
