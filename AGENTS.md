# Using bridgesim from an AI agent

This file is for AI agents (Claude Code, Codex, an MCP client, a script) helping a
Troitsky team design and test a popsicle-stick bridge. It says what to call, what comes
back, and how to turn the answers into a loop that gets the simulation closer to reality.

Everything below works on a **SolidWorks part file (`.SLDPRT`)** read directly (no
SolidWorks needed), a Parasolid `.x_b`, or an STL with all bodies in one file.

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
| `bridgesim inspect BRIDGE.SLDPRT [-o overrides.yaml] [--strict] --json` | Bodies (how each is modelled, centroid and bounding box, layup, mass), glued joints, loading position, rule check from the solids, `warnings`, `questions` | seconds |
| `bridgesim crush BRIDGE.SLDPRT [-o overrides.yaml] [--seed N] [--fidelity quick\|standard\|fine] [--out DIR] --json` | Simulated crushing test; with `--out`: `crush.json`, `curve.csv`, `events.csv` (every event), `replay.html` | minutes |
| `bridgesim compare DIR/crush.json my_test.csv --json` | Reality gap vs a physical crush test | seconds |
| `bridgesim lab [--data validation/physical] --json` | Simulated coupon tests (stick bending, lap joint, stick column) and their gap to the team's coupon data | ~20 s |
| `bridgesim calibrate validation/physical --out my_material.yaml` | Fit E, strengths, glue strength from coupon CSVs | seconds |
| `bridgesim montecarlo BRIDGE.SLDPRT -n 8 [--fidelity quick] [--processes 4] --json` | Spread of F_u over imperfections and material scatter; value for the Load Prediction Form | n x crush / processes |
| `bridgesim schema inspect\|crush\|reality_gap\|lab\|montecarlo\|question` | JSON Schema of each report | instant |
| `bridgesim mcp` | The same as MCP tools (`uv sync --extra mcp`) | - |

Fidelity: `quick` (coarser mesh, jack 200 mm/s) to screen design changes, `standard`
(100 mm/s) for the answer, `fine` (finer mesh, 50 mm/s) to check that the answer has
converged. Rank designs with one fidelity and one seed; never compare F_u across
fidelities.

### Orientation and units

bridgesim expects millimetres, X along the span and Y up (a bridge drawn on the
SolidWorks Top Plane). `.SLDPRT`/`.x_b` are always read in mm. If the bridge was drawn
standing on the Front Plane, or an STL was exported in metres, pass `--up-axis z`,
`--span-axis z`, `--units m` (MCP: `up_axis`, `span_axis`, `units`). Wrong orientation or
units are detected: you get an `important` question `geometry:orientation`, or exit 2 with
the option to use in the message.

### Error contract

Exit codes: `0` ok, `2` bad input (file missing, unreadable CAD, overrides that cannot be
applied, wrong orientation), `3` the simulation itself failed. With `--json`, stdout holds
only the report; an error is one JSON object on stderr:
`{"error": "...", "kind": "bad_input" | "simulation_failed", "exit_code": 2, "type": "..."}`.
Progress lines go to stderr (one line per percent step), so they never break the JSON.

MCP tools return the same error object as their result instead of raising.

### Overrides are checked

An override that matches no body, a joint name that does not exist, an unreadable layup,
an unknown `params` key: each becomes a line in `warnings` starting with `overrides:`,
with the closest valid name suggested (e.g. `did you mean 'Top Chord*'?`). An unknown
`params` key is always an error (exit 2). With `--strict` (MCP `strict=true`) every
override problem is an error: use it in automated loops so a typo cannot silently leave
the model unchanged.

## MCP server

Register it with Claude Code:

```bash
claude mcp add bridgesim -- uv run --directory /path/to/troitsky-analysis bridgesim mcp
```

| Tool | Use |
|---|---|
| `inspect_bridge(cad_path, overrides_yaml, up_axis, span_axis, units, strict)` | seconds |
| `crush_bridge(cad_path, overrides_yaml, seed, fidelity, out_dir, ...)` | blocking, minutes |
| `start_crush(cad_path, out_dir, ...)` then `job_status(out_dir)` | background: poll until `status` is `done` (report included) or `failed` (error included). Use this when tool calls time out |
| `start_montecarlo(cad_path, out_dir, n, fidelity)` then `job_status(out_dir)` | background spread of F_u |
| `compare_with_test(crush_json_path, test_csv_path)` | reality gap |
| `lab_tests(material, data_dir)`, `calibrate_material(data_dir, out_path)` | coupon level |
| `get_schema(name)` | JSON Schemas |

`job_status` returns `{"status": "running" | "done" | "failed", "progress": 0..1,
"message": ..., "report": {...} | null, "error": ... | null}`. A job whose process died is
reported `failed`, never left `running`.

## Speed and determinism

`inspect` takes seconds. A crush of a 100-body bridge takes roughly 5-20 minutes at
`standard` fidelity on a laptop with numba (`uv sync --extra sim`), a few minutes at
`quick`; a small bridge takes under a minute. Results are deterministic for a given
`--seed`, which sets the imperfections (member bow), the material scatter and the plate
placement. Each random value is drawn from a stream named after its body or joint, so
editing one part of the CAD does not reshuffle the imperfections of the others: two
designs crushed with the same seed differ only by the design change. `--seed -1` gives
the perfect, centred bridge. Use `montecarlo` for the spread.

## Questions: when the design is not what it should be

`inspect` and `crush` return `questions`. Each has an `id`, a `topic`
(`material`, `connection`, `geometry`, `rules`, `design`), a `severity`
(`important` > `check` > `info`), the `question` in plain words, what was `assumed`
meanwhile, and `answer_with`: an overrides fragment showing how to answer (placeholders in
`<angle brackets>` are for the designer to fill). Typical ones:

* a body whose name does not say its layup or grain direction (`layup:<body>`,
  `orientation:<body>`);
* wrong orientation or units (`geometry:orientation`);
* a part that touches nothing, or a floss end tied to nothing (`floating:`, `loose:`,
  `floss_end:`);
* two solids that overlap in the CAD (`overlap:`) - usually a modelling error;
* a member whose glued area is much weaker than the member (`weak_joint:`) - a missing
  gusset?
* a rigid block directly under the loading plate (`block_under_plate:`);
* a failing rule (`rule:<section>`), the mass with an assumed density
  (`material:density`), placeholder material values (`material:measure`).

Ask the `important` ones first. Do not guess an answer the designer can give.

Bodies are located by `centroid_mm`, `bbox_min_mm`, `bbox_max_mm` and `axis` in the
inspect report, and events by `where_mm` in the crush report, so you can tell the
designer *where* a part or a failure is ("the gusset at x = -310 mm, far side").

## Overrides YAML

```yaml
bodies:                     # by exact body name or glob pattern
  "Body 21531": {layup: "10-ply UD"}            # what an unnamed body is made of
  "Saddle*": {kind: block}                      # bar | shell | block | cable | ignore
  "Hub Tie Strap*": {glue_factor: 0.7}          # poorly glued part (x glue strength)
  "Spacer*": {glue_factor: 0}                   # touching but NOT glued
  "Top Chord*": {strength_factor: 0.9, grain: span}
glue:
  "Top Chord (10-ply UD 20.9x47.5) - Far | T-Node (10-ply) - R Far": 0.5   # one joint
params:                     # any key of materials/simulation.yaml, or material.<key>
  floss.strands: 6
  material.density_kg_m3: 590
  material.glue.tau_g_MPa: 2.6
```

Body names are exactly as `inspect` lists them (SolidWorks body names; mirrored copies
get "(mirror copy)"). Joint names are `A | B` as listed in the inspect `joints`.

## What the crush report means

* `Fu_kgf`: predicted ultimate load (§12.5): the peak jack force before the bridge is
  judged failed (load drop, 50 mm deflection, or contact with the platform).
* `failure_sequence`: what broke, in order, with the load at that moment. The first
  entries near the peak are the reason the bridge failed: strengthen *that*.
* `events`: the first event of each kind per part, with `where_mm`; `events.csv` in the
  output folder has every event (`n_events_total`).
* `Fu_first_joint_failure_kgf`: load at the first glue-joint failure (§12.5 counts a
  joint failure as a failure when the judges see it).
* `criteria`: when each §12.5 criterion (deflection, contact, joint/shear, instability)
  was first met.
* `quality.trustworthy`: `false` when the run was not quasi-static
  (`kinetic_to_work_max` > 0.1), needed more than 5 % added mass, did not fail within the
  jack travel, or had to restart; `quality.reasons` says which. Re-run with a slower
  `--speed` or a finer `--fidelity` before relying on an untrustworthy F_u.
* `assumed_parameters`: values still not measured. With placeholders the absolute load
  is uncertain (maybe +-40 %), while design comparisons (A vs B) are much more reliable.
* `rules`: the 2027 dimensional rules measured on the solids (penalties, bans).
* `warnings`: overrides without effect, CAD faces that had to be closed over, etc.

## Quantifying how close to reality the model is

1. Coupon level: fill `validation/physical/stick_bending_template.csv` and
   `glue_joint_template.csv`, run `bridgesim lab --data validation/physical --json`:
   each specimen gets simulated-vs-measured errors and a 0-100 reality-gap score.
   `bridgesim calibrate` then fits the material; re-run `lab` to confirm.
2. Bridge level: crush a prototype (or a sub-assembly), log load and mid-span deflection
   in `crush_test_template.csv` (the whole record, the drop after the peak included; mark
   the row of the failure with `event = failure` and the broken part in `notes`), then
   `bridgesim compare results/crush.json my_test.csv --json`: errors on ultimate load,
   stiffness (rising branch only), deflection at peak, curve shape and failure location,
   a 0-100 score, the §12.7 prediction error, and advice on what to measure next.

## Python API

```python
from bridgesim.sim import study
rep = study.inspect("Bridge_v6.SLDPRT")                     # InspectReport
crush, raw = study.crush("Bridge_v6.SLDPRT", overrides={"params": {"floss.strands": 6}},
                         out_dir="results", fidelity="quick")  # CrushReport, CrushResult
gap = study.compare(crush, "my_test.csv")                    # RealityGap
print(crush.model_dump_json(indent=2))
```

## Limits to tell the designer about

Read [ASSUMPTIONS-SIM.md](ASSUMPTIONS-SIM.md). In short: material values are placeholders
until measured; glue lines are perfect over the whole touching area unless a
`glue_factor` says otherwise; members are fibre beams (no local crushing under
concentrated loads, no splitting along the grain except through shear); thick plates use
thin-plate elements; time is compressed (the jack moves ~mm/s instead of mm/min; the
materials have no rate effect).
