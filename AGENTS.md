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
4. improve   -> change the CAD (or overrides); `ab` says whether the change really helps
5. calibrate -> after coupon tests (sticks, glue joints): fit the material
6. compare   -> after a real crush test: reality-gap score and what to measure next
```

## Commands (all accept `--json`: one JSON document on stdout, nothing else)

| Command | What it does | Time |
|---|---|---|
| `bridgesim inspect BRIDGE.SLDPRT [-o overrides.yaml] [--strict] --json` | Bodies (how each is modelled, centroid and bounding box, layup, mass), glued joints, loading position, rule check from the solids, `warnings`, `questions` | seconds |
| `bridgesim crush BRIDGE.SLDPRT [-o overrides.yaml] [--seed N] [--fidelity quick\|standard\|fine] [--rate-check] [--out DIR] --json [--brief]` | Simulated crushing test; with `--out`: `crush.json`, `curve.csv`, `events.csv` (every event), `replay.html`. `--rate-check` runs it again at half the jack speed (3x the time). `--brief`: the report without the curve and event log (~20 kB instead of ~200 kB; the MCP crush tools return this by default) | minutes |
| `bridgesim compare DIR/crush.json my_test.csv --json` | Reality gap vs a physical crush test | seconds |
| `bridgesim sensitivity BRIDGE.SLDPRT [--rel 0.2] [--fidelity quick] --json` | Which material values move F_u most: each one lowered and raised by `rel`; `measure_first` lists the tests that would narrow the prediction most | (1 + 2 x 8) x crush / processes |
| `bridgesim lab [--data validation/physical] --json` | Simulated coupon tests (stick bending, lap joint, stick column) and their gap to the team's coupon data | ~1 min |
| `bridgesim calibrate validation/physical --out my_material.yaml` | Fit E, f_t/f_c and tau_g so the *simulated* coupons reproduce the measured ones (`lab --data` then shows ~0 % error) | ~1 min |
| `bridgesim ab A.SLDPRT B.SLDPRT [-n 3] [--overrides-a a.yaml] [--overrides-b b.yaml] [--fidelity quick] --json` | Is B better than A? Both crushed with the same seeds: change of F_u and efficiency with a 90 % interval, `significant`, `verdict`, first failure of each. B may be A with other overrides | 2n x crush / processes |
| `bridgesim montecarlo BRIDGE.SLDPRT -n 8 [--fidelity quick] [--processes 4] --json` | Spread of F_u over imperfections and material scatter; `recommended_prediction_kgf` for the Load Prediction Form (the 1/F-weighted median, which minimises the §12.7 error). `note` says what the spread leaves out (placeholder material values) | n x crush / processes |
| `bridgesim schema inspect\|crush\|reality_gap\|lab\|montecarlo\|design_comparison\|sensitivity\|question` | JSON Schema of each report | instant |
| `bridgesim mcp` | The same as MCP tools (`uv sync --extra mcp`) | - |

Fidelity: `quick` (coarser mesh, jack 200 mm/s, ~1-2 min) to screen design changes,
`standard` (100 mm/s, ~4-6 min) for the answer, `fine` (finer mesh, 50 mm/s, ~10 min) to
check that the answer has converged. On Bridge_v6 (seed 0): quick 2329, standard 2035,
fine 1940 kgf, standard and fine failing at the same joint; so quick can be ~15 % off and
is only for ranking, and the standard F_u is within ~5 % of the converged one. Rank designs
with one fidelity and one seed (or `ab`); never compare F_u across fidelities.

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

MCP tools report errors as a result flagged as an error (`isError`), whose text is
`Error executing tool <name>: ` followed by the same JSON error object.

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
| `crush_bridge(cad_path, overrides_yaml, seed, fidelity, out_dir, strict, rate_check, ...)` | blocking, minutes |
| `start_crush(cad_path, out_dir, ...)` then `job_status(out_dir)` | background: poll until `status` is `done` (report included) or `failed` (error included). Use this when tool calls time out |
| `start_montecarlo(cad_path, out_dir, n, fidelity)` then `job_status(out_dir)` | background spread of F_u |
| `start_compare_designs(cad_a, cad_b, out_dir, n, overrides_a_yaml, overrides_b_yaml)` then `job_status(out_dir)` | background A/B comparison |
| `start_sensitivity(cad_path, out_dir, overrides_yaml, rel)` then `job_status(out_dir)` | background: what to measure first |
| `compare_with_test(crush_json_path, test_csv_path)` | reality gap |
| `lab_tests(material, data_dir)`, `calibrate_material(data_dir, out_path)` | coupon level |
| `get_schema(name)` | JSON Schemas |

`job_status` returns `{"status": "running" | "done" | "failed" | "unknown", "progress":
0..1 | null, "message": ..., "report": {...} | null, "error": ... | null}`. Jobs run
detached and record their exit in `out_dir/status.json`, so any server (a restarted one
too) reports them: a finished job is `done` with its report, a job whose process died is
`failed`, never left `running`.

## Speed and determinism

`inspect` takes seconds. A crush of a 100-body bridge takes roughly 5-20 minutes at
`standard` fidelity on a laptop with numba (`uv sync --extra sim`), a few minutes at
`quick`; a small bridge takes under a minute. Results are deterministic for a given
`--seed`, which sets the imperfections (member bow), the material scatter and the plate
placement. Each random value is drawn from a stream named after its body or joint, so
editing one part of the CAD does not reshuffle the imperfections of the others: two
designs crushed with the same seed differ only by the design change. `--seed -1` gives
the perfect, centred bridge. Use `montecarlo` for the spread, and `ab` to decide between
two designs: one seed each can mislead, because the scatter between seeds (glue strength
varies 30 % joint to joint) is often larger than a small design change.

## Questions: when the design is not what it should be

`inspect` and `crush` return `questions`. Each has an `id`, a `topic`
(`material`, `connection`, `geometry`, `rules`, `design`), a `severity`
(`important` > `check` > `info`), the `question` in plain words, what was `assumed`
meanwhile, and `answer_with`: how to answer, as an overrides fragment (`bodies`, `glue`,
`params`) with placeholders in `<angle brackets>` for the designer to fill, and/or `cad`
(a change to make in SolidWorks) or `cli` (a command to run). Typical ones:

* a body whose name does not say its layup or grain direction (`layup:<body>`,
  `orientation:<body>`);
* wrong orientation or units (`geometry:orientation`);
* a part that touches nothing, or a floss end tied to nothing (`floating:`, `loose:`,
  `floss_end:`);
* two solids that overlap in the CAD (`overlap:`) - usually a modelling error;
* a member whose glued area is much weaker than the member (`weak_joint:`) - a missing
  gusset?
* laminated members longer than one stick (`butt_joints`, one question for all of
  them): how far apart the butt joints of neighbouring plies are (it sets each member's
  tension strength);
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
  "Deck Stringer*": {butt_stagger_mm: 30}       # joints of neighbouring plies 30 mm apart
  "Bottom Chord*": {butt_stagger_mm: 0}         # plies without butt joints (long strips)
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
  judged failed (load drop, 50 mm deflection, or contact with the platform); 0 if it
  collapses under its own weight (`failure_mode: instability`).
* `failure_sequence` (text) and `failures` (structured: `kind`, `parts`, `load_kgf`,
  `jack_travel_mm`, `where_mm`, `fraction_separated`): what broke, in order, up to the
  collapse. The entries near the peak are the reason the bridge failed: strengthen
  *that*. A joint is `joint_cracking` when 30 % of its glued area has let go and
  `joint_failure` (separated) at 90 %; `member_shear` means the member split along its
  glue lines or grain (it keeps carrying axial load). Which part goes first can change
  with the mesh when two parts are close to failing together: check with
  `--fidelity fine` before redesigning around one part, and use `ab` to confirm.
* `events`: the first event of each kind per part, with `where_mm`; `events.csv` in the
  output folder has every event (`n_events_total`).
* `Fu_first_joint_failure_kgf`: load at the first glue-joint failure (§12.5 counts a
  joint failure as a failure when the judges see it).
* `criteria`: when each §12.5 criterion (deflection, contact, joint/shear, instability)
  was first met.
* `quality.trustworthy`: `false` when the loading was not quasi-static
  (`kinetic_to_work_max` > 0.1: kinetic energy of the bridge over the jack's work while
  the load rises to 95 % of its peak), F_u moved more than 5 % at half speed
  (`quality.rate_check`, with `--rate-check`), more than 5 % mass was added, the bridge
  did not fail within the jack travel, or the run had to restart; `quality.reasons` says
  which. Re-run with `--rate-check` or a finer `--fidelity` before relying on an
  untrustworthy F_u.
* `assumed_parameters`: values still not measured. With placeholders the absolute load
  is uncertain (maybe +-40 %), while design comparisons (A vs B) are much more reliable.
* `rules`: the 2027 dimensional rules measured on the solids (penalties, bans).
* `warnings`: overrides without effect, CAD faces that had to be closed over, etc.

## Quantifying how close to reality the model is

0. Decide what to measure: `bridgesim sensitivity` ranks the placeholder values by how
   much they move F_u for *this* design (`measure_first`).
1. Coupon level: fill `validation/physical/stick_bending_template.csv` and
   `glue_joint_template.csv`, run `bridgesim lab --data validation/physical --json`:
   each specimen gets simulated-vs-measured errors and a 0-100 reality-gap score.
   `bridgesim calibrate` then fits the material; re-run `lab` to confirm.
2. Bridge level: crush a prototype (or a sub-assembly), log load against mid-span
   deflection (`midspan_deflection_mm`) or the jack stroke (`jack_travel_mm`) in
   `crush_test_template.csv` (the whole record; mark the row of the failure with
   `event = failure` and the part that broke first in `notes`, with its side: "top chord,
   far side, left"). The peak load and the broken part alone are enough (that is what the
   competition gives you). Then `bridgesim compare results/crush.json my_test.csv --json`:
   errors on ultimate load, stiffness (secant 10-50 % of the peak), deflection at peak,
   curve shape and failure location (side included), a 0-100 score (0 when off by 1.5x
   in load or 2x in stiffness), the §12.7 prediction error, and advice on what to measure
   next.

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
until measured, and the model is verified (the equations are solved right) but not yet
validated against a physical crush test, so give F_u with its uncertainty and lean on
design comparisons; glue lines are perfect over the whole touching area unless a
`glue_factor` says otherwise; butt joints in laminated members are assumed in a brick
pattern unless `butt_stagger_mm` says otherwise; members are fibre beams (no splitting
along the grain except through shear); plates crack ply by ply; thick plates use
thin-plate elements; parts that come into contact during the test push on each other
without friction; time is compressed (the jack moves ~mm/s instead of mm/min; the
materials have no rate effect, and `--rate-check` shows the speed does not matter).
