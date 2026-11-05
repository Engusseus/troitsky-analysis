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
              (screen first, in seconds: stiffness and the most loaded parts, linear)
4. improve   -> change the CAD (or overrides); `ab` says whether the change really helps
5. calibrate -> after coupon tests (sticks, glue joints): fit the material
6. compare   -> after a real crush test: reality-gap score and what to measure next
7. calibrate-bridge -> fit the model to that test, then validate it on the next one
```

Every `inspect` and `crush` report ends with **`next_steps`**: this loop worked out for
the report in hand, most useful first. Each step has an `action` (`fix_cad`,
`ask_designer`, `screen`, `crush`, `rerun`, `strengthen`, `confirm`, `measure`, `predict`,
`compare`), `why` in plain words, the `ids` of the questions concerned, the `parts` and
`where_mm` to point the designer at, and the exact `cli` to run (the MCP tool of the
same name otherwise). E.g. after a crush: `strengthen` the parts that broke at the peak
(with a what-if in `overrides` and the `ab` command that tests it), `confirm` a quick
result at standard fidelity, `rerun` an untrustworthy one, `predict` with `montecarlo`.
Files in a `cli` are absolute paths, so it runs from any folder. Placeholders are in
`<angle brackets>`; the overrides file is the one you passed (`overrides.yaml` when they
came inline, e.g. through MCP). With `--out DIR`, a `strengthen` step's what-if is also
saved as `DIR/whatif.yaml`: your overrides plus the what-if (a part you already scaled
gets both factors), ready for its `ab ... --overrides-b DIR/whatif.yaml` command
(`artifacts.whatif_yaml`).

## Commands (all accept `--json`: one JSON document on stdout, nothing else)

| Command | What it does | Time |
|---|---|---|
| `bridgesim inspect BRIDGE.SLDPRT [-o overrides.yaml] [--strict] [--body PATTERN]... [--joint PATTERN]... --json [--brief]` | Bodies (how each is modelled, centroid and bounding box, layup, mass), glued joints, loading position, rule check from the solids, `warnings`, `questions`, `next_steps`. `--brief` (and the MCP default): questions, counts, failed rules, warnings and next steps only (~25 kB for a 100-body bridge instead of ~100 kB). `--body`/`--joint` (glob, case ignored; MCP `body`, `joint`): only those bodies, the joints with such a part, and the questions about them, in full | seconds |
| `bridgesim crush BRIDGE.SLDPRT [-o overrides.yaml] [--seed N] [--fidelity quick\|standard\|fine] [--rate-check] [--stop-at-failure] [--out DIR] --json [--brief]` | Simulated crushing test; with `--out`: `crush.json`, `curve.csv`, `events.csv` (every event), `replay.html`. `--rate-check` runs it again at half the jack speed (3x the time). `--stop-at-failure` (the MCP default; `full_collapse=true` to follow the collapse): stop 2 mm of jack travel after the bridge is judged failed, ~1/3 faster, the replay ends there. `--brief`: what an agent needs for its next step, ~12 kB instead of ~200 kB (the MCP crush tools return this by default): no curve, event log or failure sequence; the first 10 `failures` (kind, `what`, load, travel, `where_mm`); the `important` questions in full and `n_questions` by severity (the `check` ids are in the `ask_designer` step; `inspect` has their text and `answer_with`); `next_steps` without empty fields. `crush.json` has everything. A what-if to test: `DIR/whatif.yaml` | minutes |
| `bridgesim screen BRIDGE.SLDPRT [-o overrides.yaml] [--load 100] [--top 25] --json` | Linear pre-screen, no simulation: the bridge's stiffness under the plate (`stiffness_kgf_per_mm`) and every joint, member (tension, compression, shear, Euler buckling over its length) and plate (per ply and mode) with its `utilisation` at `--load` and the plate load at which it reaches its limit if the bridge stayed linear (`limits`, most loaded first; `first_limit_kgf`: where damage starts, often far below F_u, since the load then finds other paths, which only crush follows). For the stiffness, the most loaded parts, and how a what-if moves them in seconds; judge F_u and design changes with crush and `ab` | seconds |
| `bridgesim compare DIR/crush.json my_test.csv --json` | Reality gap vs a physical crush test. Given `montecarlo.json` instead of `crush.json`: where the test falls in the predicted range | seconds |
| `bridgesim calibrate-bridge BRIDGE.SLDPRT my_test.csv [-o overrides.yaml] [--fidelity standard] [--out fitted.overrides.yaml] --json` | After a physical crush test of this bridge: a global stiffness and a global strength factor fitted so the simulated crush reproduces it (one crush per step, up to 4); writes your overrides plus the fitted `params`. The test is then used up for validation | 2-4 x crush |
| `bridgesim sensitivity BRIDGE.SLDPRT [--rel 0.2] [--fidelity quick] [--seeds 1] [--param KEY]... --json` | Which material values move F_u most: each one lowered and raised by `rel`; `measure_first` lists the tests that would narrow the prediction most. `--param` (repeat it) chooses the values, any key of `materials/simulation.yaml` or `material.<key>` (default: the 9 that usually matter most); `--seeds 3` averages over seeds | (1 + 2 x params) x seeds x crush / processes |
| `bridgesim lab [--data validation/physical] --json` | Simulated coupon tests (stick bending, lap joint, stick column) and their gap to the team's coupon data | ~1 min |
| `bridgesim calibrate validation/physical --out my_material.yaml` | Fit E, f_t/f_c and tau_g so the *simulated* coupons reproduce the measured ones (`lab --data` then shows ~0 % error); f_t, f_c, f_c90, the end-grain factor and the glue's shear stiffness and toughness from the direct coupon tests; with 3+ specimens of a kind also the scatter (`my_material.overrides.yaml`, `params:`). A folder without any test results is an input error (exit 2) | ~1 min |
| `bridgesim ab A.SLDPRT B.SLDPRT [-n 3] [--overrides-a a.yaml] [--overrides-b b.yaml] [--fidelity standard] --json` | Is B better than A? Both crushed with the same seeds: change of F_u and efficiency with a 90 % interval, `significant`, `verdict`, first failure of each. B may be A with other overrides: a what-if (`section_scale`, `glue_factor`) before changing the CAD | 2n x crush / processes |
| `bridgesim montecarlo BRIDGE.SLDPRT -n 8 [--fidelity standard] [--processes 4] [--sample-placeholders] [--out DIR] --json` | Spread of F_u over imperfections and material scatter (`min_kgf`, `max_kgf`, p05/p50/p95: below 20 runs read min and max); `recommended_prediction_kgf` for the Load Prediction Form (the 1/F-weighted median, which minimises the §12.7 error). `note` says what the spread leaves out: the placeholder material values, unless `--sample-placeholders` (MCP `sample_placeholders`) draws them per run (lognormal, 20 %; `sampled_placeholders`) so the range includes them | n x crush / processes |
| `bridgesim schema inspect\|crush\|screen\|reality_gap\|lab\|montecarlo\|design_comparison\|sensitivity\|bridge_calibration\|question` | JSON Schema of each report | instant |
| `bridgesim mcp` | The same as MCP tools (`uv sync --extra mcp`) | - |

Fidelity: `quick` (coarser mesh, jack 200 mm/s, ~3 min for a 100-body bridge on 4 cores)
to screen design changes, `standard` (100 mm/s, ~5-8 min) for the answer, `fine` (finer
mesh, 50 mm/s, ~15 min) to check that the answer has converged. Mesh convergence on
Bridge_v6 (`validation/README.md` §3.4, all at 100 mm/s): the perfect bridge gives 2710,
2684, 2673 kgf with elements x2, x1, x0.75 (-1.0 %, then -0.4 %); seed by seed (a seed is
the same imperfect bridge at every fidelity) the standard mesh is within 1.3 % of the fine
one and quick 2-5 % high, all three ranking the seeds alike. The spread *between* seeds is
~20 %: three failure sites (the deck stringers under the plate, the end-post joint, the
hub tie strap) compete within ~15 %. So rank designs with one fidelity and the same seeds
(`ab`, n >= 3), give F_u as the `montecarlo` range, and never compare F_u across
fidelities. Do not go finer than `fine`: crack onset at a stress concentration is local
(ASSUMPTIONS-SIM.md). `ab`, `montecarlo`, `sensitivity` and `calibrate-bridge`
stop each crush 2 mm of jack travel after the bridge is judged failed (F_u and the
failures are known by then; no replay), which saves about a third of the time.

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

MCP tools return their report as structured content (`structuredContent`, a JSON
object with a published output schema) and the same JSON as text. Errors are a result
flagged as an error (`isError`), whose text is `Error executing tool <name>: ` followed by
the same JSON error object. Give absolute paths: a relative one is taken from the
server's folder (where the client started it), and the error for a missing file says so.

### Overrides are checked

An override that matches no body, a joint name that does not exist, an unreadable layup,
an unknown `params` key: each becomes a line in `warnings` starting with `overrides:`,
with the closest valid name suggested (e.g. `did you mean 'Top Chord*'?`). An unknown
`params` key is always an error (exit 2), and so is a `params` value outside its
physical range (a negative stiffness, a friction coefficient of 7, a CoV above 1: the
message names the key and the range). With `--strict` (MCP `strict=true`) every
override problem is an error: use it in automated loops so a typo cannot silently leave
the model unchanged.

## MCP server

Register it with Claude Code:

```bash
claude mcp add bridgesim -- uv run --directory /path/to/troitsky-analysis bridgesim mcp
```

| Tool | Use |
|---|---|
| `inspect_bridge(cad_path, overrides_yaml, up_axis, span_axis, units, strict, brief=true, body, joint)` | seconds |
| `screen_bridge(cad_path, overrides_yaml, load_kgf, top, ...)` | seconds: linear pre-screen |
| `crush_bridge(cad_path, overrides_yaml, seed, fidelity, out_dir, strict, rate_check, full_collapse=false, ...)` | blocking, minutes; stops 2 mm after the failure unless `full_collapse` |
| `start_crush(cad_path, out_dir, ...)` then `job_status(out_dir)` | background: poll until `status` is `done` (report included) or `failed` (error included). Use this when tool calls time out |
| `start_montecarlo(cad_path, out_dir, n, fidelity, sample_placeholders)` then `job_status(out_dir)` | background spread of F_u |
| `start_compare_designs(cad_a, cad_b, out_dir, n, overrides_a_yaml, overrides_b_yaml)` then `job_status(out_dir)` | background A/B comparison |
| `start_sensitivity(cad_path, out_dir, overrides_yaml, rel, params, seeds)` then `job_status(out_dir)` | background: what to measure first |
| `start_calibrate_bridge(cad_path, test_csv_path, out_dir, overrides_yaml, fidelity)` then `job_status(out_dir)` | background: fit stiffness and strength to a physical crush test (writes `out_dir/fitted.overrides.yaml`) |
| `compare_with_test(crush_json_path, test_csv_path)` | reality gap (a `montecarlo.json` as the first argument: the test against the predicted range) |
| `lab_tests(material, data_dir)`, `calibrate_material(data_dir, out_path)` | coupon level |
| `cancel_job(out_dir)` | stop a background job |
| `get_schema(name)` | JSON Schemas |

`job_status` returns `{"status": "queued" | "running" | "done" | "failed" | "cancelled" |
"unknown", "progress": 0..1 | null, "message": ..., "report": {...} | null, "error": ... |
null}`. Jobs run detached and record their exit in `out_dir/status.json`, so any server (a
restarted one too) reports them: a finished job is `done` with its report, a job whose
process died is `failed`, never left `running`. Jobs share the machine: a few run at once
(half the CPU cores, `BRIDGESIM_JOB_SLOTS` to change it), each with its share of the cores,
and the others wait as `queued`, so starting several is safe. `cancel_job(out_dir)` stops
one. From the CLI, give each of two crushes run side by side `--threads 2` (on 4 cores):
without it each takes every core and both crawl.

## Speed and determinism

`inspect` takes seconds. A crush of a 100-body bridge takes roughly 5-10 minutes at
`standard` fidelity on 4 cores with numba (`uv sync --extra sim`), about 3 minutes at
`quick`; a small bridge takes under a minute. Results are deterministic for a given
`--seed`, which sets the imperfections (member bow, support corners out of level), the
material scatter and the plate placement. Each random value is drawn from a stream
named after its body or joint (a joint by its bodies and where they overlap in the CAD,
never by the mesh), so a seed is the same imperfect bridge at quick, standard and fine
fidelity, and editing one part of the CAD does not reshuffle the imperfections of the
others: two designs crushed with the same seed differ only by the design change. `--seed -1` gives
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

* a body whose name does not say its layup (`layup:<body>`, important: its stiffness
  and strength are guessed) or its grain direction (`orientation:<body>`);
* wrong orientation or units (`geometry:orientation`);
* a part that touches nothing, or a floss end tied to nothing (`floating:`, `loose:`,
  `floss_end:`);
* two solids that overlap in the CAD (`overlap:`) - usually a modelling error;
* faces read only approximately (`cad_approximated:<body>`, important when they are a
  twentieth of the body's surface): a spline, cone or freeform surface flattened, or a
  face that could not be decoded closed over;
* plates made of narrow arms (`narrow_plates`: an X, a frame), with about one shell
  element across an arm: their arms are ~4x too stiff in in-plane bending; if the part is
  really members, draw each as its own body (it becomes a bar);
* a member whose glued area is much weaker than the member (`weak_joint:`) - a missing
  gusset?
* laminated members longer than one stick (`butt_joints`, one question for all of
  them): how far apart the butt joints of neighbouring plies are (it sets each member's
  tension strength);
* a rigid block directly under the loading plate (`rigid_under_plate:`), and the compact
  parts in the load path modelled as rigid blocks (`rigid_blocks`);
* glued joints that are mostly end grain, a member's end glued flat (`end_grain_joints`);
* a `kind` set in the overrides that does not fit the body's shape (`kind:`);
* a failing rule (`rule:<section>`), the mass with an assumed density
  (`material:density`), placeholder material values (`material:measure`).

Ask the `important` ones first. Do not guess an answer the designer can give. A question
is closed by its answer (setting what it asks), or, when the designer confirms the design
is as drawn, by listing its id (or a pattern such as `weak_joint:*`) under `acknowledge:`
in the overrides; acknowledged ids are reported in `acknowledged_questions`.

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
  "Deck Stringer*": {section_scale: 1.25}       # what-if: section x1.25 (area x1.56)
glue:
  "Top Chord (10-ply UD 20.9x47.5) - Far | T-Node (10-ply) - R Far": 0.5   # one joint
params:                     # any key of materials/simulation.yaml, or material.<key>
  floss.strands: 6
  material.density_kg_m3: 590
  material.glue.tau_g_MPa: 2.6
  glue.end_grain_factor: 0.8                     # end-grain joints sized before gluing
acknowledge:                # questions the designer has answered "as drawn"
  - "weak_joint:*"
  - rigid_blocks
```

Other top-level keys are reported in `warnings` (an `answer_with` may hold `cad` or `cli`:
things to do, not overrides).

**What-ifs before changing the CAD.** `section_scale` scales a member's cross-section
(both sides) or a plate's thickness, and its mass (the mass rule is checked with it);
`glue` / `glue_factor` above 1 stands for a bigger glued area or a gusset at a joint. The
CAD and its glued areas are unchanged, and each what-if is listed in `warnings`. A
`strengthen` step in `next_steps` comes with such a what-if in its `overrides` (on every
copy of the parts that broke, since a design change is made on all of them) and the `ab`
command that tests it on the same seeds: if it helps, make the change in SolidWorks and
compare the two files with `ab`. (On Bridge_v6 the what-if of a quick crush raised F_u by
7.0 % with 3 of 3 seeds, 90 % interval -7.4 % to +21.5 %: +17 % on the seed where the
strengthened joint governed, 1-3 % where three parts under the plate fail together, so
`ab` called it not proven: `validation/README.md` §3.4.)

Body names are exactly as `inspect` lists them (SolidWorks body names; mirrored copies
get "(mirror copy)"). Joint names are `A | B` as listed in the inspect `joints`; both
sides may be glob patterns (`"End Post* | T1 Cover*": 1.5` is the joint on all four
corners). A value still holding an answer_with placeholder (`<...>`) is an error that
names the key, and a layup override must read `N-ply UD`, `N-ply 0-90`, `N-ply +-45`
(optionally `along the span`), `block`, `toothpick` or `floss`.

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
* `Fu_first_joint_failure_kgf`: load when the first glued joint separated or a member
  split in shear: §12.5 counts "shear, including joint failure" as failure, so a strict
  judge who sees it could stop the test there. `montecarlo` gives both readings:
  `recommended_prediction_kgf` (judged at the collapse, the usual practice) and
  `recommended_prediction_strict_kgf` (at the first visible joint failure).
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
1. Coupon level: fill the templates in `validation/physical/` for the tests you made:
   `stick_bending_template.csv` (E, bending strength) and `glue_joint_template.csv`
   (tau_g), which `calibrate` fits through the simulated coupons, and the direct ones:
   `stick_tension_template.csv` (f_t), `stick_compression_template.csv` (f_c, short
   stacks), `crush_across_template.csv` (f_c90), `end_grain_template.csv` (the
   end-grain glue factor) and `lap_slip_template.csv` (a lap test's load-slip record:
   glue shear stiffness and toughness). Run `bridgesim lab --data validation/physical
   --json` for each specimen's simulated-vs-measured errors and a 0-100 reality-gap
   score, then `bridgesim calibrate validation/physical --out my_material.yaml`: the
   material YAML plus `my_material.overrides.yaml` (the measured simulation values and
   the scatter as `params:`); re-run `lab` to confirm.
2. Bridge level: crush a prototype (or a sub-assembly), log load against mid-span
   deflection (`midspan_deflection_mm`) or the jack stroke (`jack_travel_mm`) in
   `crush_test_template.csv` (the whole record; mark the row of the failure with
   `event = failure` and the part that broke first in `notes`, with its side: "top chord,
   far side, left"). The peak load and the broken part alone are enough (that is what the
   competition gives you). Then `bridgesim compare results/crush.json my_test.csv --json`:
   errors on ultimate load, stiffness (secant 10-50 % of the peak), deflection at peak,
   curve shape and failure location (side included), a 0-100 score (0 when off by 1.5x
   in load or 2x in stiffness), the §12.7 prediction error, and advice on what to measure
   next. Do it **blind**: crush (and `montecarlo --out DIR`) before the physical test,
   then `bridgesim compare DIR/montecarlo.json my_test.csv` says whether the test falls
   inside the predicted range (see `validation/README.md` §3.3 for the acceptance
   criteria at each level).
3. Close the gap: `bridgesim calibrate-bridge BRIDGE.SLDPRT my_test.csv` fits a global
   stiffness and a global strength factor to that test and writes the overrides to use
   from then on. A strength factor far from 1 means the governing mechanism is off:
   compare `first_failure` with the broken part and measure that material. A test used to
   calibrate no longer validates: check the fitted model on the next test.

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
pattern unless `butt_stagger_mm` says otherwise; members are shear-flexible fibre beams
(no splitting along the grain except through shear), tapered ones with their real
profile (a part at least twice as long as its section is a member); compact blocks are
rigid; plates crack ply by ply; thick plates use thin-plate elements, and plates made of
narrow arms are meshed coarsely across them (`narrow_plates`); the bearing faces seat over 0.3 mm and the support corners are up to 0.5 mm out
of level (`contact.seating_mm`, `scatter.support_level_mm`: measure them on the built
bridge); parts that come into contact during the test push on each other without
friction; time is compressed (the jack moves ~mm/s instead of mm/min; the
materials have no rate effect, and `--rate-check` shows the speed does not matter).
