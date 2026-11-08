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
`ask_designer`, `screen`, `crush`, `rerun`, `strengthen`, `diagnose`, `confirm`, `measure`,
`predict`, `compare`), `why` in plain words, the `ids` of the questions concerned, the `parts` and
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
| `bridgesim screen BRIDGE.SLDPRT [-o overrides.yaml] [--load 100] [--top 25] --json` | Linear pre-screen, no simulation (on the standard crush's mesh): the bridge's stiffness under the plate (`stiffness_kgf_per_mm`), its lowest natural frequencies on its supports with the plate on it (`modes`: what a tap test of the built bridge measures, with the parts that move most in each; the supports and the plate touch it through a numerical contact stiffness, which moves them as much as the wood does (contact 4x softer: the first mode of Bridge_v10 47 -> 27 Hz, the ambitious review of round 21), so read them as a rough guide, not a check of E or the glue; the supports hold by friction, an end sliding when the thrust there is more than friction holds) and every joint (glue and `bearing`), member (tension, compression, shear, Euler buckling over its length) and plate (per ply and mode) with its `utilisation` at `--load` and the plate load at which it reaches its limit if the bridge stayed linear (`limits`, most loaded first; `first_limit_kgf`: where damage starts, often far below F_u, since the load then finds other paths, which only crush follows: on Bridge_v10 its top 25 hold neither node M's diagonal laps, which govern the crush, nor node A's, as 70-100 joints soften and pass their load on before the peak; the adversarial review of round 25). For the stiffness, the most loaded parts, and how a what-if moves them in seconds; judge F_u, where it fails and design changes with crush and `ab` | seconds |
| `bridgesim compare DIR/crush.json my_test.csv --json` | Reality gap vs a physical crush test. Given `montecarlo.json` instead of `crush.json`: where the test falls in the predicted range | seconds |
| `bridgesim calibrate-bridge BRIDGE.SLDPRT my_test.csv [-o overrides.yaml] [--fidelity standard] [--seeds 3] [--out fitted.overrides.yaml] --json` | After a physical crush test of this bridge: a global stiffness and a global strength factor fitted so the simulated crush reproduces it (up to 4 steps; each crushes seeds 1..`--seeds` and fits their median, so the factors do not absorb one seed's imperfections; `steps[].Fu_kgf_runs`); writes your overrides plus the fitted `params`. The test is then used up for validation | 2-4 x seeds x crush / processes |
| `bridgesim sensitivity BRIDGE.SLDPRT [--rel 0.2] [--fidelity quick] [--seeds 1] [--param KEY]... --json` | Which material values move F_u most: each one lowered and raised by `rel`; `measure_first` lists the tests that would narrow the prediction most. `--param` (repeat it) chooses the values, any key of `materials/simulation.yaml` or `material.<key>` (default: the 22 that usually matter most); `--seeds 3` averages over seeds | (1 + 2 x params) x seeds x crush / processes |
| `bridgesim lab [--data validation/physical] --json` | Simulated coupon tests (stick bending, lap joint, stick column) and their gap to the team's coupon data | ~1 min |
| `bridgesim calibrate validation/physical --out my_material.yaml` | Fit E, f_t/f_c and tau_g so the *simulated* coupons reproduce the measured ones (`lab --data` then shows ~0 % error); f_t, f_c, f_c90, the end-grain factor, the glue's shear stiffness and toughness and the interlaminar shear strength (short-beam shear) from the direct coupon tests; with 3+ specimens of a kind also the scatter (`my_material.overrides.yaml`, `params:`). A folder without any test results is an input error (exit 2) | ~1 min |
| `bridgesim ab A.SLDPRT B.SLDPRT [-n 3] [--overrides-a a.yaml] [--overrides-b b.yaml] [--fidelity standard] --json` | Is B better than A? Both crushed with the same seeds: change of F_u and efficiency with a 90 % interval, `significant`, `verdict`, what governed each run (`runs_a/b[].governing`, the most frequent in `governing_a/b`) and `failure_moved`: the seeds where B failed somewhere else than A (strengthening one part often moves the failure to the next). B may be A with other overrides: a what-if (`section_scale`, `glue_factor`) before changing the CAD | 2n x crush / processes |
| `bridgesim diagnose BRIDGE.SLDPRT [-o overrides.yaml] [-n 2] [--fidelity quick] [--candidates 3] [--out DIR] --json` | What really limits the bridge, measured: crushes it with seeds 1..n, strengthens each part that broke near the peak in turn (glue x1.5 for a joint, section x1.25 for a member or plate, on every copy) on the same seeds, and ranks them by the paired rise of F_u (`candidates`, `limiting`, `verdict`; `governing_after`: where the failure moves). The `governing` label of one crush names the first part in a cascade; diagnose says which one the load depends on, or that no single change helps (then strengthen them together). `DIR/whatif.yaml`: the what-if to confirm with `ab` | (1 + candidates) x n x crush / processes |
| `bridgesim montecarlo BRIDGE.SLDPRT -n 8 [--fidelity standard] [--processes 4] [--sample-placeholders] [--out DIR] --json` | Spread of F_u over imperfections and material scatter (`min_kgf`, `max_kgf`, p05/p50/p95: below 20 runs read min and max); `recommended_prediction_kgf` for the Load Prediction Form (the 1/F-weighted median, which minimises the §12.7 error, over the runs times the mesh and model-form error, which pulls it below the runs' own `recommended_prediction_runs_kgf`), with `recommended_interval_kgf` (how far other runs could move it: wide with few runs, run more) and `prediction_by_model_cov` (the same at a model-form error of 0.15, the shipped 0.40 and 0.76: how much that judgement moves it) and `prediction_if_biased_kgf` (if the model read this bridge as it reads the published white-glue tests, `uncertainty.published_bias` 0.93 since round 26, 0.67 in round 25: shown, not applied); a run flagged only as not quasi-static is crushed again at half the jack speed (`slowed_runs`); other untrustworthy runs are listed apart (`Fu_untrustworthy_kgf`) and left out (when fewer than half are trustworthy all are used and `note` starts with NOT RELIABLE: re-run at the crush's `quality.min_fidelity`). `note` says what the spread leaves out: the placeholder material values, unless `--sample-placeholders` (MCP `sample_placeholders`) draws them per run (lognormal, each with its own prior, 20-50 %, a Latin hypercube over the runs; `sampled_placeholders`) so the range includes them. `total_p05_kgf`/`total_p95_kgf`: that spread times the mesh error and the model-form error (`uncertainty.mesh_cov` 5 %, `uncertainty.model_cov` 40 %: the scatter of the published white-glue truss tests about the model, pooled with how far F_u moved with modelling choices tests have not settled), the range to give before any physical test | n x crush / processes |
| `uv run python validation/check_runs.py DIR` | Every crush.json under DIR (single crushes, each montecarlo run) with its `code_sha`, F_u and governing part; exit 1 if any was made on other code than the one checked out: quote evidence only from a folder it passes | seconds |
| `bridgesim schema inspect\|inspect_brief\|crush\|crush_brief\|screen\|reality_gap\|lab\|montecarlo\|design_comparison\|diagnosis\|sensitivity\|bridge_calibration\|question\|job_status` | JSON Schema of each report (the `_brief` forms are what `--brief` and the MCP tools return by default; every MCP tool but `get_schema` publishes its schema as its output schema) | instant |
| `bridgesim mcp` | The same as MCP tools (`uv sync --extra mcp`) | - |

Fidelity: `quick` (coarse mesh, jack 200 mm/s, ~3 min for a 100-body bridge on 4 cores)
to screen design changes, `standard` (100 mm/s, ~5-10 min) for the answer, `fine` (finest
mesh, 50 mm/s, ~30-40 min) to check that the answer has converged. These speeds are for a
stiff bridge (its lowest mode under the plate rings with a period up to 10 ms; Bridge_v6
~8 ms): a softer or lighter one is crushed slower in proportion, at most 16x, speeding
up again once its deflection is large, so it stays quasi-static (`quality.speed_mm_s`,
`quality.fundamental_period_ms`). Mesh convergence on
Bridge_v6 (`validation/README.md` §3.4, five seeds at 100 mm/s; a seed is the same
imperfect bridge at every fidelity): fine is +4.6 % above standard on average (-1.1 % to
+11 % seed by seed); on Bridge_v10, each fidelity at its own speed, round 26's model:
quick / standard / fine 1694 / 1823 / 1805 kgf on seed 0, quick / standard 1832 / 1862
on seed 1, standard and fine at node M (the diagonal's gusset laps), quick at the
diagonal's foot on the stub (`validation/README.md` §3.19; round 25's model 1877 / 1938 /
1845 and 2137 / 1829, round 24's 1830 / 1725 / 1701, round 23's 2079 / 2021 / 2014, round 22's 1693 /
1651 / 1651; round 20's at one speed, 100 mm/s, 2162 / 2061 / 2093, standard at 50 mm/s
-1.3 %). The meshes are not monotone and their order changes with the design: fine read
+2.3 % above standard on Bridge_v6's 10 seed pairs, 0 to -4.8 % on Bridge_v10 (rounds
22-26), so `montecarlo` takes out only the jack speed's share (`uncertainty.mesh_bias_*`)
and keeps ~5 % scatter, and quick -7 to +17 % from standard (rounds 25-26), sometimes
failing at another joint: when failure sites compete, confirm a quick `diagnose` or
`sensitivity` at standard. The spread
*between* seeds is ~20 %: several failure sites (the hub core, the End Post joints, the
cross-beam's joints to the deck stringers) compete within a few %, and which goes first
changes with the mesh and the seed (`competing` lists them; `montecarlo` counts them in
`governing_parts`). So rank designs with one fidelity and the same seeds (`ab`, n >= 3),
give F_u as the `montecarlo` range, and never compare F_u across fidelities. A small,
light bridge (members ~6-10 mm, glued joints of a few tens of mm^2) converges less well:
on a light truss (round 18) quick read 17.9, standard 34.5 and fine 44.6 kgf
(`validation/README.md` §3.10), and in round 21 standard 46.2 and fine 35.8, failing
elsewhere: use fine for it (a crush whose elements are more than 3x the failing
members' depth, 1.5x below 10 mm, is flagged untrustworthy; `quality.min_fidelity` names
the coarsest fidelity fine enough, and the `diagnose` and `predict` steps use it). `ab`, `montecarlo`, `sensitivity` and `calibrate-bridge`
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
| `start_diagnose(cad_path, out_dir, overrides_yaml, n, fidelity, candidates)` then `job_status(out_dir)` | background: which part really limits the bridge |
| `start_sensitivity(cad_path, out_dir, overrides_yaml, rel, params, seeds)` then `job_status(out_dir)` | background: what to measure first |
| `start_calibrate_bridge(cad_path, test_csv_path, out_dir, overrides_yaml, fidelity, seeds)` then `job_status(out_dir)` | background: fit stiffness and strength to a physical crush test (writes `out_dir/fitted.overrides.yaml`) |
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
  and strength are guessed; meanwhile it is UD sticks in plies of a stick's thickness,
  glued together) or its grain direction (`orientation:<body>`);
* wrong orientation or units (`geometry:orientation`);
* a part that touches nothing, or a floss end tied to nothing (`floating:`, `loose:`,
  `floss_end:`);
* two solids that overlap in the CAD (`overlap:`) - usually a modelling error;
* bodies the CAD file names that could not be read at all (`cad_unread`, important:
  they are missing from the model);
* faces read only approximately (`cad_approximated:<body>`, important when they are a
  twentieth of the body's surface): a spline, cone or freeform surface flattened, or a
  face that could not be decoded closed over;
* plates made of narrow arms (an L-shaped leg, an X, a curved rib) are modelled as
  frames of members along their arms (`frame_plates`, info: which ones; a cross-ply or
  angle-ply one is taken as one sheet whose plies keep the plate's long axis in every
  arm, `frame_plies:<part>`, `grain: arm` if they were laid along each arm); one whose glued
  parts lie off its arms stays a plate with about one shell element across an arm, ~4x
  too stiff in in-plane bending (`narrow_plates`): if the part is really members, draw
  each as its own body (it becomes a bar);
* members whose glued area is much weaker than the member in tension (`weak_joints`, one
  question for all of them) - a missing gusset? (`screen` shows which joints are really
  loaded: a member in compression needs far less);
* laminated members and stick plates longer than one stick (`butt_joints`, one question
  for all of them): how far apart the butt joints of neighbouring plies are (it sets each
  member's tension strength: in the default brick pattern every other ply is cut at the
  same section, so a 13-ply member keeps 0.8 x 6/13 = 37 % of its tension strength, a
  deck about 40 %; and how the ends meet: a gap over 0.5 mm, as rounded stick ends
  leave, does not close and leaves the joint out in compression too);
* a loading plate that does not fit where it goes (`plate_fit`, important: parts closer
  together than the plate is wide; meanwhile they are taken as clear of it);
* a rigid block directly under the loading plate (`rigid_under_plate:`), and the compact
  parts in the load path modelled as rigid blocks (`rigid_blocks`);
* glued joints that are mostly end grain, a member's end glued flat (`end_grain_joints`):
  whether they are reinforced, and how well the cut ends fit (`scatter.bearing_gap_mm`: a
  cut end out of square stands off its seat at one edge, dry where the gap is wider than
  the glue bridges, until it closes);
* a `kind` set in the overrides that does not fit the body's shape (`kind:`);
* a failing rule (`rule:<section>`), the mass with an assumed density
  (`material:density`), placeholder material values (`material:measure`).

Ask the `important` ones first. Do not guess an answer the designer can give. A question
is closed by its answer (setting what it asks), or, when the designer confirms the design
is as drawn, by listing its id (or a pattern such as `floating:*`) under `acknowledge:`
in the overrides; acknowledged ids are reported in `acknowledged_questions`.

Bodies are located by `centroid_mm`, `bbox_min_mm`, `bbox_max_mm` and `axis` in the
inspect report, and events by `where_mm` in the crush report, so you can tell the
designer *where* a part or a failure is ("the gusset at x = -310 mm, far side").

## Overrides YAML

```yaml
bodies:                     # by exact body name or glob pattern
  "Body 21531": {layup: "10-ply UD"}            # what an unnamed body is made of
  "Saddle*": {kind: block}                      # bar | shell | frame | block | cable | ignore
  "Hub Tie Strap*": {glue_factor: 0.7}          # poorly glued part (x glue strength)
  "Spacer*": {glue_factor: 0}                   # touching but NOT glued
  "Top Chord*": {strength_factor: 0.9, grain: span}
  "Hub Cover*": {grain: arm}                    # a +-45 frame's plies laid along each arm
  "Heel Seat*": {grain: y}                      # a block's fibres along y: it bears on end grain
  "Deck Stringer*": {butt_stagger_mm: 30}       # joints of neighbouring plies 30 mm apart
  "Bottom Chord*": {butt_stagger_mm: 0}         # plies without butt joints (long strips)
  "Top Chord*": {butt_gap_mm: 0}                # ply ends sanded tight: joints bear
  "Floor Beam*": {kind: bar, axis: z}            # a member spanning along z (its outline
                                                 # may be longer the other way)
  "Deck Stringer*": {section_scale: 1.25}       # what-if: section x1.25 (area x1.56)
glue:
  "Top Chord (10-ply UD 20.9x47.5) - Far | T-Node (10-ply) - R Far": 0.5   # one joint
  "Floor Beam (block) | Floor Beam Tenon*": solid   # one piece drawn as two bodies:
                                                    # the wood's strength across the face
params:                     # any key of materials/simulation.yaml, or material.<key>
  floss.strands: 6
  material.density_kg_m3: 590
  material.glue.tau_g_MPa: 2.6
  glue.end_grain_factor: 0.8                     # end-grain joints sized before gluing
measured:                   # params you measured: montecarlo stops redrawing them (a value
  - material.glue.tau_g_MPa # under params alone stays a placeholder, drawn around it)
acknowledge:                # questions the designer has answered "as drawn"
  - weak_joints
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
compare the two files with `ab`. (On Bridge_v6 the what-if of a quick crush, the hub
parts that broke at the peak strengthened, raised F_u by 3.3 %, 90 % interval -0.3 % to
+6.9 %, stronger on all three seeds but not proven, and the failure moved on all three;
`diagnose` then measured that no single part carries the bridge: the inner hub tie
straps +2.8 %: `validation/README.md` §3.4.) When `competing` is not empty,
run `diagnose` before redesigning.

Body names are exactly as `inspect` lists them (SolidWorks body names; mirrored copies
get "(mirror copy)", named after the original with its side words, Near and Far, L and R,
swapped to the side where the copy stands). Joint names are `A | B` as listed in the inspect `joints`; both
sides may be glob patterns (`"End Post* | T1 Cover*": 1.5` is the joint on all four
corners). A value still holding an answer_with placeholder (`<...>`) is an error that
names the key, and a layup override must read `N-ply UD`, `N-ply 0-90`, `N-ply +-45`
(optionally `along the span`), `block`, `toothpick` or `floss`.

## What the crush report means

* `Fu_kgf`: predicted ultimate load (§12.5): the peak jack force before the bridge is
  judged failed (the load falling 30 % below its peak for at least one period of the
  bridge's ringing, 50 mm deflection, or contact with the platform); 0 if it collapses under its own weight (`failure_mode: instability`).
  The load (curve, F_u, each event's load) is the jack force smoothed without lag
  (ASSUMPTIONS-SIM.md: the load reading).
* `governing`: the break that started the fall of the load, i.e. the first break from
  where the load last stood within 5 % of F_u (breaks before it were survived: the load
  came back); of the breaks within half a period of the bridge's ringing (one moment:
  the load takes that long to fall), the joint that could carry the most among those
  sharing a part with the first to let go (one node; a joint on the other truss in the
  same moment is the cascade). `competing`:
  the other parts that broke within 10 % of F_u before the failure call. A joint whose glue can carry under 2 % of F_u neither governs nor
  competes (its break is a symptom). When it is not empty the failure could start at one of them on another
  seed or mesh, so run `diagnose` before redesigning around the `governing` part.
* `failure_sequence` (text) and `failures` (structured: `kind`, `parts`, `load_kgf`,
  `jack_travel_mm`, `where_mm`, `fraction_separated`): what broke, in order, up to the
  collapse. The entries near the peak are the reason the bridge failed: strengthen
  *that*. A joint is `joint_cracking` when 30 % of its glued area has let go and
  `joint_failure` (separated) at 90 %; `member_crushing` that a member's fibres fold in
  compression past f_c (it bends on, more softly); `bearing_crushing` that the wood under
  a face pressed together (a member's end bearing on a knee, a tenon on its sill) has
  crushed 0.1 mm deep: it bears more than its grain takes, and more glue there changes
  nothing (a `strengthen` step then says so and offers no glue what-if for that face: a
  bigger bearing face, or the grain along the force); `member_shear` means the member split along its
  glue lines or grain (it keeps carrying axial load). Which part goes first can change
  with the mesh when two parts are close to failing together: check with
  `--fidelity fine` before redesigning around one part, and use `ab` to confirm.
* `events`: the first event of each kind per part, with `where_mm`; `events.csv` in the
  output folder has every event (`n_events_total`).
* `test_time_at_Fu_s`: when the real test would reach F_u (the jack's travel to the peak at
  `crusher.test_rate_mm_per_min`, 10 mm/min until someone times the crusher). Each glue
  line's strength is taken at its own time under load (from when it first took a tenth
  of its strength): a PVA line is ~26 % stronger per decade faster (`glue.rate_exponent`
  0.1: 10^0.1), against the 60 s its coupons took (`glue.coupon_time_s`); a line that
  picks up load when its neighbour breaks is loaded fast.
* `Fu_first_joint_failure_kgf`: the highest load up to the moment the first glued joint
  separated or a member split in shear (what a strict judge who stops the test there
  reads: §12.5 counts "shear, including joint failure" as failure); it equals F_u when
  the first joint goes at or after the peak. `montecarlo` gives both readings:
  `recommended_prediction_kgf` (judged at the collapse, the usual practice) and
  `recommended_prediction_strict_kgf` (at the first visible joint failure).
* `criteria`: when each §12.5 criterion (deflection, contact, joint/shear, instability)
  was first met.
* `quality.rate_check` (with `--rate-check`): F_u at half the jack speed, the change,
  and `Fu_extrapolated_kgf` ((4 F(v/2) - F(v)) / 3: F_u for a jack at rest, like the
  real test's mm/min, along the square of the speed: the inertia it adds grows with the
  kinetic energy).
* `quality.trustworthy`: `false` when the loading was not quasi-static
  (`kinetic_to_work_max` > 0.05: kinetic energy of the bridge over the jack's work while
  the load rises to 95 % of its peak), F_u moved more than 5 % at half speed
  (`quality.rate_check`, with `--rate-check`), more than 5 % mass was added, the bridge
  did not fail within the jack travel, the run had to restart, the failure starts as a
  crack in plate elements past the crack band of the mode they cracked in, the mesh is
  too coarse for a light bridge (elements more than 3x the failing members' depth, 1.5x
  below 10 mm: on a light truss of 6 mm members quick read 17.9, standard 34.5 and fine
  44.6 kgf in round 18; since round 25 members get elements at most 2x their depth, 1.33x
  below 10 mm, at quick as at standard, and still read 18.5 / 37.8 / 44.8 kgf at quick /
  standard / fine, the plates and glued joints setting it (round 26, its small joints
  sampled on their own grid: 27.0 / 38.4 / 30.3, while a truss of 10 mm members reads
  46.4 / 46.4 / 46.7), so a bridge whose failing members are under 10 mm deep needs fine; `quality.min_fidelity`: the coarsest
  fidelity fine enough), the failing joint is much narrower than the elements of a plate
  it is glued to (its glue hangs on far nodes; any bridge, since round 26), the load rose
  more than 10 % past F_u after the failure call (a first, small break; the bridge then
  carried more: read both), or bodies the CAD file names were not read (`cad_unread`,
  until it is acknowledged: they carry no load); `quality.reasons` says which. Re-run with `--rate-check` or a finer `--fidelity` before relying on an
  untrustworthy F_u.
* `assumed_parameters`: values still not measured. With placeholders the absolute load
  is uncertain by tens of %: on Bridge_v10 (round 26, `montecarlo -n 12
  --sample-placeholders`) the runs spread 1119-2374 kgf around a median of ~1615, the
  recommended prediction 1317 kgf (1511 / 877 at a model-form error of 0.15 / 0.76,
  1416 if it read this bridge as it reads the published tests), and the total range
  (with the mesh and model-form error) 784-3385 kgf. Design comparisons (A vs
  B, same seeds) cancel the scatter and most placeholder values, but not the model's
  form: a change at the part whose law is uncertain inherits that law (the adversarial
  review of round 22: glue x1.5 at node M's laps read +32 % with round 22's crossed-lap
  law and +8 % without it). Before acting on an `ab` there, run it again with the
  uncertain law at its other end (`params`: `wood.f_rolling_MPa` 2.5 and 5,
  `wood.bearing_spread_across` 0.15 and 0.6, `glue.rate_exponent` 0,
  `glue.K_s_N_per_mm3` / 4).
* `rules`: the 2027 dimensional rules measured on the solids (penalties, bans).
* `warnings`: overrides without effect, CAD faces that had to be closed over, etc.

## Quantifying how close to reality the model is

Before any test of your own: `validation/published_tests.py` rebuilds three published
craft-stick truss tests and crushes them, each at its own pace; `montecarlo` of each at
fine (`validation/README.md` §3.14, round 26, 10 runs, placeholders drawn) puts B's and C's
test loads inside the runs' range (median 1.48 and 0.58 of the test; 1.21 and 0.37 in
round 25, C's end-grain joints raised by the literature's end-grain factor) and A's far
above it (0.21; its rebuild loads the chords where the paper's wire loaded the
diagonals). None is a blind validation (A's rebuild and B's putty stand-in were set with
their comparisons; C chose the end-grain law, so its place is no evidence for it). Only C
fails where its test did (3 of 9 runs, at a vertical's foot on the bottom chord); A's lateral drift and
B's diagonal buckling at its butt joints are mechanisms the model cannot form. Small trusses, outside the range of a team's bridge: they inform
`uncertainty.model_cov` (0.40, a judgement), not a bias.

0. Decide what to measure: `bridgesim sensitivity` ranks the placeholder values by how
   much they move F_u for *this* design (`measure_first`).
1. Coupon level: fill the templates in `validation/physical/` for the tests you made
   (give each glue test its `time_to_failure_s`: a lap that breaks in 10 s reads ~20 %
   stronger than one that takes 60 s, and `calibrate` brings each to `glue.coupon_time_s`;
   the crush test's `time_s` column times the crusher):
   `stick_bending_template.csv` (E, bending strength) and `glue_joint_template.csv`
   (tau_g), which `calibrate` fits through the simulated coupons, and the direct ones:
   `stick_tension_template.csv` (f_t), `stick_compression_template.csv` (f_c, short
   stacks), `crush_across_template.csv` (f_c90), `end_grain_template.csv` (the
   end-grain glue factor), `lap_slip_template.csv` (a lap test's load-slip record:
   glue shear stiffness and toughness) and `short_beam_shear_template.csv` (a laminate's
   glue lines in shear: `glue.interlaminar_MPa`, where laminated members split). Run `bridgesim lab --data validation/physical
   --json` for each specimen's simulated-vs-measured errors and a 0-100 reality-gap
   score, then `bridgesim calibrate validation/physical --out my_material.yaml`: the
   material YAML plus `my_material.overrides.yaml` (the measured simulation values and
   the scatter as `params:`); re-run `lab` to confirm.
2. Bridge level: crush a prototype (or a sub-assembly), log load against mid-span
   deflection (`midspan_deflection_mm`) or the jack stroke (`jack_travel_mm`) in
   `crush_test_template.csv` (the whole record; mark the row of the failure with
   `event = failure` and the part that broke first in `part`: its body name as `inspect`
   lists it, a glob, or `A | B` for a joint, e.g. `Knee-Diagonal R Far* | Stub Far*`; or
   in words in `notes`, with its side: "top chord, far side, left"). The peak load and
   the broken part alone are enough (that is what the
   competition gives you). Then `bridgesim compare results/crush.json my_test.csv --json`:
   errors on ultimate load, stiffness (secant 10-40 % of the peak, as EN 12512), deflection at peak,
   curve shape and failure location (side included), a 0-100 score (0 when off by 1.5x
   in load or 2x in stiffness), the §12.7 prediction error, and advice on what to measure
   next. Do it **blind**: crush (and `montecarlo --out DIR`) before the physical test,
   then `bridgesim compare DIR/montecarlo.json my_test.csv` says whether the test falls
   inside the predicted range (see `validation/README.md` §3.3 for the acceptance
   criteria at each level).
3. Close the gap: `bridgesim calibrate-bridge BRIDGE.SLDPRT my_test.csv` fits a global
   stiffness and a global strength factor to that test and writes the overrides to use
   from then on (the fitted values listed under `measured:`, so `montecarlo` no longer
   redraws them, and `uncertainty.model_cov` 0.15: a calibrated model's error, build to
   build). A strength factor far from 1 means the governing mechanism is off:
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
design comparisons; glue lines cover the whole touching area, joint to joint they vary in
strength and in stiffness alike, and they creep (in physical test time), unless a
`glue_factor` says otherwise, and each cut end bearing on another part stands off its
seat at one edge by a seeded fit-up gap (`scatter.bearing_gap_mm`); butt joints in
laminated members and stick plates are assumed in a brick pattern unless
`butt_stagger_mm` says otherwise; members are shear-flexible fibre beams
(no splitting along the grain except through shear), tapered ones with their real
profile (a part at least twice as long as its section is a member); compact blocks are
rigid; plates crack ply by ply; plates made of narrow arms are frames of members (thick
plates otherwise use thin-plate elements); the bearing faces seat over 0.3 mm, the support
corners are up to 0.5 mm out of level and the bridge leans out of its plane by up to
0.005 rad (`contact.seating_mm`, `scatter.support_level_mm`, `scatter.lean_rad`: measure
them on the built bridge); parts that come into contact during the test push on each other without
friction; time is compressed (the jack moves ~mm/s instead of mm/min; the glue's creep
and its rate-dependent strength run on the physical test time, the jack's travel at the
real rate; the wood is as dry as its coupons unless `wood.moisture_excess_pct` says the
members were laminated in the days before the test; F_u falls a few %
as the simulated jack slows, -2.4 % from quick's 200 to 100 mm/s on Bridge_v10 and
-0.5 % from 100 to 50: `--rate-check` measures it and extrapolates to a jack at rest).
