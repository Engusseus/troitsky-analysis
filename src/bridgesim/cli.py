"""Command line interface: ``bridgesim run``, ``bridgesim check``, ``bridgesim generate``."""

from __future__ import annotations

import csv
import io
import math
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError

from bridgesim import __version__
from bridgesim.analysis import AnalysisError, analyze
from bridgesim.materials import bridge_material, load_material
from bridgesim.rules import RuleSet, RulesReport, apply_crushing, evaluate
from bridgesim.schema import Bridge, Material
from bridgesim.textsafe import csv_row, friendly_error, printable
from bridgesim.units import n_to_kgf

app = typer.Typer(help="Troitsky popsicle-stick bridge analysis (bridgesim).",
                  no_args_is_help=True, add_completion=False)

MaterialOpt = Annotated[
    str | None, typer.Option("--material", "-m", help="Material YAML path or bundled name")
]
RulesOpt = Annotated[str, typer.Option("--rules", "-r", help="Rules YAML path or bundled name")]

#: Problems with an input file: reported in one short message, exit code 2.
BAD_INPUT = (ValidationError, ValueError, yaml.YAMLError, UnicodeDecodeError, OSError)


@app.callback()
def _utf8_output() -> None:
    """Write console output as UTF-8, before any command prints.

    When stdout is a pipe or a file, Python on Windows encodes it with the ANSI code page
    (cp1252), so "§" leaves as the single byte 0xA7, which PowerShell, CI logs and most
    viewers (all UTF-8) cannot decode. Reconfiguring here, rather than in a wrapper around
    ``app``, also fixes console scripts installed before this change.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _fail(what: str, exc: BaseException) -> None:
    typer.secho(f"ERROR: could not read the {what}:\n{printable(friendly_error(exc))}",
                fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2) from exc


def _inputs(file: Path, material: str | None, rules: str) -> tuple[Bridge, Material, RuleSet]:
    try:
        bridge = Bridge.from_file(file)
    except BAD_INPUT as exc:
        _fail(f"bridge file {file}", exc)
    try:
        mat = load_material(material) if material else bridge_material(bridge)
    except BAD_INPUT as exc:
        _fail("material", exc)
    try:
        rs = RuleSet.load(rules)
    except BAD_INPUT as exc:
        _fail(f"rules {rules!r}", exc)
    return bridge, mat, rs


def _kgf(N: float) -> str:
    if math.isnan(N):
        return "NOT EVALUATED"
    return f"{n_to_kgf(N):.1f} kgf" if math.isfinite(N) else "none"


def _print_rules(rep: RulesReport) -> None:
    typer.echo(f"\nRule check ({printable(rep.ruleset)}):")
    for r in rep.results:
        if r.passed is None:
            continue
        mark = "PASS" if r.passed else "FAIL"
        extra = f"  -{r.penalty:g} pts" if r.penalty else ""
        extra += f"  BANS {', '.join('§' + b for b in r.bans)}" if r.bans else ""
        extra += "  DISQUALIFICATION RISK" if r.disqualification else ""
        amb = " (ambiguous rule, stricter reading)" if r.ambiguous else ""
        typer.echo(printable(f"  [{mark}] §{r.section:<9} {r.title}: {r.measured_detail}"
                             f"{extra}{amb}"))
    typer.echo(printable(f"  Total penalty: {-rep.total_penalty or 0:g} pts; bans: "
                         f"{', '.join('§' + b for b in rep.bans) or 'none'}"))
    typer.echo(printable(f"  Not checked by the tool: {', '.join(r.title for r in rep.info)}"))


@app.command()
def run(
    file: Annotated[Path, typer.Argument(help="Bridge YAML file", exists=True)],
    material: MaterialOpt = None,
    rules: RulesOpt = "troitsky_2027",
    out: Annotated[Path | None, typer.Option(help="Write report, CSV and PNGs here")] = None,
) -> None:
    """Analyse a bridge: predicted failure load, governing member, mass, rule check."""
    bridge, mat, rs = _inputs(file, material, rules)
    bridge = apply_crushing(bridge, rs)
    try:
        res = analyze(bridge, mat, deflection_limit_mm=rs.crushing.deflection_limit_mm)
    except AnalysisError as exc:
        typer.secho(f"ERROR: {printable(exc)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    try:
        rep = evaluate(bridge, mat, rs)
    except ValueError as exc:
        _fail(f"rules {rules!r}", exc)

    assumed = mat.assumed_keys()
    if assumed:
        typer.secho(f"WARNING: {len(assumed)} material values are ASSUMED placeholders "
                    f"({', '.join(assumed)}). Results are illustrative only.",
                    fg=typer.colors.YELLOW)
    typer.echo(f"\n{printable(bridge.name)}  (bridgesim {__version__})")
    typer.echo(f"  Predicted ultimate load F_u,p : {n_to_kgf(res.Fu_pred_N):8.1f} kgf "
               f"({res.Fu_pred_N:.0f} N)")
    typer.echo(f"  Governing                     : {res.governing_label}"
               f"{printable(res.governing_detail)}")
    typer.echo(f"  Limits: strength {_kgf(res.Fu_strength_N)}, deflection "
               f"{_kgf(res.Fu_deflection_N)}, global buckling {_kgf(res.Fu_buckling_N)}")
    modes = res.buckling_load_table() if math.isfinite(res.Fu_buckling_N) else []
    if modes:
        typer.echo(f"  Global buckling mode 1        : {printable(modes[0]['shape'])}")
    if len(modes) > 1:
        typer.echo("  Next global buckling loads    : " + ", ".join(
            f"{row['F_cr_kgf']:.1f} kgf (mode {row['mode']})" for row in modes[1:]))
    working = res.buckling_load_table(braces_working=True) if res.has_buckling_bracket else []
    if working:
        typer.echo(f"  Global buckling, X working    : {_kgf(res.Fu_buckling_braces_working_N)} "
                   f"with one strap of each slack X in tension (F_u,p uses the lower "
                   f"{_kgf(res.Fu_buckling_N)})")
    typer.echo(f"  Deflection at F_u,p           : {res.delta_at_Fu_mm:.1f} mm "
               f"(limit {res.deflection_limit_mm:g})")
    extra = (f", incl. {res.mass.extra_kg:.2f} kg extra wood"
             if res.mass.extra_kg >= 0.005 else "")
    typer.echo(f"  Mass                          : {res.mass.total_kg:.2f} kg "
               f"(~{res.mass.stick_count} sticks{extra})")
    typer.echo(f"  Efficiency eta_s              : {res.efficiency:.1f} kgf/kg")
    typer.echo("\n  Most critical members (utilisation at F_u,p):")
    for m in res.critical_members(5):
        typer.echo(f"    {printable(m.id):<10} {m.group:<13} U = {m.U * res.load_factor:5.2f}  "
                   f"{m.util.mode_label}")
    if res.slack_members:
        typer.echo(f"  Note: {printable(res.slack_note)}")
    for w in res.warnings:
        typer.secho(f"  WARNING: {printable(w)}", fg=typer.colors.YELLOW)
    _print_rules(rep)

    if out:
        from bridgesim import report, viz

        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(report.to_markdown(res, rep), encoding="utf-8")
        (out / "report.html").write_text(report.to_html(res, rep), encoding="utf-8")
        written = ["report.md", "report.html"]
        tables = {"members.csv": res.member_table()}
        if modes:
            tables |= {"buckling_mode.csv": res.buckling_mode_table(1),
                       "buckling_loads.csv": modes + working}
        for name, rows in tables.items():
            with (out / name).open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(csv_row(r) for r in rows)
            written.append(name)
        figures = {"sfd_bmd_global.png": viz.global_sfd_bmd_figure(res)}
        if modes:
            figures["buckling_mode.png"] = viz.buckling_mode_figure(res)
        for name, fig in figures.items():
            (out / name).write_bytes(viz.figure_png(fig))
            written.append(name)
        typer.echo(f"\nWrote {', '.join(written)} to {out}")


@app.command()
def check(
    file: Annotated[Path, typer.Argument(help="Bridge YAML file", exists=True)],
    material: MaterialOpt = None,
    rules: RulesOpt = "troitsky_2027",
) -> None:
    """Check the competition rules only. Exit code 1 if any checked rule fails, 2 if a file
    cannot be read."""
    bridge, mat, rs = _inputs(file, material, rules)
    try:
        rep = evaluate(bridge, mat, rs)
    except ValueError as exc:
        _fail(f"rules {rules!r}", exc)
    _print_rules(rep)
    if rep.total_penalty or rep.bans or rep.disqualification_risks or not rep.all_passed:
        raise typer.Exit(code=1)


@app.command()
def generate(
    kind: Annotated[str, typer.Argument(help="Generator name (warren)")] = "warren",
    out: Annotated[Path, typer.Option(help="Output YAML")] = Path("bridge.yaml"),
    span_mm: float = 1150.0,
    n_panels: int = 8,
    truss_height_mm: float = 260.0,
    deck_top_elevation_mm: float = 200.0,
    deck_clear_width_mm: float = 170.0,
    name: str = "Warren truss",
) -> None:
    """Generate a parametric bridge YAML (default sections; edit the file to change them)."""
    from bridgesim.generators import GENERATORS

    if kind not in GENERATORS:
        raise typer.BadParameter(f"Unknown generator {kind!r}; choose from {list(GENERATORS)}")
    Params, gen = GENERATORS[kind]
    try:
        p = Params(name=name, span_mm=span_mm, n_panels=n_panels,
                   truss_height_mm=truss_height_mm, deck_top_elevation_mm=deck_top_elevation_mm,
                   deck_clear_width_mm=deck_clear_width_mm)
        bridge = gen(p)
    except (ValidationError, ValueError) as exc:
        typer.secho(f"ERROR: cannot generate this bridge:\n{printable(friendly_error(exc))}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    bridge.save(out)
    typer.echo(f"Wrote {out}")


@app.command()
def version() -> None:
    """Print versions."""
    from bridgesim import PYNITE_VERSION

    typer.echo(f"bridgesim {__version__} (Pynite {PYNITE_VERSION})")


# ------------------------------------------------------------------- crush simulation
# These commands work on CAD files (.SLDPRT, .x_b, .stl). With --json they print one
# JSON document (schema: `bridgesim schema <name>`) and nothing else, for scripts and AI
# agents. Exit codes: 0 ok, 2 bad input, 3 simulation problem.

CadArg = Annotated[Path, typer.Argument(help="SolidWorks part (.SLDPRT), Parasolid .x_b "
                                             "or STL (all bodies in one file, mm)")]
OverridesOpt = Annotated[Path | None, typer.Option(
    "--overrides", "-o", help="YAML answering the questions inspect asks (layups, glue "
                              "quality, floss strands, measured values)")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print the report as JSON only")]
SimOpt = Annotated[str | None, typer.Option("--sim-params", help="Simulation parameter "
                                                                 "YAML (default: bundled)")]


def _sim_fail(exc: BaseException) -> None:
    typer.secho(f"ERROR: {printable(friendly_error(exc))}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=3) from exc


def _print_questions(qs) -> None:
    if not qs:
        return
    typer.echo("\nQuestions for the designer (answer them in an overrides YAML):")
    for q in qs:
        typer.echo(f"  [{q.severity}] {q.question}")
        typer.echo(f"      assumed: {q.assumed}")


@app.command()
def inspect(cad: CadArg, overrides: OverridesOpt = None, material: MaterialOpt = None,
            rules: RulesOpt = "troitsky_2027", sim_params: SimOpt = None,
            json_out: JsonOpt = False) -> None:
    """Read a CAD bridge and report how bridgesim understood it (members, plates, joints,
    loading, rules) and what it had to assume. Fast: no simulation."""
    from bridgesim.sim import study

    try:
        rep = study.inspect(cad, overrides, material or "popsicle_birch", sim_params, rules)
    except BAD_INPUT as exc:
        _fail(f"CAD file {cad}", exc)
    if json_out:
        typer.echo(rep.model_dump_json(indent=2))
        return
    typer.echo(f"{rep.file}: {rep.n_bodies} bodies, {rep.mass_kg:.2f} kg")
    kinds: dict[str, int] = {}
    for b in rep.bodies:
        kinds[b.kind] = kinds.get(b.kind, 0) + 1
    typer.echo("  modelled as: " + ", ".join(f"{v} {k}" for k, v in kinds.items()))
    typer.echo(f"  glued joints: {len(rep.joints)}; model: {rep.model}")
    typer.echo(f"  loading plate at {rep.loading['plate_centre_mm']} mm on "
               f"{', '.join(rep.loading['deck_bodies'])}")
    typer.echo(f"Rules: penalty {rep.rules.total_penalty:g}, bans {rep.rules.bans or 'none'}")
    for ln in rep.rules.lines:
        if ln.passed is not None:
            typer.echo(f"  [{'PASS' if ln.passed else 'FAIL'}] §{ln.section:9s} {ln.title}: "
                       f"{ln.measured}")
    _print_questions(rep.questions)


@app.command()
def crush(cad: CadArg, overrides: OverridesOpt = None, material: MaterialOpt = None,
          rules: RulesOpt = "troitsky_2027", sim_params: SimOpt = None,
          seed: Annotated[int, typer.Option(help="Random imperfections/scatter seed "
                                                 "(-1: perfect bridge)")] = 0,
          speed: Annotated[float | None, typer.Option(
              help="Simulated jack speed, mm/s (default from --fidelity)")] = None,
          fidelity: Annotated[str, typer.Option(
              help="quick (screening, ~3-5x faster), standard, fine")] = "standard",
          out: Annotated[Path | None, typer.Option("--out", help="Folder for crush.json, "
                                                                 "curve.csv, replay.html")]
          = None,
          json_out: JsonOpt = False) -> None:
    """Simulate the §12.5 crushing test of a CAD bridge (nonlinear, dynamic): joints
    break, members buckle and crack, the bridge collapses. Writes an interactive 3D
    replay with --out."""
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions

    def prog(f, msg):
        if not json_out:
            typer.echo(f"  {f:4.0%} {msg}", err=True)

    try:
        rep, _ = study.crush(cad, overrides, material or "popsicle_birch", sim_params,
                             rules, seed=None if seed < 0 else seed, out_dir=out,
                             options=CrushOptions(speed_mm_s=speed) if speed else None,
                             progress=prog, fidelity=fidelity)
    except BAD_INPUT as exc:
        _fail(f"CAD file {cad}", exc)
    except RuntimeError as exc:
        _sim_fail(exc)
    if json_out:
        typer.echo(rep.model_dump_json(indent=2))
        return
    typer.echo(f"\n{rep.file}: predicted ultimate load F_u = {rep.Fu_kgf:.1f} kgf "
               f"({rep.Fu_N:.0f} N), {rep.failure_mode}")
    typer.echo(f"  {rep.failure_detail}")
    typer.echo(f"  mass {rep.mass_kg:.2f} kg, efficiency {rep.efficiency_kgf_per_kg:.1f} "
               f"kgf/kg, deck deflection at F_u {rep.deflection_at_Fu_mm:.1f} mm")
    typer.echo("Failure sequence:")
    for line in rep.failure_sequence[:15]:
        typer.echo(f"  {line}")
    q = rep.quality
    typer.echo(f"Numerics: dt {q.dt_s * 1e6:.2f} us, {q.steps} steps, {q.wall_s:.0f} s, added "
               f"mass {q.added_mass_fraction:.1%}, peak kinetic/work {q.kinetic_to_work_max:.3f}")
    for k, v in rep.artifacts.items():
        typer.echo(f"  {k}: {v}")
    _print_questions(rep.questions)


@app.command()
def compare(result: Annotated[Path, typer.Argument(help="crush.json from bridgesim crush")],
            test: Annotated[Path, typer.Argument(help="Your crush test CSV (format of "
                                                      "validation/physical/"
                                                      "crush_test_template.csv)")],
            json_out: JsonOpt = False) -> None:
    """How close is the simulation to your real crush test? Prints a reality-gap score."""
    from bridgesim.sim import study

    try:
        gap = study.compare(result, test)
    except BAD_INPUT as exc:
        _fail("inputs", exc)
    if json_out:
        typer.echo(gap.model_dump_json(indent=2))
        return
    typer.echo(f"{gap.subject}: reality-gap score {gap.score_0_100:.0f}/100")
    for mt in gap.metrics:
        err = "" if mt.error is None else f" ({mt.error:+.0%})"
        typer.echo(f"  {mt.name}: simulated {mt.simulated} vs measured {mt.measured} "
                   f"{mt.unit}{err}")
    for a in gap.advice:
        typer.echo(f"  - {a}")


@app.command()
def lab(material: MaterialOpt = None, sim_params: SimOpt = None,
        data: Annotated[Path | None, typer.Option(help="Folder with your test CSVs "
                                                        "(validation/physical)")] = None,
        json_out: JsonOpt = False) -> None:
    """Simulate stick bending, a glued lap joint and a stick column with the current
    material, and compare with your coupon tests if --data has them."""
    from bridgesim.sim import study

    rep = study.lab(material or "popsicle_birch", sim_params, data)
    if json_out:
        typer.echo(rep.model_dump_json(indent=2))
        return
    for t in rep.tests:
        typer.echo("  " + ", ".join(f"{k}: {v}" for k, v in t.items()))
    for g in rep.gaps:
        typer.echo(f"{g.subject}: reality-gap score {g.score_0_100:.0f}/100")


@app.command()
def calibrate(data: Annotated[Path, typer.Argument(help="Folder with your test CSVs")],
              material: MaterialOpt = None,
              out: Annotated[Path, typer.Option(help="Material YAML to write")] = Path(
                  "my_material.yaml"),
              json_out: JsonOpt = False) -> None:
    """Fit E, strengths and glue strength from your test CSVs into a material YAML."""
    from bridgesim.sim import study

    mat, log = study.calibrate(data, material or "popsicle_birch")
    out.write_text(yaml.safe_dump(mat.model_dump(mode="json"), sort_keys=False),
                   encoding="utf-8")
    if json_out:
        import json as _json
        typer.echo(_json.dumps({"material": str(out), "changes": log}))
        return
    for line in log:
        typer.echo(f"  {line}")
    typer.echo(f"Wrote {out} (use it with --material {out})")


@app.command()
def montecarlo(cad: CadArg, n: Annotated[int, typer.Option(help="Number of runs")] = 8,
               overrides: OverridesOpt = None, material: MaterialOpt = None,
               speed: Annotated[float, typer.Option(help="Simulated jack speed, mm/s")]
               = 100.0, json_out: JsonOpt = False) -> None:
    """Crush the bridge n times with different imperfections and material scatter: the
    spread of F_u, and the value to put on the Load Prediction Form (§12.7)."""
    from bridgesim.sim import study
    from bridgesim.sim.crush import CrushOptions

    rep = study.montecarlo(cad, n, overrides, material or "popsicle_birch",
                           options=CrushOptions(speed_mm_s=speed))
    if json_out:
        typer.echo(rep.model_dump_json(indent=2))
        return
    typer.echo(f"F_u over {n} runs: 5 % {rep.p05_kgf:.1f}, median {rep.p50_kgf:.1f}, 95 % "
               f"{rep.p95_kgf:.1f} kgf. Predict {rep.recommended_prediction_kgf:.1f} kgf.")
    typer.echo(f"First failures: {rep.failure_modes}")


@app.command()
def schema(name: Annotated[str, typer.Argument(help="inspect, crush, reality_gap, lab, "
                                                     "montecarlo or question")]) -> None:
    """Print the JSON Schema of a report (for scripts and AI agents)."""
    import json as _json

    from bridgesim.sim.report import SCHEMAS

    if name not in SCHEMAS:
        _fail("schema name", ValueError(f"unknown schema {name!r}; use one of "
                                        f"{', '.join(SCHEMAS)}"))
    typer.echo(_json.dumps(SCHEMAS[name].model_json_schema(), indent=2))


@app.command()
def mcp() -> None:
    """Run bridgesim as an MCP server (stdio) so an AI agent can call inspect, crush,
    compare, lab and calibrate as tools. Needs: pip install "bridgesim[mcp]"."""
    from bridgesim.sim.mcp_server import main

    main()


if __name__ == "__main__":
    app()
