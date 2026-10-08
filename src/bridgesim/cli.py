"""Command line interface: ``bridgesim run``, ``bridgesim check``, ``bridgesim generate``."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Annotated

import typer

from bridgesim import __version__
from bridgesim.analysis import AnalysisError, analyze
from bridgesim.materials import bridge_material, load_material
from bridgesim.rules import RuleSet, RulesReport, apply_crushing, evaluate
from bridgesim.schema import Bridge
from bridgesim.textsafe import csv_row, printable
from bridgesim.units import n_to_kgf

app = typer.Typer(help="Troitsky popsicle-stick bridge analysis (bridgesim).",
                  no_args_is_help=True, add_completion=False)

MaterialOpt = Annotated[
    str | None, typer.Option("--material", "-m", help="Material YAML path or bundled name")
]
RulesOpt = Annotated[str, typer.Option("--rules", "-r", help="Rules YAML path or bundled name")]


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
        typer.echo(printable(f"  [{mark}] §{r.section:<9} {r.title}: {r.measured_text}"
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
    bridge = Bridge.from_file(file)
    mat = load_material(material) if material else bridge_material(bridge)
    rs = RuleSet.load(rules)
    bridge = apply_crushing(bridge, rs)
    try:
        res = analyze(bridge, mat, deflection_limit_mm=rs.crushing.deflection_limit_mm)
    except AnalysisError as exc:
        typer.secho(f"ERROR: {printable(exc)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    rep = evaluate(bridge, mat, rs)

    assumed = mat.assumed_keys()
    if assumed:
        typer.secho(f"WARNING: {len(assumed)} material values are ASSUMED placeholders "
                    f"({', '.join(assumed)}). Results are illustrative only.",
                    fg=typer.colors.YELLOW)
    typer.echo(f"\n{printable(bridge.name)}  (bridgesim {__version__})")
    typer.echo(f"  Predicted ultimate load F_u,p : {n_to_kgf(res.Fu_pred_N):8.1f} kgf "
               f"({res.Fu_pred_N:.0f} N)")
    gov = f" in {printable(res.governing_member)}" if res.governing_member else ""
    typer.echo(f"  Governing                     : {res.governing_label}{gov}")
    typer.echo(f"  Limits: strength {_kgf(res.Fu_strength_N)}, deflection "
               f"{_kgf(res.Fu_deflection_N)}, global buckling {_kgf(res.Fu_buckling_N)}")
    typer.echo(f"  Deflection at F_u,p           : {res.delta_at_Fu_mm:.1f} mm "
               f"(limit {res.deflection_limit_mm:g})")
    typer.echo(f"  Mass                          : {res.mass.total_kg:.2f} kg "
               f"(~{res.mass.stick_count} sticks)")
    typer.echo(f"  Efficiency eta_s              : {res.efficiency:.1f} kgf/kg")
    typer.echo("\n  Most critical members (utilisation at F_u,p):")
    for m in res.critical_members(5):
        typer.echo(f"    {printable(m.id):<10} {m.group:<13} U = {m.U * res.load_factor:5.2f}  "
                   f"{m.util.mode_label}")
    for w in res.warnings:
        typer.secho(f"  WARNING: {printable(w)}", fg=typer.colors.YELLOW)
    _print_rules(rep)

    if out:
        from bridgesim import report, viz

        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(report.to_markdown(res, rep), encoding="utf-8")
        (out / "report.html").write_text(report.to_html(res, rep), encoding="utf-8")
        rows = res.member_table()
        with (out / "members.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(csv_row(r) for r in rows)
        (out / "sfd_bmd_global.png").write_bytes(
            viz.figure_png(viz.global_sfd_bmd_figure(res)))
        typer.echo(f"\nWrote report.md, report.html, members.csv, sfd_bmd_global.png to {out}")


@app.command()
def check(
    file: Annotated[Path, typer.Argument(help="Bridge YAML file", exists=True)],
    material: MaterialOpt = None,
    rules: RulesOpt = "troitsky_2027",
) -> None:
    """Check the competition rules only. Exit code 1 if any penalty or ban applies."""
    bridge = Bridge.from_file(file)
    mat = load_material(material) if material else bridge_material(bridge)
    rep = evaluate(bridge, mat, RuleSet.load(rules))
    _print_rules(rep)
    if rep.total_penalty or rep.bans or rep.disqualification_risks:
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
    p = Params(name=name, span_mm=span_mm, n_panels=n_panels, truss_height_mm=truss_height_mm,
               deck_top_elevation_mm=deck_top_elevation_mm,
               deck_clear_width_mm=deck_clear_width_mm)
    gen(p).save(out)
    typer.echo(f"Wrote {out}")


@app.command()
def version() -> None:
    """Print versions."""
    from bridgesim import PYNITE_VERSION

    typer.echo(f"bridgesim {__version__} (Pynite {PYNITE_VERSION})")


if __name__ == "__main__":
    app()
