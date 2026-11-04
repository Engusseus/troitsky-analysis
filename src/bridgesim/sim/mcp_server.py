"""bridgesim as an MCP server: AI agents call the simulation as tools.

Start it with ``bridgesim mcp`` (stdio). Register it in an MCP client, e.g. Claude Code::

    claude mcp add bridgesim -- uv run --directory /path/to/troitsky-analysis bridgesim mcp

Every tool returns one JSON document. On failure it returns
``{"error": "...", "kind": "bad_input" | "simulation_failed", "type": "..."}`` instead of
raising, so the agent always sees the reason.

Tools:

* ``inspect_bridge`` - what bridgesim understood from a CAD file, the rule check and the
  questions to ask the designer (seconds);
* ``crush_bridge`` - the simulated §12.5 crushing test (minutes, blocking);
* ``start_crush`` / ``start_montecarlo`` + ``job_status`` - the same in the background,
  for clients whose tool calls time out;
* ``compare_with_test`` - reality gap between a simulation and a physical crush test;
* ``lab_tests`` - simulated coupon tests, and their gap to the team's coupon data;
* ``calibrate_material`` - fit the material from the team's coupon data;
* ``get_schema`` - JSON Schema of any report.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

#: background jobs started by this server process: out_dir -> Popen
_JOBS: dict[str, subprocess.Popen] = {}


def _yaml_or_none(text: str | None):
    if not text:
        return None
    import yaml

    return yaml.safe_load(text)


def _error(exc: BaseException) -> str:
    from bridgesim.cad.parasolid import ParasolidError
    from bridgesim.cad.sldprt import SolidWorksFormatError

    bad = (ValueError, KeyError, FileNotFoundError, OSError, SolidWorksFormatError,
           ParasolidError)
    kind = "bad_input" if isinstance(exc, bad) else "simulation_failed"
    msg = str(exc) or type(exc).__name__
    return json.dumps({"error": msg, "kind": kind, "type": type(exc).__name__})


def _safe(fn):
    """Wrap a tool so any exception comes back as a JSON error the agent can read."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as exc:  # noqa: BLE001
            return _error(exc)

    return wrapper


def _start(cmd: list[str], out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "result.json").open("w") as so, (out / "progress.log").open("w") as se:
        proc = subprocess.Popen(cmd, stdout=so, stderr=se)
    _JOBS[str(out.resolve())] = proc
    (out / "job.json").write_text(json.dumps({"pid": proc.pid, "cmd": cmd}))
    return json.dumps({"status": "running", "out_dir": str(out),
                       "poll": "job_status(out_dir) every 30-60 s"})


def job_status_impl(out_dir: str) -> str:
    out = Path(out_dir)
    log = (out / "progress.log").read_text(encoding="utf-8", errors="replace") \
        if (out / "progress.log").exists() else ""
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    res = (out / "result.json").read_text(encoding="utf-8") \
        if (out / "result.json").exists() else ""
    proc = _JOBS.get(str(out.resolve()))
    code = proc.poll() if proc is not None else None
    if proc is None and (out / "status.json").exists():
        code = json.loads((out / "status.json").read_text()).get("exit_code")
    done_json = res.strip().startswith("{") and res.strip().endswith("}")
    if code is None and proc is None and not done_json:
        return json.dumps({"status": "unknown", "error": "no job started by this server "
                           "in that folder", "progress": lines[-1] if lines else ""})
    if code is None:
        return json.dumps({"status": "running", "progress": lines[-1] if lines else ""})
    (out / "status.json").write_text(json.dumps({"exit_code": code}))
    if code == 0 and done_json:
        return json.dumps({"status": "done", "report": json.loads(res)})
    err = None
    for ln in reversed(lines):
        if ln.startswith("{") and '"error"' in ln:
            with contextlib.suppress(json.JSONDecodeError):
                err = json.loads(ln)
            break
    return json.dumps({"status": "failed", "exit_code": code,
                       "error": err or {"error": "\n".join(lines[-20:])}})


def build_server():
    try:  # mcp >= 2
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:
        try:  # mcp 1.x
            from mcp.server.fastmcp import FastMCP as Server
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise SystemExit("The MCP server needs the 'mcp' package: "
                             "pip install 'bridgesim[mcp]' (or uv sync --extra mcp)") from exc

    from bridgesim.sim import study
    from bridgesim.sim.report import SCHEMAS

    srv = Server("bridgesim", instructions=(
        "Simulates the Troitsky popsicle-stick bridge crushing test on SolidWorks parts. "
        "Call inspect_bridge first, ask the designer its `questions`, pass the answers as "
        "overrides_yaml to crush_bridge or start_crush. Check report.quality.trustworthy. "
        "After a real test, compare_with_test gives the reality gap. Every tool returns "
        "JSON; failures come back as {error, kind}. Full guide: AGENTS.md."))

    @srv.tool()
    @_safe
    def inspect_bridge(cad_path: str, overrides_yaml: str = "", material: str = "",
                       up_axis: str = "y", span_axis: str = "x", units: str = "",
                       strict: bool = False) -> str:
        """Read a SolidWorks part (.SLDPRT), Parasolid (.x_b) or STL bridge. Returns JSON:
        bodies (how each is modelled, where it is, layup, mass), glued joints, loading
        position, Troitsky 2027 rule check, `warnings` (e.g. overrides that had no effect)
        and `questions` the designer should answer (answer them via overrides_yaml).
        bridgesim expects X along the span and Y up; say otherwise with up_axis/span_axis.
        """
        rep = study.inspect(cad_path, _yaml_or_none(overrides_yaml),
                            material or "popsicle_birch", units=units or None, up=up_axis,
                            span=span_axis, strict=strict)
        return rep.model_dump_json()

    @srv.tool()
    @_safe
    def crush_bridge(cad_path: str, overrides_yaml: str = "", seed: int = 0,
                     fidelity: str = "standard", out_dir: str = "", material: str = "",
                     up_axis: str = "y", span_axis: str = "x", units: str = "") -> str:
        """Simulate the Troitsky crushing test (rulebook §12.5) on the CAD bridge
        (blocking; minutes). Returns JSON with Fu_kgf, failure_mode, failure_sequence,
        curve, quality (check quality.trustworthy) and artifacts (replay.html, events.csv
        when out_dir is given). fidelity: quick (screening), standard, fine. Use
        start_crush instead if your tool calls time out."""
        rep, _ = study.crush(cad_path, _yaml_or_none(overrides_yaml),
                             material or "popsicle_birch", seed=None if seed < 0 else seed,
                             out_dir=out_dir or None, fidelity=fidelity,
                             units=units or None, up=up_axis, span=span_axis)
        return rep.model_dump_json()

    @srv.tool()
    @_safe
    def start_crush(cad_path: str, out_dir: str, overrides_yaml: str = "", seed: int = 0,
                    fidelity: str = "standard", material: str = "", up_axis: str = "y",
                    span_axis: str = "x", units: str = "") -> str:
        """Start a crush simulation in the background and return at once. Poll
        job_status(out_dir) until status is 'done' (report included) or 'failed' (error
        included)."""
        if not Path(cad_path).exists():
            raise FileNotFoundError(f"no such file: {cad_path}")
        out = Path(out_dir)
        cmd = [sys.executable, "-m", "bridgesim.cli", "crush", cad_path, "--out", str(out),
               "--seed", str(seed), "--fidelity", fidelity, "--up-axis", up_axis,
               "--span-axis", span_axis, "--json"]
        if units:
            cmd += ["--units", units]
        if overrides_yaml:
            out.mkdir(parents=True, exist_ok=True)
            (out / "overrides.yaml").write_text(overrides_yaml, encoding="utf-8")
            cmd += ["--overrides", str(out / "overrides.yaml")]
        if material:
            cmd += ["--material", material]
        return _start(cmd, out)

    @srv.tool()
    @_safe
    def start_montecarlo(cad_path: str, out_dir: str, n: int = 8, overrides_yaml: str = "",
                         fidelity: str = "quick", material: str = "") -> str:
        """Start n crushes with different imperfections and material scatter in the
        background (spread of F_u and the value for the Load Prediction Form). Poll
        job_status(out_dir)."""
        if not Path(cad_path).exists():
            raise FileNotFoundError(f"no such file: {cad_path}")
        out = Path(out_dir)
        cmd = [sys.executable, "-m", "bridgesim.cli", "montecarlo", cad_path, "-n", str(n),
               "--fidelity", fidelity, "--out", str(out), "--json"]
        if overrides_yaml:
            out.mkdir(parents=True, exist_ok=True)
            (out / "overrides.yaml").write_text(overrides_yaml, encoding="utf-8")
            cmd += ["--overrides", str(out / "overrides.yaml")]
        if material:
            cmd += ["--material", material]
        return _start(cmd, out)

    @srv.tool()
    @_safe
    def job_status(out_dir: str) -> str:
        """Status of a background job: running (with the latest progress line), done
        (with the report), failed (with the error and exit code)."""
        return job_status_impl(out_dir)

    @srv.tool()
    @_safe
    def compare_with_test(crush_json_path: str, test_csv_path: str) -> str:
        """Reality gap between a simulated crush (crush.json) and a physical crush test
        CSV (columns load_kgf, midspan_deflection_mm, event, notes; a row with event
        'failure' and the broken part in notes). Returns per-metric errors, a 0-100 score
        and advice on what to measure to close the gap."""
        return study.compare(crush_json_path, test_csv_path).model_dump_json()

    @srv.tool()
    @_safe
    def lab_tests(material: str = "", data_dir: str = "") -> str:
        """Simulate stick 3-point bending, a glued lap joint and a stick column; with
        data_dir (folder of the team's coupon CSVs) also the reality gap per specimen."""
        return study.lab(material or "popsicle_birch", None, data_dir or None
                         ).model_dump_json()

    @srv.tool()
    @_safe
    def calibrate_material(data_dir: str, out_path: str = "my_material.yaml",
                           material: str = "") -> str:
        """Fit E, strengths and glue strength from coupon CSVs; writes a material YAML to
        pass as `material` to the other tools."""
        import yaml

        mat, log = study.calibrate(data_dir, material or "popsicle_birch")
        Path(out_path).write_text(yaml.safe_dump(mat.model_dump(mode="json"),
                                                 sort_keys=False), encoding="utf-8")
        return json.dumps({"material": out_path, "changes": log})

    @srv.tool()
    @_safe
    def get_schema(name: str) -> str:
        """JSON Schema of a report: inspect, crush, reality_gap, lab, montecarlo, question."""
        if name not in SCHEMAS:
            raise ValueError(f"unknown schema {name!r}; use one of {', '.join(SCHEMAS)}")
        return json.dumps(SCHEMAS[name].model_json_schema())

    return srv


def main() -> None:
    build_server().run()
