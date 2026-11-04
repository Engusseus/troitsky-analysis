"""bridgesim as an MCP server: AI agents call the simulation as tools.

Start it with ``bridgesim mcp`` (stdio). Register it in an MCP client, e.g. Claude Code::

    claude mcp add bridgesim -- uv run --directory /path/to/troitsky-analysis bridgesim mcp

Every tool returns one JSON document. On failure the result is flagged as an error
(``isError``) and its text is ``Error executing tool <name>: `` followed by
``{"error": "...", "kind": "bad_input" | "simulation_failed", "type": "..."}``, so the
agent always sees the reason. Paths are made absolute against the server's working
directory (wherever the client started it): give absolute paths; a missing relative one
says where it was looked for.

Tools:

* ``inspect_bridge`` - what bridgesim understood from a CAD file, the rule check and the
  questions to ask the designer (seconds);
* ``crush_bridge`` - the simulated §12.5 crushing test (minutes, blocking);
* ``start_crush`` / ``start_montecarlo`` / ``start_compare_designs`` /
  ``start_sensitivity`` + ``job_status`` - in the background (jobs survive a restart of
  this server), for clients whose tool calls time out;
* ``compare_with_test`` - reality gap between a simulation and a physical crush test;
* ``lab_tests`` - simulated coupon tests, and their gap to the team's coupon data;
* ``calibrate_material`` - fit the material from the team's coupon data;
* ``get_schema`` - JSON Schema of any report.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import time
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


def _infile(p: str, what: str = "file") -> str:
    """An input path, made absolute; a missing one says where a relative path was looked
    for (the server's folder, not the agent's)."""
    q = Path(p).expanduser()
    q = q if q.is_absolute() else Path.cwd() / q
    if not q.exists():
        hint = "" if Path(p).expanduser().is_absolute() else (
            f" (a relative path is taken from the server's folder {Path.cwd()}: "
            "pass an absolute path)")
        raise FileNotFoundError(f"no such {what}: {p}{hint}")
    return str(q.resolve())


def _outpath(p: str) -> Path:
    q = Path(p).expanduser()
    return (q if q.is_absolute() else Path.cwd() / q).resolve()


def _tool_error_cls():
    """The SDK's ToolError: raised by a tool, the client gets a result flagged as an
    error (isError) with our message, the JSON object {"error", "kind", "type"}."""
    try:  # mcp >= 2
        from mcp.server.mcpserver.exceptions import ToolError
    except ImportError:
        try:  # mcp 1.x
            from mcp.server.fastmcp.exceptions import ToolError
        except ImportError:  # pragma: no cover - no mcp installed
            ToolError = RuntimeError  # noqa: N806
    return ToolError


def _safe(fn):
    """Wrap a tool so any exception comes back as a flagged MCP error (isError) whose
    text is a JSON object the agent can read."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            out = fn(*a, **k)
        except Exception as exc:  # noqa: BLE001
            raise _tool_error_cls()(_error(exc)) from None
        # a JSON object, so clients get structured content (and the same JSON as text)
        return json.loads(out) if isinstance(out, str) else out

    import inspect
    from typing import Any

    # dict[str, Any] (not a bare dict) so the SDK publishes it as structured output
    wrapper.__annotations__ = {**fn.__annotations__, "return": dict[str, Any]}
    wrapper.__signature__ = inspect.signature(fn).replace(return_annotation=dict[str, Any])
    return wrapper


def _start(cmd: list[str], out: Path) -> str:
    """Start ``cmd`` detached, through bridgesim.sim.jobrun, which records the exit code
    in out/status.json: the job's state survives a restart of this server."""
    out.mkdir(parents=True, exist_ok=True)
    for f in ("status.json", "result.json", "progress.log"):
        (out / f).unlink(missing_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "bridgesim.sim.jobrun", str(out), "--",
                             *cmd], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    _JOBS[str(out.resolve())] = proc
    (out / "job.json").write_text(json.dumps({"pid": proc.pid, "cmd": cmd,
                                              "started": time.time()}))
    return json.dumps({"status": "running", "out_dir": str(out),
                       "poll": "job_status(out_dir) every 30-60 s"})


def _alive(pid: int) -> bool:
    """Whether a process with this pid is running (and not a zombie)."""
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    try:  # a finished child of this server that was not reaped yet
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _progress_of(lines: list[str]) -> tuple[float | None, str]:
    """Last progress line '  42% jack 3.1 mm, ...' -> (0.42, 'jack 3.1 mm, ...')."""
    for ln in reversed(lines):
        mt = re.match(r"\s*(\d+)%\s+(.*)", ln)
        if mt:
            return int(mt.group(1)) / 100, mt.group(2)
    return None, lines[-1] if lines else ""


def job_status_impl(out_dir: str) -> str:
    """{"status": running | done | failed | unknown, "progress": 0..1 or null,
    "message": last progress message, "report": the report when done, "error": the error
    object when failed}. Read from the job's files, so it works from any server."""
    out = _outpath(out_dir)
    log = (out / "progress.log").read_text(encoding="utf-8", errors="replace") \
        if (out / "progress.log").exists() else ""
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    frac, msg = _progress_of(lines)

    def reply(status, **kw):
        return json.dumps({"status": status, "progress": frac, "message": msg,
                           "report": None, "error": None} | kw)

    proc = _JOBS.get(str(out.resolve()))
    if proc is not None:
        proc.poll()  # reap it if it has finished
    status = json.loads((out / "status.json").read_text()) \
        if (out / "status.json").exists() else None
    if status is None:
        job = json.loads((out / "job.json").read_text()) if (out / "job.json").exists() \
            else None
        if job is None:
            return reply("unknown", error={"error": "no job was started in that folder",
                                           "kind": "bad_input"})
        if _alive(int(job["pid"])):
            return reply("running")
        return reply("failed", error={"error": "the job's process died without finishing "
                                               "(killed, or the machine restarted)",
                                      "kind": "simulation_failed"})
    code = status.get("exit_code")
    res = (out / "result.json").read_text(encoding="utf-8") \
        if (out / "result.json").exists() else ""
    if code == 0 and res.strip().startswith("{"):
        return reply("done", progress=1.0, report=json.loads(res))
    err = None
    for ln in reversed(lines):
        if ln.startswith("{") and '"error"' in ln:
            with contextlib.suppress(json.JSONDecodeError):
                err = json.loads(ln)
            break
    return reply("failed", exit_code=code,
                 error=err or {"error": "\n".join(lines[-20:]), "kind": "simulation_failed"})


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
                       strict: bool = False, brief: bool = True) -> str:
        """Read a SolidWorks part (.SLDPRT), Parasolid (.x_b) or STL bridge (use absolute
        paths). Returns JSON: Troitsky 2027 rule check, `warnings` (e.g. overrides that had
        no effect), `questions` the designer should answer (answer them via overrides_yaml,
        or list confirmed ones under `acknowledge:`), the loading position and counts of
        the bodies by kind. brief=false adds every body (how it is modelled, where it is,
        layup, mass) and glued joint (~100 kB). bridgesim expects X along the span and Y
        up; say otherwise with up_axis/span_axis."""
        rep = study.inspect(_infile(cad_path, "CAD file"), _yaml_or_none(overrides_yaml),
                            material or "popsicle_birch", units=units or None, up=up_axis,
                            span=span_axis, strict=strict)
        return json.dumps(study.inspect_brief(rep)) if brief else rep.model_dump_json()

    @srv.tool()
    @_safe
    def crush_bridge(cad_path: str, overrides_yaml: str = "", seed: int = 0,
                     fidelity: str = "standard", out_dir: str = "", material: str = "",
                     up_axis: str = "y", span_axis: str = "x", units: str = "",
                     strict: bool = False, rate_check: bool = False,
                     brief: bool = True) -> str:
        """Simulate the Troitsky crushing test (rulebook §12.5) on the CAD bridge
        (blocking; minutes). Returns JSON with Fu_kgf, failure_mode, failure_sequence,
        curve, quality (check quality.trustworthy) and artifacts (replay.html, events.csv
        when out_dir is given). fidelity: quick (screening), standard, fine. strict: make
        overrides that have no effect an error. rate_check: crush again at half speed and
        report the change of F_u (quality.rate_check; 3x the time). brief (default):
        without the curve and the event log, which are in out_dir's curve.csv and
        events.csv (brief=false returns them inline, ~200 kB). Use start_crush instead
        if your tool calls time out."""
        rep, _ = study.crush(_infile(cad_path, "CAD file"), _yaml_or_none(overrides_yaml),
                             material or "popsicle_birch", seed=None if seed < 0 else seed,
                             out_dir=str(_outpath(out_dir)) if out_dir else None,
                             fidelity=fidelity,
                             units=units or None, up=up_axis, span=span_axis,
                             strict=strict, rate_check=rate_check)
        return json.dumps(study.brief(rep)) if brief else rep.model_dump_json()

    @srv.tool()
    @_safe
    def start_crush(cad_path: str, out_dir: str, overrides_yaml: str = "", seed: int = 0,
                    fidelity: str = "standard", material: str = "", up_axis: str = "y",
                    span_axis: str = "x", units: str = "", strict: bool = False,
                    rate_check: bool = False, brief: bool = True) -> str:
        """Start a crush simulation in the background and return at once. Poll
        job_status(out_dir) until status is 'done' (report included) or 'failed' (error
        included)."""
        cad_path = _infile(cad_path, "CAD file")
        out = _outpath(out_dir)
        cmd = [sys.executable, "-m", "bridgesim.cli", "crush", cad_path, "--out", str(out),
               "--seed", str(seed), "--fidelity", fidelity, "--up-axis", up_axis,
               "--span-axis", span_axis, "--json"]
        if units:
            cmd += ["--units", units]
        if strict:
            cmd += ["--strict"]
        if rate_check:
            cmd += ["--rate-check"]
        if brief:
            cmd += ["--brief"]
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
        cad_path = _infile(cad_path, "CAD file")
        out = _outpath(out_dir)
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
    def start_compare_designs(cad_a: str, cad_b: str, out_dir: str, n: int = 3,
                              overrides_a_yaml: str = "", overrides_b_yaml: str = "",
                              fidelity: str = "standard", material: str = "") -> str:
        """Is design B really better than design A? Starts, in the background, n crushes
        of each with the same seeds; the report (job_status(out_dir)) gives the change of
        F_u and efficiency with a 90 % confidence interval, `significant`, a `verdict`
        and what fails first in each. cad_b may be cad_a with other overrides."""
        cad_a, cad_b = _infile(cad_a, "CAD file"), _infile(cad_b, "CAD file")
        out = _outpath(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "bridgesim.cli", "ab", cad_a, cad_b, "-n", str(n),
               "--fidelity", fidelity, "--out", str(out), "--json"]
        for tag, y in (("a", overrides_a_yaml), ("b", overrides_b_yaml)):
            if y:
                (out / f"overrides_{tag}.yaml").write_text(y, encoding="utf-8")
                cmd += [f"--overrides-{tag}", str(out / f"overrides_{tag}.yaml")]
        if material:
            cmd += ["--material", material]
        return _start(cmd, out)

    @srv.tool()
    @_safe
    def start_sensitivity(cad_path: str, out_dir: str, overrides_yaml: str = "",
                          rel: float = 0.2, fidelity: str = "quick", material: str = "") -> str:
        """Which material values move F_u most? Starts, in the background, crushes with
        each placeholder value lowered and raised by rel; the report (job_status) ranks
        them and lists what to measure first."""
        cad_path = _infile(cad_path, "CAD file")
        out = _outpath(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "bridgesim.cli", "sensitivity", cad_path, "--rel",
               str(rel), "--fidelity", fidelity, "--json"]
        if overrides_yaml:
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
        return study.compare(_infile(crush_json_path, "crush report"),
                             _infile(test_csv_path, "test record")).model_dump_json()

    @srv.tool()
    @_safe
    def lab_tests(material: str = "", data_dir: str = "") -> str:
        """Simulate stick 3-point bending, a glued lap joint and a stick column; with
        data_dir (folder of the team's coupon CSVs) also the reality gap per specimen."""
        return study.lab(material or "popsicle_birch", None,
                         _infile(data_dir, "data folder") if data_dir else None
                         ).model_dump_json()

    @srv.tool()
    @_safe
    def calibrate_material(data_dir: str, out_path: str = "my_material.yaml",
                           material: str = "") -> str:
        """Fit E, the strengths and the glue strength from coupon CSVs so the simulated
        coupons reproduce them; writes a material YAML to pass as `material` to the other
        tools, and the specimens' scatter (scatter.* params) to put in overrides_yaml.
        Use absolute paths."""
        import yaml

        mat, log, params = study.calibrate(_infile(data_dir, "data folder"),
                                           material or "popsicle_birch")
        out_path = str(_outpath(out_path))
        Path(out_path).write_text(yaml.safe_dump(mat.model_dump(mode="json"),
                                                 sort_keys=False), encoding="utf-8")
        return json.dumps({"material": out_path, "changes": log,
                           "overrides_yaml": yaml.safe_dump({"params": params})
                           if params else ""})

    @srv.tool()
    @_safe
    def get_schema(name: str) -> str:
        """JSON Schema of a report: inspect, crush, reality_gap, lab, montecarlo,
        design_comparison, sensitivity or question."""
        if name not in SCHEMAS:
            raise ValueError(f"unknown schema {name!r}; use one of {', '.join(SCHEMAS)}")
        return json.dumps(SCHEMAS[name].model_json_schema())

    return srv


def main() -> None:
    build_server().run()
