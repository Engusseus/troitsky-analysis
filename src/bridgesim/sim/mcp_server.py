"""bridgesim as an MCP server: AI agents call the simulation as tools.

Start it with ``bridgesim mcp`` (stdio). Register it in an MCP client, e.g. Claude Code::

    claude mcp add bridgesim -- uv run --directory /path/to/troitsky-analysis bridgesim mcp

Tools (all return JSON documents whose schemas ``get_schema`` gives):

* ``inspect_bridge`` - what bridgesim understood from a CAD file, the rule check and the
  questions to ask the designer (seconds);
* ``crush_bridge`` - the simulated §12.5 crushing test (minutes), with a 3D replay file;
* ``start_crush`` / ``crush_status`` - the same in the background, for clients whose tool
  calls time out;
* ``compare_with_test`` - reality gap between a simulation and a physical crush test;
* ``lab_tests`` - simulated coupon tests, and their gap to the team's coupon data;
* ``calibrate_material`` - fit the material from the team's coupon data;
* ``get_schema`` - JSON Schema of any report.
"""

from __future__ import annotations

import json
from pathlib import Path


def _yaml_or_none(text: str | None):
    if not text:
        return None
    import yaml

    return yaml.safe_load(text)


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
        "overrides_yaml to crush_bridge. After a real test, compare_with_test gives the "
        "reality gap. Full guide: AGENTS.md in the repository."))

    @srv.tool()
    def inspect_bridge(cad_path: str, overrides_yaml: str = "", material: str = "") -> str:
        """Read a SolidWorks part (.SLDPRT), Parasolid (.x_b) or STL bridge. Returns JSON:
        bodies (how each is modelled: bar/shell/block/cable, layup, mass), glued joints
        with areas, loading position, Troitsky 2027 rule check, and `questions` the
        designer should answer (answer them by passing overrides_yaml)."""
        rep = study.inspect(cad_path, _yaml_or_none(overrides_yaml),
                            material or "popsicle_birch")
        return rep.model_dump_json()

    @srv.tool()
    def crush_bridge(cad_path: str, overrides_yaml: str = "", seed: int = 0,
                     fidelity: str = "standard", out_dir: str = "",
                     material: str = "") -> str:
        """Simulate the Troitsky crushing test (rulebook §12.5) on the CAD bridge:
        nonlinear explicit dynamics with breaking glue joints, cracking and buckling
        members, floss, contact with the platform and the loading plate. Takes minutes.
        Returns JSON with Fu_kgf, failure_mode, failure_sequence, curve, quality and
        artifacts (replay.html path when out_dir is given). Use start_crush for long runs.
        fidelity: quick (screening), standard, fine."""
        rep, _ = study.crush(cad_path, _yaml_or_none(overrides_yaml),
                             material or "popsicle_birch", seed=seed,
                             out_dir=out_dir or None, fidelity=fidelity)
        return rep.model_dump_json()

    @srv.tool()
    def start_crush(cad_path: str, out_dir: str, overrides_yaml: str = "", seed: int = 0,
                    fidelity: str = "standard", material: str = "") -> str:
        """Start a crush simulation in the background (it takes minutes) and return at
        once. Poll crush_status(out_dir) until status is 'done'. fidelity: quick
        (screening a design change), standard, fine."""
        import subprocess
        import sys

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "bridgesim.cli", "crush", cad_path, "--out", str(out),
               "--seed", str(seed), "--fidelity", fidelity, "--json"]
        if overrides_yaml:
            (out / "overrides.yaml").write_text(overrides_yaml, encoding="utf-8")
            cmd += ["--overrides", str(out / "overrides.yaml")]
        if material:
            cmd += ["--material", material]
        with (out / "result.json").open("w") as so, (out / "progress.log").open("w") as se:
            proc = subprocess.Popen(cmd, stdout=so, stderr=se)
        (out / "pid").write_text(str(proc.pid))
        return json.dumps({"status": "running", "out_dir": str(out),
                           "poll": "crush_status(out_dir)"})

    @srv.tool()
    def crush_status(out_dir: str) -> str:
        """Status of a crush started with start_crush: running (with the latest progress
        line), done (with the full crush report), or failed (with the error)."""
        import os

        out = Path(out_dir)
        log = (out / "progress.log").read_text(encoding="utf-8", errors="replace") \
            if (out / "progress.log").exists() else ""
        last = [ln for ln in log.splitlines() if ln.strip()][-1:] or [""]
        res = (out / "result.json").read_text(encoding="utf-8") \
            if (out / "result.json").exists() else ""
        if res.strip().startswith("{") and res.strip().endswith("}"):
            return json.dumps({"status": "done", "report": json.loads(res)})
        pid = int((out / "pid").read_text()) if (out / "pid").exists() else None
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                alive = False
        if alive:
            return json.dumps({"status": "running", "progress": last[0].strip()})
        return json.dumps({"status": "failed", "error": log[-2000:]})

    @srv.tool()
    def compare_with_test(crush_json_path: str, test_csv_path: str) -> str:
        """Reality gap between a simulated crush (crush.json) and a physical crush test
        CSV (columns load_kgf, midspan_deflection_mm, event, notes). Returns per-metric
        errors, a 0-100 score and advice on what to measure to close the gap."""
        return study.compare(crush_json_path, test_csv_path).model_dump_json()

    @srv.tool()
    def lab_tests(material: str = "", data_dir: str = "") -> str:
        """Simulate stick 3-point bending, a glued lap joint and a stick column; with
        data_dir (folder of the team's coupon CSVs) also the reality gap per specimen."""
        return study.lab(material or "popsicle_birch", None, data_dir or None
                         ).model_dump_json()

    @srv.tool()
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
    def get_schema(name: str) -> str:
        """JSON Schema of a report: inspect, crush, reality_gap, lab, montecarlo, question."""
        return json.dumps(SCHEMAS[name].model_json_schema())

    return srv


def main() -> None:
    build_server().run()
