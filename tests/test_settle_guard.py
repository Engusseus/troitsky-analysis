"""A crush whose integration throws a part while the bridge settles under its own weight
restarts with a smaller time step: it is not read as a bridge that collapses under its
own weight (F_u 0). Found on Bridge_v12 at standard fidelity: a node of the centre gusset
plate left at 24 m/s, its part flew on at ~390 m/s, below the 1 km/s the step's own check
looks for, and the run came back as `instability`, F_u 0, trustworthy."""
from __future__ import annotations

from pathlib import Path

import pytest

from bridgesim.cad.synthetic import simple_beam_bridge
from bridgesim.schema import Material
from bridgesim.sim import crush
from bridgesim.sim import solver as S
from bridgesim.sim.idealize import build_model
from bridgesim.sim.materials import SimParams

pytest.importorskip("numba")

ROOT = Path(__file__).resolve().parents[1]
MAT = Material.from_yaml(ROOT / "materials" / "popsicle_birch.yaml")


def test_a_part_thrown_while_settling_restarts_the_run(monkeypatch):
    m = build_model(simple_beam_bridge(), MAT, SimParams.load(None), seed=0, mesh_scale=1.6)
    real = S.fit_ramp
    calls = {"n": 0, "kicked": 0}

    def kicking(model, st, g):
        real(model, st, g)
        calls["n"] += 1
        # once, in the step before the settle loop looks at the speeds: 50 m/s, 15x what
        # anything reaches falling from this bridge's top, 20x below the step's own check
        if calls["n"] == 200 and not calls["kicked"]:
            calls["kicked"] = 1
            st.v[0, 1] = 5e4

    monkeypatch.setattr(S, "fit_ramp", kicking)
    res = crush.run_crush(m, crush.CrushOptions(speed_mm_s=300, max_travel_mm=0.2,
                                                auto_speed=False))
    assert calls["kicked"] == 1
    assert len(res.notes) == 1
    assert "under self-weight" in res.notes[0] and "smaller step" in res.notes[0]
    # the second attempt, not kicked, settles as the bridge does (test_sim: 4-8 mm)
    assert res.criteria["instability"] is None
    assert 4.0 < res.settle_sag_mm < 8.0
