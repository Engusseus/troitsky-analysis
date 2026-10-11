"""Tension-only members: slender braces that bow in compression instead of failing."""

import csv
import math

import numpy as np
import pytest
from typer.testing import CliRunner

from bridgesim.analysis import AnalysisError, analyze
from bridgesim.cli import app
from bridgesim.generators import generate_warren
from bridgesim.report import model_assumptions, to_markdown
from bridgesim.schema import Bridge
from bridgesim.stability import braces_working_terms, crossing_pairs
from bridgesim.viz import bridge_figure, member_chain_diagrams

STRAP = {"id": "top_brace", "b_mm": 19.0, "d_mm": 4.1}  # 2 jumbo sticks flat


def _warren(strap: bool = True, flag: frozenset = frozenset(), drop: frozenset = frozenset()):
    data = generate_warren().model_dump(mode="json")
    if strap:
        data["sections"] = [STRAP if s["id"] == "top_brace" else s for s in data["sections"]]
    data["members"] = [{**m, "tension_only": m["group"] in flag} for m in data["members"]
                       if m["group"] not in drop]
    return Bridge.model_validate(data)


@pytest.fixture(scope="module")
def straps():
    return analyze(_warren())


@pytest.fixture(scope="module")
def slack_straps():
    return analyze(_warren(flag=frozenset({"top_brace"})))


def test_thin_straps_govern_by_buckling_and_the_results_say_why(straps) -> None:
    """Strap 19 x 4.1 mm over L = sqrt(143.75^2 + 182^2) = 231.9 mm:
    I_min = 19 x 4.1^3 / 12 = 109.1 mm^4, P_cr = pi^2 x 10000 x 109.1 / 231.9^2 = 200.2 N.
    The chords shortening squeeze it, so it fails long before anything that carries load."""
    gov = straps.member(straps.governing_member)
    assert gov.group == "top_brace" and gov.mode == "buckling"
    assert gov.cap.P_cr == pytest.approx(math.pi**2 * 1e4 * 19 * 4.1**3 / 12 / 231.92**2,
                                         rel=1e-3)
    assert any(f"{gov.id} (top_brace) buckles" in w and "tension_only: true" in w
               for w in straps.warnings)


def test_slack_straps_give_the_same_answer_as_no_straps(slack_straps, straps) -> None:
    """Every top strap is shortened by the chords, so with tension_only all of them drop out
    and the model is exactly the bridge without top bracing: bracing can no longer lower
    the predicted load."""
    none = analyze(_warren(drop=frozenset({"top_brace"})))
    tb = [m for m in slack_straps.members if m.group == "top_brace"]
    assert sorted(slack_straps.slack_members) == sorted(m.id for m in tb)
    assert all(m.U == 0 and m.N_N == 0 and m.mode == "slack" for m in tb)
    assert slack_straps.Fu_pred_N == pytest.approx(none.Fu_pred_N, rel=1e-9)
    assert slack_straps.Fu_buckling_N == pytest.approx(none.Fu_buckling_N, rel=1e-9)
    assert slack_straps.Fu_pred_N > straps.Fu_pred_N
    assert slack_straps.governing_member not in slack_straps.slack_members
    assert not any("tension_only: true" in w for w in slack_straps.warnings)
    assert slack_straps.buckling.member_energy[tb[0].id] == 0.0
    # Slack members still weigh what they weigh.
    assert slack_straps.mass.total_kg == pytest.approx(analyze(_warren()).mass.total_kg)


def test_tension_only_members_in_tension_are_checked_as_usual(default_result) -> None:
    """The bottom chords stretch, so the bottom X-braces are in tension and stay in."""
    r = analyze(_warren(strap=False, flag=frozenset({"bottom_brace"})))
    assert r.slack_members == []
    base = {m.id: m for m in default_result.members}
    for m in r.members:
        assert m.N_N == pytest.approx(base[m.id].N_N, rel=1e-6, abs=1e-9)
        assert m.U == pytest.approx(base[m.id].U, rel=1e-6, abs=1e-12)
    assert r.Fu_pred_N == pytest.approx(default_result.Fu_pred_N, rel=1e-9)


def test_the_default_design_gets_no_brace_hint(default_result) -> None:
    assert not any("tension_only" in w for w in default_result.warnings)


def test_a_node_joined_only_by_slack_members_is_explained() -> None:
    """Split the first top X at its crossing: when all four halves go slack the crossing node
    has no stiffness left, and the error says how to fix the file."""
    data = _warren(strap=False).model_dump(mode="json")
    nodes = {n["id"]: n for n in data["nodes"]}
    a, b = nodes["T0n"], nodes["T1f"]
    data["nodes"].append({"id": "X0", **{k: (a[k] + b[k]) / 2 for k in ("x_mm", "y_mm", "z_mm")}})
    data["members"] = [m for m in data["members"] if m["id"] not in ("tb0a", "tb0b")]
    for mid, i in (("h1", "T0n"), ("h2", "T1f"), ("h3", "T0f"), ("h4", "T1n")):
        data["members"].append({"id": mid, "i": i, "j": "X0", "section": "top_brace",
                                "group": "top_brace", "tension_only": True})
    with pytest.raises(AnalysisError, match="split X-brace"):
        analyze(Bridge.model_validate(data))


def test_flag_round_trips_and_is_only_written_when_set() -> None:
    plain = generate_warren().to_yaml()
    assert "tension_only" not in plain
    flagged = _warren(flag=frozenset({"top_brace"}))
    text = flagged.to_yaml()
    assert "tension_only: true" in text
    assert Bridge.from_yaml_str(text) == flagged


def test_report_and_view_show_slack_members(slack_straps) -> None:
    text = to_markdown(slack_straps)
    assert "12 tension-only member\\(s\\) would be in compression" in text
    assert any(a.startswith("Tension-only members (tb0a") for a in model_assumptions(slack_straps))
    fig = bridge_figure(slack_straps.bridge, slack_straps)
    slack_trace = next(t for t in fig.data if t.name == "slack")
    assert slack_trace.line.dash == "dash"
    assert sum(x is None for x in slack_trace.x) == len(slack_straps.slack_members)


def test_diagrams_show_no_force_in_a_slack_member() -> None:
    """Pynite still works out forces for a member it has switched off, from its end
    movements; the poster diagrams must show the zero the analysis uses."""
    data = generate_warren().model_dump(mode="json")
    data["members"].append({"id": "cb", "i": "B0n", "j": "T1n", "section": "diagonal",
                            "tension_only": True})  # a counter-diagonal, squeezed
    r = analyze(Bridge.model_validate(data))
    assert r.slack_members == ["cb"]
    _, N, V, M, _ = member_chain_diagrams(r, ["cb"])
    assert not N.any() and not V.any() and not M.any()
    _, N, _, _, _ = member_chain_diagrams(r, ["d0un"])  # its neighbour still carries load
    assert abs(N).max() > 100


def test_crossing_slack_straps_are_found_in_pairs(slack_straps, material) -> None:
    """The default Warren has an X in every top panel but the one at mid-span: six pairs
    tb{i}a / tb{i}b. Members that only meet at a node (a strap and a chord) are no pair."""
    pairs = crossing_pairs(slack_straps.bridge, slack_straps.slack_members, material.stick)
    assert pairs == [(f"tb{i}a", f"tb{i}b") for i in (0, 1, 2, 4, 5, 6)]
    assert crossing_pairs(slack_straps.bridge, ["tb0a", "tc0n", "d0un"], material.stick) == []


def test_half_stiffness_pair_equals_one_strap_working_in_racking(slack_straps, material) -> None:
    """Rack the first X: T1n and T1f move Delta = 1 mm across. tb0a (T0n -> T1f) stretches
    and tb0b (T0f -> T1n) shortens, each by Delta s / L (s = 182 mm across, L = 231.9 mm).
    One strap working stores 1/2 (E A / L) (Delta s / L)^2; both at half stiffness store
    the same, with E = 10000 MPa and A = 19 x 4.1 = 77.9 mm^2."""
    terms, ids = braces_working_terms(slack_straps.fe_model, slack_straps.bridge,
                                      slack_straps.slack_members, material.stick)
    assert ids == sorted(slack_straps.slack_members, key=lambda m: (m[2], m[3]))
    move = {"T1n": np.array([0, 0, 1.0]), "T1f": np.array([0, 0, 1.0])}
    energy = 0.0
    for nodes, k in terms[:2]:  # tb0a, tb0b
        u = np.concatenate([move.get(n, np.zeros(3)) for n in nodes])
        energy += 0.5 * u @ k @ u
    L, s = math.hypot(143.75, 182.0), 182.0
    assert energy == pytest.approx(0.5 * 1e4 * 19 * 4.1 / L * (s / L) ** 2)


def test_buckling_is_bracketed_when_slack_straps_cross(slack_straps) -> None:
    """Lower end: straps out, exactly the bridge without top bracing; F_u,p uses it. Upper
    end: one strap of each X working, higher."""
    none = analyze(_warren(drop=frozenset({"top_brace"})))
    upper = slack_straps.Fu_buckling_braces_working_N
    assert slack_straps.Fu_buckling_N == pytest.approx(none.Fu_buckling_N, rel=1e-9)
    assert upper > slack_straps.Fu_buckling_N
    assert slack_straps.Fu_pred_N == pytest.approx(none.Fu_pred_N, rel=1e-9)
    assert len(slack_straps.braces_working_members) == 12
    note = slack_straps.slack_note
    assert "is a lower bound" in note and "The real value lies in between" in note
    rows = slack_straps.buckling_load_table(braces_working=True)
    assert rows[0]["case"] == "X braces working"
    assert rows[0]["F_cr_N"] == pytest.approx(upper, abs=0.05)
    assert slack_straps.buckling_load_table()[0]["case"] == "as analysed"


def test_no_bracket_without_a_crossing_slack_pair() -> None:
    """A lone slack counter-diagonal would only work in one direction of sway: no upper end
    is claimed for it."""
    data = generate_warren().model_dump(mode="json")
    data["members"].append({"id": "cb", "i": "B0n", "j": "T1n", "section": "diagonal",
                            "tension_only": True})
    r = analyze(Bridge.model_validate(data))
    assert r.slack_members == ["cb"]
    assert r.buckling_braces_working is None and r.Fu_buckling_braces_working_N is None
    assert r.buckling_load_table(braces_working=True) == []


def test_cli_prints_and_exports_the_bracket(tmp_path, slack_straps) -> None:
    path = tmp_path / "straps.yaml"
    slack_straps.bridge.save(path)
    res = CliRunner().invoke(app, ["run", str(path), "--out", str(tmp_path / "out")])
    assert res.exit_code == 0, res.output
    assert "Global buckling, X working    :" in res.output
    with (tmp_path / "out" / "buckling_loads.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["case"] for r in rows] == ["as analysed"] * 3 + ["X braces working"] * 3
    assert "F_cr with one strap of each slack X working" in (tmp_path / "out" / "report.html"
                                                             ).read_text(encoding="utf-8")


def test_a_failed_second_solve_is_warned_about_not_shown_as_a_bracket(monkeypatch) -> None:
    """If only the solve with the X braces working fails, the lower end stands, a warning
    says why there is no upper end, and no output claims one."""
    import bridgesim.analysis as analysis_mod
    from bridgesim.stability import BucklingResult, global_buckling

    def fail_second(model, bridge, **kw):
        if len(kw.get("extra", ())) > 0:
            return BucklingResult(math.nan, note="solver failed")
        return global_buckling(model, bridge, **kw)

    monkeypatch.setattr(analysis_mod, "global_buckling", fail_second)
    r = analysis_mod.analyze(_warren(flag=frozenset({"top_brace"})))
    assert math.isfinite(r.Fu_buckling_N) and math.isnan(r.Fu_buckling_braces_working_N)
    assert not r.has_buckling_bracket
    assert any(w.startswith("Global buckling with one strap of each slack X working: solver "
                            "failed") for w in r.warnings)
    assert "The real value lies in between" not in r.slack_note
    text = to_markdown(r)
    assert "one strap of each slack X working \\(upper end" not in text
    assert "X braces working |" not in text
