"""Tension-only members: slender braces that bow in compression instead of failing."""

import math

import pytest

from bridgesim.analysis import AnalysisError, analyze
from bridgesim.generators import generate_warren
from bridgesim.report import model_assumptions, to_markdown
from bridgesim.schema import Bridge
from bridgesim.viz import bridge_figure

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
