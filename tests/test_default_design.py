"""The default design (``examples/warren_2027.yaml`` == ``generate_warren()``) end to end."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bridgesim.analysis import AnalysisResult, analyze
from bridgesim.generators.warren import WarrenParams, generate_warren
from bridgesim.model import build_model
from bridgesim.rules import evaluate
from bridgesim.schema import Bridge, Material

N_PANELS = WarrenParams().n_panels


def test_example_matches_generator(example_bridge: Bridge, default_bridge: Bridge) -> None:
    """The committed example is the default generator output, renamed."""
    assert example_bridge.name != default_bridge.name
    assert example_bridge.model_copy(update={"name": default_bridge.name}) == default_bridge


def test_generator_topology(default_bridge: Bridge, default_params: WarrenParams) -> None:
    """Counts for n = 8 panels with all bracing:

        nodes: 2 (n+1) bottom + 2 n top + (n+1) centre + 4 pier bases = 18+16+9+4 = 47
        members per truss: 3n (chords + 2 diagonals) + (n-1) top chords + (n+1) floor-beam
        halves + 2 piers = 24 + 7 + 9 + 2 = 42, x2 = 84; struts n = 8; top bracing
        2 (n-2) = 12 (mid panel left open); bottom bracing 2n = 16; pier bracing 4.
        total = 84 + 8 + 12 + 16 + 4 = 124
    """
    n = default_params.n_panels
    assert len(default_bridge.nodes) == 2 * (n + 1) + 2 * n + (n + 1) + 4
    per_truss = 3 * n + (n - 1) + (n + 1) + 2
    assert len(default_bridge.members) == 2 * per_truss + n + 2 * (n - 2) + 2 * n + 4
    assert default_bridge.supports.pinned == ["P0n", "P0f"]
    assert default_bridge.supports.roller == [f"P{n}n", f"P{n}f"]


@pytest.mark.parametrize("which", ["example", "default"])
def test_passes_all_computable_rules(request: pytest.FixtureRequest, material: Material,
                                     which: str) -> None:
    """Every computable 2027 rule passes: no penalty, no bans, no disqualification."""
    bridge = request.getfixturevalue(f"{which}_bridge")
    rep = evaluate(bridge, material)
    failing = [(r.key, r.measured_text) for r in rep.checked if not r.passed]
    assert rep.all_passed, failing
    assert rep.total_penalty == 0
    assert rep.bans == []
    assert rep.disqualification_risks == []
    assert len(rep.checked) == 14


def test_example_analyses_like_generated(example_bridge: Bridge, material: Material,
                                         default_result: AnalysisResult) -> None:
    """Same geometry -> same predicted ultimate load."""
    r = analyze(example_bridge, material)
    assert r.Fu_pred_N == pytest.approx(default_result.Fu_pred_N, rel=1e-12)


def test_analysis_is_stable_and_finite(default_bridge: Bridge, material: Material) -> None:
    """Pynite's stability check (nodal and global residual) passes and every nodal
    displacement is finite.
    """
    model = build_model(default_bridge, material)
    model.analyze_linear(check_stability=True)
    r = analyze(default_bridge, material)
    disp = np.array(list(r.displacements_mm.values()))
    assert np.isfinite(disp).all()


def fu_limits(r: AnalysisResult) -> dict[str, float]:
    """Every limit-state load the result reports (Fu_strength_N, Fu_deflection_N and any
    other ``Fu_*_N`` field, e.g. a global-buckling load), excluding Fu_pred_N itself.
    """
    return {k: float(v) for k, v in vars(r).items()
            if k.startswith("Fu_") and k.endswith("_N") and k != "Fu_pred_N"
            and isinstance(v, int | float)}


#: Member-level check modes (checks.utilisation); anything else is a global limit state.
MEMBER_MODES = {"tension", "crushing", "buckling", "bending_z", "bending_y", "shear", "joint"}


def test_predicted_load_and_mass(default_result: AnalysisResult) -> None:
    """F_u,p is finite and positive and equals the smallest reported limit-state load
    (F_u,p = min(F_u,strength, F_u,delta, ...)), the bridge deflects downward, and the
    estimated mass is under the 6.00 kg free limit.
    """
    r = default_result
    limits = fu_limits(r)
    assert {"Fu_strength_N", "Fu_deflection_N"} <= set(limits)
    assert math.isfinite(r.Fu_pred_N) and r.Fu_pred_N > 0
    assert r.Fu_pred_N == min(limits.values())
    assert r.Fu_strength_N > 0 and r.Fu_deflection_N > 0
    assert r.delta_ref_mm > 0
    assert 0 < r.mass.total_kg < 6.0
    assert r.efficiency == pytest.approx(r.Fu_pred_kgf / r.mass.total_kg)


def test_no_support_warnings(default_result: AnalysisResult) -> None:
    """No uplift, no friction-exceeding thrust, a valid load path. The only warning is that
    F_u,p is close enough to global buckling for P-delta to matter (an honest caveat)."""
    assert [w for w in default_result.warnings if "Support" in w] == []
    assert len(default_result.warnings) == 1
    assert "of the global buckling load" in default_result.warnings[0]


def test_vertical_equilibrium(default_result: AnalysisResult) -> None:
    """Sum of vertical support reactions = applied load = P_ref = 1000 N; horizontal
    reactions balance (sum FX = sum FZ = 0). Nodal loads also sum to P_ref.
    """
    r = default_result
    assert sum(x.FY_N for x in r.reactions) == pytest.approx(r.P_ref_N, rel=1e-9)
    assert sum(x.FX_N for x in r.reactions) == pytest.approx(0.0, abs=1e-6)
    assert sum(x.FZ_N for x in r.reactions) == pytest.approx(0.0, abs=1e-6)
    assert sum(r.nodal_loads_N.values()) == pytest.approx(r.P_ref_N, rel=1e-12)
    # double symmetry: each pier base carries a quarter of the load
    assert [x.FY_N for x in r.reactions] == pytest.approx([r.P_ref_N / 4] * 4, rel=1e-6)


def _mirror_pairs(n: int) -> list[tuple[str, str]]:
    """Member pairs mirrored about mid-span (x -> span - x) and about the centreline
    (near 'n' <-> far 'f').

        bc_i <-> bc_(n-1-i),  tc_i <-> tc_(n-2-i),  d_i u <-> d_(n-1-i) d,
        fb_i <-> fb_(n-i),  ts_i <-> ts_(n-1-i),  pier0 <-> pier_n
    """
    pairs = []
    for s in ("n", "f"):
        pairs += [(f"bc{i}{s}", f"bc{n - 1 - i}{s}") for i in range(n)]
        pairs += [(f"tc{i}{s}", f"tc{n - 2 - i}{s}") for i in range(n - 1)]
        pairs += [(f"d{i}u{s}", f"d{n - 1 - i}d{s}") for i in range(n)]
        pairs += [(f"fb{i}{s}", f"fb{n - i}{s}") for i in range(n + 1)]
        pairs += [(f"pier0{s}", f"pier{n}{s}")]
    pairs += [(f"ts{i}", f"ts{n - 1 - i}") for i in range(n)]
    for i in range(n):
        for name in (f"bc{i}", f"d{i}u", f"d{i}d"):
            pairs.append((f"{name}n", f"{name}f"))
    return pairs


def test_symmetric_member_forces(default_result: AnalysisResult) -> None:
    """Symmetric bridge + symmetric load -> mirror members carry equal axial forces and
    equal bending moments (e.g. bc3n vs bc4n, d0un vs d7dn, bc3n vs bc3f).
    """
    r = default_result
    tol = 1e-6 * r.P_ref_N
    pairs = _mirror_pairs(N_PANELS)
    assert ("bc3n", "bc4n") in pairs
    for a, b in pairs:
        ma, mb = r.member(a), r.member(b)
        assert abs(ma.N_N - mb.N_N) <= tol, (a, b)
        assert ma.Mz_Nmm == pytest.approx(mb.Mz_Nmm, abs=tol * 100), (a, b)
        assert ma.My_Nmm == pytest.approx(mb.My_Nmm, abs=tol * 100), (a, b)


def test_chords_act_as_expected(default_result: AnalysisResult) -> None:
    """Simply supported through truss: the bottom chord at mid-span is in tension and the
    top chord over mid-span in compression.
    """
    r = default_result
    mid = N_PANELS // 2
    assert r.member(f"bc{mid}n").N_N > 0
    assert r.member(f"tc{mid - 1}n").N_N < 0


def test_governing_member_reaches_capacity_at_fu(default_result: AnalysisResult) -> None:
    """Load-factor method: F_u,strength = P_ref / U_max, so the most utilised member
    reaches capacity exactly at F_u,strength: U_max * F_u,strength / P_ref = 1 (to 1e-9),
    and no member exceeds capacity at F_u,p <= F_u,strength. When strength governs,
    F_u,p = F_u,strength, the reported governing member/mode are that member's, and
    U_gov * F_u,p / P_ref = 1; otherwise a global limit state (deflection, ...) governs.
    """
    r = default_result
    gov = max(r.members, key=lambda m: m.U)
    assert gov.U * r.Fu_strength_N / r.P_ref_N == pytest.approx(1.0, abs=1e-9)
    assert all(m.U * r.load_factor <= 1.0 + 1e-9 for m in r.members)
    if r.Fu_pred_N == r.Fu_strength_N:
        assert r.governing_member == gov.id
        assert r.governing_mode == gov.mode
        assert gov.U * r.Fu_pred_N / r.P_ref_N == pytest.approx(1.0, abs=1e-9)
    else:
        assert r.Fu_pred_N < r.Fu_strength_N
        assert r.governing_mode not in MEMBER_MODES


def test_pinned_joint_variant(material: Material) -> None:
    """joint_fixity = pinned (web and bracing members moment-released) is still stable and
    gives a positive, finite F_u.
    """
    bridge = generate_warren(WarrenParams(joint_fixity="pinned"))
    r = analyze(bridge, material)
    assert math.isfinite(r.Fu_pred_N) and r.Fu_pred_N > 0
    assert r.Fu_pred_N == min(fu_limits(r).values())
    for m in r.members:
        if m.group == "diagonal":  # in-plane moment released at both ends, no member loads
            assert abs(m.Mz_Nmm) < 1e-6 * r.P_ref_N, m.id
