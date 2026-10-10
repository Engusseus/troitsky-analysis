"""Sections with their own material values (plates, cross-laminated or other wood)."""

import math

import pytest
from pydantic import ValidationError

from bridgesim.analysis import analyze
from bridgesim.checks import capacities
from bridgesim.generators import generate_warren
from bridgesim.report import model_assumptions, to_markdown
from bridgesim.schema import Bridge, Section

PLATE = {"E_MPa": 5000.0, "G_MPa": 100.0, "f_t_MPa": 20.0, "f_c_MPa": 15.0,
         "f_b_MPa": 25.0, "f_v_MPa": 3.0}


def _beam(**own) -> Bridge:
    """The simply supported beam of test_beam_validation.py: 10 x 20 mm (d vertical), span
    1000 mm, P = 1000 N at mid-span, section values ``own``."""
    return Bridge.model_validate({
        "nodes": [{"id": f"N{k}", "x_mm": 500.0 * k, "y_mm": 100.0, "z_mm": 0.0}
                  for k in range(3)],
        "sections": [{"id": "rect", "b_mm": 10.0, "d_mm": 20.0, **own}],
        "members": [{"id": "m1", "i": "N0", "j": "N1", "section": "rect"},
                    {"id": "m2", "i": "N1", "j": "N2", "section": "rect"}],
        "deck": {"x_start_mm": 0.0, "x_end_mm": 1000.0, "top_elevation_mm": 110.0,
                 "clear_width_mm": 150.0},
        "supports": {"pinned": ["N0"], "roller": ["N2"], "extra_restraints": {"N0": ["RX"]}},
        "load": {"P_ref_N": 1000.0, "plate_length_mm": 1e-3, "x_center_mm": 500.0,
                 "deck_support_nodes": ["N1"]},
    })


def test_section_E_sets_the_stiffness(material) -> None:
    """Same beam as test_beam_validation: delta = P L^3 / (48 E I) = 312.5 mm at E = 10000;
    a section with E = 5000 (about a 0/90 stick plate along the bridge) bends twice as much,
    625 mm. Other sections keep the bridge material."""
    base = analyze(_beam(), material)
    soft = analyze(_beam(E_MPa=5000.0), material)
    assert base.delta_ref_mm == pytest.approx(312.5, rel=5e-3)
    assert soft.delta_ref_mm == pytest.approx(625.0, rel=5e-3)


def test_section_values_reach_the_solver(material) -> None:
    r = analyze(_beam(E_MPa=5000.0, G_MPa=100.0), material)
    pm = r.fe_model.members["m1"]
    assert (pm.material.E, pm.material.G) == (5000.0, 100.0)
    r = analyze(_beam(f_t_MPa=20.0), material)  # strength only: stiffness unchanged
    assert r.fe_model.members["m1"].material.E == material.E_MPa.value


def test_section_values_set_the_capacities(material) -> None:
    """10 x 20 mm rectangle: A = 200 mm^2, S_z = 10 x 20^2 / 6 = 666.7 mm^3,
    S_y = 20 x 10^2 / 6 = 333.3 mm^3, I_min = 20 x 10^3 / 12 = 1666.7 mm^4. With the plate
    values: N_t = 20 A = 4000 N, N_c = 15 A = 3000 N, M_z = 25 S_z = 16 667 N mm,
    M_y = 25 S_y = 8333 N mm, V = 3 A / 1.5 = 400 N, and over L = 500 mm
    P_cr = pi^2 x 5000 x 1666.7 / 500^2 = 329.0 N."""
    cap = capacities(Section(id="p", b_mm=10, d_mm=20, **PLATE), material, 500.0, 1.0)
    assert cap.N_t_R == pytest.approx(4000.0)
    assert cap.N_c_crush == pytest.approx(3000.0)
    assert cap.M_R_z == pytest.approx(25 * 10 * 20**2 / 6)
    assert cap.M_R_y == pytest.approx(25 * 20 * 10**2 / 6)
    assert cap.V_R == pytest.approx(400.0)
    assert cap.P_cr == pytest.approx(math.pi**2 * 5000 * 20 * 10**3 / 12 / 500**2)
    plain = capacities(Section(id="w", b_mm=10, d_mm=20, f_t_MPa=20.0), material, 500.0, 1.0)
    assert plain.N_t_R == pytest.approx(4000.0)
    assert plain.N_c_crush == pytest.approx(material.f_c_MPa.value * 200)


@pytest.mark.parametrize("key, value", [("E_MPa", 0.0), ("G_MPa", 1e6), ("f_t_MPa", 0.0),
                                        ("f_v_MPa", 1e5), ("E_MPa", math.inf)])
def test_unphysical_section_values_are_rejected(key: str, value: float) -> None:
    with pytest.raises(ValidationError):
        Section(id="p", b_mm=10, d_mm=2, **{key: value})


def test_round_trip_and_report() -> None:
    data = generate_warren().model_dump(mode="json")
    data["sections"].append({"id": "plate", "b_mm": 60.0, "d_mm": 4.0, **PLATE})
    data["members"].append({"id": "pl", "i": "C3", "j": "C4", "section": "plate"})
    b = Bridge.model_validate(data)
    assert Bridge.from_yaml_str(b.to_yaml()) == b
    assert "E_MPa" not in generate_warren().to_yaml()
    r = analyze(b)
    text = to_markdown(r)
    assert r"E = 5000, G = 100, f\_t = 20" in text and "bridge material" in text
    assert any(a.startswith("Sections with their own material values") and "plate (E_MPa = 5000"
               in a for a in model_assumptions(r))
