"""bridgesim web app (Streamlit). Runs locally (`streamlit run app/streamlit_app.py`) and in
the browser via stlite. Do not import PyVista or sectionproperties here (Pyodide)."""

from __future__ import annotations

import copy
import csv
import hashlib
import html
import inspect
import io
import math
import sys
from pathlib import Path

try:
    import bridgesim  # noqa: F401
except ImportError:  # running from a source checkout without `uv sync`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import streamlit as st
import yaml
from pydantic import ValidationError

from bridgesim import PYNITE_VERSION, __version__, report, viz
from bridgesim.analysis import AnalysisResult, analyze
from bridgesim.generators.warren import WarrenParams, generate_warren
from bridgesim.materials import load_material, material_from_yaml_str
from bridgesim.paths import list_yaml
from bridgesim.rules import RuleSet, RulesReport, apply_crushing, evaluate
from bridgesim.schema import MEMBER_GROUPS, Bridge, Material
from bridgesim.textsafe import csv_cell, csv_row, friendly_error, md, md_keep_bold
from bridgesim.units import N_PER_KGF, n_to_kgf

st.set_page_config(page_title="bridgesim: Troitsky bridge analysis", page_icon="🌉",
                   layout="wide")

GROUP_LABELS = {
    "top_chord": "Top chord", "bottom_chord": "Bottom chord", "diagonal": "Diagonals",
    "vertical": "Verticals", "floor_beam": "Floor beams", "top_strut": "Top struts",
    "top_brace": "Top bracing", "bottom_brace": "Bottom bracing", "pier": "Piers",
    "pier_brace": "Pier bracing", "other": "Other",
}
PROP_LABELS = {
    "E_MPa": ("E, elastic modulus", "MPa"), "G_MPa": ("G, shear modulus", "MPa"),
    "nu": ("ν, Poisson's ratio", ""), "f_t_MPa": ("f_t, tension strength", "MPa"),
    "f_c_MPa": ("f_c, compression strength", "MPa"), "f_b_MPa": ("f_b, bending strength", "MPa"),
    "f_v_MPa": ("f_v, shear strength", "MPa"), "density_kg_m3": ("Density", "kg/m³"),
    "glue.tau_g_MPa": ("τ_g, glue shear strength", "MPa"),
    "glue.mass_fraction": ("Glue mass fraction", "–"),
    "glue.overlap_mm": ("Glued overlap per member end", "mm"),
    "glue.faces": ("Glued faces per member end", "–"),
}


def _stretch() -> dict:
    """Full-width kwargs that work on old (stlite) and new Streamlit versions."""
    params = inspect.signature(st.plotly_chart).parameters
    if "width" in params and params["width"].default == "stretch":
        return {}
    return {"use_container_width": True}


def _wide_button() -> dict:
    """Full-width button kwargs for old and new Streamlit."""
    params = inspect.signature(st.button).parameters
    if "width" in params:
        return {"width": "stretch"}
    return {"use_container_width": True}


def _pts(x: float) -> str:
    return "0" if not x else f"−{x:g}"


# --------------------------------------------------------------------------- state


def _init_state() -> None:
    ss = st.session_state
    if "material" not in ss:
        ss.material = load_material("popsicle_birch")
    if "params" not in ss:
        ss.params = WarrenParams().model_dump()
    ss.setdefault("source", "Parametric Warren truss")
    ss.setdefault("uploaded_bridge", None)
    ss.setdefault("saved", {})
    ss.setdefault("result", None)
    ss.setdefault("rules_report", None)
    ss.setdefault("error", None)
    ss.setdefault("form_ver", 0)  # bumped when inputs are replaced from outside the form
    ss.setdefault("rules_key", None)
    ss.setdefault("seen_digest", None)  # uploader content already processed (or ignored)
    ss.setdefault("bridge_from_uploader", False)


BAD_FILE = (ValidationError, ValueError, yaml.YAMLError, UnicodeDecodeError)


def _lines_md(text: str) -> str:
    """Escaped Markdown that keeps one line per problem."""
    return "  \n".join(md(line) for line in text.splitlines())


def _digest(uploaded) -> str | None:
    return hashlib.sha256(uploaded.getvalue()).hexdigest() if uploaded is not None else None


def _keyed(widget, label: str, key: str, default, **kwargs):
    """Create a form widget whose value lives in session state under ``key``, seeded once.

    Passing ``value=`` instead would make the widget's identity depend on it: after a submit
    changed the default, the next edit to that field would be silently dropped. Keys carry
    ``form_ver`` so loading a design or material (``_replace_inputs``) re-seeds every field.
    """
    if key not in st.session_state:
        st.session_state[key] = default
    return widget(label, key=key, **kwargs)


def _replace_inputs() -> None:
    """Give the form widgets fresh keys so they show newly loaded values."""
    st.session_state.form_ver += 1


def _current_bridge(ruleset: RuleSet) -> Bridge:
    """The bridge being analysed, carrying the active material and the rules' crusher."""
    ss = st.session_state
    if ss.source == "Upload bridge YAML" and ss.uploaded_bridge is not None:
        bridge = ss.uploaded_bridge
    else:
        bridge = generate_warren(WarrenParams.model_validate(ss.params), ss.material.stick)
    bridge = bridge.model_copy(update={"material": ss.material})
    return apply_crushing(bridge, ruleset)


def _run_analysis(ruleset: RuleSet) -> None:
    ss = st.session_state
    try:
        bridge = _current_bridge(ruleset)
        ss.result = analyze(bridge, ss.material,
                            ruleset.crushing.deflection_limit_mm)
        ss.rules_report = evaluate(bridge, ss.material, ruleset)
        ss.error = None
    except Exception as exc:  # show any failure in the UI rather than crashing
        ss.result, ss.rules_report = None, None
        ss.error = f"{type(exc).__name__}:\n{friendly_error(exc)}"


# --------------------------------------------------------------------------- sidebar


def _rules_picker() -> RuleSet:
    files = {p.stem: p for p in list_yaml("rules")}
    with st.sidebar:
        st.markdown("### Rules")
        names = list(files)
        choice = st.selectbox("Rules file", names, index=names.index("troitsky_2027")
                              if "troitsky_2027" in names else 0,
                              help="Rules live in rules/*.yaml. Next year's team edits one file.")
        up = st.file_uploader("…or upload a rules YAML", type=["yaml", "yml"], key="rules_up")
    try:
        if up is not None:
            rs = RuleSet.from_yaml_str(up.getvalue().decode("utf-8"))
            st.sidebar.caption(f"Using the uploaded rules “{md(rs.name)}”; the picker above "
                               "applies once the file is removed.")
            return rs
    except BAD_FILE as exc:
        st.sidebar.error("Rules file not valid:  \n" + _lines_md(friendly_error(exc)))
    return RuleSet.load(files[choice])


def _design_panel(ruleset: RuleSet) -> None:
    ss = st.session_state
    sb = st.sidebar
    sb.markdown("### Design")
    source = sb.radio("Bridge source", ["Parametric Warren truss", "Upload bridge YAML"],
                      index=0 if ss.source == "Parametric Warren truss" else 1,
                      help="Upload a bridge.yaml exported from this app or written by hand.")
    if source != ss.source:  # results on screen belong to the other source
        ss.source, ss.result, ss.error = source, None, None
    if ss.source == "Upload bridge YAML":
        up = sb.file_uploader("bridge.yaml", type=["yaml", "yml"], key="bridge_up")
        digest = _digest(up)
        if up is not None and digest != ss.seen_digest:
            try:
                ss.uploaded_bridge = Bridge.from_yaml_str(up.getvalue().decode("utf-8"))
                ss.seen_digest = digest  # only after it parsed, so a fixed file is re-read
                ss.bridge_from_uploader = True
                if isinstance(ss.uploaded_bridge.material, Material):
                    ss.material = ss.uploaded_bridge.material
                    _replace_inputs()
                ss.result, ss.error = None, None  # re-analyse with the new geometry
            except BAD_FILE as exc:
                sb.error("Could not read the bridge file:  \n" + _lines_md(friendly_error(exc)))
                # fall back to the parametric design, without an error from the old bridge
                ss.uploaded_bridge, ss.result, ss.error = None, None, None
        elif up is None and ss.seen_digest is not None:  # the file was removed
            ss.seen_digest = None
            if ss.bridge_from_uploader:  # a restored snapshot is not tied to the uploader
                ss.uploaded_bridge, ss.result, ss.error = None, None, None
        if ss.uploaded_bridge is not None:
            sb.success(f"Loaded “{md(ss.uploaded_bridge.name)}”: {len(ss.uploaded_bridge.nodes)}"
                       f" nodes, {len(ss.uploaded_bridge.members)} members")
        if ss.uploaded_bridge is None:
            sb.info("No bridge uploaded yet; showing the parametric design.")

    with sb.expander("Material values", expanded=False):
        mup = st.file_uploader("Load material YAML", type=["yaml", "yml"], key="mat_up")
        if mup is not None and st.button("Use this material file"):
            try:
                ss.material = material_from_yaml_str(mup.getvalue().decode("utf-8"))
                _replace_inputs()
                ss.result, ss.error = None, None  # re-analyse with the new material
                st.success(f"Material “{md(ss.material.name)}” loaded")
            except BAD_FILE as exc:
                st.error("Material file not valid:  \n" + _lines_md(friendly_error(exc)))

    p = copy.deepcopy(ss.params)  # edits only reach ss.params when Analyze is pressed
    v = ss.form_ver
    parametric = ss.source == "Parametric Warren truss" or ss.uploaded_bridge is None
    with sb.form("design"):
        if parametric:
            st.caption("Through Warren truss. All lengths in mm.")
            p["name"] = _keyed(st.text_input, "Design name", f"name_{v}", p["name"])
            c1, c2 = st.columns(2)
            mm = dict(step=5.0, format="%.0f")
            p["span_mm"] = _keyed(c1.number_input, "Span c/c (mm)", f"span_{v}",
                                  float(p["span_mm"]), min_value=500.0, max_value=1500.0,
                                  help="Pier centre to centre (§8.2.1.1)", **mm)
            p["n_panels"] = int(_keyed(c2.number_input, "Panels (even)", f"panels_{v}",
                                       int(p["n_panels"]), min_value=2, max_value=24, step=2))
            p["truss_height_mm"] = _keyed(c1.number_input, "Truss height (mm)", f"h_{v}",
                                          float(p["truss_height_mm"]), min_value=40.0,
                                          max_value=520.0, help="Chord centre to centre", **mm)
            p["deck_top_elevation_mm"] = _keyed(
                c2.number_input, "Deck top (mm)", f"deck_{v}", float(p["deck_top_elevation_mm"]),
                min_value=20.0, max_value=500.0, help="Table to top of deck (§8.2.2.1)", **mm)
            p["deck_clear_width_mm"] = _keyed(
                c1.number_input, "Deck width (mm)", f"width_{v}",
                float(p["deck_clear_width_mm"]), min_value=50.0, max_value=340.0,
                help="Clear flat width between the trusses (§8.2.3.1)", **mm)
            p["deck_overhang_mm"] = _keyed(
                c2.number_input, "Overhang (mm)", f"over_{v}", float(p["deck_overhang_mm"]),
                min_value=0.0, max_value=250.0, help="Deck length beyond each pier centreline",
                **mm)
            st.markdown("X-bracing")
            for key, lab, tip in (
                    ("top_bracing", "Top bracing", "Every top panel except mid-span (§8.9)"),
                    ("bottom_bracing", "Bottom (floor) bracing", "Every bottom panel"),
                    ("pier_bracing", "Pier bracing", "Between the two piers at each end")):
                on = _keyed(st.checkbox, lab, f"br_{v}_{key}", p[key] == "x", help=tip)
                p[key] = "x" if on else "none"
            p["joint_fixity"] = _keyed(
                st.radio, "Joints", f"fix_{v}", p["joint_fixity"], options=["rigid", "pinned"],
                horizontal=True, help="Rigid = glued joints transfer moment (default). "
                                      "Pinned releases web/bracing moments for comparison.")
            with st.expander("Sections (sticks per member)"):
                st.caption("10 × 2 mm sticks glued face to face. *flat*: stack grows in depth "
                           "(b = 10, d = 2n). *on edge*: stack grows in width (b = 2n, d = 10). "
                           "d is the depth in the member's main bending plane.")
                for g, spec in p["sections"].items():
                    a, b = st.columns([1, 1.7])
                    spec["sticks"] = int(_keyed(a.number_input, GROUP_LABELS.get(g, g),
                                                f"n_{v}_{g}", int(spec["sticks"]),
                                                min_value=1, max_value=60, step=1))
                    spec["layout"] = _keyed(b.selectbox, "layout", f"l_{v}_{g}", spec["layout"],
                                            options=["flat", "on_edge"],
                                            format_func=lambda x: x.replace("_", " "),
                                            label_visibility="hidden")
        else:
            st.caption(f"Using uploaded bridge “{md(ss.uploaded_bridge.name)}”. Geometry and "
                       "sections come from the file; material values below still apply.")

        st.markdown("**Material** · tick *measured* once a value comes from your own tests")
        mat = ss.material
        new_vals = {}
        for key, prop in mat.props().items():
            label, unit = PROP_LABELS.get(key, (key, ""))
            val = _keyed(st.number_input, f"{label} ({unit})" if unit and unit != "–" else label,
                         f"mv_{v}_{key}", float(prop.value), format="%g",
                         help=md(prop.note) if prop.note else None)
            meas = _keyed(st.checkbox, ":green[● measured]" if prop.source == "measured"
                          else ":orange[○ assumed] · tick when measured",
                          f"ms_{v}_{key}", prop.source == "measured")
            new_vals[key] = (val, "measured" if meas else "assumed")
        submitted = st.form_submit_button("Analyze", type="primary", **_wide_button())

    if submitted:
        data = mat.model_dump()
        for key, (val, src) in new_vals.items():
            tgt = data["glue"] if key.startswith("glue.") else data
            k = key.removeprefix("glue.")
            tgt[k] = {**tgt[k], "value": val, "source": src}
        try:
            ss.material = Material.model_validate(data)
            ss.params = WarrenParams.model_validate(p).model_dump()
        except ValidationError as exc:
            ss.error = f"Invalid input:\n{friendly_error(exc)}"
            return
        _run_analysis(ruleset)


# --------------------------------------------------------------------------- main panels


def _results_card(r: AnalysisResult) -> None:
    st.markdown("#### Results")
    st.metric("Predicted ultimate load F_u,p", f"{r.Fu_pred_kgf:,.1f} kgf",
              help=f"{r.Fu_pred_N:,.0f} N. Linear first-failure estimate (load-factor method).")
    st.caption(f"= {r.Fu_pred_N:,.0f} N · 1 kgf = {N_PER_KGF} N")
    c1, c2 = st.columns(2)
    c1.metric("Mass", f"{r.mass.total_kg:.2f} kg", help=f"≈ {r.mass.stick_count} sticks" + (
        f", incl. {r.mass.extra_kg:.2f} kg of extra wood (gussets, plates)"
        if r.mass.extra_kg >= 0.005 else ""))
    c2.metric("η_s = F_u/m", f"{r.efficiency:,.0f}", help="kgf per kg (rulebook §12.6)")
    gov = r.governing_label + (f" in **{md(r.governing_member)}**" if r.governing_member else "")
    st.markdown(f"**Governs:** {gov}")
    pct = r.delta_at_Fu_mm / r.deflection_limit_mm
    st.markdown(f"**δ at F_u,p:** {r.delta_at_Fu_mm:.1f} / {r.deflection_limit_mm:g} mm")
    st.progress(min(1.0, pct))
    def lim(N: float) -> str:
        if math.isnan(N):
            return "NOT EVALUATED"
        return f"{n_to_kgf(N):,.0f} kgf" if math.isfinite(N) else "∞"

    st.caption(f"Limits: strength {lim(r.Fu_strength_N)} · deflection "
               f"{lim(r.Fu_deflection_N)} · global buckling {lim(r.Fu_buckling_N)}")
    for w in r.warnings:
        st.warning(md(w), icon="⚠️")


def _rules_card(rep: RulesReport) -> None:
    st.markdown("#### Rule check")
    if rep.bans or rep.total_penalty or rep.disqualification_risks:
        msg = f"**Penalty: {_pts(rep.total_penalty)} pts**"
        if rep.bans:
            msg += f"  \n**BANNED from §{', §'.join(md(b) for b in rep.bans)}**"
        if rep.disqualification_risks:
            msg += "  \n**Disqualification risk:** " + ", ".join(
                md(r.title) for r in rep.disqualification_risks)
        st.error(msg, icon="🚫")
    elif any(not r.passed for r in rep.checked):
        n = sum(not r.passed for r in rep.checked)
        st.warning(f"{n} checked rule(s) fail, but this rules file sets no penalty for them",
                   icon="⚠️")
    else:
        st.success("All checked rules pass · 0 pts", icon="✅")
    for r in rep.checked:
        icon = "✓" if r.passed else "✗"
        line = f"{icon} {md(r.title)} · {md(r.measured_detail)}"
        if not r.passed:
            line = f":red[{line}]"
            extra = f" · {_pts(r.penalty)} pts" if r.penalty else ""
            extra += f" · bans §{', §'.join(md(b) for b in r.bans)}" if r.bans else ""
            extra += " · DQ risk" if r.disqualification else ""
            line += f" :red[{extra}]"
        st.markdown(f"{line} · §{md(r.section)}", help=(md(r.note) if r.note else None))
    with st.expander(f"Not checked by the tool ({len(rep.info)})"):
        for r in rep.info:
            st.markdown(f"· **{md(r.title)}** (§{md(r.section)}): {md(r.note)}")


def _viewport(bridge: Bridge, r: AnalysisResult | None) -> None:
    views = {"Utilisation": "utilization", "Deformed": "deformed",
             "Buckling mode": "buckling", "Undeformed": "undeformed"}
    c1, c2 = st.columns([2, 1.2])
    label = c1.radio("View", list(views), horizontal=True, label_visibility="collapsed")
    scale = None
    if views[label] == "deformed" and r is not None:
        auto = viz.default_deform_scale(r)
        mult = c2.slider("Exaggeration", 0.1, 5.0, 1.0, 0.1,
                         help=f"Multiplier on the automatic scale (×{auto:,.0f})")
        scale = auto * mult
    fig = viz.bridge_figure(bridge, r, views[label], deform_scale=scale)
    st.plotly_chart(fig, **_stretch())
    if views[label] == "utilization":
        st.caption("Members coloured by utilisation at the predicted failure load "
                   "(1.0 = fails). Red = governing member. Hover a member for its forces.")
    elif views[label] == "buckling":
        st.caption("First global (system) buckling mode from a linear eigenvalue analysis. "
                   "Darker members take a larger share of the mode's strain energy. Elastic, "
                   "perfectly straight sticks: real crooked sticks buckle earlier.")


def _critical_table(r: AnalysisResult) -> None:
    df = pd.DataFrame(r.member_table())
    top = df.sort_values("U_ref", ascending=False).head(10)
    cols = ["member", "group", "section", "N_at_Fu_N", "U_at_Fu", "governing_check",
            "member_fails_at_kgf", "L_mm"]
    st.dataframe(top[cols], hide_index=True, **_stretch(),
                 column_config={
                     "N_at_Fu_N": st.column_config.NumberColumn("N at F_u,p (N, + tension)",
                                                                format="%.0f"),
                     "U_at_Fu": st.column_config.ProgressColumn(
                         "Utilisation at F_u,p", min_value=0.0, max_value=1.0, format="%.2f"),
                     "member_fails_at_kgf": st.column_config.NumberColumn(
                         "Fails alone at (kgf)", format="%.0f"),
                     "governing_check": "Governing check",
                     "L_mm": st.column_config.NumberColumn("Length (mm)", format="%.0f"),
                 })
    with st.expander("All members"):
        st.dataframe(df, hide_index=True, **_stretch())


def _diagrams(bridge: Bridge, r: AnalysisResult) -> None:
    st.caption("Shear and bending-moment diagrams for the Technical Poster (§12.4). Values "
               "scale linearly with load.")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    # Only groups whose members form a connected chain in a truss plane (e.g. chords) can be
    # drawn end to end; others would show unrelated members as one continuous beam.
    groups_present = [g for g in MEMBER_GROUPS
                      if any(viz.is_connected_chain(bridge, viz.chain_for_group(bridge, g, s))
                             for s in ("n", "f"))]
    options = ["Whole bridge (beam analogy)", "Member group (one truss plane)", "Single member"]
    if not groups_present:
        options.remove("Member group (one truss plane)")
    what = c1.selectbox("Diagram", options)
    load_choice = c2.radio("At load", ["F_u,p", "P_ref", "custom"], horizontal=True)
    load_N = {"F_u,p": r.Fu_pred_N, "P_ref": r.P_ref_N}.get(load_choice)
    if load_N is None:
        fu = r.Fu_pred_kgf
        start = float(min(5000.0, max(1.0, round(fu)))) if math.isfinite(fu) else 100.0
        load_N = c3.number_input("Load (kgf)", 1.0, 5000.0, start, 1.0) * N_PER_KGF
    if what.startswith("Whole"):
        fig = viz.global_sfd_bmd_figure(r, load_N)
        name = "sfd_bmd_whole_bridge.png"
    elif what.startswith("Member group"):
        g = c3.selectbox("Group", groups_present, format_func=lambda x: GROUP_LABELS.get(x, x))
        plane = {"n": "−Z plane", "f": "+Z plane"}
        side = st.radio("Truss plane", ["n", "f"], horizontal=True, format_func=plane.get)
        ids = viz.chain_for_group(bridge, g, side)
        if not viz.is_connected_chain(bridge, ids):
            st.info("This group does not form a connected chain in that plane; use "
                    "Single member instead.")
            return
        fig = viz.member_diagrams_figure(r, ids, f"{GROUP_LABELS.get(g, g)} ({plane[side]})",
                                         load_N)
        name = f"diagram_{g}_{side}.png"
    else:
        mid = c3.selectbox("Member", [m.id for m in bridge.members],
                           index=[m.id for m in bridge.members].index(r.governing_member)
                           if r.governing_member else 0)
        fig = viz.member_diagrams_figure(r, [mid], f"Member {mid}", load_N)
        name = f"diagram_{mid}.png"
    png = viz.figure_png(fig)
    st.image(png, width=820)
    st.download_button("Download PNG", png, file_name=name, mime="image/png")


def _snapshot_metrics(snap: dict, ruleset: RuleSet, rules_key: str) -> dict:
    """Metrics of a saved design under the active rules (recomputed if the rules changed)."""
    if snap.get("rules_key") != rules_key:
        try:
            material = Material.model_validate(snap["material"])
            b = Bridge.from_yaml_str(snap["bridge_yaml"]).model_copy(
                update={"material": material})
            b = apply_crushing(b, ruleset)
            res = analyze(b, material, ruleset.crushing.deflection_limit_mm)
            snap["metrics"] = _metrics(res, evaluate(b, material, ruleset))
        except Exception as exc:  # show why instead of stale numbers
            snap["metrics"] = {"F_u,p (kgf)": f"not recomputed: {type(exc).__name__}"}
        snap["rules_key"] = rules_key
    return snap["metrics"]


def _compare(r: AnalysisResult, rep: RulesReport, bridge: Bridge, ruleset: RuleSet) -> None:
    ss = st.session_state
    c1, c2 = st.columns([2, 1])
    name = c1.text_input("Save the current design as", value=f"Design {chr(65 + len(ss.saved))}")
    if c2.button("+ Save for comparison", **_wide_button()):
        ss.saved[name] = {
            "bridge_yaml": bridge.to_yaml(), "material": ss.material.model_dump(),
            "params": copy.deepcopy(ss.params), "source": ss.source,
            "metrics": _metrics(r, rep), "rules_key": ss.rules_key,
        }
    rows = {"Current": _metrics(r, rep)} | {
        k: _snapshot_metrics(v, ruleset, ss.rules_key) for k, v in ss.saved.items()}
    st.caption(f"All rows evaluated with the active rules ({md(ruleset.name)}).")
    st.dataframe(pd.DataFrame(rows).astype(str), **_stretch())
    if ss.saved:
        c3, c4, c5 = st.columns([2, 1, 1])
        pick = c3.selectbox("Saved design", list(ss.saved))
        if c4.button("Load", **_wide_button()):
            snap = ss.saved[pick]
            ss.material = Material.model_validate(snap["material"])
            ss.params = copy.deepcopy(snap["params"])
            if snap["source"] == "Upload bridge YAML":
                ss.uploaded_bridge = Bridge.from_yaml_str(snap["bridge_yaml"])
                ss.bridge_from_uploader = False
                # A file still sitting in the uploader must not replace the restored design.
                ss.seen_digest = _digest(ss.get("bridge_up"))
            ss.source = snap["source"]
            ss.result = None
            _replace_inputs()
            st.rerun()
        if c5.button("Delete", **_wide_button()):
            ss.saved.pop(pick, None)
            st.rerun()


def _metrics(r: AnalysisResult, rep: RulesReport) -> dict:
    return {
        "F_u,p (kgf)": round(r.Fu_pred_kgf, 1),
        "Mass (kg)": round(r.mass.total_kg, 2),
        "η_s (kgf/kg)": round(r.efficiency, 1),
        "Governs": r.governing_label + (f" in {r.governing_member}" if r.governing_member else ""),
        "δ at F_u,p (mm)": round(r.delta_at_Fu_mm, 1),
        "Global buckling (kgf)": round(n_to_kgf(r.Fu_buckling_N), 0)
        if math.isfinite(r.Fu_buckling_N)
        else ("not evaluated" if math.isnan(r.Fu_buckling_N) else "∞"),
        "Penalty (pts)": -rep.total_penalty or 0,
        "Bans": ", ".join(rep.bans) or "–",
        "Sticks (est.)": r.mass.stick_count,
        "Members": len(r.members),
    }


def _export(bridge: Bridge, r: AnalysisResult, rep: RulesReport) -> None:
    c1, c2, c3 = st.columns(3)
    c1.download_button("bridge.yaml", bridge.to_yaml(), "bridge.yaml", "text/yaml",
                       help="Includes the material values in use. Re-load it with Bridge "
                            "source → Upload bridge YAML, or run it with the CLI.")
    import yaml

    c1.download_button("material.yaml", yaml.safe_dump(st.session_state.material.model_dump(),
                                                        sort_keys=False),
                       "material.yaml", "text/yaml")
    rows = r.member_table()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(csv_row(r) for r in rows)
    c2.download_button("Results CSV (members)", buf.getvalue(), "members.csv", "text/csv")
    rbuf = io.StringIO()
    rw = csv.writer(rbuf)
    rw.writerow(["section", "rule", "measured", "limit", "passed", "penalty", "bans", "note"])
    for x in rep.results:
        rw.writerow([csv_cell(v) for v in (x.section, x.title, x.measured_detail, x.limit,
                                           x.passed, x.penalty, " ".join(x.bans), x.note)])
    c2.download_button("Rule check CSV", rbuf.getvalue(), "rules.csv", "text/csv")
    c3.download_button("Report (HTML)", report.to_html(r, rep), "design_validation.html",
                       "text/html", help="Assumptions, Method, Results (§10.2). Print to PDF.")
    c3.download_button("Report (Markdown)", report.to_markdown(r, rep), "design_validation.md",
                       "text/markdown")


def _method(r: AnalysisResult) -> None:
    st.markdown("**Modelling assumptions** (also in ASSUMPTIONS.md and the report)")
    for a in report.model_assumptions(r):
        st.markdown(f"- {md_keep_bold(a)}")
    st.markdown("**Method: load-factor approach**")
    st.latex(r"U_i = \frac{|N|}{N_R} + \frac{|M_y|}{M_{R,y}} + \frac{|M_z|}{M_{R,z}},\qquad "
             r"N_{c,R} = \min\!\left(f_c A,\ \frac{\pi^2 E I_{min}}{(KL)^2}\right)")
    st.latex(r"(K + \lambda K_g)\,\phi = 0 \;\Rightarrow\; F_{cr} = \lambda_{cr} P_{ref}")
    st.latex(r"F_{u,p} = \min\left(\frac{P_{ref}}{\max_i U_i},\ "
             r"P_{ref}\frac{" + f"{r.deflection_limit_mm:g}" + r"\,\mathrm{mm}}{\delta_{ref}},"
             r"\ F_{cr}\right),\qquad "
             r"\eta_s = \frac{F_{u,p}\,[\mathrm{kgf}]}{m\,[\mathrm{kg}]}")
    st.caption(f"Software: bridgesim {__version__} (MIT) with the Pynite {PYNITE_VERSION} "
               "3D frame solver (MIT). Cite both in your Design Validation (§10.2).")


# --------------------------------------------------------------------------- page


def main() -> None:
    _init_state()
    ss = st.session_state
    ruleset = _rules_picker()
    rules_key = ruleset.model_dump_json()
    if rules_key != ss.rules_key:  # rules file changed: re-evaluate
        ss.rules_key, ss.result, ss.error = rules_key, None, None
    _design_panel(ruleset)
    if ss.result is None and ss.error is None:
        _run_analysis(ruleset)

    st.markdown(f"## 🌉 bridgesim <span style='font-size:.55em;opacity:.6'>v{__version__} · "
                f"rules: {html.escape(ruleset.name)}</span>", unsafe_allow_html=True)
    assumed = ss.material.assumed_keys()
    if assumed:
        st.warning(f"**{len(assumed)} of {len(ss.material.props())} material values are "
                   "assumed placeholders, not test data.** Predicted loads are illustrative "
                   "until you enter your own measurements (tick *measured*).", icon="🧪")
    if ss.error:
        st.error(_lines_md(ss.error))  # may quote values from an uploaded file
        return
    r: AnalysisResult = ss.result
    rep: RulesReport = ss.rules_report
    bridge = r.bridge

    left, right = st.columns([2.6, 1])
    with left:
        _viewport(bridge, r)
    with right:
        _results_card(r)
        st.divider()
        _rules_card(rep)

    tabs = st.tabs(["Load–deflection", "Critical members", "Shear & moment diagrams",
                    "Compare designs", "Export", "Assumptions & method"])
    with tabs[0]:
        st.plotly_chart(viz.load_deflection_figure(r), **_stretch())
        st.caption("v0.1 is linear: the curve is a straight line to the first failure. "
                   "A progressive-collapse curve is planned for v0.3.")
    with tabs[1]:
        _critical_table(r)
    with tabs[2]:
        _diagrams(bridge, r)
    with tabs[3]:
        _compare(r, rep, bridge, ruleset)
    with tabs[4]:
        _export(bridge, r, rep)
    with tabs[5]:
        _method(r)
    st.caption("Not affiliated with the Troitsky organizers. The official rulebook is "
               "authoritative. Open source (MIT).")


main()
