"""bridgesim web app (Streamlit). Runs locally (`streamlit run app/streamlit_app.py`) and in
the browser via stlite. Do not import PyVista or sectionproperties here (Pyodide)."""

from __future__ import annotations

import csv
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
from bridgesim.rules import RuleSet, RulesReport, evaluate
from bridgesim.schema import MEMBER_GROUPS, Bridge, Material
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
    ss.setdefault("upload_key", None)


BAD_FILE = (ValidationError, ValueError, yaml.YAMLError, UnicodeDecodeError)


def _replace_inputs() -> None:
    """Give the form widgets fresh keys so they show newly loaded values."""
    st.session_state.form_ver += 1


def _current_bridge() -> Bridge:
    ss = st.session_state
    if ss.source == "Upload bridge YAML" and ss.uploaded_bridge is not None:
        return ss.uploaded_bridge
    return generate_warren(WarrenParams.model_validate(ss.params), ss.material.stick)


def _run_analysis(ruleset: RuleSet) -> None:
    ss = st.session_state
    try:
        bridge = _current_bridge()
        ss.result = analyze(bridge, ss.material,
                            ruleset.crushing.get("deflection_limit_mm", 50.0))
        ss.rules_report = evaluate(bridge, ss.material, ruleset)
        ss.error = None
    except Exception as exc:  # show any failure in the UI rather than crashing
        ss.result, ss.rules_report = None, None
        ss.error = f"{type(exc).__name__}: {exc}"


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
            return RuleSet.from_yaml_str(up.getvalue().decode("utf-8"))
    except BAD_FILE as exc:
        st.sidebar.error(f"Rules file not valid: {exc}")
    return RuleSet.load(files[choice])


def _design_panel(ruleset: RuleSet) -> None:
    ss = st.session_state
    sb = st.sidebar
    sb.markdown("### Design")
    ss.source = sb.radio("Bridge source", ["Parametric Warren truss", "Upload bridge YAML"],
                         index=0 if ss.source == "Parametric Warren truss" else 1,
                         help="Upload a bridge.yaml exported from this app or written by hand.")
    if ss.source == "Upload bridge YAML":
        up = sb.file_uploader("bridge.yaml", type=["yaml", "yml"], key="bridge_up")
        if up is not None and ss.upload_key != (up.name, up.size):
            ss.upload_key = (up.name, up.size)
            try:
                ss.uploaded_bridge = Bridge.from_yaml_str(up.getvalue().decode("utf-8"))
                if isinstance(ss.uploaded_bridge.material, Material):
                    ss.material = ss.uploaded_bridge.material
                    _replace_inputs()
                ss.result = None  # re-analyse with the new geometry
            except BAD_FILE as exc:
                sb.error(f"Could not read the bridge file:\n\n{exc}")
                ss.uploaded_bridge = None
        if ss.uploaded_bridge is not None:
            sb.success(f"Loaded “{ss.uploaded_bridge.name}”: {len(ss.uploaded_bridge.nodes)}"
                       f" nodes, {len(ss.uploaded_bridge.members)} members")
        if ss.uploaded_bridge is None:
            sb.info("No bridge uploaded yet; showing the parametric design.")

    with sb.expander("Material values", expanded=False):
        mup = st.file_uploader("Load material YAML", type=["yaml", "yml"], key="mat_up")
        if mup is not None and st.button("Use this material file"):
            try:
                ss.material = material_from_yaml_str(mup.getvalue().decode("utf-8"))
                _replace_inputs()
                ss.result = None
                st.success(f"Material “{ss.material.name}” loaded")
            except BAD_FILE as exc:
                st.error(f"Material file not valid: {exc}")

    p = ss.params
    v = ss.form_ver
    parametric = ss.source == "Parametric Warren truss" or ss.uploaded_bridge is None
    with sb.form("design"):
        if parametric:
            st.caption("Through Warren truss. All lengths in mm.")
            p["name"] = st.text_input("Design name", p["name"])
            c1, c2 = st.columns(2)
            p["span_mm"] = c1.number_input("Span c/c (mm)", 500.0, 1500.0,
                                           float(p["span_mm"]), 5.0, format="%.0f",
                                           help="Pier centre to centre (§8.2.1.1)")
            p["n_panels"] = int(c2.number_input("Panels (even)", 2, 24, int(p["n_panels"]), 2))
            p["truss_height_mm"] = c1.number_input("Truss height (mm)", 40.0, 520.0,
                                                   float(p["truss_height_mm"]), 5.0,
                                                   format="%.0f", help="Chord centre to centre")
            p["deck_top_elevation_mm"] = c2.number_input(
                "Deck top (mm)", 20.0, 500.0, float(p["deck_top_elevation_mm"]), 5.0,
                format="%.0f", help="Table to top of deck (§8.2.2.1)")
            p["deck_clear_width_mm"] = c1.number_input(
                "Deck width (mm)", 50.0, 340.0, float(p["deck_clear_width_mm"]), 5.0,
                format="%.0f", help="Clear flat width between the trusses (§8.2.3.1)")
            p["deck_overhang_mm"] = c2.number_input(
                "Overhang (mm)", 0.0, 250.0, float(p["deck_overhang_mm"]), 5.0, format="%.0f",
                help="Deck length beyond each pier centreline")
            st.markdown("X-bracing")
            c3, c4, c5 = st.columns(3)
            opts = ["x", "none"]
            p["top_bracing"] = c3.selectbox("Top", opts, opts.index(p["top_bracing"]),
                                            help="Every top panel except mid-span (§8.9)")
            p["bottom_bracing"] = c4.selectbox("Bottom", opts, opts.index(p["bottom_bracing"]))
            p["pier_bracing"] = c5.selectbox("Piers", opts, opts.index(p["pier_bracing"]))
            fix = ["rigid", "pinned"]
            p["joint_fixity"] = st.radio("Joints", fix, fix.index(p["joint_fixity"]),
                                         horizontal=True,
                                         help="Rigid = glued joints transfer moment (default). "
                                              "Pinned releases web/bracing moments for comparison.")
            with st.expander("Sections (sticks per member)"):
                st.caption("10 × 2 mm sticks glued face to face. *flat*: stack grows in depth "
                           "(b = 10, d = 2n). *on edge*: stack grows in width (b = 2n, d = 10). "
                           "d is the depth in the member's main bending plane.")
                for g, spec in p["sections"].items():
                    a, b = st.columns([1, 1.3])
                    spec["sticks"] = int(a.number_input(GROUP_LABELS.get(g, g), 1, 60,
                                                        int(spec["sticks"]), 1, key=f"n_{v}_{g}"))
                    lay = ["flat", "on_edge"]
                    spec["layout"] = b.selectbox("layout", lay, lay.index(spec["layout"]),
                                                 key=f"l_{v}_{g}", label_visibility="hidden")
        else:
            st.caption(f"Using uploaded bridge “{ss.uploaded_bridge.name}”. Geometry and "
                       "sections come from the file; material values below still apply.")

        st.markdown("**Material** · tick *measured* once a value comes from your own tests")
        mat = ss.material
        new_vals = {}
        for key, prop in mat.props().items():
            label, unit = PROP_LABELS.get(key, (key, ""))
            a, b = st.columns([1.6, 1])
            val = a.number_input(f"{label} ({unit})" if unit and unit != "–" else label,
                                 value=float(prop.value), format="%g", key=f"mv_{v}_{key}",
                                 help=prop.note or None)
            meas = b.checkbox("measured", prop.source == "measured", key=f"ms_{v}_{key}")
            b.markdown(":green[● measured]" if meas else ":orange[○ assumed]")
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
            ss.error = f"Invalid input: {exc}"
            return
        _run_analysis(ruleset)


# --------------------------------------------------------------------------- main panels


def _results_card(r: AnalysisResult) -> None:
    st.markdown("#### Results")
    st.metric("Predicted ultimate load F_u,p", f"{r.Fu_pred_kgf:,.1f} kgf",
              help=f"{r.Fu_pred_N:,.0f} N. Linear first-failure estimate (load-factor method).")
    st.caption(f"= {r.Fu_pred_N:,.0f} N · 1 kgf = {N_PER_KGF} N")
    c1, c2 = st.columns(2)
    c1.metric("Mass", f"{r.mass.total_kg:.2f} kg", help=f"≈ {r.mass.stick_count} sticks")
    c2.metric("η_s = F_u/m", f"{r.efficiency:,.0f}", help="kgf per kg (rulebook §12.6)")
    gov = r.governing_label + (f" in **{r.governing_member}**" if r.governing_member else "")
    st.markdown(f"**Governs:** {gov}")
    pct = r.delta_at_Fu_mm / r.deflection_limit_mm
    st.markdown(f"**δ at F_u,p:** {r.delta_at_Fu_mm:.1f} / {r.deflection_limit_mm:g} mm")
    st.progress(min(1.0, pct))
    def lim(N: float) -> str:
        return f"{n_to_kgf(N):,.0f} kgf" if math.isfinite(N) else "∞"

    st.caption(f"Limits: strength {lim(r.Fu_strength_N)} · deflection "
               f"{lim(r.Fu_deflection_N)} · global buckling {lim(r.Fu_buckling_N)}")
    for w in r.warnings:
        st.warning(w, icon="⚠️")


def _rules_card(rep: RulesReport) -> None:
    st.markdown("#### Rule check")
    if rep.bans or rep.total_penalty or rep.disqualification_risks:
        msg = f"**Penalty: {_pts(rep.total_penalty)} pts**"
        if rep.bans:
            msg += f"  \n**BANNED from §{', §'.join(rep.bans)}**"
        if rep.disqualification_risks:
            msg += "  \n**Disqualification risk:** " + ", ".join(
                r.title for r in rep.disqualification_risks)
        st.error(msg, icon="🚫")
    else:
        st.success("All checked rules pass · 0 pts", icon="✅")
    for r in rep.checked:
        icon = "✓" if r.passed else "✗"
        line = f"{icon} {r.title} · {r.measured_text}"
        if not r.passed:
            line = f":red[{line}]"
            extra = f" · {_pts(r.penalty)} pts" if r.penalty else ""
            extra += f" · bans §{', §'.join(r.bans)}" if r.bans else ""
            extra += " · DQ risk" if r.disqualification else ""
            line += f" :red[{extra}]"
        st.markdown(f"{line} <span style='opacity:.6'>§{r.section}</span>",
                    unsafe_allow_html=True, help=(r.note or None))
    with st.expander(f"Not checked by the tool ({len(rep.info)})"):
        for r in rep.info:
            st.markdown(f"· **{r.title}** (§{r.section}): {r.note}")


def _viewport(bridge: Bridge, r: AnalysisResult | None) -> None:
    views = {"Utilization": "utilization", "Deformed": "deformed",
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
    groups_present = [g for g in MEMBER_GROUPS if any(m.group == g for m in bridge.members)]
    what = c1.selectbox("Diagram", ["Whole bridge (beam analogy)", "Member group (one truss plane)",
                                    "Single member"])
    load_choice = c2.radio("At load", ["F_u,p", "P_ref", "custom"], horizontal=True)
    load_N = {"F_u,p": r.Fu_pred_N, "P_ref": r.P_ref_N}.get(load_choice)
    if load_N is None:
        load_N = c3.number_input("Load (kgf)", 1.0, 5000.0, round(r.Fu_pred_kgf), 1.0) * N_PER_KGF
    if what.startswith("Whole"):
        fig = viz.global_sfd_bmd_figure(r, load_N)
        name = "sfd_bmd_whole_bridge.png"
    elif what.startswith("Member group"):
        g = c3.selectbox("Group", groups_present, format_func=lambda x: GROUP_LABELS.get(x, x))
        side = st.radio("Truss plane", ["n", "f"], horizontal=True,
                        format_func=lambda s: "−Z plane" if s == "n" else "+Z plane")
        ids = viz.chain_for_group(bridge, g, side)
        if not ids:
            st.info("No members of this group in that plane.")
            return
        fig = viz.member_diagrams_figure(r, ids, f"{GROUP_LABELS.get(g, g)} ({side} plane)",
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


def _compare(r: AnalysisResult, rep: RulesReport, bridge: Bridge) -> None:
    ss = st.session_state
    c1, c2 = st.columns([2, 1])
    name = c1.text_input("Save the current design as", value=f"Design {chr(65 + len(ss.saved))}")
    if c2.button("+ Save for comparison", **_wide_button()):
        ss.saved[name] = {
            "bridge_yaml": bridge.to_yaml(), "material": ss.material.model_dump(),
            "params": dict(ss.params), "source": ss.source,
            "metrics": _metrics(r, rep),
        }
    rows = {"Current": _metrics(r, rep)} | {k: v["metrics"] for k, v in ss.saved.items()}
    st.dataframe(pd.DataFrame(rows).astype(str), **_stretch())
    if ss.saved:
        c3, c4, c5 = st.columns([2, 1, 1])
        pick = c3.selectbox("Saved design", list(ss.saved))
        if c4.button("Load", **_wide_button()):
            snap = ss.saved[pick]
            ss.material = Material.model_validate(snap["material"])
            ss.params = snap["params"]
            if snap["source"] == "Upload bridge YAML":
                ss.uploaded_bridge = Bridge.from_yaml_str(snap["bridge_yaml"])
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
        "Governs": r.governing_label + (f" ({r.governing_member})" if r.governing_member else ""),
        "δ at F_u,p (mm)": round(r.delta_at_Fu_mm, 1),
        "Global buckling (kgf)": round(n_to_kgf(r.Fu_buckling_N), 0)
        if math.isfinite(r.Fu_buckling_N) else "∞",
        "Penalty (pts)": -rep.total_penalty or 0,
        "Bans": ", ".join(rep.bans) or "–",
        "Sticks (est.)": r.mass.stick_count,
        "Members": len(r.members),
    }


def _export(bridge: Bridge, r: AnalysisResult, rep: RulesReport) -> None:
    c1, c2, c3 = st.columns(3)
    c1.download_button("bridge.yaml", bridge.to_yaml(), "bridge.yaml", "text/yaml",
                       help="Re-load it with Bridge source → Upload bridge YAML")
    import yaml

    c1.download_button("material.yaml", yaml.safe_dump(st.session_state.material.model_dump(),
                                                        sort_keys=False),
                       "material.yaml", "text/yaml")
    rows = r.member_table()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    c2.download_button("Results CSV (members)", buf.getvalue(), "members.csv", "text/csv")
    rbuf = io.StringIO()
    rw = csv.writer(rbuf)
    rw.writerow(["section", "rule", "measured", "limit", "passed", "penalty", "bans", "note"])
    for x in rep.results:
        rw.writerow([x.section, x.title, x.measured_text, x.limit, x.passed, x.penalty,
                     " ".join(x.bans), x.note])
    c2.download_button("Rule check CSV", rbuf.getvalue(), "rules.csv", "text/csv")
    c3.download_button("Report (HTML)", report.to_html(r, rep), "design_validation.html",
                       "text/html", help="Assumptions, Method, Results (§10.2). Print to PDF.")
    c3.download_button("Report (Markdown)", report.to_markdown(r, rep), "design_validation.md",
                       "text/markdown")


def _method(r: AnalysisResult) -> None:
    st.markdown("**Modelling assumptions** (also in ASSUMPTIONS.md and the report)")
    for a in report.MODEL_ASSUMPTIONS:
        st.markdown(f"- {a}")
    st.markdown("**Method: load-factor approach**")
    st.latex(r"U_i = \frac{|N|}{N_R} + \frac{|M_y|}{M_{R,y}} + \frac{|M_z|}{M_{R,z}},\qquad "
             r"N_{c,R} = \min\!\left(f_c A,\ \frac{\pi^2 E I_{min}}{(KL)^2}\right)")
    st.latex(r"(K + \lambda K_g)\,\phi = 0 \;\Rightarrow\; F_{cr} = \lambda_{cr} P_{ref}")
    st.latex(r"F_{u,p} = \min\left(\frac{P_{ref}}{\max_i U_i},\ "
             r"P_{ref}\frac{50\,\mathrm{mm}}{\delta_{ref}},\ F_{cr}\right),\qquad "
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
                f"rules: {ruleset.name}</span>", unsafe_allow_html=True)
    assumed = ss.material.assumed_keys()
    if assumed:
        st.warning(f"**{len(assumed)} of {len(ss.material.props())} material values are "
                   "assumed placeholders, not test data.** Predicted loads are illustrative "
                   "until you enter your own measurements (tick *measured*).", icon="🧪")
    if ss.error:
        st.error(ss.error)
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
        _compare(r, rep, bridge)
    with tabs[4]:
        _export(bridge, r, rep)
    with tabs[5]:
        _method(r)
    st.caption("Not affiliated with the Troitsky organizers. The official rulebook is "
               "authoritative. Open source (MIT).")


main()
