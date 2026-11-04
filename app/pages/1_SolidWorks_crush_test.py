"""Drop a SolidWorks part in, see how bridgesim understood it, and crush it."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

from bridgesim.sim import study
from bridgesim.sim.crush import CrushOptions

st.set_page_config(page_title="bridgesim: SolidWorks crush test", page_icon="🌉",
                   layout="wide")
st.title("SolidWorks crush test")
st.caption("Drop your bridge (SolidWorks .SLDPRT, Parasolid .x_b, or STL with all bodies, "
           "mm). bridgesim rebuilds every body, decides how to model it, finds the glued "
           "joints, checks the 2027 rules on the solids and simulates the §12.5 crushing "
           "test: joints break, members buckle and crack, the bridge collapses.")

up = st.file_uploader("Bridge CAD file", type=["sldprt", "x_b", "stl"])
if up is None:
    st.info("Waiting for a file. Material values are placeholders until you measure your "
            "sticks and glue (see the Measuring section of the README).")
    st.stop()

data = up.getvalue()
digest = hashlib.sha256(data).hexdigest()[:16]
work = Path(tempfile.gettempdir()) / f"bridgesim_{digest}"
work.mkdir(exist_ok=True)
cad = work / up.name
cad.write_bytes(data)

ov_text = st.text_area(
    "Overrides (YAML): answers to the questions below", key="ov",
    value=st.session_state.get("ov", "bodies: {}\nparams: {}\n"), height=120,
    help="e.g.  bodies: {\"Body 21531\": {layup: \"10-ply UD\"}}  params: {floss.strands: 6}")
try:
    overrides = yaml.safe_load(ov_text) or {}
except yaml.YAMLError as exc:
    st.error(f"Overrides are not valid YAML: {exc}")
    st.stop()


@st.cache_data(show_spinner="Reading the CAD file and building the model…")
def _inspect(path: str, ov: str):
    return study.inspect(path, yaml.safe_load(ov) or {}).model_dump(mode="json")


rep = _inspect(str(cad), ov_text)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bodies", rep["n_bodies"])
c2.metric("Mass", f"{rep['mass_kg']:.2f} kg")
c3.metric("Glued joints", len(rep["joints"]))
c4.metric("Rule penalty", f"{rep['rules']['total_penalty']:g} pts")

qs = rep["questions"]
if qs:
    st.subheader(f"Questions ({len(qs)})")
    for q in sorted(qs, key=lambda q: {"important": 0, "check": 1, "info": 2}[q["severity"]]):
        box = st.warning if q["severity"] == "important" else st.info
        box(f"**{q['question']}**  \nAssumed meanwhile: {q['assumed']}"
            + (f"  \nAnswer with: `{q['answer_with']}`" if q["answer_with"] else ""))

with st.expander("How each body is modelled"):
    st.dataframe([{k: b[k] for k in ("name", "kind", "layup", "layup_source", "mass_g",
                                     "E_eff_MPa", "why")} for b in rep["bodies"]],
                 hide_index=True)
with st.expander("Glued joints"):
    st.dataframe(rep["joints"], hide_index=True)
with st.expander("Rules (2027), measured on the solids"):
    st.dataframe([{"§": ln["section"], "rule": ln["title"],
                   "result": "pass" if ln["passed"] else ("fail" if ln["passed"] is False
                                                          else "not checked"),
                   "measured": ln["measured"], "penalty": ln["penalty"], "cause": ln["cause"]}
                  for ln in rep["rules"]["lines"]], hide_index=True)

st.subheader("Crushing test")
cc1, cc2 = st.columns(2)
seed = cc1.number_input("Imperfection / scatter seed (-1 = perfect bridge)", value=0,
                        step=1, min_value=-1)
speed = cc2.select_slider("Simulated jack speed (mm/s)", options=[50, 100, 150, 200],
                          value=100, help="Slower is closer to quasi-static but takes "
                                          "longer; the report says if it was quasi-static.")
if st.button("Run the crushing test", type="primary"):
    bar = st.progress(0.0, text="Starting…")
    out = work / f"run_{seed}_{speed}"
    crush_rep, _ = study.crush(cad, overrides, seed=None if seed < 0 else int(seed),
                               out_dir=out, options=CrushOptions(speed_mm_s=float(speed)),
                               progress=lambda f, m: bar.progress(min(1.0, f), text=m))
    bar.empty()
    st.session_state["crush"] = crush_rep.model_dump(mode="json")

res = st.session_state.get("crush")
if res and res["sha256"].startswith(digest):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted ultimate load F_u", f"{res['Fu_kgf']:.1f} kgf")
    m2.metric("Failure", res["failure_mode"])
    m3.metric("Efficiency η_s", f"{res['efficiency_kgf_per_kg']:.0f} kgf/kg")
    m4.metric("Deflection at F_u", f"{res['deflection_at_Fu_mm']:.1f} mm")
    st.write(res["failure_detail"])
    html = Path(res["artifacts"]["replay_html"]).read_text(encoding="utf-8")
    components.html(html, height=720, scrolling=False)
    fig = go.Figure(go.Scatter(x=res["curve"]["deflection_mm"], y=res["curve"]["load_kgf"],
                               mode="lines", name="simulated"))
    fig.update_layout(xaxis_title="mid-span deflection (mm)", yaxis_title="jack load (kgf)",
                      height=360, margin=dict(l=10, r=10, t=10, b=10))
    tup = st.file_uploader("Compare with your real crush test (CSV, format of "
                           "validation/physical/crush_test_template.csv)", type=["csv"])
    if tup is not None:
        tpath = work / "test.csv"
        tpath.write_bytes(tup.getvalue())
        gap = study.compare(Path(res["artifacts"]["report_json"]), tpath)
        st.metric("Reality-gap score", f"{gap.score_0_100:.0f} / 100")
        st.dataframe([g.model_dump() for g in gap.metrics], hide_index=True)
        for a in gap.advice:
            st.info(a)
        import csv as _csv
        rows = list(_csv.DictReader(tpath.read_text().splitlines()))
        fig.add_trace(go.Scatter(x=[float(r["midspan_deflection_mm"]) for r in rows],
                                 y=[float(r["load_kgf"]) for r in rows], mode="markers",
                                 name="your test"))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Failure sequence")
    st.write("\n".join(f"1. {s}" for s in res["failure_sequence"]) or "nothing broke")
    q = res["quality"]
    st.caption(f"Numerics: time step {q['dt_s'] * 1e6:.2f} µs, {q['steps']} steps, "
               f"{q['wall_s']:.0f} s, added mass {q['added_mass_fraction']:.1%}, peak "
               f"kinetic/work {q['kinetic_to_work_max']:.3f} (quasi-static below 0.1).")
    d1, d2, d3 = st.columns(3)
    d1.download_button("crush.json", Path(res["artifacts"]["report_json"]).read_bytes(),
                       "crush.json")
    d2.download_button("replay.html", html.encode(), "replay.html")
    d3.download_button("curve.csv", Path(res["artifacts"]["curve_csv"]).read_bytes(),
                       "curve.csv")
