""""Assumptions, Method, Results" report for Design Validation (rulebook §10.2).

§10.2 asks that judges can see what was analysed or tested, the assumptions made, the
process followed and what the results show, and that any software used is identified.
The report follows that order. Output: Markdown (with $$ LaTeX) or standalone HTML.
"""

from __future__ import annotations

import datetime as _dt
import html
import math
from typing import TYPE_CHECKING

from bridgesim import PYNITE_VERSION, __version__
from bridgesim.checks import MODE_LABELS
from bridgesim.units import N_PER_KGF, n_to_kgf

if TYPE_CHECKING:
    from bridgesim.analysis import AnalysisResult
    from bridgesim.rules import RulesReport

#: Modelling assumptions surfaced in the UI and report (ASSUMPTIONS.md has the long form).
MODEL_ASSUMPTIONS: list[str] = [
    "Linear-elastic, small-displacement 3D frame analysis (no P-Delta, no initial "
    "imperfections, no joint slip). The predicted load is the load at which the FIRST "
    "member, joint, global buckling or the deflection limit is reached; load "
    "redistribution after first failure is ignored.",
    "Every member is a prismatic Euler-Bernoulli frame element on its centre line. "
    "Laminated sticks act as one solid rectangle (perfect glue lines); splices along "
    "members longer than one stick (115 mm) are not weakened.",
    "Joints are rigid by default (glued joints behave closer to rigid than pinned). A "
    "'pinned' switch releases in-plane moments of web members and bracing for comparison.",
    "Member buckling: Euler load over the full member length about the weaker axis, with "
    "the effective-length factor K stated above. Global (system) buckling: linear "
    "eigenvalue analysis of the whole frame, with "
    "pier bases held by friction. Both assume perfectly straight sticks, so they are "
    "unconservative for crooked sticks (imperfections arrive in v0.2).",
    "Member utilisation is the linear sum |N|/N_R + |M_y|/M_R,y + |M_z|/M_R,z (no "
    "amplification). Member shear (1.5 V/A <= f_v) and glued-joint shear are separate "
    "checks. Torsion is ignored.",
    "Glued joint (placeholder): capacity = tau_g x overlap x member width x faces, "
    "checked against the resultant member-end force. Chords are treated as continuous.",
    "The deck is non-structural: it only transfers the crusher-plate load to the "
    "floor-beam centre nodes as simply supported strips. Its stiffness is ignored.",
    "Supports rest on the platform without anchorage (§8.3): one end restrains DX, DY, DZ, "
    "the other DY, DZ; rotations free. Hold-down or large horizontal reactions are "
    "reported as warnings, never hidden.",
    "Self-weight is ignored (a few percent of the failure load).",
    "Mass = density x member volumes (centre-line lengths) + deck plate, plus a glue mass "
    "fraction. Stick count = wood volume / 2300 mm^3.",
]


def _fmt_load(N: float) -> str:
    if math.isnan(N):
        return "NOT EVALUATED (see warnings)"
    if not math.isfinite(N):
        return "∞"
    return f"{N:,.0f} N ({n_to_kgf(N):,.1f} kgf)"


def _sections(result: AnalysisResult, rules: RulesReport | None):
    """Yield (heading, blocks) where a block is ('p', text) | ('ul', items) |
    ('table', headers, rows) | ('math', tex)."""
    b, mat, r = result.bridge, result.material, result
    gen = b.metadata.get("params", {})
    meta = [
        f"Bridge: **{b.name}** ({len(b.nodes)} nodes, {len(b.members)} members, "
        f"joints {b.joint_fixity})",
        f"Software: **bridgesim {__version__}** (MIT, open source) using the "
        f"**Pynite {PYNITE_VERSION}** 3D frame finite-element solver (MIT).",
        f"Rules file: {rules.ruleset if rules else 'not evaluated'}. "
        "The official rulebook is authoritative.",
        f"Generated: {_dt.date.today().isoformat()}",
    ]
    yield "Summary", [
        ("ul", meta),
        ("table", ["Quantity", "Value"], [
            ["Predicted ultimate load F_u,p", _fmt_load(r.Fu_pred_N)],
            ["Governing", f"{r.governing_label}"
             + (f" in member {r.governing_member}" if r.governing_member else "")],
            ["Strength-based load F_u,strength", _fmt_load(r.Fu_strength_N)],
            [f"Deflection-based load F_u,δ ({r.deflection_limit_mm:g} mm)",
             _fmt_load(r.Fu_deflection_N)],
            ["Global elastic buckling load F_cr", _fmt_load(r.Fu_buckling_N)
             + (f", mode led by {r.buckling.key_member}" if r.buckling and r.buckling.key_member
                else "")],
            ["Mid-span deflection at F_u,p", f"{r.delta_at_Fu_mm:.1f} mm "
             f"(limit {r.deflection_limit_mm:g} mm)"],
            ["Estimated mass m", f"{r.mass.total_kg:.2f} kg (~{r.mass.stick_count} sticks)"],
            ["Structural efficiency η_s = F_u/m", f"{r.efficiency:.1f} kgf/kg"],
            ["Rule penalties", "not evaluated" if rules is None else
             f"{-rules.total_penalty or 0:g} pts; bans: "
             + (", ".join(f"§{x}" for x in rules.bans) or "none")],
        ]),
        ("p", "**All material values marked 'assumed' are placeholders, not test data. "
              "The predicted load is only as good as these numbers.**"
         if mat.assumed_keys() else "All material values are marked as measured."),
    ]

    geo = [["Generator", str(b.metadata.get("generator", "custom"))]]
    for k in ("span_mm", "n_panels", "truss_height_mm", "deck_top_elevation_mm",
              "deck_clear_width_mm"):
        if k in gen:
            geo.append([k, f"{gen[k]:g}" if isinstance(gen[k], float) else str(gen[k])])
    sec_rows = []
    for s in b.sections:
        pr = s.props(mat.stick)
        n_mem = sum(1 for m in b.members if m.section == s.id)
        sec_rows.append([s.id, s.label(), f"{pr.b_mm:g} × {pr.d_mm:g}", f"{pr.A_mm2:.0f}",
                         f"{pr.I_min_mm4:.0f}", str(n_mem)])
    pl = r.plate
    yield "1. What was analysed", [
        ("p", "A 3D frame model of the whole bridge (both trusses, floor beams, bracing and "
              "piers) under the Crushing Day load case: a point load at mid-span spread by "
              f"the {b.load.plate_length_mm:g} mm × {b.load.plate_width_mm:g} mm crusher "
              f"plate (§12.5), centred at x = {pl.x_center_mm:.0f} mm."),
        ("table", ["Parameter", "Value"], geo),
        ("table", ["Section", "Make-up", "b × d (mm)", "A (mm²)", "I_min (mm⁴)", "Members"],
         sec_rows),
    ]

    mrows = [[k, f"{p.value:g}", p.source, p.note] for k, p in mat.props().items()]
    yield "2. Assumptions", [
        ("p", f"Material: **{mat.name}**. Stick {mat.stick.length_mm:g} × "
              f"{mat.stick.width_mm:g} × {mat.stick.thickness_mm:g} mm."),
        ("table", ["Property", "Value", "Source", "Note"], mrows),
        ("ul", model_assumptions(r)),
    ]

    loads = ", ".join(f"{nid}: {F:.1f} N" for nid, F in r.nodal_loads_N.items())
    yield "3. Method", [
        ("p", "1. Load path. The plate load is a uniform line load over its length. The deck "
              "spans as simply supported strips between floor beams, so each strip passes "
              "its share to the two adjacent floor-beam centre nodes by the lever rule:"),
        ("math", r"R = \frac{P_{ref}}{L_{plate}}(b-a),\quad "
                 r"F_i \mathrel{+}= R\,\frac{x_{i+1}-\bar{x}}{p},\quad "
                 r"F_{i+1} \mathrel{+}= R\,\frac{\bar{x}-x_i}{p}"),
        ("p", f"At P_ref = {r.P_ref_N:g} N: {loads}."),
        ("p", "2. Linear static 3D frame analysis (Pynite) gives the axial force N, shears and "
              "bending moments M_y, M_z along every member and all node displacements."),
        ("p", "3. Capacities (wood parallel to grain):"),
        ("math", r"N_{t,R} = f_t A,\quad N_{c,R} = \min\!\left(f_c A,\ "
                 r"\frac{\pi^2 E I_{min}}{(KL)^2}\right),\quad M_{R} = f_b S"),
        ("p", "4. Utilisation of each member at P_ref (linear interaction):"),
        ("math", r"U_i = \frac{|N|}{N_R} + \frac{|M_y|}{M_{R,y}} + \frac{|M_z|}{M_{R,z}}"),
        ("p", "5. Load-factor method. The analysis is linear, so every demand scales with "
              "load and the predicted ultimate load is the smallest of"),
        ("math", r"F_{u,strength} = \frac{P_{ref}}{\max_i U_i},\qquad "
                 r"F_{u,\delta} = P_{ref}\,\frac{" + f"{r.deflection_limit_mm:g}"
                 + r"\ \text{mm}}{\delta_{ref}},\qquad "
                 r"F_{u,p} = \min(F_{u,strength},\ F_{u,\delta})"),
        ("p", "6. Global elastic buckling (instability, §12.5): the smallest load factor λ for "
              "which the structure loses stiffness, with K the elastic and K_g the geometric "
              "stiffness at P_ref. Pier bases are held by friction during buckling:"),
        ("math", r"(K + \lambda K_g)\,\phi = 0,\qquad F_{cr} = \lambda_{cr} P_{ref},\qquad "
                 r"F_{u,p} = \min(F_{u,strength},\ F_{u,\delta},\ F_{cr})"),
        ("p", f"with δ_ref = {r.delta_ref_mm:.3f} mm at node {r.delta_node}. "
              f"1 kgf = {N_PER_KGF} N; efficiency η_s = F_u,p [kgf] / m [kg] (§12.6)."),
    ]

    crit = [[m.id, m.group, m.section_label, f"{m.N_N * r.load_factor:+.0f}",
             f"{m.U * r.load_factor:.2f}", MODE_LABELS[m.mode],
             f"{n_to_kgf(r.P_ref_N / m.U):.0f}" if m.U else "∞"]
            for m in r.critical_members(10)]
    reac = [[x.node, f"{x.FX_N * r.load_factor:.1f}", f"{x.FY_N * r.load_factor:.1f}",
             f"{x.FZ_N * r.load_factor:.1f}"] for x in r.reactions]
    blocks = [
        ("p", f"Predicted ultimate load **F_u,p = {_fmt_load(r.Fu_pred_N)}**, governed by "
              f"**{r.governing_label}**"
              + (f" in member **{r.governing_member}**." if r.governing_member else ".")),
        ("p", "Ten most critical members (values at F_u,p; utilisation 1.00 = fails):"),
        ("table", ["Member", "Group", "Section", "N at F_u,p (N, + tension)", "U at F_u,p",
                   "Governing check", "Fails alone at (kgf)"], crit),
        ("p", "Support reactions at F_u,p (N):"),
        ("table", ["Node", "FX", "FY (up +)", "FZ"], reac),
    ]
    if r.warnings:
        blocks.append(("ul", [f"⚠ {w}" for w in r.warnings]))
    if rules is not None:
        rows = [[{True: "✓", False: "✗", None: "·"}[x.passed], f"§{x.section}", x.title,
                 x.measured_text, x.limit, f"{-x.penalty:g}" if x.penalty else "0",
                 ", ".join(f"§{y}" for y in x.bans) + (" DQ risk" if x.disqualification else ""),
                 x.note] for x in rules.results]
        blocks += [
            ("p", f"Rule check ({rules.ruleset}): total {-rules.total_penalty or 0:g} points; "
                  f"bans: {', '.join('§' + x for x in rules.bans) or 'none'}."),
            ("table", ["", "Section", "Rule", "Measured", "Limit", "Penalty", "Bans", "Note"],
             rows),
        ]
    yield "4. Results", blocks

    yield "5. Limitations and how to strengthen this validation", [
        ("ul", [
            "Replace every 'assumed' material value with your own test results (3-point "
            "bending for E, tension/compression and glued-joint tests to failure).",
            "Crush-test a sub-assembly or a short prototype and compare with this model.",
            "This is a first-failure linear estimate: real bridges can carry more after a "
            "brace buckles (redistribution), or less (imperfections, poor joints).",
            "Report the predicted load as a range informed by material scatter.",
        ]),
    ]


def model_assumptions(result: AnalysisResult) -> list[str]:
    """MODEL_ASSUMPTIONS plus lines stating the material and crushing inputs actually used."""
    m = result.material
    load = result.bridge.load
    vals = (f"E = {m.E_MPa.value:g}, f_t = {m.f_t_MPa.value:g}, f_c = {m.f_c_MPa.value:g}, "
            f"f_b = {m.f_b_MPa.value:g}, f_v = {m.f_v_MPa.value:g} MPa; density "
            f"{m.density_kg_m3.value:g} kg/m^3; glue shear {m.glue.tau_g_MPa.value:g} MPa")
    assumed = m.assumed_keys()
    if assumed:
        src = (f"{len(assumed)} of {len(m.props())} values are ASSUMED placeholders, not test "
               f"data ({', '.join(assumed)})")
    else:
        src = "all values are marked as measured by the team"
    crusher = (
        f"Crusher plate {load.plate_length_mm:g} mm x {load.plate_width_mm:g} mm, uniform, "
        f"centred at x = {result.plate.x_center_mm:.0f} mm (§12.5); failure at "
        f"{result.deflection_limit_mm:g} mm mid-span deflection. The load is applied at the "
        "floor-beam centres (z = 0), which over-estimates floor-beam bending."
    )
    ks = sorted({mem.K for mem in result.bridge.members})
    if ks == [1.0]:
        k_line = "Effective-length factor K = 1 for every member (pin-ended columns)."
    else:
        custom = [f"{mem.id} (K = {mem.K:g})" for mem in result.bridge.members if mem.K != 1.0]
        shown = ", ".join(custom[:12]) + (f", … ({len(custom)} in total)" if len(custom) > 12
                                          else "")
        k_line = f"Effective-length factor K = 1 except: {shown}."
    return [f"Material “{m.name}”: {vals}. {src}.", crusher, k_line, *MODEL_ASSUMPTIONS]


# --------------------------------------------------------------------------- renderers


def _md_table(headers, rows) -> str:
    def esc(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(esc(h) for h in headers) + " |",
           "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def to_markdown(result: AnalysisResult, rules: RulesReport | None = None) -> str:
    parts = [f"# Design validation: {result.bridge.name}",
             "_Assumptions, Method, Results (Troitsky rulebook §10.2)_"]
    for heading, blocks in _sections(result, rules):
        parts.append(f"## {heading}")
        for blk in blocks:
            kind = blk[0]
            if kind == "p":
                parts.append(blk[1])
            elif kind == "ul":
                parts.append("\n".join(f"- {it}" for it in blk[1]))
            elif kind == "table":
                parts.append(_md_table(blk[1], blk[2]))
            elif kind == "math":
                parts.append(f"$$\n{blk[1]}\n$$")
    return "\n\n".join(parts) + "\n"


def _inline(s: str) -> str:
    s = html.escape(s)
    while "**" in s:
        s = s.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
    return s


def to_html(result: AnalysisResult, rules: RulesReport | None = None) -> str:
    body = [f"<h1>Design validation: {html.escape(result.bridge.name)}</h1>",
            "<p class='sub'>Assumptions, Method, Results (Troitsky rulebook §10.2)</p>"]
    for heading, blocks in _sections(result, rules):
        body.append(f"<h2>{html.escape(heading)}</h2>")
        for blk in blocks:
            kind = blk[0]
            if kind == "p":
                body.append(f"<p>{_inline(blk[1])}</p>")
            elif kind == "ul":
                body.append("<ul>" + "".join(f"<li>{_inline(i)}</li>" for i in blk[1]) + "</ul>")
            elif kind == "table":
                head = "".join(f"<th>{html.escape(str(h))}</th>" for h in blk[1])
                rows = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r)
                               + "</tr>" for r in blk[2])
                body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
            elif kind == "math":
                body.append(f"<div class='math'>\\[{html.escape(blk[1])}\\]</div>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Design validation: {html.escape(result.bridge.name)}</title>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 980px; margin: 2rem auto;
       padding: 0 16px; color: #1b1b1a; background: #fcfcfb; line-height: 1.45; }}
h1 {{ margin-bottom: 0; }} .sub {{ color: #52514e; margin-top: .2rem; }}
h2 {{ border-bottom: 1px solid #e4e3df; padding-bottom: .2rem; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: .8rem 0; font-size: .9rem; }}
th, td {{ border: 1px solid #e4e3df; padding: 4px 8px; text-align: left; vertical-align: top; }}
th {{ background: #f0efec; }} .math {{ overflow-x: auto; }}
@media print {{ body {{ margin: 0; }} }}
</style></head><body>
{chr(10).join(body)}
</body></html>
"""
