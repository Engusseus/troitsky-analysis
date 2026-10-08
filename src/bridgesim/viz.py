"""Figures: Plotly 3D bridge view and load-deflection curve; Matplotlib SFD/BMD for posters.

Colour use: utilisation is a magnitude, so it uses one sequential hue (blue, light to dark).
The governing member is a *state* and uses the reserved "critical" red together with a
text label, so meaning never rests on colour alone. Never import PyVista here: this module
must stay importable in the browser (Pyodide).
"""

from __future__ import annotations

import html
import io
import math
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure

from bridgesim.model import COMBO
from bridgesim.units import n_to_kgf

if TYPE_CHECKING:
    from bridgesim.analysis import AnalysisResult
    from bridgesim.schema import Bridge

# Sequential blue ramp, reference palette steps 250 -> 700 (lightest step stays visible
# against a light surface).
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SEQ_SCALE = [[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)]
CRITICAL = "#d03b3b"
NEUTRAL = "#8a8985"
GHOST = "#c9c8c3"
SERIES = "#2a78d6"
DECK = "#d9b98c"
PLATE = "#52514e"
INK = "#52514e"

VIEWS = ("utilization", "deformed", "buckling", "undeformed")


def _plot_xyz(p: np.ndarray) -> tuple[float, float, float]:
    """Bridge (X, Y up, Z) -> Plotly scene (x, y, z up)."""
    return float(p[0]), float(p[2]), float(p[1])


def default_deform_scale(result: AnalysisResult) -> float:
    """Scale so the largest displacement draws as 8% of the bridge length."""
    dmax = max(
        (math.sqrt(dx * dx + dy * dy + dz * dz) for dx, dy, dz in result.displacements_mm.values()),
        default=0.0,
    )
    xs = [n.x_mm for n in result.bridge.nodes]
    size = (max(xs) - min(xs)) or 1.0
    return 0.08 * size / dmax if dmax > 0 else 1.0


def _quad(x0, x1, z0, z1, y, color, name, opacity, hover):
    return go.Mesh3d(
        x=[x0, x1, x1, x0], y=[z0, z0, z1, z1], z=[y, y, y, y], i=[0, 0], j=[1, 2], k=[2, 3],
        color=color, opacity=opacity, name=name, hoverinfo="text", hovertext=hover,
        showlegend=False, flatshading=True,
    )


def bridge_figure(
    bridge: Bridge,
    result: AnalysisResult | None = None,
    view: str = "utilization",
    deform_scale: float | None = None,
    height: int = 620,
) -> go.Figure:
    """3D view of the bridge.

    ``view``: ``utilization`` colours members by utilisation at the predicted failure load
    (U x F_u,p / P_ref, 1.0 = this member fails at F_u,p); ``deformed`` draws the
    displaced shape at P_ref multiplied by ``deform_scale`` over a grey ghost;
    ``buckling`` draws the first global buckling mode (members coloured by their share of
    the mode's strain energy); ``undeformed`` draws plain geometry.
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    nodes = {n.id: np.array(n.xyz, float) for n in bridge.nodes}
    traces: list[go.BaseTraceType] = []
    title = html.escape(bridge.name)

    pos = dict(nodes)
    buck = result.buckling if result is not None else None
    if view == "buckling" and (buck is None or not buck.mode_mm):
        view = "undeformed"
        title += "  ·  no global buckling mode (nothing in compression)"
    if view == "buckling":
        xs = [n.x_mm for n in bridge.nodes]
        s = deform_scale or 0.06 * ((max(xs) - min(xs)) or 1.0)
        pos = {k: v + s * np.array(buck.mode_mm.get(k, (0, 0, 0))) for k, v in nodes.items()}
        F_cr = n_to_kgf(result.Fu_buckling_N)
        title += (f"  ·  first global buckling mode, F_cr = {F_cr:,.0f} kgf"
                  f" (λ = {buck.lambda_cr:.2f}); shape not to scale")
    if view in ("deformed", "buckling") and result is not None:
        if view == "deformed":
            s = deform_scale or default_deform_scale(result)
            disp = result.displacements_mm
            pos = {k: v + s * np.array(disp.get(k, (0, 0, 0))) for k, v in nodes.items()}
            title += f"  ·  deformed shape at P_ref = {result.P_ref_N:g} N, scale ×{s:,.0f}"
        gx, gy, gz = [], [], []
        for m in bridge.members:
            for p in (nodes[m.i], nodes[m.j]):
                x, y, z = _plot_xyz(p)
                gx.append(x), gy.append(y), gz.append(z)
            gx.append(None), gy.append(None), gz.append(None)
        traces.append(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", name="undeformed",
                                   line=dict(color=GHOST, width=2), hoverinfo="skip",
                                   showlegend=False))

    res = {m.id: m for m in result.members} if result is not None else {}
    lf = result.load_factor if result is not None else 1.0
    color_by_u = view in ("utilization", "buckling") and result is not None
    gov = result.governing_member if result is not None else None
    u_gov = res[gov].U if gov in res else None

    lx, ly, lz, lc = [], [], [], []
    hx, hy, hz, hc, ht = [], [], [], [], []
    crit_x, crit_y, crit_z, crit_ids = [], [], [], []
    for m in bridge.members:
        a, b = pos[m.i], pos[m.j]
        u = res[m.id].U * lf if m.id in res else 0.0  # utilisation at F_u,p
        energy = buck.member_energy.get(m.id, 0.0) if buck is not None else 0.0
        cval = energy if view == "buckling" else u  # what the colour scale shows
        if result is None or view in ("buckling", "undeformed"):
            is_crit = False
        elif result.governing_mode == "global_buckling":
            is_crit = m.id == gov
        else:
            is_crit = (
                u_gov is not None and m.id in res and res[m.id].U >= 0.999 * u_gov
                and result.governing_mode != "deflection"
            )
        target = (crit_x, crit_y, crit_z) if is_crit else (lx, ly, lz)
        for p in (a, b):
            x, y, z = _plot_xyz(p)
            target[0].append(x), target[1].append(y), target[2].append(z)
            if target[0] is lx:
                lc.append(cval)
        target[0].append(None), target[1].append(None), target[2].append(None)
        if target[0] is lx:
            lc.append(cval)
        if is_crit:
            crit_ids.append(m.id)
        mid = _plot_xyz((a + b) / 2)
        hx.append(mid[0]), hy.append(mid[1]), hz.append(mid[2]), hc.append(cval)
        if m.id in res:
            r = res[m.id]
            ht.append(
                f"<b>{html.escape(m.id)}</b> ({m.group})<br>{html.escape(r.section_label)}, "
                f"L = {r.L_mm:.0f} mm"
                f"<br>N at P_ref = {r.N_N:+.1f} N ({'tension' if r.N_N >= 0 else 'compression'})"
                f"<br>M_z = {r.Mz_Nmm:.0f} N·mm, M_y = {r.My_Nmm:.0f} N·mm"
                f"<br>U at P_ref = {r.U:.3f}; at F_u,p = {u:.2f}"
                f"<br>Governing check: {r.util.mode_label}"
                + (f"<br>Fails alone at {n_to_kgf(result.P_ref_N / r.U):.0f} kgf"
                   if r.U > 0 else "")
                + (f"<br>Share of buckling-mode energy: {energy:.2f}"
                   if view == "buckling" else "")
            )
        else:
            ht.append(f"<b>{html.escape(m.id)}</b> ({m.group})")

    line_kw: dict = dict(width=5)
    if color_by_u:
        line_kw.update(color=lc, colorscale=SEQ_SCALE, cmin=0.0, cmax=1.0, showscale=True,
                       colorbar=dict(title=dict(text="Utilisation<br>at F_u,p" if view != "buckling"
                                         else "Share of<br>mode energy"), thickness=14,
                                     len=0.6, tickvals=[0, 0.25, 0.5, 0.75, 1.0]))
    else:
        line_kw.update(color=SERIES if view in ("deformed", "buckling") else NEUTRAL)
    traces.append(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=line_kw, hoverinfo="skip",
                               name="members", showlegend=False))
    if crit_ids:
        traces.append(go.Scatter3d(x=crit_x, y=crit_y, z=crit_z, mode="lines",
                                   line=dict(color=CRITICAL, width=9), hoverinfo="skip",
                                   name="governing", showlegend=False))
        first = bridge.members[[m.id for m in bridge.members].index(crit_ids[0])]
        lab = _plot_xyz((pos[first.i] + pos[first.j]) / 2)
        # Label at table level under the member, where the clear-span box keeps space free.
        traces.append(go.Scatter3d(
            x=[lab[0]], y=[lab[1]], z=[15.0], mode="text",
            text=[html.escape(f"governs: {', '.join(crit_ids[:2])}"
                              f"{'…' if len(crit_ids) > 2 else ''}")],
            textfont=dict(color=CRITICAL, size=13), hoverinfo="skip", showlegend=False))
    traces.append(go.Scatter3d(
        x=hx, y=hy, z=hz, mode="markers", hovertext=ht, hoverinfo="text", name="",
        marker=dict(size=4, color=hc if color_by_u else NEUTRAL, colorscale=SEQ_SCALE,
                    cmin=0, cmax=1, opacity=0.9 if color_by_u else 0.35),
        showlegend=False))

    # Supports, deck and crusher plate.
    sx, sy, sz, st = [], [], [], []
    for nid in bridge.supports.pinned + bridge.supports.roller:
        x, y, z = _plot_xyz(pos[nid])
        sx.append(x), sy.append(y), sz.append(z)
        kind = "pinned (DX, DY, DZ)" if nid in bridge.supports.pinned else "roller (DY, DZ)"
        st.append(f"{html.escape(nid)}: {kind}")
    traces.append(go.Scatter3d(x=sx, y=sy, z=sz, mode="markers", hovertext=st, hoverinfo="text",
                               marker=dict(size=7, symbol="diamond", color=INK),
                               showlegend=False, name="supports"))
    d = bridge.deck
    zc = d.z_center_mm
    traces.append(_quad(d.x_start_mm, d.x_end_mm, zc - d.clear_width_mm / 2,
                        zc + d.clear_width_mm / 2, d.top_elevation_mm, DECK, "deck", 0.35,
                        f"Deck (non-structural): {d.length_mm:.0f} × {d.clear_width_mm:.0f} mm, "
                        f"top at {d.top_elevation_mm:.0f} mm"))
    from bridgesim.loads import plate_placement

    pl = plate_placement(bridge)
    py = d.top_elevation_mm + 1.0
    if view in ("deformed", "buckling") and result is not None and result.delta_node in pos:
        py += (pos[result.delta_node] - nodes[result.delta_node])[1]
    traces.append(_quad(pl.x_start_mm, pl.x_end_mm, zc - pl.width_mm / 2, zc + pl.width_mm / 2,
                        py, PLATE, "plate", 0.9,
                        f"Crusher plate {bridge.load.plate_length_mm:g} × "
                        f"{bridge.load.plate_width_mm:g} mm at x = {pl.x_center_mm:.0f} mm"))
    traces.append(go.Cone(x=[pl.x_center_mm], y=[zc], z=[py + 45], u=[0], v=[0], w=[-40],
                          sizemode="absolute", sizeref=30, anchor="tip", showscale=False,
                          colorscale=[[0, PLATE], [1, PLATE]], hoverinfo="skip"))

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)), height=height,
        margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="X along bridge (mm)", showbackground=False),
            yaxis=dict(title="Z across (mm)", showbackground=False),
            zaxis=dict(title="Y up (mm)", showbackground=False),
            camera=dict(eye=dict(x=0.8, y=-2.35, z=1.05), up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=-0.08)),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def load_deflection_figure(result: AnalysisResult, height: int = 300) -> go.Figure:
    """Linear load-deflection estimate up to F_u,p, with the 50 mm deflection limit."""
    lim = result.deflection_limit_mm
    Fu = result.Fu_pred_kgf
    dFu = result.delta_at_Fu_mm
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, dFu], y=[0, Fu], mode="lines+markers", line=dict(color=SERIES, width=2),
        marker=dict(size=[0, 9], color=SERIES), name="linear estimate",
        hovertemplate="δ = %{x:.2f} mm<br>F = %{y:.1f} kgf<extra>linear estimate</extra>"))
    fig.add_annotation(x=dFu, y=Fu, text=f"F_u,p = {Fu:.0f} kgf ({result.governing_label})",
                       showarrow=False, yshift=14, xanchor="right" if dFu > lim * 0.5 else "left",
                       font=dict(size=12))
    fig.add_vline(x=lim, line=dict(color=INK, dash="dash", width=1))
    fig.add_annotation(x=lim, y=Fu * 0.5, text=f"{lim:g} mm failure limit (§12.5)",
                       showarrow=False, textangle=-90, xshift=-10, font=dict(size=11, color=INK))
    fig.update_layout(
        title=dict(text="Load–deflection at mid-span (linear estimate, v0.1)", font=dict(size=14)),
        height=height, margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
        xaxis=dict(title="Mid-span deflection (mm)", range=[0, max(lim, dFu) * 1.08],
                   zeroline=False),
        yaxis=dict(title="Load (kgf)", range=[0, Fu * 1.15], zeroline=False),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# --------------------------------------------------------------------------- SFD / BMD


def global_shear_moment(result: AnalysisResult, load_N: float | None = None, n: int = 600):
    """Whole-bridge (beam analogy) shear force and bending moment along X.

    Free-body of everything left of a vertical cut at x, using the analysed vertical
    support reactions R_k and the applied deck loads F_k (downward), scaled linearly to
    ``load_N``:

        V(x) = sum_{x_k <= x} (R_k - F_k),    M(x) = sum_{x_k <= x} (R_k - F_k)(x - x_k)

    Returns (x, V, M) with V in N and M in N·mm.
    """
    load = result.Fu_pred_N if load_N is None else load_N
    k = load / result.P_ref_N
    nodes = result.bridge.node_map()
    pts = [(nodes[r.node].x_mm, r.FY_N * k) for r in result.reactions]
    pts += [(nodes[nid].x_mm, -F * k) for nid, F in result.nodal_loads_N.items()]
    xs = [p[0] for p in pts]
    x = np.linspace(min(xs), max(xs), n)
    x = np.unique(np.concatenate([x, np.array(xs) - 1e-6, np.array(xs) + 1e-6]))
    V = np.zeros_like(x)
    M = np.zeros_like(x)
    for xk, Fk in pts:
        on = x >= xk
        V[on] += Fk
        M[on] += Fk * (x[on] - xk)
    return x, V, M


def _mpl_text(text: str) -> str:
    """Escape '$' so Matplotlib never parses YAML-derived names as MathText."""
    return text.replace("$", r"\$")


def _style_axis(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.axhline(0, color=INK, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)


def global_sfd_bmd_figure(result: AnalysisResult, load_N: float | None = None) -> Figure:
    """Poster-ready whole-bridge SFD/BMD (rulebook §12.4) at ``load_N`` (default F_u,p)."""
    load = result.Fu_pred_N if load_N is None else load_N
    x, V, M = global_shear_moment(result, load)
    fig = Figure(figsize=(8, 5.2), dpi=150, layout="constrained")
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    ax1.plot(x, V, color=SERIES, linewidth=2)
    ax1.fill_between(x, V, color=SERIES, alpha=0.12, linewidth=0)
    _style_axis(ax1, "Shear V (N)")
    ax2.plot(x, M / 1000.0, color=SERIES, linewidth=2)
    ax2.fill_between(x, M / 1000.0, color=SERIES, alpha=0.12, linewidth=0)
    _style_axis(ax2, "Moment M (N·m)")
    ax2.set_xlabel("x along bridge (mm)", color=INK)
    i = int(np.argmax(np.abs(M)))
    ax2.plot([x[i]], [M[i] / 1000], "o", color=SERIES, markersize=5)
    ax2.text(0.02, 0.88, f"M_max = {M[i] / 1000:.1f} N·m at x = {x[i]:.0f} mm",
             transform=ax2.transAxes, fontsize=9, color=INK)
    j = int(np.argmax(np.abs(V)))
    ax1.text(0.02, 0.88 if V[j] < 0 else 0.06, f"|V|_max = {abs(V[j]):.0f} N",
             transform=ax1.transAxes, fontsize=9, color=INK)
    fig.suptitle(
        f"{_mpl_text(result.bridge.name)}: whole-bridge shear and bending moment\n"
        f"at {n_to_kgf(load):.0f} kgf ({load:.0f} N) crusher load, sagging moment positive",
        fontsize=11, color="#0b0b0b")
    return fig


def member_chain_diagrams(
    result: AnalysisResult, member_ids: list[str], load_N: float | None = None
):
    """Axial force, shear V_y and moment M_z along a chain of members (or one member).

    Values from the solved Pynite model at P_ref, scaled linearly to ``load_N``. Members are
    laid end to end in the given order; x is the cumulative distance along the chain.
    Axial force is tension-positive; shear and moment use Pynite's local sign convention.
    """
    load = result.Fu_pred_N if load_N is None else load_N
    k = load / result.P_ref_N
    model = result.fe_model
    xs, N, V, M, bounds = [], [], [], [], [0.0]
    offset = 0.0
    for mid in member_ids:
        pm = model.members[mid]
        L = pm.L()
        xa = np.linspace(0, L, 41)
        N.append(-pm.axial_array(41, COMBO, x_array=xa)[1] * k)
        V.append(pm.shear_array("Fy", 41, COMBO, x_array=xa)[1] * k)
        M.append(pm.moment_array("Mz", 41, COMBO, x_array=xa)[1] * k)
        xs.append(xa + offset)
        offset += L
        bounds.append(offset)
    return np.concatenate(xs), np.concatenate(N), np.concatenate(V), np.concatenate(M), bounds


def member_diagrams_figure(
    result: AnalysisResult, member_ids: list[str], title: str, load_N: float | None = None
) -> Figure:
    """Poster-ready axial/shear/moment diagrams for a member or a chain (e.g. a chord)."""
    load = result.Fu_pred_N if load_N is None else load_N
    x, N, V, M, bounds = member_chain_diagrams(result, member_ids, load)
    fig = Figure(figsize=(8, 6.4), dpi=150, layout="constrained")
    axs = fig.subplots(3, 1, sharex=True)
    for ax, y, lab in ((axs[0], N, "Axial N (N)\n+ tension"), (axs[1], V, "Shear V_y (N)"),
                       (axs[2], M / 1000, "Moment M_z (N·m)")):
        ax.plot(x, y, color=SERIES, linewidth=2)
        ax.fill_between(x, y, color=SERIES, alpha=0.12, linewidth=0)
        for b in bounds[1:-1]:
            ax.axvline(b, color="#e4e3df", linewidth=0.8)
        _style_axis(ax, lab)
    axs[2].set_xlabel("distance along member(s) (mm); thin lines = joints", color=INK)
    fig.suptitle(f"{_mpl_text(title)} at {n_to_kgf(load):.0f} kgf crusher load "
                 "(local member axes)",
                 fontsize=11, color="#0b0b0b")
    return fig


def figure_png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    return buf.getvalue()


def chain_for_group(bridge: Bridge, group: str, side: str = "n") -> list[str]:
    """Members of a chord-like ``group`` in one truss plane, ordered along X.

    ``side`` selects the plane: "n" = negative Z, "f" = positive Z.
    """
    nodes = bridge.node_map()
    sel = []
    for m in bridge.members:
        if m.group != group:
            continue
        zc = 0.5 * (nodes[m.i].z_mm + nodes[m.j].z_mm)
        if (zc < 0) == (side == "n"):
            sel.append((min(nodes[m.i].x_mm, nodes[m.j].x_mm), m.id))
    return [mid for _, mid in sorted(sel)]
