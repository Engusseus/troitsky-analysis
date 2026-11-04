"""Replay of a crush simulation as a self-contained interactive HTML page (three.js).

The page shows the real CAD bodies, deformed exactly as the simulation moved them: every
surface point of a body is attached ("skinned") to the structural nodes of that body (two
nodes along a member, three corners of a shell element, or the rigid body's single node)
with its offset in the node frames, so members bend, plates dish and pieces fly apart when
a member or joint breaks. Glue lines are drawn as points coloured by their damage, floss
as lines, plus the loading plate, the jack and the base platform.

Playback can follow *real test time* (the competition jack speed from
``materials/simulation.yaml``) or run faster, and is synchronised with the load-deflection
curve and the event log.
"""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bridgesim.cad import CadBody
from bridgesim.sim.crush import KGF, CrushResult
from bridgesim.sim.model import Model

KIND_COLOUR = {"bar": "#c8955a", "shell": "#d9b27c", "block": "#a8794a", "cable": "#f2f2f2"}


def _subdivide(V: np.ndarray, T: np.ndarray, max_edge: float) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(6):
        e = np.max(np.stack([np.linalg.norm(V[T[:, (k + 1) % 3]] - V[T[:, k]], axis=1)
                             for k in range(3)]), axis=0)
        if e.max() <= max_edge:
            break
        big = e > max_edge
        Tb = T[big]
        mids: dict[tuple[int, int], int] = {}
        newV = [V]
        out = [T[~big]]
        rows = []
        for tri in Tb:
            m3 = []
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(a, b), max(a, b))
                if key not in mids:
                    mids[key] = len(V) + len(newV) - 1
                    newV.append(((V[a] + V[b]) / 2)[None])
                m3.append(mids[key])
            a, b, c = tri
            ab, bc, ca = m3
            rows += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        out.append(np.array(rows, int))
        V = np.vstack(newV)
        T = np.vstack(out)
    return V, T


def build_skin(model: Model, bodies: list[CadBody], max_edge: float = 20.0) -> dict:
    """Render meshes and, for every vertex, up to 3 (node, weight, offset) influences."""
    meshes = []
    for bi, (b, info) in enumerate(zip(bodies, model.bodies, strict=True)):
        if info.kind in ("ignore", "cable") or not info.nodes:
            continue
        V, T = b.mesh.vertices, b.mesh.triangles
        if info.kind in ("bar", "shell"):
            V, T = _subdivide(V, T, max_edge)
        nodes = np.array(info.nodes)
        XN = model.X[nodes]
        nv = len(V)
        idx = np.zeros((nv, 3), int)
        w = np.zeros((nv, 3))
        if info.kind == "block" or len(nodes) == 1:
            idx[:, 0] = nodes[0]
            w[:, 0] = 1.0
        elif info.kind == "bar":
            a = XN[0]
            u = XN[-1] - a
            L = np.linalg.norm(u)
            u /= L
            s = np.clip((V - a) @ u, 0, L)
            sn = (XN - a) @ u
            k = np.clip(np.searchsorted(sn, s) - 1, 0, len(nodes) - 2)
            t = np.clip((s - sn[k]) / np.maximum(sn[k + 1] - sn[k], 1e-9), 0, 1)
            idx[:, 0], idx[:, 1] = nodes[k], nodes[k + 1]
            w[:, 0], w[:, 1] = 1 - t, t
        else:  # shell: nearest 3 nodes, inverse-distance weights
            tree = cKDTree(XN)
            dd, kk = tree.query(V, k=min(3, len(nodes)))
            dd = np.atleast_2d(dd)
            kk = np.atleast_2d(kk)
            ww = 1 / np.maximum(dd, 0.5) ** 2
            ww /= ww.sum(1, keepdims=True)
            idx[:, : kk.shape[1]] = nodes[kk]
            w[:, : kk.shape[1]] = ww
        off = V[:, None, :] - model.X[idx]  # (nv, 3, 3) offsets in the (identity) node frames
        meshes.append({"body": bi, "name": b.name, "kind": info.kind, "V": V, "T": T,
                       "idx": idx, "w": w, "off": off})
    return {"meshes": meshes}


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def replay_html(res: CrushResult, bodies: list[CadBody], title: str = "Crush test replay",
                max_frames: int = 140) -> str:
    m = res.model
    skin = build_skin(m, bodies)
    frames = res.frames
    if len(frames) > max_frames:
        sel = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = [frames[i] for i in sel]
    nN = m.n_nodes
    # node displacement (0.01 mm, int16 if possible) and quaternions (int16)
    U = np.stack([f.x - m.X for f in frames]).astype(np.float32)
    scale = 0.01 if np.abs(U).max() < 320 else 0.05
    Uq = np.clip(np.round(U / scale), -32767, 32767).astype(np.int16)
    Q = np.stack([f.q for f in frames])
    Qq = np.clip(np.round(Q * 32767), -32767, 32767).astype(np.int16)
    # meshes
    mesh_js = []
    for me in skin["meshes"]:
        # compact: node index uint32, weights 0-255, offsets in 0.01 mm (int16)
        off = np.clip(np.round(me["off"] * 100), -32767, 32767).astype(np.int16)
        mesh_js.append({
            "name": me["name"], "kind": me["kind"], "body": me["body"],
            "T": _b64(me["T"].astype(np.uint32)),
            "idx": _b64(me["idx"].astype(np.uint32)),
            "w": _b64(np.clip(np.round(me["w"] * 255), 0, 255).astype(np.uint8)),
            "off": _b64(off), "nv": len(me["V"]), "nt": len(me["T"])})
    # body state per frame: 0 ok, 1 damaged glue, 2 cracked/broken
    nb = len(m.bodies)
    bstate = np.zeros((len(frames), nb), np.uint8)
    for fi, f in enumerate(frames):
        if len(m.b_body):
            cracked = f.beam_state > 0
            bstate[fi, np.unique(m.b_body[cracked])] = 2
        if len(m.s_body):
            bstate[fi, np.unique(m.s_body[f.shell_state > 0])] = 2
        if len(m.patches):
            for pid in np.where(f.patch_damage > 0.5)[0]:
                p = m.patches[pid]
                for bb in (p.body_a, p.body_b):
                    bstate[fi, bb] = max(bstate[fi, bb], 1)
    # glue points
    ga = m.c_a
    gp_node = m.a_node[ga]
    gp_off = m.a_r0[ga]
    gdmg = np.stack([f.patch_damage[m.c_patch] if len(m.c_patch) else np.zeros(0)
                     for f in frames]) if len(frames) else np.zeros((0, 0))
    gdmg = np.clip(np.round(gdmg * 255), 0, 255).astype(np.uint8)
    # cables
    cab = [{"a": int(m.a_node[a]), "ao": m.a_r0[a].tolist(), "b": int(m.a_node[b]),
            "bo": m.a_r0[b].tolist()} for a, b in zip(m.k_a, m.k_b, strict=True)]
    cab_ok = np.stack([f.cable_ok for f in frames]).astype(np.uint8) if len(m.k_a) else \
        np.zeros((len(frames), 0), np.uint8)
    rate = float(m.params.get("test_rate_mm_per_min", 10.0))
    data = {
        "title": title,
        "nN": nN, "X": _b64(m.X.astype(np.float32)), "scale": scale,
        "U": _b64(Uq), "Q": _b64(Qq), "nF": len(frames),
        "t": [f.t for f in frames], "travel": [f.travel_mm for f in frames],
        "force": [f.force_N / KGF for f in frames], "defl": [f.deflection_mm for f in frames],
        "meshes": mesh_js, "bstate": _b64(bstate), "nb": nb,
        "glue_node": _b64(gp_node.astype(np.uint32)), "glue_off": _b64(gp_off.astype(np.float32)),
        "glue_dmg": _b64(gdmg), "nG": int(len(ga)),
        "cables": cab, "cable_ok": _b64(cab_ok),
        "plate": {"node": int(m.plate_node), "half": m.plate_half.tolist()},
        "jack_r": float(m.jack_radius), "table_y": float(m.table_y),
        "curve": {"travel": res.curve["travel"][::4], "force": [f / KGF for f in
                                                               res.curve["force"][::4]]},
        "events": [e.as_dict() for e in res.events if e.kind != "glue_damage"][:400],
        "Fu": res.Fu_kgf, "mode": res.failure_mode, "detail": res.failure_detail,
        "rate": rate, "colours": KIND_COLOUR,
    }
    vendor = Path(__file__).with_name("vendor")
    three = (vendor / "three.min.js").read_text(encoding="utf-8")
    orbit = (vendor / "OrbitControls.js").read_text(encoding="utf-8")
    # three.js r147 (MIT, see vendor/LICENSE-three.txt) is inlined so the replay works
    # offline, e.g. at the competition
    return (_TEMPLATE.replace("__TITLE__", title).replace("__THREE__", three)
            .replace("__ORBIT__", orbit).replace("__DATA__", json.dumps(data)))


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f4ef;--fg:#1d1d1f;--muted:#6b6b70;--panel:#ffffff;--line:#d9d6cf;--accent:#c0392b}
@media (prefers-color-scheme: dark){:root{--bg:#141416;--fg:#ececec;--muted:#9a9aa0;--panel:#1d1d21;--line:#333338;--accent:#ff6b5b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
#wrap{display:grid;grid-template-columns:minmax(0,1fr) 360px;height:100vh}
#view{position:relative;min-height:320px}#hud{position:absolute;left:12px;top:10px;background:color-mix(in srgb,var(--panel) 85%,transparent);padding:8px 12px;border-radius:8px;border:1px solid var(--line)}
#hud .big{font-size:22px;font-weight:600}#side{background:var(--panel);border-left:1px solid var(--line);display:flex;flex-direction:column;overflow:hidden}
#side h2{font-size:15px;margin:12px 14px 6px}#chart{width:100%;height:210px}
#ctrl{padding:8px 14px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}button,select{font:inherit;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
input[type=range]{width:100%}#events{overflow:auto;flex:1;padding:0 14px 14px;font-size:12.5px}
.ev{padding:4px 0;border-bottom:1px solid var(--line);color:var(--muted)}.ev.on{color:var(--fg)}.ev b{color:var(--accent)}
#legend{font-size:12px;color:var(--muted);padding:0 14px}
@media (max-width:800px){#wrap{grid-template-columns:1fr;grid-template-rows:60vh auto;height:auto}#side{border-left:0}}
</style></head><body>
<div id="wrap"><div id="view"><div id="hud"><div>Jack load</div><div class="big" id="load">0 kgf</div>
<div id="sub"></div></div></div>
<div id="side"><h2 id="ttl"></h2><div id="legend">Wood colour = intact &middot; <span style="color:#e0a800">amber</span> = a glue line on it is failing &middot; <span style="color:#d0342c">red</span> = cracked or broken. Dots: glue points (green intact &rarr; red failed).</div>
<canvas id="chart"></canvas>
<div id="ctrl"><button id="play">Play</button><select id="speed"><option value="real">Real test time</option><option value="10">10x</option><option value="60" selected>60x</option><option value="sim">Simulated time</option></select>
<label><input type="checkbox" id="glue" checked> glue points</label><input type="range" id="slider" min="0" value="0"></div>
<h2>What happened</h2><div id="events"></div></div></div>
<script>__THREE__</script>
<script>__ORBIT__</script>
<script>
const D=__DATA__;
function dec(b,T){const s=atob(b),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new T(u.buffer)}
document.getElementById('ttl').textContent=D.title;
const X=dec(D.X,Float32Array),U=dec(D.U,Int16Array),Q=dec(D.Q,Int16Array),BS=dec(D.bstate,Uint8Array);
const nN=D.nN,nF=D.nF;
const view=document.getElementById('view');
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(window.devicePixelRatio);
view.appendChild(renderer.domElement);
const scene=new THREE.Scene();
const dark=matchMedia('(prefers-color-scheme: dark)').matches;scene.background=new THREE.Color(dark?0x141416:0xf6f4ef);
const cam=new THREE.PerspectiveCamera(35,1,1,20000);cam.position.set(900,700,1500);
const ctl=new THREE.OrbitControls(cam,renderer.domElement);ctl.target.set(0,250,0);
scene.add(new THREE.HemisphereLight(0xffffff,0x666666,1.1));const dl=new THREE.DirectionalLight(0xffffff,1.2);dl.position.set(600,1200,900);scene.add(dl);
const plat=new THREE.Mesh(new THREE.BoxGeometry(1420,15,500),new THREE.MeshStandardMaterial({color:0x8a9099,metalness:.6,roughness:.5}));plat.position.set(0,D.table_y-7.5,0);scene.add(plat);
const meshes=[];
for(const m of D.meshes){const T=dec(m.T,Uint32Array),idx=dec(m.idx,Uint32Array),w8=dec(m.w,Uint8Array),o16=dec(m.off,Int16Array);
 const w=new Float32Array(w8.length);for(let i=0;i<w8.length;i++)w[i]=w8[i]/255;const off=new Float32Array(o16.length);for(let i=0;i<o16.length;i++)off[i]=o16[i]/100;
 const Tw=new Uint32Array(T.length);Tw.set(T);
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(new Float32Array(3*m.nv),3));g.setIndex(new THREE.BufferAttribute(Tw,1));g.computeVertexNormals();
 const mat=new THREE.MeshStandardMaterial({color:D.colours[m.kind]||'#c8955a',roughness:.85,flatShading:true,side:THREE.DoubleSide});
 const o=new THREE.Mesh(g,mat);scene.add(o);meshes.push({o,g,idx,w,off,T,Tw,rest:null,nv:m.nv,body:m.body,base:new THREE.Color(D.colours[m.kind]||'#c8955a')});}
// plate + jack
const ph=D.plate.half;const plate=new THREE.Mesh(new THREE.BoxGeometry(2*ph[0],2*ph[1],2*ph[2]),new THREE.MeshStandardMaterial({color:0xb0302a,metalness:.5,roughness:.4}));scene.add(plate);
const jack=new THREE.Mesh(new THREE.CylinderGeometry(D.jack_r,D.jack_r,600,32),new THREE.MeshStandardMaterial({color:0x9aa0a8,metalness:.8,roughness:.3}));scene.add(jack);
// glue points
const gN=dec(D.glue_node,Uint32Array),gO=dec(D.glue_off,Float32Array),gD=dec(D.glue_dmg,Uint8Array);
const gg=new THREE.BufferGeometry();const gpos=new Float32Array(D.nG*3),gcol=new Float32Array(D.nG*3);
gg.setAttribute('position',new THREE.BufferAttribute(gpos,3));gg.setAttribute('color',new THREE.BufferAttribute(gcol,3));
const gpts=new THREE.Points(gg,new THREE.PointsMaterial({size:4,vertexColors:true,depthTest:false}));gpts.renderOrder=2;scene.add(gpts);
// cables
const cOK=dec(D.cable_ok,Uint8Array);const cg=new THREE.BufferGeometry();const cpos=new Float32Array(D.cables.length*6);cg.setAttribute('position',new THREE.BufferAttribute(cpos,3));
const clines=new THREE.LineSegments(cg,new THREE.LineBasicMaterial({color:dark?0xffffff:0x333333}));scene.add(clines);
const P=new Float32Array(nN*3),R=new Float32Array(nN*9);
function nodes(f){const s=D.scale;for(let i=0;i<nN;i++){P[3*i]=X[3*i]+U[(f*nN+i)*3]*s;P[3*i+1]=X[3*i+1]+U[(f*nN+i)*3+1]*s;P[3*i+2]=X[3*i+2]+U[(f*nN+i)*3+2]*s;
 let w=Q[(f*nN+i)*4]/32767,x=Q[(f*nN+i)*4+1]/32767,y=Q[(f*nN+i)*4+2]/32767,z=Q[(f*nN+i)*4+3]/32767;const n=Math.hypot(w,x,y,z)||1;w/=n;x/=n;y/=n;z/=n;
 const r=R,o=9*i;r[o]=1-2*(y*y+z*z);r[o+1]=2*(x*y-z*w);r[o+2]=2*(x*z+y*w);r[o+3]=2*(x*y+z*w);r[o+4]=1-2*(x*x+z*z);r[o+5]=2*(y*z-x*w);r[o+6]=2*(x*z-y*w);r[o+7]=2*(y*z+x*w);r[o+8]=1-2*(x*x+y*y);}}
function tp(n,ox,oy,oz,out,k){const o=9*n;out[k]=P[3*n]+R[o]*ox+R[o+1]*oy+R[o+2]*oz;out[k+1]=P[3*n+1]+R[o+3]*ox+R[o+4]*oy+R[o+5]*oz;out[k+2]=P[3*n+2]+R[o+6]*ox+R[o+7]*oy+R[o+8]*oz;}
const tmp=new Float32Array(3);
function skin(m){const pos=m.g.attributes.position.array;for(let v=0;v<m.nv;v++){let x=0,y=0,z=0;for(let k=0;k<3;k++){const w=m.w[3*v+k];if(w<=0)continue;const n=m.idx[3*v+k],o=(3*v+k)*3;tp(n,m.off[o],m.off[o+1],m.off[o+2],tmp,0);x+=w*tmp[0];y+=w*tmp[1];z+=w*tmp[2];}pos[3*v]=x;pos[3*v+1]=y;pos[3*v+2]=z;}return pos;}
function edge(p,a,b){const dx=p[3*a]-p[3*b],dy=p[3*a+1]-p[3*b+1],dz=p[3*a+2]-p[3*b+2];return Math.sqrt(dx*dx+dy*dy+dz*dz);}
// a body that breaks apart: its skin triangles would stretch between the pieces; a
// triangle stretched more than 1.6x (+2 mm) is torn and not drawn
function tear(m,pos){const T=m.T,Tw=m.Tw,r=m.rest;let k=0;for(let t=0;t<T.length;t+=3){const a=T[t],b=T[t+1],c=T[t+2];
  if(edge(pos,a,b)>1.6*r[t]+2||edge(pos,b,c)>1.6*r[t+1]+2||edge(pos,c,a)>1.6*r[t+2]+2)continue;Tw[k++]=a;Tw[k++]=b;Tw[k++]=c;}
 m.g.setDrawRange(0,k);m.g.index.needsUpdate=true;}
nodes(0);for(const m of meshes){const pos=skin(m),T=m.T,r=new Float32Array(T.length);for(let t=0;t<T.length;t+=3){r[t]=edge(pos,T[t],T[t+1]);r[t+1]=edge(pos,T[t+1],T[t+2]);r[t+2]=edge(pos,T[t+2],T[t]);}m.rest=r;}
function show(f){nodes(f);
 for(const m of meshes){const pos=skin(m);tear(m,pos);
  m.g.attributes.position.needsUpdate=true;m.g.computeVertexNormals();const st=BS[f*D.nb+m.body];m.o.material.color.copy(st==2?new THREE.Color('#d0342c'):st==1?new THREE.Color('#e0a800'):m.base);}
 for(let i=0;i<D.nG;i++){tp(gN[i],gO[3*i],gO[3*i+1],gO[3*i+2],gpos,3*i);const d=gD[f*D.nG+i]/255;gcol[3*i]=0.2+0.8*d;gcol[3*i+1]=0.8*(1-d)+0.1;gcol[3*i+2]=0.2;}
 gg.attributes.position.needsUpdate=true;gg.attributes.color.needsUpdate=true;
 D.cables.forEach((c,i)=>{tp(c.a,c.ao[0],c.ao[1],c.ao[2],cpos,6*i);tp(c.b,c.bo[0],c.bo[1],c.bo[2],cpos,6*i+3);if(!cOK[f*D.cables.length+i]){cpos[6*i+3]=cpos[6*i];cpos[6*i+4]=cpos[6*i+1];cpos[6*i+5]=cpos[6*i+2];}});cg.attributes.position.needsUpdate=true;
 if(D.plate.node>=0){const n=D.plate.node;plate.position.set(P[3*n],P[3*n+1],P[3*n+2]);const o=9*n;plate.setRotationFromMatrix(new THREE.Matrix4().set(R[o],R[o+1],R[o+2],0,R[o+3],R[o+4],R[o+5],0,R[o+6],R[o+7],R[o+8],0,0,0,0,1));
  jack.position.set(P[3*n],P[3*n+1]+ph[1]+300,P[3*n+2]);}
 document.getElementById('load').textContent=D.force[f].toFixed(1)+' kgf';
 document.getElementById('sub').textContent=`jack ${D.travel[f].toFixed(1)} mm · deck ${D.defl[f].toFixed(1)} mm down · test time ${(D.travel[f]/D.rate*60).toFixed(0)} s`;
 drawChart(f);events(f);slider.value=f;}
// chart
const cv=document.getElementById('chart'),cx=cv.getContext('2d');
function drawChart(f){const w=cv.clientWidth,h=cv.clientHeight;cv.width=w*devicePixelRatio;cv.height=h*devicePixelRatio;cx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
 const tr=D.curve.travel,fo=D.curve.force,xm=Math.max(...tr,1),ym=Math.max(...fo,1)*1.1,l=44,b=26;cx.clearRect(0,0,w,h);const css=getComputedStyle(document.body);
 cx.strokeStyle=css.getPropertyValue('--line');cx.fillStyle=css.getPropertyValue('--muted');cx.font='11px system-ui';
 for(let i=0;i<=4;i++){const y=h-b-(h-b-10)*i/4;cx.beginPath();cx.moveTo(l,y);cx.lineTo(w-8,y);cx.stroke();cx.fillText((ym*i/4).toFixed(0),4,y+4);}
 cx.fillText('jack travel (mm) →',l,h-6);cx.fillText('kgf',4,10);
 cx.strokeStyle=css.getPropertyValue('--fg');cx.lineWidth=1.5;cx.beginPath();tr.forEach((t,i)=>{const x=l+(w-l-8)*t/xm,y=h-b-(h-b-10)*fo[i]/ym;i?cx.lineTo(x,y):cx.moveTo(x,y)});cx.stroke();
 const x=l+(w-l-8)*D.travel[f]/xm;cx.strokeStyle=css.getPropertyValue('--accent');cx.beginPath();cx.moveTo(x,10);cx.lineTo(x,h-b);cx.stroke();}
// events
const evd=document.getElementById('events');const evs=D.events.map(e=>{const d=document.createElement('div');d.className='ev';d.innerHTML=`<b>${e.load_kgf.toFixed(1)} kgf</b> · ${e.jack_travel_mm.toFixed(1)} mm · ${e.kind.replace('_',' ')}: ${e.what}${e.detail?' — '+e.detail:''}`;evd.appendChild(d);return [e,d]});
const sum=document.createElement('div');sum.className='ev on';sum.innerHTML=`<b>Predicted ultimate load ${D.Fu.toFixed(1)} kgf</b> (${D.mode}) ${D.detail}`;evd.prepend(sum);
function events(f){const t=D.travel[f];evs.forEach(([e,d])=>d.classList.toggle('on',e.jack_travel_mm<=t+1e-9));}
const slider=document.getElementById('slider');slider.max=nF-1;slider.oninput=()=>{cur=+slider.value;show(cur)};
document.getElementById('glue').onchange=e=>{gpts.visible=e.target.checked};
let cur=0,playing=false,last=0;const btn=document.getElementById('play');btn.onclick=()=>{playing=!playing;btn.textContent=playing?'Pause':'Play';last=performance.now()};
function tick(now){if(playing){const sp=document.getElementById('speed').value;const dt=(now-last)/1000;last=now;
 let want;if(sp==='sim'){want=D.t[cur]+dt*0.05}else{const mult=sp==='real'?1:+sp;want=D.travel[cur]+dt*mult*D.rate/60;}
 let nf=cur;if(sp==='sim'){while(nf<nF-1&&D.t[nf]<want)nf++}else{while(nf<nF-1&&D.travel[nf]<want)nf++}
 if(nf!==cur){cur=nf;show(cur)}if(cur>=nF-1){playing=false;btn.textContent='Play'}}
 ctl.update();renderer.render(scene,cam);requestAnimationFrame(tick)}
function resize(){const w=view.clientWidth,h=view.clientHeight||window.innerHeight*0.6;renderer.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix();drawChart(cur)}
window.addEventListener('resize',resize);resize();show(0);requestAnimationFrame(tick);
</script></body></html>
"""

__all__ = ["replay_html", "build_skin", "math"]
