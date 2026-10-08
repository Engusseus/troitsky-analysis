# Spike: running the Streamlit app in the browser (stlite + Pynite)

**Date:** 2026-10-07
**Question:** Can `app/streamlit_app.py` run fully client-side on GitHub Pages via stlite (Streamlit on Pyodide), with the FEA solver PyniteFEA running inside Pyodide?

**Verdict: Pages deploy viable, with one workaround and one item still to check on a real deploy.**
PyniteFEA 3.2.0 imports and solves a frame correctly inside Pyodide and inside stlite. It has to be installed with `deps=False`, which means a small stlite-only entry script (recipe below). `st.dataframe` still needs to be checked on a real deploy (see caveats).

## Versions tested

| Component | Version | Notes |
|---|---|---|
| stlite | `@stlite/browser` **1.9.2** (latest on npm) | `https://cdn.jsdelivr.net/npm/@stlite/browser@1.9.2/build/stlite.js` (+ `stlite.css`) |
| Pyodide used by stlite | **0.29.3** (Python 3.13.2, ABI `2025_0`) | Fixed in the stlite bundle: `cdn.jsdelivr.net/pyodide/v0.29.3/full/pyodide.mjs`. The bundled Streamlit wheel is `streamlit-1.62.0-cp313`, so stlite 1.9.2 cannot use the newer Pyodide line. |
| Latest Pyodide | 314.0.7 (Python 3.14, ships numpy 2.4.6) | Its numpy satisfies Pynite's pin, but stlite does not use it yet. Not tested in a browser. |
| Streamlit (inside stlite) | 1.62.0 | stlite fork. It replaces `pyarrow` with a stub and serialises dataframes with fastparquet. |
| PyniteFEA | **3.2.0** | API checked against the installed source (below). |
| numpy / scipy / matplotlib (Pyodide 0.29.3) | **2.2.5** / 1.14.1 / 3.8.4 | From the official 0.29.3 `pyodide-lock.json`. |
| pydantic / pydantic-core (Pyodide 0.29.3) | **2.12.5** / 2.41.5 | pydantic v2 is available. The sandbox ran 2.10.6 (see test setup). |
| pyyaml / pandas (Pyodide 0.29.3) | 6.0.2 / 2.3.3 | Both are in the official lock. |
| plotly, prettytable | from PyPI (pure Python, currently 7.1.0 / 3.18.0) | Not in the Pyodide distribution, so micropip fetches them from PyPI. |

## How the test was run (read this before trusting the numbers)

The sandbox's egress policy blocks `cdn.jsdelivr.net` (`CONNECT tunnel failed, response 403`), and GitHub release assets are blocked too. The test harness (Playwright with headless Chromium) therefore served the jsdelivr URLs locally, using `page.route("https://cdn.jsdelivr.net/**")`:

- **Pyodide core:** the official `pyodide@0.29.3` npm tarball (`pyodide.mjs`, `pyodide.asm.wasm`, `python_stdlib.zip`).
- **stlite:** the official `@stlite/browser@1.9.2` npm tarball. stlite's built-in Streamlit wheel ships inside it.
- **Package wheels and lockfile:** a public GitHub mirror of a Pyodide 0.28.x distribution (`nghiaphamhb/Ruvie-Agent@6efea24/references/static/pyodide`). It has the same ABI (`2025_0`, emscripten 4.0.9) and the same numpy 2.2.5, scipy 1.14.1 and matplotlib 3.8.4 builds, but pydantic is 2.10.6 instead of 2.12.5.
- **Gaps in the mirror (sandbox only):** 14 packages that stlite pulls in were missing from the mirror, so the lockfile was patched.
  - 9 pure-Python packages were fetched from PyPI at the same versions: altair, jsonschema, referencing, jinja2, fsspec, narwhals, pyodide-http, attrs and jsonschema-specifications.
  - 5 compiled packages were replaced with **empty stub wheels**: markupsafe, rpds-py, pyrsistent, cramjam and fastparquet.
  - pyyaml was missing too, so `import yaml` was not exercised in the browser.
- **PyPI** (pypi.org, files.pythonhosted.org) was reached for real, through the proxy.

The timings below therefore leave out the real download of about 64 MB from jsdelivr. Before relying on the verdict, re-run the template once on a real Pages URL. It takes about 5 minutes; see the checklist at the end.

## Results

### Step 0: CPython reference
Pynite 3.2.0 gives N2.DY = **-0.249991** on the 2-member frame, both with numpy 2.5.3 / scipy 1.18.1 and with numpy **2.2.5** / scipy 1.14.1. So the `numpy>=2.4.0` pin is not needed for this API surface.

Facts from the PyniteFEA 3.2.0 source:
- The API matches the spike snippet: `add_section(name, A, Iy, Iz, J)`, `add_member(name, i, j, material_name, section_name)`, `add_material(name, E, G, nu, rho)`, `def_support(...)`, `add_node_load(node, 'FY', P)` and `analyze()`. `nodes['N2'].DY` is a dict keyed by combo, for example `DY['Combo 1']`.
- `import Pynite` loads **numpy, scipy, matplotlib (pyplot) and prettytable** at import time, because `Pynite/__init__.py` imports `ShearWall`, which does `import matplotlib.pyplot` and `from prettytable import PrettyTable`. So matplotlib and prettytable are effectively required.
- `import Pynite` does **not** load pyvista, vtk, IPython or jinja2. Those appear only in `Rendering.py`, `VTKWriter.py`, `Visualization.py` and `Reporting.py`, which are not imported by default. Never import those modules in the web app.

### Step 1: plain Pyodide 0.29.3 page

| Recipe | Result |
|---|---|
| `await micropip.install("PyniteFEA==3.2.0")` | **FAIL**: `ValueError: Can't find a pure Python 3 wheel for 'numpy>=2.4.0'.` (from `micropip/transaction.py`, `find_wheel`) |
| `loadPackage(["numpy","scipy","matplotlib"])`, then `micropip.install("prettytable")`, then `micropip.install("PyniteFEA==3.2.0", deps=False)` | **OK**: `N2.DY=-0.249991 numpy=2.2.5 scipy=1.14.1` |

Timings for the working recipe, from a cold page:

| Milestone | Time |
|---|---|
| Pyodide core loaded | 3.5 s |
| numpy, scipy and matplotlib (14 packages) loaded | 20.8 s (includes fetching from the GitHub mirror) |
| prettytable and PyniteFEA installed from PyPI | 22.5 s |
| First `import Pynite` plus solve done | 26.5 s (the import and solve alone take 4.0 s) |

Older PyniteFEA versions did not need to be tried.

### Step 2: stlite 1.9.2

| Attempt | Result |
|---|---|
| `requirements: [..., "PyniteFEA"]` | Would fail with the numpy error above. stlite passes all `requirements` to one `micropip.install(..., keep_going=True)` call, and Pynite's numpy pin fails it. |
| `installs: [{requirements: ["PyniteFEA==3.2.0"], options: {deps: false}}]` passed to `mount()` | **Silently ignored.** The app then fails with `ModuleNotFoundError: No module named 'Pynite'`. The worker supports `installs`, but `mount()` in `@stlite/browser` 1.9.2 does not forward it (the string `installs` does not appear in `stlite.js`). |
| Top-level `await micropip.install("PyniteFEA==3.2.0", deps=False)` in the entrypoint script | **OK.** |
| Same, but in a stlite-only wrapper `web/stlite_entry.py` that then runs `app/streamlit_app.py` with `runpy.run_path(..., run_name="__main__")` | **OK. This is the recommended form.** |

What the test app checked:
- It solved the frame and wrote the result with `st.write`.
- It printed the versions in use: numpy 2.2.5, scipy 1.14.1, matplotlib 3.8.4 (backend `agg`), pydantic 2.10.6 (sandbox mirror), PyniteFEA 3.2.0, Streamlit 1.62.0 and Python 3.13.2.
- It imported a mounted `src/bridgesim` package and read a mounted YAML file as text.
- It rendered a Plotly chart (`.js-plotly-plot .main-svg` present in the DOM).

Timings, measured with Playwright from navigation (CDN files served locally, so network time is excluded):

| Run | Result text visible | Plotly chart visible |
|---|---|---|
| App with in-script await, run 1 | 18.3 s | 19.5 s |
| App with in-script await, run 2 | 16.1 s | 17.5 s |
| `web/stlite_entry.py` wrapper | 20.1 s | 22.5 s |

Breakdown of a typical run:

| Phase | Time |
|---|---|
| Pyodide boot | about 2.5 s |
| Pyodide packages (37 packages, numpy, scipy, matplotlib and pandas among them) | about 7 s |
| Streamlit runtime load | about 3 s |
| PyniteFEA install | 0.2 s |
| First script run (Pynite import plus solve) | 2.5–3 s |

The page transferred about 64 MB from the jsdelivr paths (52 requests), plus PyPI downloads. Expect **about 25–40 s to interactive on a first visit** over a normal connection, and much less on later visits once the browser cache is warm. Streamlit reruns, such as widget changes, skip all of that and cost only the script time.

The only other console noise was a harmless warning: `Failed to scan component manifests: No module named 'toml'`.

## Working install recipe

- **`mount()` `requirements`:**
  `["numpy", "scipy", "matplotlib", "pandas", "pydantic", "pyyaml", "plotly", "prettytable"]`
  - Do **not** list `PyniteFEA` or `streamlit` here. stlite ignores `streamlit` and always uses its built-in copy.
  - Do **not** list `typer` either. Only `bridgesim/cli.py` imports it, and the web app must not.
- **PyniteFEA:** install it in the stlite entry script with a top-level await, as below. The real `app/streamlit_app.py` stays plain CPython, so `streamlit run` keeps working.

```python
# web/stlite_entry.py  (stlite only; top-level await is legal here)
import importlib.util, runpy
if importlib.util.find_spec("Pynite") is None:          # skip on Streamlit reruns
    import micropip
    # PyniteFEA 3.2.0 pins numpy>=2.4, Pyodide 0.29.3 has numpy 2.2.5 -> no deps
    await micropip.install("PyniteFEA==3.2.0", deps=False)
runpy.run_path("app/streamlit_app.py", run_name="__main__")
```

**Making `bridgesim` importable:** mount the package files into the Pyodide filesystem at their repository paths. The working directory is `/home/pyodide`, which plays the role of the repo root. Then:
- The existing `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))` in `app/streamlit_app.py` works unchanged. This was tested.
- `bridgesim.paths.data_dir()` falls back to `<repo root>/rules|materials|examples`, so mounting the YAML files at `rules/...`, `materials/...` and `examples/...` makes `list_yaml()` and `glob` work.

A pure-Python `bridgesim` wheel passed by absolute URL in `requirements` would also work in principle, but it was **not tested**. Mounting the files needs no build step, so prefer it.

## `web/index.html` template

Deploy a GitHub Pages artifact that keeps the repo's relative layout:

```
_site/
  index.html          # this file (from web/index.html)
  files.json          # manifest generated in CI
  app/  src/bridgesim/  rules/  materials/  examples/  web/stlite_entry.py
```

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>bridgesim (in-browser)</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@1.9.2/build/stlite.css" />
</head>
<body>
  <div id="root"></div>
  <script type="module">
    import { mount } from "https://cdn.jsdelivr.net/npm/@stlite/browser@1.9.2/build/stlite.js";
    // Absolute URLs: stlite fetches {url} entries itself; resolve against the page.
    const abs = (p) => new URL(p, location.href).href;
    const paths = await (await fetch(abs("files.json"))).json();   // ["app/streamlit_app.py", "src/bridgesim/__init__.py", ...]
    const files = Object.fromEntries(paths.map((p) => [p, { url: abs(p) }]));
    mount(
      {
        entrypoint: "web/stlite_entry.py",
        requirements: ["numpy", "scipy", "matplotlib", "pandas", "pydantic", "pyyaml", "plotly", "prettytable"],
        files,
      },
      document.getElementById("root"),
    );
  </script>
</body>
</html>
```

CI step that builds the site, to run before `actions/upload-pages-artifact`:

```bash
mkdir -p _site && cp web/index.html _site/
paths=$(find app src/bridgesim rules materials examples web/stlite_entry.py -type f \
          \( -name '*.py' -o -name '*.yaml' \) -not -path '*/__pycache__/*' -not -name 'cli.py' | sort)
for p in $paths; do mkdir -p "_site/$(dirname "$p")"; cp "$p" "_site/$p"; done
printf '%s\n' $paths | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().split()))' > _site/files.json
```

The spike tested an inline `files` map with the same `{url: absolute}` form. The `files.json` manifest is a thin variation on that.

## Gotchas for the real app

1. **numpy is 2.2.5 in the browser.** Do not use numpy 2.3 or newer APIs anywhere in `bridgesim`. Pin `PyniteFEA==3.2.0` exactly: it was verified on numpy 2.2.5, and a future Pynite might really need numpy 2.4.
2. **Never import** `Pynite.Rendering`, `Pynite.Visualization`, `Pynite.VTKWriter` or `Pynite.Reporting` (they need pyvista, vtk, IPython or jinja2). Also never import `bridgesim.cli` (it needs typer, which is not installed in the web build). Anything that shells out (`subprocess`), opens a browser or writes outside the working directory will not work.
3. **matplotlib works.** Pyodide 0.29.3 ships 3.8.4, and the backend in the stlite worker is `agg`. Pynite imports `matplotlib.pyplot` at import time anyway. For `st.pyplot` or PNG downloads, build `matplotlib.figure.Figure` objects directly, as `viz.py` already does.
4. **pydantic v2 is available**: 2.12.5 in Pyodide 0.29.3 (2.10.6 in the sandbox mirror). Keep `pydantic>=2.6`.
5. **`st.dataframe` was not verified.** stlite stubs out pyarrow and serialises dataframes with fastparquet and cramjam, both of which were stubbed in the sandbox. The real app calls `st.dataframe` three times, so check it on the first real deploy. If it misbehaves, fall back to `st.table` or a Markdown table.
6. **The `installs` option of `mount()` is ignored** in 1.9.2, so use the entry-script `await micropip.install(..., deps=False)`. Top-level `await` is a `SyntaxError` under CPython `streamlit run`, which is why it lives in `web/stlite_entry.py` and not in `app/streamlit_app.py`.
7. **Keep the stlite version pinned** (`@1.9.2`). Each stlite release fixes its Pyodide version, and so the numpy version. Re-check the numpy pin whenever you upgrade; a stlite release built on Pyodide 314 would ship numpy 2.4.6.
8. **Runtime dependency on cdn.jsdelivr.net and pypi.org.** Visitors' browsers download about 60–70 MB on first load. Networks that block jsdelivr, such as some school networks, will not load the app. In that case use the fallback below.
9. **prettytable from PyPI needs `wcwidth>=0.3.5`**, but Pyodide's lock has 0.2.13. micropip resolved this without error in the sandbox. If it ever conflicts, pin `prettytable<3.17`.

## Recommendation

**(a) A Pages deploy is viable.** Use the template above: stlite 1.9.2 on Pyodide 0.29.3, the eight-package `requirements` list, and `web/stlite_entry.py` installing `PyniteFEA==3.2.0` with `deps=False`.

Before announcing it, do a one-time real-network check on the Pages URL:
1. The app boots and the Warren example solves.
2. `st.dataframe` tables render.
3. YAML loading works.
4. Note the cold-load time.

**Fallback:** keep supporting local `uv run streamlit run app/streamlit_app.py`. If the browser build shows a blocker, such as dataframes or load time on school networks, deploy the same app unchanged to Streamlit Community Cloud. It runs normal CPython with numpy 2.4 or newer, so it needs no `deps=False` workaround.
