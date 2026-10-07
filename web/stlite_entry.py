# stlite-only entry point (top-level await is legal in stlite, not in CPython).
# PyniteFEA 3.2.0 pins numpy>=2.4 but Pyodide 0.29.3 ships numpy 2.2.5, which works for
# the API bridgesim uses (see docs/web-spike.md), so install it without dependencies.
import importlib.util
import runpy

if importlib.util.find_spec("Pynite") is None:  # skip on Streamlit reruns
    import micropip

    await micropip.install("PyniteFEA==3.2.0", deps=False)  # noqa: F704, PLE1142

runpy.run_path("app/streamlit_app.py", run_name="__main__")
