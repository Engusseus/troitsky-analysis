# Contributing to bridgesim

Thanks for helping! bridgesim is maintained by students for students. Small, focused
pull requests are easiest to review.

## Development setup

```bash
uv sync                      # installs Python deps and bridgesim in editable mode
uv run pytest -q             # all tests (about 20 s)
uv run ruff check src app tests
uv run streamlit run app/streamlit_app.py
```

## Ground rules

* **Units in names**: `span_mm`, `load_N`, `E_MPa`. Internally mm, N, MPa.
* **Every engineering formula** gets a docstring with the formula and its source.
* **Tests first** for engineering maths: compare against a hand calculation or a textbook
  result and put that calculation in the test docstring.
* **Never hide a failing check.** Don't clamp values or drop warnings to make results look
  better.
* New modelling assumptions go in `ASSUMPTIONS.md` and `report.MODEL_ASSUMPTIONS`.
* The app must stay browser-compatible (Pyodide): do not import PyVista,
  sectionproperties, or packages with compiled extensions that Pyodide lacks.
* Pynite is pinned exactly in `pyproject.toml`. When upgrading, re-run the full test
  suite (`tests/test_orientation.py` locks the local-axis convention).

## Updating the rules for a new year (next year's team)

1. Copy `rules/troitsky_2027.yaml` to `rules/troitsky_2028.yaml`.
2. Read the new rulebook's Section 8 (and 12.5 for the crusher) and update numbers,
   section references, penalties and bans. Keep each `measure` key unchanged; they map
   to functions in `src/bridgesim/measure.py`.
3. If a rule needs a new measurement, add it to `measure.py` with a test.
4. Add boundary tests for the changed bands in `tests/test_rules.py`.
5. The app lists every file in `rules/` in the *Rules file* picker automatically.

## Adding a generator

Add `src/bridgesim/generators/<name>.py` with a Pydantic params model and a function that
returns a `Bridge`, register it in `generators/__init__.py`, and add a test that the
default design passes all rules.

## Physical test data

Put stick, joint and crush test data in `validation/physical/` as CSV (templates there).
Fitting materials from these files is planned for v0.2.
