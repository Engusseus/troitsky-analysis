"""Smoke test of the Streamlit app with streamlit.testing.v1.AppTest."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_app_runs_without_exceptions(app: AppTest) -> None:
    """Default design renders: no exceptions or errors, results card and rule check shown."""
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    labels = [m.label for m in app.metric]
    assert "Predicted ultimate load F_u,p" in labels
    assert any("kgf" in m.value for m in app.metric)
    assert any("All checked rules pass" in s.value for s in app.success)


def test_assumed_material_banner(app: AppTest) -> None:
    """While any material value is assumed, a banner says so."""
    assert any("assumed placeholders" in w.value for w in app.warning)


def test_changing_span_and_reanalysing(app: AppTest) -> None:
    """Edit the span in the design form, press Analyze, and get a new result."""
    before = next(m.value for m in app.metric if m.label.startswith("Predicted"))
    span = next(n for n in app.sidebar.number_input if n.label.startswith("Span"))
    span.set_value(1100.0)
    next(b for b in app.sidebar.button if b.label == "Analyze").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    after = next(m.value for m in app.metric if m.label.startswith("Predicted"))
    assert after != before


def test_short_span_is_flagged() -> None:
    """A 900 mm span violates §8.2.1.1 and must show a penalty and bans, not hide them."""
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    span = next(n for n in app.sidebar.number_input if n.label.startswith("Span"))
    span.set_value(900.0)
    next(b for b in app.sidebar.button if b.label == "Analyze").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any("BANNED" in e.value for e in app.error)


def test_validation_errors_are_shown_as_inert_text() -> None:
    """Pydantic quotes the bad value; a Markdown image in it must not render."""
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    name = next(t for t in app.sidebar.text_input if t.label == "Design name")
    name.set_value("![x](https://example.invalid/p.png)\x07")
    next(b for b in app.sidebar.button if b.label == "Analyze").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    errors = [e.value for e in app.error]
    assert errors and all("![x](" not in e for e in errors)
    assert any(r"\!\[x\]" in e for e in errors)
