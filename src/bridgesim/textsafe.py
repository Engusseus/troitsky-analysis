"""Helpers for showing text that came from user-supplied YAML (bridges, materials, rules).

Files are shared between teams, so names and notes are untrusted: they must not carry
terminal control sequences, HTML, or spreadsheet formulas into what bridgesim displays or
exports.
"""

from __future__ import annotations

import html
import re
import unicodedata

_MD_SPECIAL = re.compile(r"([\\`*_\[\]()#+!|~>:<{}$])")


def has_control_chars(text: str, allow: str = "") -> bool:
    """True if ``text`` contains control/format characters (Unicode Cc, Cf) not in ``allow``."""
    return any(unicodedata.category(c) in ("Cc", "Cf") and c not in allow for c in text)


def printable(text: object) -> str:
    """Drop control/format characters (ESC, OSC, bidi overrides...) for terminal output."""
    s = str(text)
    return "".join(c for c in s if unicodedata.category(c) not in ("Cc", "Cf") or c in "\n\t")


def md(text: object) -> str:
    """Escape text for Streamlit Markdown: no HTML, links, colour directives or maths."""
    return _MD_SPECIAL.sub(r"\\\1", html.escape(printable(text), quote=False))


def md_keep_bold(text: object) -> str:
    """Like :func:`md`, but keeps ``**bold**`` markers (they cannot carry links or HTML)."""
    return md(text).replace(r"\*\*", "**")


def csv_cell(value: object) -> object:
    """Neutralise spreadsheet formulas: prefix text starting with = + - @ (or tab/CR) by '."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def csv_row(row: dict) -> dict:
    return {k: csv_cell(v) for k, v in row.items()}


def friendly_error(exc: BaseException, limit: int = 8) -> str:
    """Explain why a file was rejected, one line per problem, for people not programmers.

    Pydantic errors become ``where: what (got value)``, with list positions counted from 1
    (``nodes #3 › x_mm``) and without links to the Pydantic docs.
    """
    from pydantic import ValidationError

    if not isinstance(exc, ValidationError):
        return str(exc)
    errors = exc.errors(include_url=False)
    lines = []
    for e in errors[:limit]:
        where = " › ".join(f"#{p + 1}" if isinstance(p, int) else str(p) for p in e["loc"])
        where = where.replace(" › #", " #")
        msg = e["msg"].removeprefix("Value error, ")
        got = e.get("input")
        if e["type"] != "missing" and isinstance(got, (str, int, float, bool)):
            shown = repr(got)
            msg += f" (got {shown if len(shown) <= 40 else shown[:37] + '...'})"
        lines.append(f"{where}: {msg}" if where else msg)
    if len(errors) > limit:
        lines.append(f"... and {len(errors) - limit} more")
    return "\n".join(lines)
