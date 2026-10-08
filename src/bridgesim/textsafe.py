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
