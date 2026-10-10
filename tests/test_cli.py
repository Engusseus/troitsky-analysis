"""Command line behaviour that only shows up in a real process (pipes, encodings)."""

import os
import subprocess
import sys
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "warren_2027.yaml"


def test_piped_output_is_utf8_whatever_the_locale() -> None:
    """On Windows a piped or redirected stdout uses the ANSI code page, so "§" used to leave
    as the single cp1252 byte 0xA7 and show as "�" in PowerShell and CI logs. Forcing cp1252
    here reproduces that on any OS; the CLI must still write "§" as UTF-8 (0xC2 0xA7)."""
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    out = subprocess.run([sys.executable, "-m", "bridgesim.cli", "check", str(EXAMPLE_PATH)],
                         capture_output=True, env=env, timeout=120).stdout
    assert "[PASS] §8.2.1.1".encode() in out
    assert b"\xa7" not in out.replace("§".encode(), b"")
