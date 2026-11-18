"""Published coupon tests in validation/literature/: the files parse, bridgesim reads the
coupon files it should and no others, and the data are those of the sources (README
there). Fast: no simulation."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import pytest
import yaml

from bridgesim.sim.study import COUPONS, _coupon_rows, coupon_files

LIT = Path(__file__).resolve().parents[1] / "validation" / "literature"
ENS = LIT / "ens_paris_saclay"
OTHER = ENS / "other"


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("path", sorted(LIT.rglob("*.csv")), ids=lambda p: p.name)
def test_csv_parses(path):
    """Every row has exactly the header's fields (a comma in an unquoted note would add
    one, csv.DictReader putting it under the key None)."""
    rows = _rows(path)
    assert rows
    for r in rows:
        assert None not in r, f"{path.name}: a row with more fields than the header"
        assert None not in r.values(), f"{path.name}: a row with fewer fields than the header"


@pytest.mark.parametrize("path", sorted(LIT.rglob("*.yaml")), ids=lambda p: p.name)
def test_yaml_parses(path):
    assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)


def test_coupon_files_reads_exactly_the_coupon_csvs():
    """The ENS folder holds three coupon files, recognised by kind with no warning; the
    files in other/ are not in a template's format and are not read (the folder's CSVs
    only; and from other/ itself none is taken for a coupon of any kind)."""
    found, warn = coupon_files(ENS)
    got = {k: [f.name for f in v] for k, v in found.items() if v}
    assert got == {"stick_tension": ["stick_tension.csv"],
                   "stick_bending": ["stick_bending.csv"],
                   "glue_joint": ["glue_laps.csv"]}
    assert warn == []
    found_other, _ = coupon_files(OTHER)
    assert all(not v for v in found_other.values())
    # and no file there has the columns of a template, whatever its name
    for f in OTHER.glob("*.csv"):
        cols = set(_rows(f)[0])
        assert not any(need <= cols for _w, need in COUPONS.values()), f.name


def test_tension_tests_are_the_76_of_the_annex():
    """Eduscol 14237: 76 sticks, mean 1389 N (p. 2), 409-2204 N."""
    rows = _rows(ENS / "stick_tension.csv")
    F = [float(r["failure_load_N"]) for r in rows]
    assert len(F) == 76
    assert statistics.fmean(F) == pytest.approx(1389, abs=1)
    assert statistics.stdev(F) == pytest.approx(339.7, abs=0.1)
    assert (min(F), max(F)) == (409, 2204)
    assert len(_coupon_rows(ENS, "stick_tension")) == 76


def test_lap_file_holds_only_glue_failures():
    """glue_laps.csv: laps whose glue line failed (the rouge report's 'la colle se casse');
    the laps that broke in the wood are in other/wood_failed_laps.csv only, without a
    failure_load_N column, so no coupon reader can take them for glue strengths."""
    laps = _rows(ENS / "glue_laps.csv")
    wood = _rows(OTHER / "wood_failed_laps.csv")
    assert laps and wood
    assert all("la colle se casse" in r["notes"] for r in laps)
    assert {r["specimen"] for r in laps}.isdisjoint(r["specimen"] for r in wood)
    assert all(r["glue_line_failed"] != "yes" for r in wood)
    assert "failure_load_N" not in wood[0]
    assert [r["failure_load_N"] for r in _coupon_rows(ENS, "glue_joint")] == ["1700"]


def test_bending_files_agree_with_table_1():
    """edu_14235 Table 1 (p. 13): the single sticks flat broke at 29, 50 and 32 N, and
    the table has 31 results (3 per configuration, 4 for three glued sticks flat)."""
    single = [float(r["failure_load_N"]) for r in _rows(ENS / "stick_bending.csv")]
    assert single == [29, 50, 32]
    table = [r for r in _rows(OTHER / "laminated_bending.csv") if r["in_table_1"] == "yes"]
    assert len(table) == 31
    flat1 = [float(r["peak_load_N"]) for r in table
             if r["sticks"] == "1" and r["orientation"] == "flat"]
    assert flat1 == single
