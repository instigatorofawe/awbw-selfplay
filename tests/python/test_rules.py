import subprocess
import sys
from pathlib import Path

import pytest

from awbw import rules

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def r():
    return rules.load()


def test_units_match_awbw_ids(r):
    assert len(r.units) == 25
    assert [u.index for u in r.units] == list(range(25))
    assert r.unit("Infantry").id == 1 and r.unit(1141438).name == "Mega Tank"
    assert r.unit("Md.Tank").cost == 16000 and r.unit("Piperunner").move_type == "P"
    assert r.unit("Sub").hidden_fuel_per_day == 5 and r.unit("Stealth").hidden_fuel_per_day == 8


def test_damage_chart_and_weapons(r):
    tank, inf, bcop, cruiser = (r.unit(n).index for n in ("Tank", "Infantry", "B-Copter", "Cruiser"))
    # The AWBW chart shows the weapon used; Tank's cannon can't hit infantry or copters.
    assert r.primary[tank][tank] == 55 and r.secondary[tank][tank] == 6
    assert r.primary[tank][inf] is None and r.secondary[tank][inf] == 75
    assert r.primary[tank][bcop] is None and r.secondary[tank][bcop] == 10
    # Cruiser: missiles (ammo) against ships, anti-air gun (no ammo) against air.
    assert r.primary[cruiser][r.unit("Sub").index] == 90
    assert r.secondary[cruiser][bcop] == 115 and r.primary[cruiser][bcop] is None
    assert all(x is None for x in r.primary[inf]) and r.secondary[inf][inf] == 55


def test_transports(r):
    names = lambda idx: sorted(r.units[i].name for i in idx)  # noqa: E731
    assert names(r.unit("APC").carry) == ["Infantry", "Mech"]
    assert names(r.unit("Cruiser").carry) == ["B-Copter", "T-Copter"]
    assert r.unit("Tank").index in r.unit("Lander").carry
    assert names(r.select("transport")) == ["APC", "Black Boat", "Lander", "T-Copter"]


def test_terrain(r):
    assert r.terrain[1].kind == "plain" and r.terrain[1].defense == 1
    assert r.terrain[42].kind == "hq" and r.terrain[42].country == 1 and r.terrain[42].defense == 4
    assert all(t.country is None or t.country in r.countries for t in r.terrain.values())
    clear = r.move_costs["clear"]
    assert clear["mountain"].get("T") is None and clear["wood"]["W"] == 3 and clear["reef"]["S"] == 2


def test_co_modifiers(r):
    max_ = r.cos[7]
    tank, art, inf = (r.unit(n).index for n in ("Tank", "Artillery", "Infantry"))
    assert (max_.d2d.atk[tank], max_.d2d.atk[art], max_.d2d.atk[inf]) == (20, -10, 0)
    assert max_.d2d.range[art] == -1 and max_.cop.move[tank] == 1 and max_.scop.atk[tank] == 50
    assert (max_.cop_stars, max_.scop_stars) == (3, 6)
    assert r.cos[30].cop_stars is None and r.cos[30].scop_stars == 10  # Von Bolt has no COP
    assert r.cos[23].scop.terrain_atk["comtower"] == 130  # Kindle's urban group
    assert (r.cos[25].scop.luck_good, r.cos[25].scop.luck_bad) == (89, 39)  # Flak
    assert r.cos[3].scop.counter == 150 and r.cos[3].d2d.cost == 120  # Kanbei


def test_generated_header_is_current():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "gen_tables.py"), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
