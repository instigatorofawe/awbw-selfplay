import pytest

from awbw import _core, rules

R = rules.load()
KIND = {k: i for i, k in enumerate(R.terrain_kinds)}
U = {u.name: u.index for u in R.units}


def unit(name, hp=100, co=1, power=0, terrain="plain", **kw):
    t = next(t for t in R.terrain.values() if t.kind == terrain)
    ammo = kw.pop("ammo", R.units[U[name]].ammo)
    return _core.Combatant(U[name], hp, ammo, co, power, KIND[terrain], t.defense, **kw)


def test_tank_vs_tank_on_plains():
    # (55 + luck) * 0.9, rounded: the classic 49-57%.
    assert [_core.damage(unit("Tank"), unit("Tank"), luck) for luck in range(10)] == [
        49, 50, 51, 52, 53, 54, 54, 55, 56, 57,
    ]  # fmt: skip


def test_rounding_takes_exact_x95_up():
    # Raw (1 + 9) * 0.5 * 0.99 = 4.95: AWBW deals 5 (seen in replay 1738979).
    att = unit("Infantry", hp=44, co=5)
    de = unit("Md.Tank", hp=5, co=16)
    assert _core.damage(att, de, 9) == 5
    assert _core.damage(att, de, 8) == 4  # raw 4.455


def test_modifiers():
    hawke_inf = unit("Infantry", co=12, towers=2)
    assert _core.attack_value(hawke_inf, unit("Infantry")) == 130
    grimm = unit("Infantry", co=20, power=1)
    assert _core.attack_value(grimm, unit("Infantry")) == 160  # +50 COP, +10 for any power
    assert _core.defense_value(grimm, unit("Infantry")) == 90  # -20 d2d, +10 power
    jake_on_rubble = unit("Tank", co=22, terrain="rubble")
    assert _core.attack_value(jake_on_rubble, unit("Tank")) == 110
    koal_on_bridge = unit("Tank", co=21, terrain="bridge")
    assert _core.attack_value(koal_on_bridge, unit("Tank")) == 100
    # Javier: +20% defense against indirect fire only.
    javier = unit("Tank", co=27)
    assert _core.defense_value(javier, unit("Artillery")) == 120
    assert _core.defense_value(javier, unit("Tank")) == 100
    # Air units get no terrain stars.
    assert _core.damage(unit("Tank"), unit("B-Copter", terrain="mountain"), 0) == 10


def test_weapon_choice():
    tank = unit("Tank")
    assert _core.weapon(tank, U["Infantry"]) == 1  # machine gun
    assert _core.weapon(tank, U["Tank"]) == 0  # cannon
    assert _core.weapon(unit("Tank", ammo=0), U["Tank"]) == 1  # out of shells
    assert _core.weapon(unit("Mech", ammo=0), U["Tank"]) == 1
    assert _core.weapon(unit("Infantry"), U["Bomber"]) == -1


@pytest.mark.parametrize(
    ("att", "de", "dist", "ok"),
    [
        ("Artillery", "Tank", 1, False),
        ("Artillery", "Tank", 2, True),
        ("Artillery", "Tank", 4, False),
        ("Tank", "Tank", 2, False),
        ("Infantry", "Bomber", 1, False),
    ],
)
def test_range(att, de, dist, ok):
    assert _core.can_attack(unit(att), unit(de), dist) is ok


def test_ranges_with_co():
    assert _core.can_attack(unit("Artillery", co=2), unit("Tank"), 4)  # Grit +1
    assert not _core.can_attack(unit("Artillery", co=7), unit("Tank"), 3)  # Max -1


def test_outcomes_distribution():
    out = _core.combat_outcomes(unit("Tank"), unit("Tank"), 1)
    assert abs(sum(p for _, _, p in out) - 1) < 1e-12
    assert len({(a, d) for a, d, _ in out}) == len(out)  # merged
    assert {d for _, d, _ in out} == {100 - x for x in (49, 50, 51, 52, 53, 54, 55, 56, 57)}
    assert all(a < 100 for a, _, _ in out)  # the survivor always counters


def test_no_counter_from_indirect_fire_or_dead_units():
    out = _core.combat_outcomes(unit("Artillery"), unit("Artillery"), 2)
    assert all(a == 100 for a, _, _ in out)
    out = _core.combat_outcomes(unit("Mega Tank"), unit("Infantry", hp=10), 1)
    assert out == [(100, 0, 1.0)]


def test_hidden_units():
    stealth = unit("Stealth", hidden=True)
    assert not _core.can_attack(unit("Anti-Air"), stealth, 1)
    assert _core.can_attack(unit("Fighter"), stealth, 1)
    # A hidden Stealth attacking an Anti-Air takes no counter (replay 1735903).
    out = _core.combat_outcomes(stealth, unit("Anti-Air", hp=90), 1)
    assert all(a == 100 for a, _, _ in out)
    assert not _core.can_attack(unit("Battleship"), unit("Sub", hidden=True), 2)


def test_sonja_counter_break_strikes_first_without_bonus():
    # Replay 1719682: the Counter Break strike gets no 1.5x counter bonus.
    sonja_tank = unit("Tank", hp=82, co=18, power=2)
    mech = unit("Mech", hp=68, co=8, terrain="wood")
    out = _core.combat_outcomes(mech, sonja_tank, 1)
    assert any((a + 9) // 10 == 1 and (d + 9) // 10 == 8 for a, d, _ in out)
    # Normal Sonja counters do get it.
    plain = _core.damage(unit("Tank", hp=50, co=18), unit("Tank"), 0, False)
    boosted = _core.damage(unit("Tank", hp=50, co=18), unit("Tank"), 0, True)
    assert boosted > plain


def test_bad_luck_cannot_heal():
    flak = unit("Infantry", hp=10, co=25, power=2)  # luck -39..+89
    assert _core.damage(flak, unit("Md.Tank", terrain="mountain"), -39) == 0
