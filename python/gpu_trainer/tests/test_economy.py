"""Ported C# xUnit golden-value tests (csharp/OGameSim.Tests/*.cs).

Each parametrize row is lifted verbatim from an ``[InlineData]`` in the C# tests. They run against the
scalar reference (``ogame/reference.py``) AND the lookup tables (``ogame/luts.py``) — both must reproduce
the exact C# values. This is the no-.NET correctness anchor for the economy.
"""

import math
from decimal import Decimal

import pytest

from ogame import reference as ref
from ogame.luts import build_luts

LUTS = build_luts("cpu")


def _level_mine(make, level):
    m = make()
    for _ in range(level):
        m.upgrade()
    return m


# ----------------------------------------------------------------------------- MetalMine (MetalMineTests.cs)
@pytest.mark.parametrize("level,expected", [(0, 720), (1, 792), (10, 18672), (30, 376896), (50, 4226064)])
def test_metal_mine_production(level, expected):
    assert _level_mine(ref.MetalMine, level).todays_production.metal == expected
    assert LUTS.metal_prod[level].item() == expected


@pytest.mark.parametrize(
    "level,metal,crystal",
    [(0, 60, 15), (9, 2306, 576), (29, 7670042, 1917510), (49, 25504860008, 6376215002)],
)
def test_metal_mine_cost(level, metal, crystal):
    c = _level_mine(ref.MetalMine, level).upgrade_cost
    assert (c.metal, c.crystal, c.deuterium) == (metal, crystal, 0)
    assert LUTS.metal_cost_mse[level].item() == metal + 2 * crystal


# ---------------------------------------------------------------------------- CrystalMine (CrystalMineTests.cs)
@pytest.mark.parametrize("level,expected", [(0, 360), (1, 528), (10, 12432), (20, 64560), (34, 416928)])
def test_crystal_mine_production(level, expected):
    assert _level_mine(ref.CrystalMine, level).todays_production.crystal == expected
    assert LUTS.crystal_prod[level].item() == expected


@pytest.mark.parametrize(
    "level,metal,crystal",
    [(0, 48, 24), (9, 3299, 1650), (19, 362678, 181339), (33, 261336858, 130668429)],
)
def test_crystal_mine_cost(level, metal, crystal):
    c = _level_mine(ref.CrystalMine, level).upgrade_cost
    assert (c.metal, c.crystal, c.deuterium) == (metal, crystal, 0)
    assert LUTS.crystal_cost_mse[level].item() == metal + 2 * crystal


# --------------------------------------------------------------- DeuteriumSynthesizer (DeuteriumSynthesizerTests.cs)
@pytest.mark.parametrize(
    "level,expected,temperature",
    [
        (0, 0, 120), (0, 0, 0), (0, 0, -120),
        (1, 240, 120), (1, 360, 0), (1, 504, -120),
        (10, 5952, 120), (10, 8952, 0), (10, 11928, -120),
        (20, 30984, 120), (20, 46488, 0), (20, 61992, -120),
        (42, 529920, 120), (42, 794904, 0), (42, 1059864, -120),
    ],
)
def test_deuterium_production(level, expected, temperature):
    m = _level_mine(lambda: ref.DeuteriumSynthesizer(temperature), level)
    assert m.todays_production.deuterium == expected


@pytest.mark.parametrize(
    "level,metal,crystal",
    [(0, 225, 75), (9, 8650, 2883), (19, 498789, 166263), (41, 3731849658, 1243949886)],
)
def test_deuterium_cost(level, metal, crystal):
    c = _level_mine(lambda: ref.DeuteriumSynthesizer(0), level).upgrade_cost
    assert (c.metal, c.crystal, c.deuterium) == (metal, crystal, 0)


def test_deuterium_lut_uses_game_temperature():
    # the LUT specialises to the game temperature (-115 -> avg -135 -> factor 0.95)
    m = _level_mine(lambda: ref.DeuteriumSynthesizer(ref.GAME_PLANET_MAX_TEMPERATURE), 10)
    assert LUTS.deut_prod[10].item() == m.todays_production.deuterium


# ----------------------------------------------------------------------------- Astrophysics (AstrophysicsTests.cs)
@pytest.mark.parametrize(
    "level,metal,crystal,deut",
    [
        (0, 4000, 8000, 4000), (1, 7000, 14000, 7000), (2, 12250, 24500, 12250),
        (3, 21437, 42875, 21437), (8, 351855, 703711, 351855), (14, 10106311, 20212622, 10106311),
        (24, 2722533045, 5445066090, 2722533045), (30, 78199045470, 156398090941, 78199045470),
    ],
)
def test_astrophysics_cost(level, metal, crystal, deut):
    a = ref.Astrophysics()
    for _ in range(level):
        a.upgrade()
    c = a.upgrade_cost
    assert (c.metal, c.crystal, c.deuterium) == (metal, crystal, deut)
    assert LUTS.astro_cost_mse[level].item() == metal + 2 * crystal + 3 * deut


# ----------------------------------------------------------------------------- Plasma (PlasmaTechnologyTests.cs)
@pytest.mark.parametrize(
    "level,metal,crystal,deut",
    [
        (0, 2000, 4000, 1000), (1, 4000, 8000, 2000), (9, 1024000, 2048000, 512000),
        (15, 65536000, 131072000, 32768000), (19, 1048576000, 2097152000, 524288000),
    ],
)
def test_plasma_cost(level, metal, crystal, deut):
    p = ref.PlasmaTechnology()
    for _ in range(level):
        p.upgrade()
    c = p.upgrade_cost
    assert (c.metal, c.crystal, c.deuterium) == (metal, crystal, deut)
    assert LUTS.plasma_cost_mse[level].item() == metal + 2 * crystal + 3 * deut


@pytest.mark.parametrize(
    "level,metal,crystal,deut",
    [(0, 0, 0, 0), (1, 0.01, 0.0066, 0.0033), (9, 0.09, 0.0594, 0.0297),
     (15, 0.15, 0.099, 0.0495), (20, 0.2, 0.132, 0.066)],
)
def test_plasma_modifier(level, metal, crystal, deut):
    p = ref.PlasmaTechnology()
    for _ in range(level):
        p.upgrade()
    assert p.modifier.metal == Decimal(str(metal))
    assert p.modifier.crystal == Decimal(str(crystal))
    assert p.modifier.deuterium == Decimal(str(deut))


# ----------------------------------------------------------------------------- Resources (ResourcesTests.cs)
@pytest.mark.parametrize(
    "m,c,d,om,oc,od,expected",
    [(10, 0, 0, 20, 0, 0, False), (0, 10, 0, 0, 20, 0, False), (0, 0, 10, 0, 0, 20, False),
     (10, 0, 0, 10, 0, 0, True), (0, 10, 0, 0, 10, 0, True), (0, 0, 10, 0, 0, 10, True)],
)
def test_resources_can_subtract(m, c, d, om, oc, od, expected):
    assert ref.Resources(m, c, d).can_subtract(ref.Resources(om, oc, od)) is expected


def test_resources_star_operator():
    r = ref.Resources(23, 59, 131).apply_modifier(ref.ResourcesModifier(2, 7, 13))
    assert (r.metal, r.crystal, r.deuterium) == (46, 413, 1703)


# ----------------------------------------------------------------------------- Player (PlayerTests.cs)
@pytest.mark.parametrize(
    "astro_level,planet_count",
    [(0, 1), (1, 2), (2, 2), (3, 3), (13, 8), (14, 8), (22, 12), (23, 13), (30, 16), (31, 17)],
)
def test_player_planet_count(astro_level, planet_count):
    p = ref.Player()
    for _ in range(astro_level):
        p.astrophysics.upgrade()
    assert len(p.planets) == planet_count


def test_player_spend_collapses_to_mse():
    p = ref.Player()
    p.add_resources(ref.Resources(10, 20, 30))  # 140 MSE
    assert p.try_spend_resources(ref.Resources(1, 2, 3)) is True  # 14 MSE
    assert p.resources == ref.Resources(126, 0, 0)


def test_player_exact_spend():
    p = ref.Player()
    p.add_resources(ref.Resources(1, 2, 3))
    assert p.try_spend_resources(ref.Resources(1, 2, 3)) is True
    assert p.resources == ref.Resources(0, 0, 0)


def test_player_overspend_unchanged():
    p = ref.Player()
    p.add_resources(ref.Resources(1, 2, 3))
    assert p.try_spend_resources(ref.Resources(10, 20, 30)) is False
    assert p.resources == ref.Resources(1, 2, 3)


def test_player_gains_production():
    p = ref.Player()
    p.planets[0].metal_mine.upgrade()
    p.proceed_to_next_day()
    assert p.resources.metal == 792


def test_player_gains_modified_production():
    p = ref.Player()
    p.planets[0].metal_mine.upgrade()
    p.plasma_technology.upgrade()
    p.proceed_to_next_day()
    assert p.resources.metal == math.floor(792 * 1.01)
