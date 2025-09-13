import pytest
import torch
from ogame_env.foo_torch import CrystalMine


def build_mine(level):
    mine = CrystalMine(torch.device("cpu"))
    for _ in range(level):
        mine.upgrade()
    return mine


@pytest.mark.parametrize(
    "level, expected",
    [(0, 360), (1, 528), (10, 12432), (20, 64560), (34, 416928)],
)
def test_crystal_mine_production(level, expected):
    mine = build_mine(level)
    prod = mine.todays_production.values.tolist()
    assert prod == [0.0, expected, 0.0]


@pytest.mark.parametrize(
    "level, metal, crystal, deut",
    [
        (0, 48, 24, 0),
        (9, 3299, 1650, 0),
        (19, 362678, 181339, 0),
        (33, 261336858, 130668429, 0),
    ],
)
def test_crystal_mine_upgrade_cost(level, metal, crystal, deut):
    mine = build_mine(level)
    cost = mine.upgrade_cost.values.tolist()
    assert cost == [metal, crystal, deut]
