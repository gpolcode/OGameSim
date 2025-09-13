import pytest
import torch
from ogame_env.foo_torch import MetalMine


def build_mine(level):
    mine = MetalMine(torch.device("cpu"))
    for _ in range(level):
        mine.upgrade()
    return mine


@pytest.mark.parametrize(
    "level, expected", [(0, 720), (1, 792), (10, 18672), (30, 376896), (50, 4226064)]
)
def test_metal_mine_production(level, expected):
    mine = build_mine(level)
    prod = mine.todays_production.values.tolist()
    assert prod == [expected, 0.0, 0.0]


@pytest.mark.parametrize(
    "level, metal, crystal, deut",
    [
        (0, 60, 15, 0),
        (9, 2306, 576, 0),
        (29, 7670042, 1917510, 0),
        (49, 25504860008, 6376215002, 0),
    ],
)
def test_metal_mine_upgrade_cost(level, metal, crystal, deut):
    mine = build_mine(level)
    cost = mine.upgrade_cost.values.tolist()
    assert cost == [metal, crystal, deut]
