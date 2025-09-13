import pytest
import torch
from ogame_env.foo_torch import DeuteriumSynthesizer


def build_synth(level, temp):
    synth = DeuteriumSynthesizer(temp, torch.device("cpu"))
    for _ in range(level):
        synth.upgrade()
    return synth


@pytest.mark.parametrize(
    "level, expected, temp",
    [
        (0, 0, 120),
        (0, 0, 0),
        (0, 0, -120),
        (1, 240, 120),
        (1, 360, 0),
        (1, 504, -120),
        (10, 5952, 120),
        (10, 8952, 0),
        (10, 11928, -120),
        (20, 30984, 120),
        (20, 46488, 0),
        (20, 61992, -120),
        (42, 529920, 120),
        (42, 794904, 0),
        (42, 1059864, -120),
    ],
)
def test_deuterium_production(level, expected, temp):
    synth = build_synth(level, temp)
    prod = synth.todays_production.values.tolist()
    assert prod == [0.0, 0.0, expected]


@pytest.mark.parametrize(
    "level, metal, crystal, deut",
    [
        (0, 225, 75, 0),
        (9, 8650, 2883, 0),
        (19, 498789, 166263, 0),
        (41, 3731849658, 1243949886, 0),
    ],
)
def test_deuterium_upgrade_cost(level, metal, crystal, deut):
    synth = build_synth(level, 0)
    cost = synth.upgrade_cost.values.tolist()
    assert cost == [metal, crystal, deut]
