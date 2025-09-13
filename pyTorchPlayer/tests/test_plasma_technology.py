import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from ogame_env.foo_torch import PlasmaTechnology


def build(level):
    tech = PlasmaTechnology(torch.device("cuda"))
    for _ in range(level):
        tech.upgrade()
    return tech


@pytest.mark.parametrize(
    "level, metal, crystal, deut",
    [
        (0, 0.0, 0.0, 0.0),
        (1, 0.01, 0.0066, 0.0033),
        (9, 0.09, 0.0594, 0.0297),
        (15, 0.15, 0.099, 0.0495),
        (20, 0.2, 0.132, 0.066),
    ],
)
def test_plasma_modifier(level, metal, crystal, deut):
    tech = build(level)
    vals = tech.modifier.values.tolist()
    assert vals == pytest.approx([metal, crystal, deut])


@pytest.mark.parametrize(
    "level, metal, crystal, deut",
    [
        (0, 0.01, 0.0066, 0.0033),
        (8, 0.09, 0.0594, 0.0297),
        (14, 0.15, 0.099, 0.0495),
        (19, 0.2, 0.132, 0.066),
    ],
)
def test_plasma_upgraded_modifier(level, metal, crystal, deut):
    tech = build(level)
    vals = tech.upgraded_modifier.values.tolist()
    assert vals == pytest.approx([metal, crystal, deut])


@pytest.mark.parametrize(
    "level, metal, crystal, deut",
    [
        (0, 2000, 4000, 1000),
        (1, 4000, 8000, 2000),
        (9, 1024000, 2048000, 512000),
        (15, 65536000, 131072000, 32768000),
        (19, 1048576000, 2097152000, 524288000),
    ],
)
def test_plasma_upgrade_cost(level, metal, crystal, deut):
    tech = build(level)
    cost = tech.upgrade_cost.values.tolist()
    assert cost == [metal, crystal, deut]
