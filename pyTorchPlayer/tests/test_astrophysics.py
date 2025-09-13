import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from ogame_env.foo_torch import Astrophysics


def build(level):
    tech = Astrophysics(torch.device("cuda"))
    for _ in range(level):
        tech.upgrade()
    return tech


@pytest.mark.parametrize(
    "level, metal, crystal, deut",
    [
        (0, 4000, 8000, 4000),
        (1, 7000, 14000, 7000),
        (2, 12250, 24500, 12250),
        (3, 21437, 42875, 21437),
        (8, 351855, 703711, 351855),
        (14, 10106311, 20212622, 10106311),
        (24, 2722533045, 5445066090, 2722533045),
        (30, 78199045470, 156398090941, 78199045470),
    ],
)

def test_astrophysics_upgrade_cost(level, metal, crystal, deut):
    tech = build(level)
    cost = tech.upgrade_cost.values.tolist()
    assert cost == [metal, crystal, deut]
