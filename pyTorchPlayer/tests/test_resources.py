import pytest
import torch

from ogame_env.foo_torch import Resources, ResourcesModifier


def make_resources(metal, crystal, deuterium):
    return Resources(torch.tensor([metal, crystal, deuterium], dtype=torch.float64))


@pytest.mark.parametrize(
    "metal, crystal, deut, o_metal, o_crystal, o_deut",
    [
        (10, 0, 0, 20, 0, 0),
        (0, 10, 0, 0, 20, 0),
        (0, 0, 10, 0, 0, 20),
    ],
)
def test_can_subtract_false(metal, crystal, deut, o_metal, o_crystal, o_deut):
    subject = make_resources(metal, crystal, deut)
    other = make_resources(o_metal, o_crystal, o_deut)
    assert not subject.can_subtract(other)


@pytest.mark.parametrize(
    "metal, crystal, deut, o_metal, o_crystal, o_deut",
    [
        (10, 0, 0, 10, 0, 0),
        (0, 10, 0, 0, 10, 0),
        (0, 0, 10, 0, 0, 10),
    ],
)
def test_can_subtract_true(metal, crystal, deut, o_metal, o_crystal, o_deut):
    subject = make_resources(metal, crystal, deut)
    other = make_resources(o_metal, o_crystal, o_deut)
    assert subject.can_subtract(other)


def test_minus_operator_removes_resources():
    subject = make_resources(23, 59, 131)
    other = make_resources(2, 7, 13)
    result = subject - other
    assert result.values.tolist() == [21, 52, 118]


def test_plus_operator_adds_resources():
    subject = make_resources(23, 59, 131)
    other = make_resources(2, 7, 13)
    result = subject + other
    assert result.values.tolist() == [25, 66, 144]


def test_mul_operator_multiplies_resources():
    subject = make_resources(23, 59, 131)
    modifier = ResourcesModifier(torch.tensor([2, 7, 13], dtype=torch.float64))
    result = subject * modifier
    assert result.values.tolist() == [46, 413, 1703]
