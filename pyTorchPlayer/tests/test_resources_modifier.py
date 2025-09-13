import torch
from ogame_env.foo_torch import ResourcesModifier


def test_minus_operator_reduces_modifier():
    subject = ResourcesModifier(torch.tensor([23, 59, 131], dtype=torch.float64))
    other = ResourcesModifier(torch.tensor([2, 7, 13], dtype=torch.float64))
    result = subject - other
    assert result.values.tolist() == [21, 52, 118]
