import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from ogame_env.foo_torch import ResourcesModifier


def test_minus_operator_reduces_modifier():
    subject = ResourcesModifier(torch.tensor([23, 59, 131], dtype=torch.float64, device=torch.device("cuda")))
    other = ResourcesModifier(torch.tensor([2, 7, 13], dtype=torch.float64, device=torch.device("cuda")))
    result = subject - other
    assert result.values.tolist() == [21, 52, 118]
