import torch

from state import OGameBatch

def test_apply_upgrades():
    # ARRANGE
    initial_states = OGameBatch(1)    

    # ACT
    initial_states.metal += 100

    # ASSERT
    expected = torch.tensor([100], dtype=torch.float32)
    torch.testing.assert_close(initial_states.metal, expected)

