import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from ogame_env.envs.grid_world import GridWorldEnv


def test_env_returns_tensors_and_uses_gpu():
    env = GridWorldEnv()
    obs, _ = env.reset()
    assert torch.is_tensor(obs)
    assert obs.device.type == "cuda"
    step_obs, _, _, _, _ = env.step(torch.tensor(0, device=obs.device))
    assert torch.is_tensor(step_obs)
    assert step_obs.device.type == "cuda"
