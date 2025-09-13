import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from ogame_env.envs.grid_world import GridWorldEnv


def test_env_returns_tensor_and_uses_gpu():
    env = GridWorldEnv()
    obs, _ = env.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.device.type == "cuda"
    assert env.device.type == "cuda"
    step_obs, _, _, _, _ = env.step(0)
    assert isinstance(step_obs, torch.Tensor)
    assert step_obs.device.type == "cuda"
