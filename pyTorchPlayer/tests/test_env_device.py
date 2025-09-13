import numpy as np
import torch
from ogame_env.envs.grid_world import GridWorldEnv


def test_env_returns_numpy_and_uses_device():
    env = GridWorldEnv()
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert env.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    step_obs, _, _, _, _ = env.step(0)
    assert isinstance(step_obs, np.ndarray)
