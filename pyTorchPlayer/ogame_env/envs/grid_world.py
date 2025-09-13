import math
from typing import Optional, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from ..foo_torch import Player, apply_action, update_state, get_player_stats

# Gymnasium expects NumPy observations, but the internal state lives on CUDA.
# The environment therefore converts the device tensor state to a CPU NumPy
# array whenever observations are returned.  All computation remains on the GPU.


class GridWorldEnv(gym.Env[np.ndarray, Union[int, torch.Tensor]]):
    stepCounter = 0
    maxSteps = 8000
    metadata = {
        "render_modes": [],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, device: Optional[torch.device] = None):
        if device is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required")
            device = torch.device("cuda")
        self.device = device
        self.player = Player(self.device)
        self.state = update_state(self.player)

        self.action_space = spaces.Discrete(63)
        # expose float32 observations so Gymnasium can validate without casting
        high = np.full((125,), np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=high, shape=(125,), dtype=np.float32)

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = int(action.item())
        reward, terminated = apply_action(self.player, action)
        self.state = update_state(self.player)

        self.stepCounter += 1
        terminated = terminated or self.stepCounter > self.maxSteps
        infos = {}

        reward_value = reward.item()
        if terminated:
            stats = get_player_stats(self.player)
            infos = {
                "final_info": {
                    "episodic_length": float(self.stepCounter),
                    "points": float(self.player.points.item()),
                    "astrophysics": float(self.player.astrophysics.level.item()),
                    "plasma_technology": float(self.player.plasma.level.item()),
                    "metal_max": float(stats["MetalMax"]),
                    "metal_mean": float(stats["MetalAverage"]),
                    "metal_min": float(stats["MetalMin"]),
                    "crystal_max": float(stats["CrystalMax"]),
                    "crystal_mean": float(stats["CrystalAverage"]),
                    "crystal_min": float(stats["CrystalMin"]),
                    "deut_max": float(stats["DeutMax"]),
                    "deut_mean": float(stats["DeutAverage"]),
                    "deut_min": float(stats["DeutMin"]),
                }
            }
            reward_value = 0.0

        obs = self.state.detach().to("cpu", dtype=torch.float32).numpy()
        return obs, reward_value, terminated, False, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.stepCounter = 0
        self.player = Player(self.device)
        self.state = update_state(self.player)

        obs = self.state.detach().to("cpu", dtype=torch.float32).numpy()
        return obs, {}

    def updateState(self):
        self.state = update_state(self.player)
