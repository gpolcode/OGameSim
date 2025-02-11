import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from pythonnet import load

load("coreclr", runtime_config="/home/elsahr/ogamesim/pyTorchPlayer/game/runtimeconfig.json")

import clr
clr.AddReference("/home/elsahr/ogamesim/pyTorchPlayer/game/Game.dll")

from OGameSim.Entities import *
from OGameSim.Production import *

class GridWorldEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    stepCounter = 0
    maxSteps = 8000
    metadata = {
        "render_modes": [],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.player = Player()
        self.action_space = spaces.Discrete(63)
        self.observation_space = spaces.Box(low=0.0, high=np.full((617,), np.inf), shape=(617,), dtype=np.float64)

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        reward = Foo.ApplyAction(self.player, action)
        Foo.UpdateState(self.player, self.state)

        self.stepCounter += 1

        terminated = self.stepCounter > self.maxSteps

        if terminated:
            reward = 0.0

        return self.state, reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.player = Player()
        self.stepCounter = 0
        self.state = np.zeros(617)
        Foo.UpdateState(self.player, self.state)

        return self.state, {}
