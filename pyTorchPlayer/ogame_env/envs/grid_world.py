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

import ctypes
from System import IntPtr

class GridWorldEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    stepCounter = 0
    maxSteps = 8000
    metadata = {
        "render_modes": [],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.player = Player()
        self.state = np.zeros(617)
        self.updateState()

        self.action_space = spaces.Discrete(63)
        self.observation_space = spaces.Box(low=0.0, high=np.full((617,), np.inf), shape=(617,), dtype=np.float64)

    def step(self, action):
        reward = Foo.ApplyAction(self.player, action.item())
        self.updateState()

        self.stepCounter += 1

        terminated = self.stepCounter > self.maxSteps
        infos = {}

        if terminated:
            infos = {
                "final_info" : [
                    {
                        "episode" : {
                            "r" : self.player.Points,
                            "l" : self.stepCounter
                        }
                    }
                ]
            }
            reward = 0.0

        return self.state, reward, terminated, False, infos

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.stepCounter = 0
        self.player = Player()
        self.state.fill(0)
        self.updateState()

        return self.state, {}

    def updateState(self):
        # Convert to .NET IntPtr
        ptr = self.state.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        net_ptr = IntPtr(ctypes.addressof(ptr.contents))

        Foo.UpdateState(self.player, net_ptr)
