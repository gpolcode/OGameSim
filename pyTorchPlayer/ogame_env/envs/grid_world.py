import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

from pythonnet import load

load("coreclr", runtime_config="./pyTorchPlayer/runtimeconfig.json")

import clr
clr.AddReference("C:/Code/gpolcode/OGameSim/Game/bin/Release/net8.0/publish/Game")

from OGameSim.Entities import *
from OGameSim.Production import *

import ctypes
from System import IntPtr, Array, Int64, Double

class GridWorldEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    stepCounter = 0
    maxSteps = 8000
    metadata = {
        "render_modes": [],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.player = Player()
        self.state = np.zeros(125)
        self.updateState()

        self.action_space = spaces.Discrete(63)
        self.observation_space = spaces.Box(low=0.0, high=np.full((125,), np.inf), shape=(125,), dtype=np.float64)

    def step(self, action):
        # Prepare arrays for the batch API
        players = Array[Player]([self.player])
        actions = Array[Int64]([int(np.asarray(action))])
        state_buffer = Array[Double](self.state.tolist())

        result = Foo.StepBatch(players, actions, state_buffer)

        # copy updated state back to numpy array
        self.state[:] = list(state_buffer)
        reward = float(result.Item1[0])
        terminated = bool(result.Item2[0])

        self.stepCounter += 1

        terminated = terminated or self.stepCounter > self.maxSteps
        infos = {}

        if terminated:
            stats = Foo.GetPlayerStats(self.player)
            infos = {
                "final_info": {
                    "episodic_length": float(self.stepCounter),
                    "points" : float(self.player.Points.ToString()),
                    "astrophysics" : float(self.player.Astrophysics.Level),
                    "plasma_technology" : float(self.player.PlasmaTechnology.Level),
                    "metal_max" : float(stats.MetalMax),
                    "metal_mean" : float(stats.MetalAverage),
                    "metal_min" : float(stats.MetalMin),
                    "crystal_max" : float(stats.CrystalMax),
                    "crystal_mean" : float(stats.CrystalAverage),
                    "crystal_min" : float(stats.CrystalMin),
                    "deut_max" : float(stats.DeutMax),
                    "deut_mean" : float(stats.DeutAverage),
                    "deut_min" : float(stats.DeutMin)
                }
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
