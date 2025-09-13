import torch
from typing import Callable, List, Tuple, Dict, Any
import gymnasium as gym


class TorchVectorEnv:
    """Simple vector environment that operates on CUDA tensors."""

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        first_env = self.envs[0]
        self.single_observation_space = first_env.observation_space
        self.single_action_space = first_env.action_space

    def reset(self, seed: int | None = None):
        obs = []
        infos = []
        for i, env in enumerate(self.envs):
            s = seed + i if seed is not None else None
            ob, info = env.reset(seed=s)
            obs.append(ob)
            infos.append(info)
        return torch.stack(obs), infos

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs, rewards, terminations, truncations, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            ob, rew, term, trunc, info = env.step(action)
            obs.append(ob)
            rewards.append(rew)
            terminations.append(term)
            truncations.append(trunc)
            infos.append(info)
        obs_t = torch.stack(obs)
        rewards_t = torch.stack(rewards)
        term_t = torch.tensor(terminations, dtype=torch.bool, device=obs_t.device)
        trunc_t = torch.tensor(truncations, dtype=torch.bool, device=obs_t.device)
        return obs_t, rewards_t, term_t, trunc_t, infos
