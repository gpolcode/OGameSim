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
        for i, env in enumerate(self.envs):
            s = seed + i if seed is not None else None
            ob, _ = env.reset(seed=s)
            obs.append(ob)
        return torch.stack(obs), {}

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs, rewards, terminations, truncations = [], [], [], []
        final_info: Dict[str, List[Any]] = {}
        for env, action in zip(self.envs, actions):
            ob, rew, term, trunc, info = env.step(action)
            obs.append(ob)
            rewards.append(rew)
            terminations.append(term)
            truncations.append(trunc)
            if "final_info" in info:
                for k, v in info["final_info"].items():
                    final_info.setdefault(k, []).append(v)
        obs_t = torch.stack(obs)
        rewards_t = torch.stack(rewards)
        term_t = torch.tensor(terminations, dtype=torch.bool, device=obs_t.device)
        trunc_t = torch.tensor(truncations, dtype=torch.bool, device=obs_t.device)
        info_dict: Dict[str, Any] = {}
        if final_info:
            info_dict["final_info"] = final_info
        return obs_t, rewards_t, term_t, trunc_t, info_dict
