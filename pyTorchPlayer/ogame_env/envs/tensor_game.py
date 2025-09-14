"""TorchRL tensor environment and GPU planning helpers.

The environment stores all state in PyTorch tensors so it can live entirely on
GPU memory. To scale to many colonies, keep a batch dimension in ``state`` or
use the :func:`batch_plan` helper to pick upgrades for multiple colonies at
once.
"""

import torch
from torchrl.data import DiscreteTensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs.common import EnvBase
from typing import Optional


class TensorGameEnv(EnvBase):
    """Small tensor-based reimplementation of a resource game.

    The environment tracks three resources (metal, crystal, deuterium) and their
    production rates. Actions allow the agent to skip a day or upgrade one of the
    mines if enough resources are available. Upgrades increase the production of
    the corresponding resource. All state and updates are handled with PyTorch
    tensors so the logic can run on CPU or GPU.

    Example
    -------
    Below is a minimal sketch of how upgrade decisions could be computed purely
    with tensor ops on the chosen device. The snippet selects the affordable mine
    with the best payback ratio (lower is better)::

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        env = TensorGameEnv(device=device)
        costs = torch.tensor([100.0, 100.0, 100.0], device=device)
        resources = env.state[:3]
        production = env.state[3:]

        payback = costs / production                       # cost per extra unit
        affordable = resources >= costs                    # which upgrades we can buy
        masked = torch.where(affordable, payback, torch.tensor(float("inf"), device=device))
        action = masked.argmin() + 1                       # 1..3 -> upgrade ids

    This strategy runs entirely on the GPU when ``device="cuda"`` and can be
    extended to more elaborate heuristics (priority weights, probabilistic
    sampling, batch environments, etc.).
    """

    def __init__(self, max_days: int = 100, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.max_days = max_days
        # state = [metal, crystal, deut, metal_prod, crystal_prod, deut_prod]
        self.state = torch.zeros(6, device=self.device)

        self.observation_spec = UnboundedContinuousTensorSpec(shape=torch.Size([6]))
        self.action_spec = DiscreteTensorSpec(4)  # 0: wait, 1-3: upgrade mines
        self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size([]))
        self.done_spec = DiscreteTensorSpec(2, dtype=torch.bool)

    def _reset(self, tensordict):
        self.day = 0
        self.state.zero_()
        # base production per resource
        self.state[3:] = torch.tensor([1.0, 1.0, 1.0], device=self.device)

        tensordict.set("observation", self.state.clone())
        tensordict.set("reward", torch.zeros((), dtype=torch.float32, device=self.device))
        tensordict.set("done", torch.zeros((), dtype=torch.bool, device=self.device))
        return tensordict

    def _step(self, tensordict):
        action = int(tensordict.get("action").item())
        cost = torch.tensor(100.0, device=self.device)
        resources = self.state[:3]
        production = self.state[3:]
        reward = torch.zeros((), device=self.device)

        if action == 1:  # upgrade metal mine
            can = resources[0] >= cost
            resources[0] = torch.where(can, resources[0] - cost, resources[0])
            production[0] = torch.where(can, production[0] + 1.0, production[0])
            reward = torch.where(can, reward, torch.tensor(-0.1, device=self.device))
        elif action == 2:  # upgrade crystal mine
            can = resources[1] >= cost
            resources[1] = torch.where(can, resources[1] - cost, resources[1])
            production[1] = torch.where(can, production[1] + 1.0, production[1])
            reward = torch.where(can, reward, torch.tensor(-0.1, device=self.device))
        elif action == 3:  # upgrade deuterium synthesizer
            can = resources[2] >= cost
            resources[2] = torch.where(can, resources[2] - cost, resources[2])
            production[2] = torch.where(can, production[2] + 1.0, production[2])
            reward = torch.where(can, reward, torch.tensor(-0.1, device=self.device))

        # end of day production
        resources += production
        self.day += 1
        reward = reward + resources.sum()
        done = self.day >= self.max_days

        tensordict.set("observation", self.state.clone())
        tensordict.set("reward", reward.to(torch.float32))
        tensordict.set("done", torch.tensor(done, dtype=torch.bool, device=self.device))
        return tensordict

    def _set_seed(self, seed: Optional[int]) -> Optional[int]:
        return seed


# ---------------------------------------------------------------------------
# Example helper functions
# ---------------------------------------------------------------------------

def select_best_upgrade(resources: torch.Tensor, production: torch.Tensor, costs: torch.Tensor) -> int:
    """Pick the upgrade with the highest immediate return.

    Parameters
    ----------
    resources: tensor of shape ``(3,)``
        Current stored resources.
    production: tensor of shape ``(3,)``
        Production per tick for each resource.
    costs: tensor of shape ``(3,)``
        Cost for the next upgrade of each mine.

    Returns
    -------
    int
        Index ``0..2`` for metal, crystal or deuterium.

    This function is device agnostic – if the inputs live on the GPU the whole
    computation stays there. It prefers upgrades with the smallest payback time
    ``cost / production`` while masking out mines we cannot currently afford.
    """

    payback = costs / production
    affordable = resources >= costs
    masked = torch.where(affordable, payback, torch.tensor(float("inf"), device=resources.device))
    return int(torch.argmin(masked))


def batch_plan(resources: torch.Tensor, production: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    """Example of vectorising decisions for multiple colonies at once.

    ``resources``/``production``/``costs`` are expected to have shape ``(N, 3)``
    where ``N`` is the number of parallel colonies. The function returns a tensor
    of length ``N`` containing the selected upgrade index for each colony.
    """

    payback = costs / production
    affordable = resources >= costs
    masked = torch.where(affordable, payback, torch.tensor(float("inf"), device=resources.device))
    return masked.argmin(dim=1)

