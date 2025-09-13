import math
from dataclasses import dataclass, field
from typing import List
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")

# Constants for resource weight conversion
RESOURCE_WEIGHTS = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=torch.device("cuda"))

# Utility functions

def convert_to_metal_value(res: torch.Tensor) -> torch.Tensor:
    return (res * RESOURCE_WEIGHTS.to(res.device)).sum()

@dataclass
class Resources:
    values: torch.Tensor

    @staticmethod
    def zeros(device=None):
        if device is None:
            device = torch.device("cuda")
        return Resources(torch.zeros(3, dtype=torch.float64, device=device))

    def clone(self):
        return Resources(self.values.clone())

    def __iadd__(self, other: "Resources"):
        self.values += other.values
        return self

    def __add__(self, other: "Resources") -> "Resources":
        return Resources(self.values + other.values)

    def __sub__(self, other: "Resources") -> "Resources":
        return Resources(self.values - other.values)

    def __mul__(self, modifier: "ResourcesModifier") -> "Resources":
        return Resources(torch.floor(self.values * modifier.values))

    def can_subtract(self, other: "Resources") -> bool:
        return convert_to_metal_value(self.values) >= convert_to_metal_value(other.values)

    def convert_to_metal(self) -> torch.Tensor:
        return convert_to_metal_value(self.values)

@dataclass
class ResourcesModifier:
    values: torch.Tensor

    def __sub__(self, other: "ResourcesModifier") -> "ResourcesModifier":
        return ResourcesModifier(self.values - other.values)

@dataclass
class Mine:
    level: torch.Tensor
    todays_production: Resources
    upgrade_cost: Resources
    upgrade_increase_per_day: Resources
    device: torch.device

    def upgrade(self):
        self.level += 1
        self.todays_production += self.upgrade_increase_per_day
        self.upgrade_cost = self.calculate_upgrade_cost()
        self.upgrade_increase_per_day = self.calculate_upgraded_production() - self.todays_production

    # Methods to override
    def calculate_upgrade_cost(self) -> Resources:
        raise NotImplementedError

    def calculate_upgraded_production(self) -> Resources:
        raise NotImplementedError

@dataclass
class MetalMine(Mine):
    def __init__(self, device):
        level = torch.zeros(1, dtype=torch.float64, device=device)
        base_prod = Resources(torch.tensor([30*24.0, 0.0, 0.0], dtype=torch.float64, device=device))
        super().__init__(level, base_prod, Resources.zeros(device), Resources.zeros(device), device)
        self.upgrade_cost = self.calculate_upgrade_cost()
        self.upgrade_increase_per_day = self.calculate_upgraded_production() - self.todays_production

    def calculate_upgrade_cost(self) -> Resources:
        metal_cost = 60 * torch.pow(1.5, self.level)
        crystal_cost = 15 * torch.pow(1.5, self.level)
        return Resources(torch.floor(torch.tensor([metal_cost, crystal_cost, 0.0], device=self.device, dtype=torch.float64)))

    def calculate_upgraded_production(self) -> Resources:
        next_level = self.level + 1
        prod_per_hour = 30 * next_level * torch.pow(1.1, next_level)
        prod_per_day = torch.round(prod_per_hour) * 24
        return Resources(torch.tensor([prod_per_day, 0.0, 0.0], device=self.device, dtype=torch.float64))

@dataclass
class CrystalMine(Mine):
    def __init__(self, device):
        level = torch.zeros(1, dtype=torch.float64, device=device)
        base_prod = Resources(torch.tensor([0.0, 15*24.0, 0.0], dtype=torch.float64, device=device))
        super().__init__(level, base_prod, Resources.zeros(device), Resources.zeros(device), device)
        self.upgrade_cost = self.calculate_upgrade_cost()
        self.upgrade_increase_per_day = self.calculate_upgraded_production() - self.todays_production

    def calculate_upgrade_cost(self) -> Resources:
        metal_cost = 48 * torch.pow(1.6, self.level)
        crystal_cost = 24 * torch.pow(1.6, self.level)
        return Resources(torch.ceil(torch.tensor([metal_cost, crystal_cost, 0.0], device=self.device, dtype=torch.float64)))

    def calculate_upgraded_production(self) -> Resources:
        next_level = self.level + 1
        prod_per_hour = 20 * next_level * torch.pow(1.1, next_level)
        prod_per_day = torch.floor(prod_per_hour) * 24
        return Resources(torch.tensor([0.0, prod_per_day, 0.0], device=self.device, dtype=torch.float64))

@dataclass
class DeuteriumSynthesizer(Mine):
    planet_max_temperature: int

    def __init__(self, planet_max_temperature: int, device):
        self.planet_max_temperature = planet_max_temperature
        level = torch.zeros(1, dtype=torch.float64, device=device)
        base_prod = Resources.zeros(device)
        super().__init__(level, base_prod, Resources.zeros(device), Resources.zeros(device), device)
        self.upgrade_cost = self.calculate_upgrade_cost()
        self.upgrade_increase_per_day = self.calculate_upgraded_production() - self.todays_production

    def calculate_upgrade_cost(self) -> Resources:
        metal_cost = 225 * torch.pow(1.5, self.level)
        crystal_cost = 75 * torch.pow(1.5, self.level)
        return Resources(torch.round(torch.tensor([metal_cost, crystal_cost, 0.0], device=self.device, dtype=torch.float64)))

    def calculate_upgraded_production(self) -> Resources:
        next_level = self.level + 1
        prod_per_hour = 20 * next_level * torch.pow(1.1, next_level) * (0.68 - 0.002 * (self.planet_max_temperature - 20))
        prod_per_day = torch.floor(prod_per_hour) * 24
        return Resources(torch.tensor([0.0, 0.0, prod_per_day], device=self.device, dtype=torch.float64))

@dataclass
class Astrophysics:
    level: torch.Tensor
    upgrade_cost: Resources
    device: torch.device

    def __init__(self, device):
        self.level = torch.zeros(1, dtype=torch.float64, device=device)
        self.device = device
        self.upgrade_cost = self._calculate_upgrade_cost()

    def _calculate_upgrade_cost(self) -> Resources:
        common_cost = torch.floor(4000 * torch.pow(1.75, self.level))
        cost = torch.tensor([
            common_cost,
            torch.floor(8000 * torch.pow(1.75, self.level)),
            common_cost,
        ], dtype=torch.float64, device=self.device)
        return Resources(cost)

    def upgrade(self):
        self.level += 1
        self.upgrade_cost = self._calculate_upgrade_cost()

@dataclass
class PlasmaTechnology:
    level: torch.Tensor
    modifier: ResourcesModifier
    upgraded_modifier: ResourcesModifier
    upgrade_cost: Resources
    device: torch.device

    def __init__(self, device):
        self.level = torch.zeros(1, dtype=torch.float64, device=device)
        self.device = device
        self.modifier = self._calculate_modifier(self.level)
        self.upgraded_modifier = self._calculate_modifier(self.level + 1)
        self.upgrade_cost = self._calculate_upgrade_cost()

    def _calculate_modifier(self, level: torch.Tensor) -> ResourcesModifier:
        vals = torch.stack([
            level * 1.0 / 100,
            level * 0.66 / 100,
            level * 0.33 / 100,
        ]).squeeze()
        return ResourcesModifier(vals.to(self.device))

    def _calculate_upgrade_cost(self) -> Resources:
        cost = torch.tensor([
            2000 * torch.pow(2, self.level),
            4000 * torch.pow(2, self.level),
            1000 * torch.pow(2, self.level),
        ], dtype=torch.float64, device=self.device)
        return Resources(cost)

    def upgrade(self):
        self.level += 1
        self.upgrade_cost = self._calculate_upgrade_cost()
        self.modifier = self._calculate_modifier(self.level)
        self.upgraded_modifier = self._calculate_modifier(self.level + 1)

@dataclass
class Planet:
    metal_mine: MetalMine
    crystal_mine: CrystalMine
    deuterium_synthesizer: DeuteriumSynthesizer

    def __init__(self, max_temperature: int, device):
        self.metal_mine = MetalMine(device)
        self.crystal_mine = CrystalMine(device)
        self.deuterium_synthesizer = DeuteriumSynthesizer(max_temperature, device)

@dataclass
class Player:
    resources: Resources
    points: torch.Tensor
    day: torch.Tensor
    astrophysics: Astrophysics
    plasma: PlasmaTechnology
    planets: List[Planet]
    max_planets: int
    num_planets: torch.Tensor
    device: torch.device

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda")
        self.device = device
        self.resources = Resources.zeros(device)
        self.points = torch.zeros(1, dtype=torch.float64, device=device)
        self.day = torch.zeros(1, dtype=torch.float64, device=device)
        self.astrophysics = Astrophysics(device)
        self.plasma = PlasmaTechnology(device)
        self.max_planets = 15
        self.planets = [Planet(-115, device) for _ in range(self.max_planets)]
        self.num_planets = torch.ones(1, dtype=torch.int64, device=device)

    def _update_planets(self):
        required = torch.ceil(self.astrophysics.level / 2).to(torch.int64) + 1
        self.num_planets = torch.clamp(required, max=self.max_planets)

    def add_resources(self, resources: Resources):
        self.resources += resources

    def try_spend_resources(self, cost: Resources) -> bool:
        if not self.resources.can_subtract(cost):
            return False
        value = self.resources.convert_to_metal()
        cost_value = cost.convert_to_metal()
        remaining = value - cost_value
        new_vals = torch.zeros(3, dtype=torch.float64, device=self.device)
        new_vals[0] = remaining
        self.resources = Resources(new_vals)
        self.points += cost.values.sum() / 1000.0
        return True

    def proceed_to_next_day(self):
        self.day += 1
        production = self.get_todays_production()
        self.add_resources(production)

    def get_todays_production(self) -> Resources:
        self._update_planets()
        count = int(self.num_planets.item())
        prods = torch.stack([
            planet.metal_mine.todays_production.values
            + planet.crystal_mine.todays_production.values
            + planet.deuterium_synthesizer.todays_production.values
            for planet in self.planets[:count]
        ])
        mine_prod = Resources(prods.sum(dim=0))
        modifier_prod = mine_prod * self.plasma.modifier
        return mine_prod + modifier_prod

# Exploration rewards
_reward_distribution = 5_000_000
_rewards = {}
_max_value = 25.0
_bucket_count = int(300_000_000 / _reward_distribution)
_reward_device = torch.device("cuda")
for i in range(_bucket_count):
    points = i * _reward_distribution
    value = _max_value / _bucket_count * i
    _rewards[points] = {
        "value": torch.tensor(value, dtype=torch.float64, device=_reward_device),
        "redeemed": False,
    }

def get_exploration_reward(player: Player) -> torch.Tensor:
    bucket = math.floor(player.points.item() / _reward_distribution) * _reward_distribution
    reward = _rewards[bucket]
    if reward["redeemed"]:
        return torch.tensor(0.0, dtype=torch.float64, device=player.device)
    reward["redeemed"] = True
    return reward["value"].to(player.device)


def _penalty(player: Player):
    return torch.tensor(-0.1, device=player.device), False


def _try_upgrade(player: Player, upgradable):
    current_points = player.points.clone()
    if player.try_spend_resources(upgradable.upgrade_cost):
        upgradable.upgrade()
        gained_points = (player.points - current_points)[0]
        upgrade_reward = torch.log10(gained_points + 1)
        exploration_reward = get_exploration_reward(player)
        return (upgrade_reward + exploration_reward), False
    return _penalty(player)


def apply_action(player: Player, action: int):
    player._update_planets()

    if action == 0:
        player.proceed_to_next_day()
        return torch.tensor(0.1, device=player.device), False
    if action == 1:
        return _try_upgrade(player, player.astrophysics)
    if action == 2:
        return _try_upgrade(player, player.plasma)

    planet_index = action // 3 - 1
    if planet_index >= int(player.num_planets.item()):
        return _penalty(player)

    mod = action % 3
    if mod == 0:
        return _try_upgrade(player, player.planets[planet_index].metal_mine)
    if mod == 1:
        return _try_upgrade(player, player.planets[planet_index].crystal_mine)
    if mod == 2:
        return _try_upgrade(player, player.planets[planet_index].deuterium_synthesizer)
    raise NotImplementedError


def update_state(player: Player) -> torch.Tensor:
    state = torch.zeros(125, dtype=torch.float64, device=player.device)
    idx = 0

    def add_resources(res: Resources):
        nonlocal idx
        state[idx] = res.convert_to_metal()
        idx += 1

    todays_prod = player.get_todays_production()
    add_resources(player.resources)
    add_resources(todays_prod)
    add_resources(player.astrophysics.upgrade_cost)
    add_resources(player.plasma.upgrade_cost)
    delta_prod = todays_prod * ResourcesModifier(player.plasma.upgraded_modifier.values - player.plasma.modifier.values)
    add_resources(delta_prod)

    player._update_planets()
    count = int(player.num_planets.item())
    for planet in player.planets[:count]:
        add_resources(planet.metal_mine.upgrade_cost)
        add_resources(planet.metal_mine.upgrade_increase_per_day)
        add_resources(planet.crystal_mine.upgrade_cost)
        add_resources(planet.crystal_mine.upgrade_increase_per_day)
        add_resources(planet.deuterium_synthesizer.upgrade_cost)
        add_resources(planet.deuterium_synthesizer.upgrade_increase_per_day)

    return state

def get_player_stats(player: Player):
    metal_levels = torch.tensor(
        [p.metal_mine.level.item() for p in player.planets],
        dtype=torch.float64,
        device=player.device,
    )
    crystal_levels = torch.tensor(
        [p.crystal_mine.level.item() for p in player.planets],
        dtype=torch.float64,
        device=player.device,
    )
    deut_levels = torch.tensor(
        [p.deuterium_synthesizer.level.item() for p in player.planets],
        dtype=torch.float64,
        device=player.device,
    )
    return {
        "MetalMax": metal_levels.max().item(),
        "MetalAverage": metal_levels.mean().item(),
        "MetalMin": metal_levels.min().item(),
        "CrystalMax": crystal_levels.max().item(),
        "CrystalAverage": crystal_levels.mean().item(),
        "CrystalMin": crystal_levels.min().item(),
        "DeutMax": deut_levels.max().item(),
        "DeutAverage": deut_levels.mean().item(),
        "DeutMin": deut_levels.min().item(),
    }

# JIT compile critical functions for improved performance.  Fall back to
# TorchScript if `torch.compile` is unavailable or fails (e.g. missing Triton
# runtime) so importing this module never crashes.
if hasattr(torch, "compile"):
    try:
        apply_action = torch.compile(apply_action, mode="reduce-overhead")
        update_state = torch.compile(update_state, mode="reduce-overhead")
    except Exception:  # pragma: no cover - best-effort compilation
        apply_action = torch.jit.script(apply_action)
        update_state = torch.jit.script(update_state)
else:  # pragma: no cover - older torch versions
    apply_action = torch.jit.script(apply_action)
    update_state = torch.jit.script(update_state)
